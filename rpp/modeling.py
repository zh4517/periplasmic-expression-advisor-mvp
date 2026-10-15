"""Fitters, cross-fitted predictions and the scores reported for them.

Everything that looks at a label is reachable from one of two paths: inside a training fold
(`selection="nested"`) or on the whole table before any split exists (`selection="leaked"`).
Keeping both is deliberate - the difference between them is the size of the mistake the published
feature-selection step makes, and it is a number worth reporting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import matthews_corrcoef
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rpp.cv import iter_splits
from rpp.dataset import CLASS_EDGES, YIELD_OFFSET

MODELS = ("ridge", "rf")
SELECTORS = {"anova": f_regression, "mi": mutual_info_regression}
RIDGE_ALPHAS = np.logspace(-2, 4, 25)


class ConstantPredictor(BaseEstimator, RegressorMixin):
    """Median of the training labels.  The bar every model here has to clear."""

    def fit(self, X, y):
        self.value_ = float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.value_)


def build_pipeline(
    model: str,
    k: int | None,
    selector: str = "anova",
    seed: int = 0,
    forced: int = 0,
) -> Pipeline:
    """Feature selection, then learner.  Called only with training-fold rows.

    ``forced`` keeps the first that many columns whatever the scores say - used so the four process
    conditions can be compared with and without, rather than being out-shouted by 1144 sequence
    columns that never mention them.
    """
    steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
    if k:
        score_func = SELECTORS[selector]
        if score_func is mutual_info_regression:
            # Stochastic estimator; pinned so two runs of the same experiment print the same table.
            from functools import partial

            score_func = partial(score_func, random_state=seed)
        if forced:
            from sklearn.compose import ColumnTransformer

            selector_model = ColumnTransformer(
                [
                    ("forced", "passthrough", slice(0, forced)),
                    ("rest", SelectKBest(score_func=score_func, k=k), slice(forced, None)),
                ],
                # Dense output; sparse_threshold=0 is what this sklearn version offers for that.
                sparse_threshold=0,
            )
        else:
            selector_model = SelectKBest(score_func=score_func, k=k)
        steps.append(("select", selector_model))
    steps.append(("scale", StandardScaler()))

    if model == "constant":
        steps.append(("learn", ConstantPredictor()))
    elif model == "ridge":
        steps.append(("learn", RidgeCV(alphas=RIDGE_ALPHAS)))
    elif model == "rf":
        steps.append(
            (
                "learn",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    random_state=seed,
                    n_jobs=-1,
                ),
            )
        )
    else:
        raise ValueError(f"unknown model {model!r}")

    return Pipeline(steps)


@dataclass
class Run:
    """One cross-fitted prediction vector plus how it was produced."""

    predictions: np.ndarray
    strategy: str
    model: str
    k: int | None
    selection: str
    feature_set: str
    repeat: int


def cross_fit(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    strategy: str,
    model: str,
    k: int | None = 20,
    selector: str = "anova",
    selection: str = "nested",
    n_splits: int = 5,
    seed: int = 0,
    forced: int = 0,
) -> np.ndarray:
    """One prediction per row, always made by a model that never saw that row.

    With ``selection="leaked"`` the columns handed in must already have been chosen using every
    label in the table, including these test rows; that is the point of the variant.
    """
    if selection not in ("nested", "leaked"):
        raise ValueError(f"unknown selection mode {selection!r}")
    if selection == "nested" and not feature_cols:
        raise ValueError("no feature columns requested")

    X = df[feature_cols].to_numpy(dtype=float)
    y = df["log10_yield"].to_numpy(dtype=float)
    predictions = np.full(len(df), np.nan)

    for train, test in iter_splits(df, strategy, n_splits=n_splits, seed=seed):
        pipeline = build_pipeline(
            model, k if selection == "nested" else None, selector, seed, forced
        )
        pipeline.fit(X[train], y[train])
        predictions[test] = pipeline.predict(X[test])

    uncovered = np.isnan(predictions)
    if uncovered.any():
        raise RuntimeError(
            f"{int(uncovered.sum())} rows were never in a test fold under {strategy} splits"
        )
    return predictions


def leaked_columns(
    df: pd.DataFrame,
    feature_cols: list[str],
    k: int,
    selector: str = "anova",
    seed: int = 0,
) -> list[str]:
    """Rank every column against every label, then hand the top k back to be cross-validated.

    This is the sequence a notebook falls into: select, then split, then report.  The selection has
    already used the rows it later claims to hold out.
    """
    X = df[feature_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=float(np.nanmedian(X)))
    y = df["log10_yield"].to_numpy(dtype=float)
    score_func = SELECTORS[selector]
    if score_func is mutual_info_regression:
        from functools import partial

        score_func = partial(score_func, random_state=seed)
    selector_model = SelectKBest(score_func=score_func, k=min(k, len(feature_cols)))
    selector_model.fit(X, y)
    keep = selector_model.get_support()
    return [name for name, flag in zip(feature_cols, keep) if flag]


def to_mg_per_l(log_predictions: np.ndarray) -> np.ndarray:
    # Rounding kills the float wobble in 10**log10(x), which otherwise puts a reported 50.0 mg/L
    # just below the medium/high edge and labels a high-yield row as medium.
    return np.round(np.maximum(10.0**np.asarray(log_predictions, dtype=float) - YIELD_OFFSET, 0.0), 6)


def classes_from_log(log_values: np.ndarray) -> np.ndarray:
    return np.select(
        [to_mg_per_l(log_values) < CLASS_EDGES[0], to_mg_per_l(log_values) < CLASS_EDGES[1]],
        ["L", "M"],
        default="H",
    )


def score(df: pd.DataFrame, predictions: np.ndarray, top_k: int = 10) -> dict[str, float]:
    """Cross-fitted scores.  Errors are on the log scale, so they read as multiples, not mg/L."""
    y = df["log10_yield"].to_numpy(dtype=float)
    truth_class = df["expression_class"].to_numpy(dtype=str)
    delta = np.abs(predictions - y)

    order = np.argsort(-predictions)
    top = order[:top_k]
    high = truth_class == "H"

    predicted_class = classes_from_log(predictions)
    return {
        "n": float(len(df)),
        "pearson_r_log": float(pearsonr(y, predictions)[0]),
        "spearman_rho_log": float(spearmanr(y, predictions)[0]),
        "median_fold_error": float(10.0 ** np.median(delta)),
        "worst_fold_error": float(10.0 ** np.quantile(delta, 0.95)),
        "class_accuracy": float((predicted_class == truth_class).mean()),
        "mcc": float(matthews_corrcoef(truth_class, predicted_class)),
        f"high_share_in_top{top_k}": float(high[top].mean()),
        f"high_recall_at_top{top_k}": float(np.isin(np.where(high)[0], top).mean()),
        "high_base_rate": float(high.mean()),
    }
