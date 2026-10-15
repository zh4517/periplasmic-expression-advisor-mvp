"""The four evidence tables this project reports.

1. **benchmark** - what the model can actually do, under splits that hold a whole construct out.
2. **leakage** - the same runs with feature selection moved in front of the split, which is what a
   careless notebook does.  The gap is the cost of that mistake, stated as a number.
3. **ablation** - Tier A (counts) against Tier B (weighted sums of those counts).  Tier B cannot
   contain information Tier A lacks, so any gain is prior knowledge acting as a regulariser, and
   any loss says a thousand-scale column list is a liability at this sample size.
4. **conditions** - do the four process conditions buy anything, on the only rows where the
   published table states them completely.
"""

from __future__ import annotations

import pandas as pd

from rpp.cv import iter_splits
from rpp.dataset import CONDITION_COLUMNS, feature_columns
from rpp.modeling import cross_fit, leaked_columns, score

FEATURE_SETS = {"A": ("A",), "B": ("B",), "A+B": ("A", "B")}
METRIC_COLUMNS = [
    "pearson_r_log",
    "spearman_rho_log",
    "median_fold_error",
    "class_accuracy",
    "mcc",
    "high_share_in_top10",
    "high_recall_at_top10",
]


def _run(
    df: pd.DataFrame,
    *,
    experiment: str,
    strategy: str,
    feature_set: str,
    model: str,
    k: int | None,
    selection: str = "nested",
    selector: str = "anova",
    repeat: int = 0,
    tiers: tuple[str, ...] = ("A", "B"),
) -> dict:
    columns = feature_columns(df, tiers=tiers)
    forced = 0
    if feature_set == "conditions":
        condition_columns = [f"{c}_num" for c in CONDITION_COLUMNS]
        # First in the matrix so build_pipeline can pin them into every fold; they are stated
        # measurements, not label-derived choices, so holding them is not a leak.
        columns = condition_columns + columns
        forced = len(condition_columns)
    if not columns:
        raise ValueError(f"no columns for feature set {feature_set}")
    if k and k >= len(columns):
        # Asking for more columns than exist means "use them all"; capping here keeps sklearn from
        # warning once per fold about a configuration that is not a mistake.
        k = None

    if selection == "leaked" and k:
        # Columns ranked against every label in the table, then cross-validated as if they were not.
        columns = leaked_columns(df, columns, k, selector, seed=repeat)
        forced = 0

    predictions = cross_fit(
        df,
        columns,
        strategy=strategy,
        model=model,
        k=k if selection == "nested" else None,
        selector=selector,
        selection=selection,
        seed=repeat,
        forced=forced,
    )

    row = {
        "experiment": experiment,
        "strategy": strategy,
        "feature_set": feature_set,
        "model": model,
        "k": k if k else "none",
        "selection": selection,
        "n_features": len(columns),
        "repeat": float(repeat),
        "n_rows": float(len(df)),
    }
    row.update(score(df, predictions))
    return row


def _expand(
    df: pd.DataFrame,
    experiment: str,
    combos: list[dict],
    strategies: list[str],
    repeats: int,
) -> list[dict]:
    rows = []
    for strategy in strategies:
        for combo in combos:
            # Only the random split has fold randomness; grouped folds are deterministic.
            for repeat in range(repeats if strategy == "random" else 1):
                rows.append(
                    _run(
                        df,
                        experiment=experiment,
                        strategy=strategy,
                        repeat=repeat,
                        **combo,
                    )
                )
    return rows


def benchmark(df: pd.DataFrame, selector: str, repeats: int) -> list[dict]:
    combos = [
        {"feature_set": "A+B", "model": model, "k": k, "selector": selector}
        for model in ("ridge", "rf")
        for k in (5, 20, 60)
    ]
    # Median-of-training-labels: the bar the feature machinery has to clear.
    combos.append({"feature_set": "A+B", "model": "constant", "k": None, "selector": selector})
    return _expand(df, "benchmark", combos, ["random", "combo", "protein", "signal_peptide"], repeats)


def leakage(df: pd.DataFrame, selector: str, repeats: int) -> list[dict]:
    combos = [
        {
            "feature_set": "A+B",
            "model": model,
            "k": k,
            "selection": selection,
            "selector": selector,
        }
        for model in ("ridge", "rf")
        for k in (5, 20)
        for selection in ("nested", "leaked")
    ]
    return _expand(df, "leakage", combos, ["random", "protein", "signal_peptide"], repeats)


def ablation(df: pd.DataFrame, selector: str, repeats: int) -> list[dict]:
    rows = []
    for feature_set in ("A", "B", "A+B"):
        combos = [
            {
                "feature_set": feature_set,
                "model": model,
                "k": k,
                "selector": selector,
                "tiers": FEATURE_SETS[feature_set],
            }
            for model in ("ridge", "rf")
            for k in (5, 20, 60)
        ]
        rows += _expand(df, "ablation", combos, ["protein", "signal_peptide"], repeats)
    return rows


def conditions(df: pd.DataFrame, selector: str, repeats: int) -> list[dict]:
    subset = df[df["conditions_numeric"]].reset_index(drop=True)
    if subset[[f"{c}_num" for c in CONDITION_COLUMNS]].isna().any().any():
        raise AssertionError("rows flagged fully specified still carry a span in some condition")
    combos = [
        {"feature_set": feature_set, "model": model, "k": k, "selector": selector}
        for feature_set in ("A+B", "conditions")
        for model in ("ridge", "rf")
        for k in (5, 20)
    ]
    combos.append({"feature_set": "A+B", "model": "constant", "k": None, "selector": selector})
    return _expand(subset, "conditions", combos, ["protein"], repeats)


EXPERIMENTS = (benchmark, leakage, ablation, conditions)


def run_all(df: pd.DataFrame, selector: str = "anova", quick: bool = False) -> pd.DataFrame:
    repeats = 1 if quick else 3
    rows = [row for experiment in EXPERIMENTS for row in experiment(df, selector, repeats)]
    results = pd.DataFrame(rows)
    results.attrs["fold_sizes"] = {
        strategy: [len(test) for _, test in iter_splits(df, strategy)]
        for strategy in ("random", "combo", "protein", "signal_peptide")
    }
    return results


def summarise(results: pd.DataFrame, experiment: str) -> pd.DataFrame:
    """Average over repeats and keep the columns that go on a slide."""
    part = results[results["experiment"] == experiment]
    keys = ["strategy", "feature_set", "model", "k", "selection", "n_features", "n_rows"]
    metrics = [c for c in METRIC_COLUMNS if c in part.columns]
    means = part.groupby(keys, as_index=False)[metrics].mean()
    spread = (
        part.groupby(keys)[metrics]
        .agg(lambda s: s.max() - s.min())
        .rename(columns={c: f"{c}_over_repeats" for c in metrics})
    )
    merged = means.merge(spread.reset_index(), on=keys)
    return merged.round(3).sort_values(["strategy", "model", "feature_set", "selection", "k"])
