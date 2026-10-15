"""The five evidence tables this project reports.

1. **benchmark** - what the model can actually do, under splits that hold a whole construct out.
2. **leakage** - the same runs with feature selection moved in front of the split, which is what a
   careless notebook does.  The gap is the cost of that mistake, stated as a number.
3. **ablation** - Tier A (counts) against Tier B (weighted sums of those counts).  Tier B cannot
   contain information Tier A lacks, so any gain is prior knowledge acting as a regulariser, and
   any loss says a thousand-scale column list is a liability at this sample size.
4. **conditions** - do the four process conditions buy anything, on the only rows where the
   published table states them completely.
5. **signal location** - how much of the columns can be *populated in only a few rows* before the
   model stops working.  This is the one that decides whether the shortlist is a finding or an
   artifact of a handful of constructs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rpp.cv import iter_splits
from rpp.dataset import CONDITION_COLUMNS, feature_columns
from rpp.modeling import cross_fit, leaked_columns, score

FEATURE_SETS = {"A": ("A",), "B": ("B",), "A+B": ("A", "B")}
SPARSITY_FLOORS = (0.0, 0.05, 0.1, 0.2, 0.5)
METRIC_COLUMNS = [
    "pearson_r_log",
    "spearman_rho_log",
    "median_fold_error",
    "class_accuracy",
    "mcc",
    "high_share_in_top10",
    "high_recall_at_top10",
    "distinct_predictions",
    "modal_share",
]


def populated_columns(df: pd.DataFrame, columns: list[str], floor: float) -> list[str]:
    """Columns carrying a nonzero value in at least ``floor`` of the rows.

    A dipeptide count that is zero in 101 of 103 rows can still win an ANOVA contest, because the
    two rows where it appears are extreme.  Selection loves exactly that, and the label it hands the
    learner is "these two proteins", not "this motif".
    """
    if floor <= 0:
        return columns
    X = df[columns].to_numpy(dtype=float)
    share = (np.nan_to_num(X, nan=0.0) != 0).mean(axis=0)
    return [name for name, s in zip(columns, share) if s >= floor]


def _is_dipeptide_column(name: str) -> bool:
    """True for the 400 occurrence and 400 frequency dipeptide columns, in any block.

    Names are ``OccD_PM``/``OFD_PM`` for the fused chain and ``sp_OccD_PM``/``target_OFD_PM`` for the
    two halves, so the block prefix has to come off before the feature family is visible.
    """
    base = name
    for block in ("sp_", "target_"):
        if base.startswith(block):
            base = base[len(block) :]
    return base.startswith(("OccD_", "OFD_"))


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
    floor: float = 0.0,
) -> dict:
    columns = feature_columns(df, tiers=tiers)
    forced = 0
    if feature_set == "conditions":
        condition_columns = [f"{c}_num" for c in CONDITION_COLUMNS]
        # First in the matrix so build_pipeline can pin them into every fold; they are stated
        # measurements, not label-derived choices, so holding them is not a leak.
        columns = condition_columns + columns
        forced = len(condition_columns)
    columns = populated_columns(df, columns, floor)
    if feature_set == "no dipeptides":
        columns = [c for c in columns if not _is_dipeptide_column(c)]
    if not columns:
        raise ValueError(f"no columns for feature set {feature_set} at floor {floor}")
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
        "selector": selector,
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


def signal_location(df: pd.DataFrame, selector: str, repeats: int) -> list[dict]:
    """Sweep how empty a column may be before it is dropped, and how the draw is taken.

    The benchmark's best configuration reads five dipeptide counts that are nonzero in a minority of
    rows.  Three ways to attack that, all of them legitimate: keep only columns a decent share of
    constructs populate; average the selection over bootstrap draws; score the columns with a
    different statistic.  If the skill survives, it is a composition effect worth reporting.  If it
    dies, the model found a few extreme rows and called them a motif - which is also why a single
    deployed draw scores 62 of 103 constructs with one identical number.
    """
    combos = [
        {
            "feature_set": f"populated>={floor:.0%}",
            "model": "ridge",
            "k": k,
            "selector": selector,
            "floor": floor,
        }
        for floor in SPARSITY_FLOORS
        for k in (5, 20)
    ]
    combos += [
        {"feature_set": "A+B", "model": "bagged", "k": k, "selector": selector}
        for k in (5, 20)
    ]
    # Same rows, same learner, columns chosen by the other scoring rule.
    other = "mi" if selector == "anova" else "anova"
    combos.append({"feature_set": "A+B", "model": "ridge", "k": 5, "selector": other})
    combos.append({"feature_set": "A+B", "model": "ridge", "k": 20, "selector": other})
    combos += [
        {"feature_set": set_name, "model": model, "k": k, "selector": selector}
        for set_name in ("A+B", "no dipeptides")
        for model in ("ridge", "rf")
        for k in (5, 20)
    ]
    combos.append(
        {"feature_set": "no dipeptides", "model": "constant", "k": None, "selector": selector}
    )
    return _expand(df, "signal_location", combos, ["protein"], repeats)


EXPERIMENTS = (benchmark, leakage, ablation, conditions, signal_location)


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
    keys = ["strategy", "feature_set", "model", "k", "selector", "selection", "n_features", "n_rows"]
    metrics = [c for c in METRIC_COLUMNS if c in part.columns]
    means = part.groupby(keys, as_index=False)[metrics].mean()
    spread = (
        part.groupby(keys)[metrics]
        .agg(lambda s: s.max() - s.min())
        .rename(columns={c: f"{c}_over_repeats" for c in metrics})
    )
    merged = means.merge(spread.reset_index(), on=keys)
    return merged.round(3).sort_values(["strategy", "model", "feature_set", "selection", "k"])
