"""Cross-validation splits, and the guard that makes grouped splits mean what they say.

A random split on this table lets the same construct sit on both sides of the fold, which measures
memorisation rather than prediction.  Grouped splits hold a whole target protein, or a whole
signal peptide, out of training.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

# "random" is included only to be compared against; it is not a valid estimate of performance on a
# new construct, because 103 rows describe 81 proteins that recur across conditions.
SPLIT_STRATEGIES = ("random", "combo", "protein", "signal_peptide")

GROUP_COLUMN = {
    "combo": "combo_group",
    "protein": "protein_group",
    "signal_peptide": "signal_peptide_group",
}


def iter_splits(
    df: pd.DataFrame,
    strategy: str,
    n_splits: int = 5,
    seed: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train index array, test index array) pairs for one split strategy."""
    if strategy not in SPLIT_STRATEGIES:
        raise ValueError(f"unknown split strategy {strategy!r}")

    if strategy == "random":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train, test in splitter.split(df, df["expression_class"]):
            yield train, test
        return

    groups = df[GROUP_COLUMN[strategy]].to_numpy()
    splitter = GroupKFold(n_splits=n_splits)
    for train, test in splitter.split(df, groups=groups):
        yield train, test


def assert_disjoint(df: pd.DataFrame, strategy: str, splits: list[tuple]) -> None:
    """Fail loudly if a group appears on both sides of any fold.

    GroupKFold guarantees this by construction; the assertion is here so that a later refactor of
    the grouping keys cannot quietly reintroduce the leak.
    """
    column = GROUP_COLUMN[strategy]
    groups = set(df[column])
    for train, test in splits:
        train_groups = set(df.iloc[train][column])
        test_groups = set(df.iloc[test][column])
        overlap = train_groups & test_groups
        if overlap:
            raise AssertionError(f"{strategy} split leaked {len(overlap)} of {len(groups)} groups")
