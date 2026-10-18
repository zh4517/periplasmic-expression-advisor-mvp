"""交叉验证切分，以及确保分组切分真正隔离相似样本的检查。

随机切分可能让同一个构建体同时出现在训练折和测试折，衡量到的是记忆能力而非预测能力。
分组切分会将完整目标蛋白或完整信号肽组留在训练集之外。
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

# random 仅用于对照，并不能有效估计新构建体上的性能；103 行记录中有 81 个蛋白会在不同条件下重复出现。
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
    """按指定策略生成训练/测试索引；分组策略模拟未见过的新蛋白或新信号肽。"""
    if strategy not in SPLIT_STRATEGIES:
        raise ValueError(f"unknown split strategy {strategy!r}")

    if strategy == "random":
        # 仅用于对照：随机切分可能让相同蛋白的近似构建体泄漏到测试集。
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train, test in splitter.split(df, df["expression_class"]):
            yield train, test
        return

    # GroupKFold 保证同一分组不会同时落入训练集和测试集。
    groups = df[GROUP_COLUMN[strategy]].to_numpy()
    splitter = GroupKFold(n_splits=n_splits)
    for train, test in splitter.split(df, groups=groups):
        yield train, test


def assert_disjoint(df: pd.DataFrame, strategy: str, splits: list[tuple]) -> None:
    """检查每一折的训练集和测试集是否共享同一个分组，发现泄漏立即报错。

    GroupKFold 从设计上保证这一点；保留断言是为了防止后续修改分组键时悄悄重新引入泄漏。
    """
    column = GROUP_COLUMN[strategy]
    groups = set(df[column])
    for train, test in splits:
        train_groups = set(df.iloc[train][column])
        test_groups = set(df.iloc[test][column])
        # 非空交集意味着“留出某蛋白”的评估承诺已经失效。
        overlap = train_groups & test_groups
        if overlap:
            raise AssertionError(f"{strategy} split leaked {len(overlap)} of {len(groups)} groups")
