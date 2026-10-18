"""本项目报告的五类证据实验。

1. **benchmark**：在整组构建体被留出的条件下，模型真正能做到什么。
2. **leakage**：把特征选择移到切分之前，模拟常见但错误的流程；差异量化该错误的代价。
3. **ablation**：比较 Tier A 计数特征与 Tier B 的加权和特征。Tier B 不含 Tier A 没有的信息，
   其增益只能来自先验正则化；若反而变差，则说明小样本下过多尺度列带来负担。
4. **conditions**：只在公开表完整给出四项工艺条件的记录上，检查这些条件是否提供额外信息。
5. **signal location**：检查只在少数行有值的特征会在多大程度上误导模型，判断候选池信号是否
   只是少量构建体的偶然产物。
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
    """仅保留在至少 floor 比例样本中非零的特征，用于检验稀有特征依赖。

    一个二肽即使在 103 行中有 101 行为零，也可能因剩余两行极端而在 ANOVA 中胜出。此时
    特征选择交给模型的往往只是“这两条蛋白”，而非真正可泛化的序列基序。
    """
    if floor <= 0:
        return columns
    X = df[columns].to_numpy(dtype=float)
    # 二肽在极少数样本出现也可能得到高 ANOVA 分数，因此需要单独做稀疏性实验。
    share = (np.nan_to_num(X, nan=0.0) != 0).mean(axis=0)
    return [name for name, s in zip(columns, share) if s >= floor]


def _is_dipeptide_column(name: str) -> bool:
    """识别完整链、信号肽或成熟蛋白区块中的二肽特征列。

    融合链使用 ``OccD_PM`` / ``OFD_PM``，两段分别使用 ``sp_OccD_PM`` / ``target_OFD_PM``；
    因此需先移除区块前缀再识别特征族。
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
    # 执行一组固定配置，返回一行可汇总的交叉验证指标。
    columns = feature_columns(df, tiers=tiers)
    forced = 0
    if feature_set == "conditions":
        condition_columns = [f"{c}_num" for c in CONDITION_COLUMNS]
        # 放在矩阵最前面并强制保留；这是已给出的实验条件，不是从标签中挑出的列。
        # 它们是已给定的实验测量值，不是根据标签挑出的列，因此强制保留不构成泄漏。
        columns = condition_columns + columns
        forced = len(condition_columns)
    columns = populated_columns(df, columns, floor)
    if feature_set == "no dipeptides":
        columns = [c for c in columns if not _is_dipeptide_column(c)]
    if not columns:
        raise ValueError(f"no columns for feature set {feature_set} at floor {floor}")
    if k and k >= len(columns):
        # 请求列数超过可用列数即表示“全部使用”；在此截断可避免 sklearn 对每折重复发出无意义警告。
        k = None

    if selection == "leaked" and k:
        # 故意让测试标签参与特征选择，量化“先选特征再验证”的泄漏增益。
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
    # 将多个模型、特征集与切分策略展开为可重复运行的实验网格。
    rows = []
    for strategy in strategies:
        for combo in combos:
            # 仅随机切分的各折受随机种子影响；分组切分在给定数据下是确定的。
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
    """比较 Ridge、随机森林、Bagged Ridge 与常数基线。"""
    combos = [
        {"feature_set": "A+B", "model": model, "k": k, "selector": selector}
        for model in ("ridge", "rf")
        for k in (5, 20, 60)
    ]
    # 训练标签中位数基线：复杂特征流程至少应超过的最低标准。
    combos.append({"feature_set": "A+B", "model": "constant", "k": None, "selector": selector})
    return _expand(df, "benchmark", combos, ["random", "combo", "protein", "signal_peptide"], repeats)


def leakage(df: pd.DataFrame, selector: str, repeats: int) -> list[dict]:
    """比较嵌套选择与泄漏选择，展示错误流程会高估多少性能。"""
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
    """比较 Tier A、Tier B、二肽与稀疏过滤对结果的影响。"""
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
    """仅在条件完整的 36 行上，比较是否加入 OD/IPTG/温度/时间。"""
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
    """检查信号肽特征只在部分行存在时，对模型结论的影响。

    基准中分数最高的配置依赖 5 个只在少数行非零的二肽。此处通过三种合理方式检查这种信号：
    仅保留较多构建体有值的列、多次 bootstrap 后平均选择结果、或换用不同统计量打分。若模型
    仍有效，可将其视为值得报告的组成效应；若失效，则原模型只是发现了几条极端记录。
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
    # 使用同一批行和同一学习器，仅将特征评分规则替换为另一种方法。
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
    """按既定顺序运行五类证据实验，并附带每种切分的测试折大小。"""
    repeats = 1 if quick else 3
    rows = [row for experiment in EXPERIMENTS for row in experiment(df, selector, repeats)]
    results = pd.DataFrame(rows)
    results.attrs["fold_sizes"] = {
        strategy: [len(test) for _, test in iter_splits(df, strategy)]
        for strategy in ("random", "combo", "protein", "signal_peptide")
    }
    return results


def summarise(results: pd.DataFrame, experiment: str) -> pd.DataFrame:
    """对同一配置的重复运行取均值，生成适合报告的结果表。"""
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
