"""将提取出的特征矩阵整理为可直接用于模型拟合的数据集。

Three decisions live here rather than in the modelling code, because they are statements about
the *data*, not about the algorithm:

* **Target.** Table S2 reports the best yield found for each SP-protein combination, sometimes as
  a span ("17.8-45").  The claim the published rows can support is "how high can this construct
  be pushed", so the target is the top of the span, on a log scale.
* **Grouping keys.** The protein and signal-peptide cells are free text from a PDF, with casing and
  line-wrap damage.  Grouped cross-validation depends on these keys being the same construct being
  spelled the same way, so the normalisation is written down here and tested.
* **Tier of each column.** Which columns are counts and which are weighted sums of counts decides
  the ablation, and it is read out of the code that generated them instead of guessed from names.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from rpp.features import META_COLUMNS, feature_tiers

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "data" / "processed" / "features_tier_ab.csv"

# 有十行报告产量为 0 mg/L，而 log10(0) 无定义，因此变换前统一加入这个伪计数；它也限制了零产量行可产生的倍数误差。
YIELD_OFFSET = 1.0

# PERISCOPE-Opt 的类别阈值：低产 < 0.5、中产 0.5–50、高产 > 50 mg/L。补充表将恰好 50.0 的行标为高产，因此上界归入高产。
CLASS_EDGES = (0.5, 50.0)

CONDITION_COLUMNS = ("od", "iptg", "temperature", "time")

DERIVED_COLUMNS = (
    "yield_best_mgl",
    "yield_low_mgl",
    "yield_is_single_value",
    "yield_span_mgl",
    "log10_yield",
    "expression_class",
    "protein_group",
    "signal_peptide_group",
    "combo_group",
    "conditions_numeric",
)


def _clean(text: str) -> str:
    """规范化 PDF 文本中的空白符，得到可稳定比较的字符串。"""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def protein_key(text: str) -> str:
    """抽取蛋白的稳定短标识，例如 ``Enhanced Green... (eGFP)`` 转为 ``egfp``。

    The parenthesised acronym is what recurs across the table; grouping on the full description
    would let 'eGFP' written two ways sit on both sides of a split.
    """
    matches = re.findall(r"\(([^()]+)\)", _clean(text))
    # 优先使用括号中的缩写，例如 eGFP；这是表内最稳定的重复标识。
    if matches:
        return re.sub(r"[^a-z0-9]", "", matches[-1].lower())
    return re.sub(r"[^a-z0-9]", "", _clean(text).lower())


def signal_peptide_key(text: str) -> str:
    """忽略大小写和标点的信号肽分组键，使 PelB 与 pelB 归为同组。"""
    return re.sub(r"[^a-z0-9]", "", _clean(text).lower())


def _numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """将纯数值单元格转数值；范围、文本或缺失值统一为 NaN。"""
    return frame.apply(lambda column: pd.to_numeric(column, errors="coerce"))


def load_dataset(path: Path = FEATURES_CSV) -> pd.DataFrame:
    """读取特征表，并派生标签、分组键与条件完整性字段。"""
    df = pd.read_csv(path)
    conditions = _numeric(df[list(CONDITION_COLUMNS)])

    yield_low = pd.to_numeric(df["yield_min"], errors="coerce")
    yield_best = pd.to_numeric(df["yield_max"], errors="coerce")
    span = yield_best - yield_low

    derived = pd.DataFrame(
        {
            "yield_low_mgl": yield_low,
            "yield_best_mgl": yield_best,
            "yield_span_mgl": span,
            "yield_is_single_value": span.fillna(1.0) == 0.0,
            # +1 让零产量也能取对数；对数尺度关注“相差几倍”而不是绝对 mg/L。
            "log10_yield": np.log10(YIELD_OFFSET + yield_best),
            "expression_class": np.select(
                [yield_best < CLASS_EDGES[0], yield_best < CLASS_EDGES[1]],
                ["L", "M"],
                default="H",
            ),
            "protein_group": df["protein"].map(protein_key),
            "signal_peptide_group": df["signal_peptide"].map(signal_peptide_key),
            "conditions_numeric": conditions.notna().all(axis=1),
            **{f"{c}_num": conditions[c] for c in CONDITION_COLUMNS},
        }
    )
    derived["combo_group"] = derived["signal_peptide_group"] + "__" + derived["protein_group"]

    df = pd.concat([df, derived], axis=1)
    df = df.sort_values("row_no").reset_index(drop=True)

    # 提取器已判断哪些行完整给出四项条件；这里从数值列重新计算，两者必须一致，否则说明其中一次 PDF 解析有误。
    claimed = df["condition_fully_specified"].astype(str).str.lower().str.startswith("true")
    # 两套解析逻辑不一致时立即停止，避免 PDF 读错后静默进入模型。
    disagreement = df.loc[claimed != df["conditions_numeric"], "row_no"].tolist()
    if disagreement:
        raise AssertionError(
            f"condition_fully_specified disagrees on rows {disagreement[:10]}"
        )
    return df


def feature_columns(
    df: pd.DataFrame,
    tiers: tuple[str, ...] = ("A", "B"),
    include_conditions: bool = False,
) -> list[str]:
    """返回属于所请求 Tier 的序列特征列。

    CSV 中既非已记录元数据、也非已知特征的列会直接报错，而非静默丢弃；列名变化应中断流程，
    不能悄悄缩小输入。
    """
    # 按 Tier A/B 选择序列特征，必要时把四个工艺变量加入输入。
    tiers_map = feature_tiers()
    known = set(META_COLUMNS) | set(DERIVED_COLUMNS) | set(f"{c}_num" for c in CONDITION_COLUMNS)
    # 新列没有明确归属时宁可报错，也不悄悄丢弃它。
    unknown = [c for c in df.columns if c not in known and c not in tiers_map]
    if unknown:
        raise ValueError(f"{len(unknown)} column(s) with no tier, e.g. {unknown[:5]}")

    columns = [c for c in df.columns if tiers_map.get(c) in tiers]
    if include_conditions:
        columns += [f"{c}_num" for c in CONDITION_COLUMNS]
    return columns


def describe(df: pd.DataFrame) -> str:
    """生成供命令行展示的数据规模、类别分布和特征数量摘要。"""
    tiers_map = feature_tiers()
    n_a = len(feature_columns(df, tiers=("A",)))
    n_b = len(feature_columns(df, tiers=("B",)))
    lines = [
        f"rows                                  : {len(df)}",
        f"yield known as a single value         : {int(df['yield_is_single_value'].sum())}",
        f"yield reported as a span              : {int((~df['yield_is_single_value']).sum())}",
        f"all four process conditions specified : {int(df['conditions_numeric'].sum())}",
        f"classes (L/M/H)                       : {df['expression_class'].value_counts().to_dict()}",
        f"distinct target proteins              : {df['protein_group'].nunique()}",
        f"distinct signal peptides              : {df['signal_peptide_group'].nunique()}",
        f"distinct SP-protein combinations      : {df['combo_group'].nunique()}",
        f"largest protein group                 : "
        f"{df['protein_group'].value_counts().index[0]} x "
        f"{int(df['protein_group'].value_counts().iloc[0])}",
        f"largest signal-peptide group          : "
        f"{df['signal_peptide_group'].value_counts().index[0]} x "
        f"{int(df['signal_peptide_group'].value_counts().iloc[0])}",
        f"features: Tier A {n_a} / Tier B {n_b} / both {n_a + n_b} "
        f"({len(tiers_map)} generated names)",
        f"observations per feature (A+B)        : {len(df) / (n_a + n_b):.3f}",
    ]
    return "\n".join(lines)
