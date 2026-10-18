"""将 Tier A 与 Tier B 组合为一行扁平化特征。"""

from __future__ import annotations

from rpp.composition import (
    AMINO_ACIDS,
    amino_acid_counts,
    amino_acid_frequencies,
    derived_composition,
    dipeptide_counts,
    dipeptide_frequencies,
    group_frequencies,
    max_run_frequencies,
    max_runs,
    normalise,
)
from rpp.physicochemistry import physicochemical_features
from rpp.signal_peptides import split_signal_peptide

# data/processed/features_tier_ab.csv 中描述记录本身而非序列的列。
META_COLUMNS = (
    "row_no",
    "protein",
    "signal_peptide",
    "media",
    "od",
    "iptg",
    "temperature",
    "time",
    "yield_mgl",
    "yield_min",
    "yield_max",
    "level_codes",
    "level_from_yield_max",
    "condition_fully_specified",
    "rsm_optimized",
    "reference",
    "sp_inferred_length",
    "sp_label_resolved",
    "sp_resolved",
)


def _composition_block(sequence: str) -> dict[str, float]:
    """汇总只由序列字符直接计数得到的 Tier A 成分特征。"""
    block: dict[str, float] = {}
    block.update(amino_acid_counts(sequence))
    block.update(amino_acid_frequencies(sequence))
    block.update(max_runs(sequence))
    block.update(max_run_frequencies(sequence))
    block.update(group_frequencies(sequence))
    block.update(derived_composition(sequence))
    return block


def extract_features(
    sequence: str,
    signal_peptide: str | None = None,
    include_dipeptides: bool = True,
) -> dict[str, float]:
    """为一条信号肽—蛋白融合序列生成一行特征。

    已知信号肽时，会分别在运输信号和成熟蛋白上重新计算同一组特征。两者影响周质产量的机制不同：
    前者影响分泌效率，后者影响折叠和可溶性。将它们混入同一个向量会模糊这种差异。
    """
    seq = normalise(sequence)
    # 该布尔值也进入特征表，让模型知道“信号肽是否成功解析”，而非默默混淆两种情况。
    features = {"sp_resolved": signal_peptide is not None}
    features.update(_composition_block(seq))
    features.update(physicochemical_features(seq))
    if include_dipeptides:
        # 二肽保留相邻残基信息，但维度高、样本少时也更容易造成不稳定，后续由特征选择控制。
        features.update(dipeptide_counts(seq))
        features.update(dipeptide_frequencies(seq))

    if signal_peptide:
        # 若能找到切点，分别为分泌信号和成熟蛋白建特征，反映两者机制不同。
        split = split_signal_peptide(seq, signal_peptide)
        if split:
            sp, mature = split
            features["sp_length"] = len(sp)
            features["mature_length"] = len(mature)
            features.update(
                {
                    f"sp_{k}": v
                    for k, v in {**_composition_block(sp), **physicochemical_features(sp)}.items()
                }
            )
            features.update(
                {
                    f"target_{k}": v
                    for k, v in {
                        **_composition_block(mature),
                        **physicochemical_features(mature),
                    }.items()
                }
            )

    return features


def feature_tiers(sequence: str = AMINO_ACIDS) -> dict[str, str]:
    """将每个特征列名映射到生成它的 Tier。

    通过调用两个特征构造器来确定列归属，而非匹配列名，因为 ``sp_`` 和 ``target_`` 区块在一个
    前缀下同时包含组成特征与尺度特征。Tier B 是按公开尺度对 Tier A 加权得到的和，并不增加信息；
    区分二者是为了检验该先验知识是否有帮助。
    """
    tiers: dict[str, str] = {}
    # 用真实特征构造函数生成列名，避免依赖脆弱的字符串命名规则。
    tier_a = {
        **_composition_block(sequence),
        **dipeptide_counts(sequence),
        **dipeptide_frequencies(sequence),
    }
    for name in tier_a:
        tiers[name] = "A"
    for name in physicochemical_features(sequence):
        tiers[name] = "B"
    # 长度虽与理化尺度列一同输出，但本质上是序列直接计数，因此归入不依赖外部尺度表的 Tier A。
    for name in ("seq_length", "sp_length", "mature_length"):
        tiers[name] = "A"
    base = dict(tiers)
    for prefix in ("sp_", "target_"):
        for name, tier in base.items():
            tiers[f"{prefix}{name}"] = tier
    return tiers
