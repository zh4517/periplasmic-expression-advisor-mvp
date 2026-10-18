"""Tier B：理化属性特征。

所有尺度均直接读取 Biopython 自带表格，而非在此手工重录，因此数值可追溯至 Biopython。

使用这些列前应理解两项限制：

1. 尺度平均值是 Tier A 氨基酸频率向量的线性组合，并未增加额外信息；其价值在于向模型提供
   有物理意义的方向，避免在约 100 个样本中单独估计 20 个权重。
2. Biopython 的 `secondary_structure_fraction` 并不是二级结构倾向性尺度，而是残基组成员计数
   （helix=EMALK、turn=NPGSD、sheet=VIYFWLT）。论文的 `*_PHD` 是 Rost & Sander 的逐残基
   倾向性，Biopython 未提供，因此明确列为未复现。

pI 取决于实现采用的 pKa 表。同一测试序列在 Biopython 中为 5.271，在 `peptides` 中为 5.826；
这也是 PERISCOPE-Opt 同时包含 `pI` 和 `pI_Protpi` 的原因。本模块只输出一种实现，并记录来源。
"""

from __future__ import annotations

import statistics
from functools import lru_cache

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils.ProtParamData import gravy_scales

from rpp.composition import normalise

# 此模块有意不复现的 PERISCOPE-Opt 特征。
NOT_REPRODUCED: dict[str, str] = {
    "Helix_PHD / Sheet_PHD / Coil_PHD / Helix_to_Sheet_PHD": (
        "Rost & Sander per-residue propensities; table not shipped by Biopython. "
        "Group-count substitutes are emitted under SS_* and are not equivalent."
    ),
    "Expno_AA_TM": (
        "Expected residues in transmembrane helices; needs the Ulmschneider propensity "
        "table, not available offline."
    ),
    "Pred_Sol": (
        "Output of a separate solubility predictor. Feeding one model's output into "
        "another inherits its bias and caps our accuracy at its accuracy."
    ),
    "Folding_rate": "Output of a separate predictor (K-Fold / PRORATE / SeqRate).",
    "pI_Protpi": (
        "Protpi is a separate implementation with its own pKa set. We emit pI_Biopython "
        "and record the tool instead of pretending the two agree."
    ),
}


@lru_cache(maxsize=1)
def _scales() -> dict[str, dict[str, float]]:
    """从 Biopython 读取疏水性等氨基酸尺度，并缓存一次读取结果。"""
    # 同一套尺度会被多条序列反复使用；缓存可避免重复构造字典。
    return dict(gravy_scales)


def mean_scale(sequence: str, scale: str) -> float:
    """计算某理化尺度在整条序列上的平均值。"""
    table = _scales()[scale]
    seq = normalise(sequence)
    # 逐位查表后求均值，例如得到平均疏水性（GRAVY）。
    return sum(table[r] for r in seq) / len(seq)


def fraction_below(sequence: str, scale: str, threshold: float = 0.0) -> float:
    """计算尺度值低于阈值的氨基酸所占比例（例如亲水残基比例）。"""
    table = _scales()[scale]
    seq = normalise(sequence)
    return sum(1 for r in seq if table[r] < threshold) / len(seq)


def physicochemical_features(sequence: str, prefix: str = "") -> dict[str, float]:
    """用 Biopython 从一条序列提取长度、pI、疏水性、稳定性等 Tier B 特征。"""
    seq = normalise(sequence)
    # ProteinAnalysis 封装了已发表的蛋白理化指标计算，并非训练数据本身。
    analysis = ProteinAnalysis(seq)
    helix, turn, sheet = analysis.secondary_structure_fraction()
    # Biopython 返回还原态和氧化态半胱氨酸对应的两个消光系数，而不是字典。
    extinction_reduced, extinction_oxidized = analysis.molar_extinction_coefficient()

    features: dict[str, float] = {
        "seq_length": len(seq),
        "molecular_weight": analysis.molecular_weight(),
        "pI_biopython": analysis.isoelectric_point(),
        "net_charge_pH7": analysis.charge_at_pH(7.0),
        "gravy_kyte_doolittle": mean_scale(seq, "KyteDoolitle"),
        "gravy_eisenberg": mean_scale(seq, "Eisenberg"),
        "fraction_hydrophilic_kd": fraction_below(seq, "KyteDoolitle"),
        "fraction_hydrophilic_esg": fraction_below(seq, "Eisenberg"),
        "instability_index": analysis.instability_index(),
        "aromaticity": analysis.aromaticity(),
        # flexibility() 返回的是滑动窗口列表，模型需要单个标量，因此取其平均值。
        "flexibility_mean": statistics.fmean(analysis.flexibility()),
        "extinction_280_reduced": extinction_reduced,
        "extinction_280_oxidized": extinction_oxidized,
        "ss_helix_fraction": helix,
        "ss_turn_fraction": turn,
        "ss_sheet_fraction": sheet,
        "ss_helix_to_sheet": helix / sheet if sheet else float("nan"),
        "n_cysteine": seq.count("C"),
        "n_proline": seq.count("P"),
    }
    if prefix:
        # 信号肽和成熟蛋白的同名指标加不同前缀，避免在特征表中相互覆盖。
        features = {f"{prefix}_{k}": v for k, v in features.items()}
    return features
