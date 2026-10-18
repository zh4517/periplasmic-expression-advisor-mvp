"""Tier A：序列组成特征，仅依赖字符计数，不需要外部数据。

论文将出现次数和出现频率视作两类特征。给定序列长度后二者信息冗余（次数 = 频率 × 长度），
为保持与 PERISCOPE-Opt 一致仍同时输出，并在报告中标记该冗余。
"""

from __future__ import annotations

from collections import Counter

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
DIPEPTIDES = [a + b for a in AMINO_ACIDS for b in AMINO_ACIDS]

# 常用氨基酸分组。论文给出了分组名称但未定义成员，因此在此显式记录采用的成员。
RESIDUE_GROUPS: dict[str, str] = {
    "Aromatic": "FWY",
    "Aliphatic": "AVLI",
    "Charged_positive": "KR",
    "Charged_negative": "DE",
    "Polar_uncharged": "NSTQG",
    "Small": "AGS",
    "Sulfur": "CM",
    "Proline": "P",
    "Cysteine": "C",
    "Hydrophobic": "AVLIMFGC",
    "Helix_forming": "EMALK",
    "Turn_forming": "NPGSD",
    "Sheet_forming": "VIYFWLT",
}


def normalise(sequence: str) -> str:
    """去除空白并转大写，同时拒绝非标准的 20 种氨基酸字符。"""
    seq = sequence.upper().replace(" ", "").replace("\n", "")
    unknown = set(seq) - set(AMINO_ACIDS)
    if unknown:
        raise ValueError(f"non-standard residues {sorted(unknown)!r} in sequence")
    return seq


def amino_acid_counts(sequence: str) -> dict[str, int]:
    """统计 20 种氨基酸在序列中的出现次数。"""
    counts = Counter(normalise(sequence))
    return {f"Occ_{a}": counts.get(a, 0) for a in AMINO_ACIDS}


def amino_acid_frequencies(sequence: str) -> dict[str, float]:
    """将氨基酸次数除以序列长度，得到可比较的频率。"""
    seq = normalise(sequence)
    counts = Counter(seq)
    return {f"OF_{a}": counts.get(a, 0) / len(seq) for a in AMINO_ACIDS}


def max_runs(sequence: str) -> dict[str, int]:
    """计算每种氨基酸连续重复出现的最长长度（论文 MNC_X 特征）。"""
    seq = normalise(sequence)
    best = {a: 0 for a in AMINO_ACIDS}
    for a in AMINO_ACIDS:
        run = 0
        for residue in seq:
            run = run + 1 if residue == a else 0
            if run > best[a]:
                best[a] = run
    return {f"MNC_{a}": v for a, v in best.items()}


def max_run_frequencies(sequence: str) -> dict[str, float]:
    """将最长重复长度按总长度归一化。"""
    seq = normalise(sequence)
    runs = max_runs(seq)
    return {f"OF_{k}": v / len(seq) for k, v in runs.items()}


def dipeptide_counts(sequence: str) -> dict[str, int]:
    """统计全部 400 种相邻二肽的出现次数。"""
    seq = normalise(sequence)
    pairs = Counter(seq[i : i + 2] for i in range(len(seq) - 1))
    return {f"OccD_{d}": pairs.get(d, 0) for d in DIPEPTIDES}


def dipeptide_frequencies(sequence: str) -> dict[str, float]:
    """以可形成的相邻位置数为分母计算二肽频率。"""
    seq = normalise(sequence)
    n = len(seq) - 1
    if n <= 0:
        return {f"OFD_{d}": 0.0 for d in DIPEPTIDES}
    pairs = Counter(seq[i : i + 2] for i in range(n))
    return {f"OFD_{d}": pairs.get(d, 0) / n for d in DIPEPTIDES}


def group_frequencies(sequence: str) -> dict[str, float]:
    """统计疏水、带电等氨基酸分组在序列中的比例。"""
    seq = normalise(sequence)
    out = {}
    for name, residues in RESIDUE_GROUPS.items():
        out[f"OF_{name}"] = sum(seq.count(r) for r in residues) / len(seq)
    return out


def derived_composition(sequence: str) -> dict[str, float]:
    """计算论文使用的非简单计数型组成特征，并汇总返回。"""
    seq = normalise(sequence)
    freq = {a: seq.count(a) / len(seq) for a in AMINO_ACIDS}
    return {
        # 天冬氨酸频率减谷氨酸频率；论文认为两种酸性残基不可互换，因此保留有符号差值。
        "OF_DmE": freq["D"] - freq["E"],
        "OF_Hphil_ACDC": sum(freq[a] for a in "NQDEKRHPTY"),
    }
