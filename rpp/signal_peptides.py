"""从融合的信号肽—蛋白序列中推断信号肽。

Table S2 将信号肽写为自由文本标签，只给出融合序列，因此需要从数据中恢复切点。表中约有 15 个
不同标签，每个标签会在多行出现，且同标签行以相同前缀开头。因此标签组内的最长公共前缀可作为
信号肽，组内首个分叉位置即为切点。

这属于推断而非真值：一组至少需要两条在某处不同的序列，否则“前缀”只是整条序列。未通过该规则的
标签一律返回未解析状态，不做猜测。
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

MIN_SP_LENGTH = 12
MAX_SP_LENGTH = 60


@dataclass
class SignalPeptide:
    """一个信号肽标签的推断结果；未满足规则时显式标记为未解析。"""
    label: str
    sequence: str
    n_members: int
    resolved: bool
    reason: str = ""


def normalise_label(label: str) -> str:
    """清理 PDF 表格中可能被换行或多余空格破坏的信号肽标签。"""
    cleaned = re.sub(r"\s+", " ", label.strip().lower())
    return cleaned


def _longest_common_prefix(sequences: list[str]) -> str:
    """取一组序列的最长公共前缀，作为候选信号肽片段。"""
    # 排序后只需比较字典序最小和最大的序列，即可确定所有序列共有的前缀。
    sequences = sorted(sequences)
    first, last = sequences[0], sequences[-1]
    i = 0
    while i < len(first) and i < len(last) and first[i] == last[i]:
        i += 1
    return first[:i]


def infer_signal_peptides(rows: list[dict]) -> dict[str, SignalPeptide]:
    """按信号肽标签分组，从融合序列的公共前缀推断信号肽及切点。"""
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        label = normalise_label(row["signal_peptide"])
        sequence = row["sequence"].strip().upper()
        if label and sequence:
            groups[label].append(sequence)

    inferred: dict[str, SignalPeptide] = {}
    for label, sequences in groups.items():
        # 没有至少两条可比较且不同的融合序列时，不能可靠推出切点，宁可拒答。
        if len(sequences) < 2:
            inferred[label] = SignalPeptide(label, "", len(sequences), False, "singleton group")
            continue
        if len(set(sequences)) == 1:
            inferred[label] = SignalPeptide(
                label, "", len(sequences), False, "identical sequences, no divergence point"
            )
            continue
        prefix = _longest_common_prefix(sequences)
        # 用保守长度范围过滤“公共前缀”：过短不像 SP，过长通常表示没有真正分叉。
        if len(prefix) < MIN_SP_LENGTH:
            inferred[label] = SignalPeptide(
                label, prefix, len(sequences), False, f"shared prefix only {len(prefix)} aa"
            )
            continue
        if len(prefix) > MAX_SP_LENGTH:
            inferred[label] = SignalPeptide(
                label, prefix, len(sequences), False, f"shared prefix {len(prefix)} aa, too long for an SP"
            )
            continue
        inferred[label] = SignalPeptide(label, prefix, len(sequences), True)

    return inferred


def split_signal_peptide(
    sequence: str, sp_sequence: str
) -> tuple[str, str] | None:
    """在融合序列确实以推断 SP 开头时，切分为（信号肽，成熟蛋白）。"""
    sequence = sequence.strip().upper()
    if not sp_sequence or not sequence.startswith(sp_sequence):
        # 切点不匹配时返回 None，避免把错误片段误当成熟蛋白。
        return None
    return sp_sequence, sequence[len(sp_sequence) :]
