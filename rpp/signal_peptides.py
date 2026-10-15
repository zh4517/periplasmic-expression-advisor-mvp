"""Recover the signal peptide from fused SP-protein sequences.

Table S2 lists the signal peptide as a free-text label and gives only the fused sequence,
so the cut point has to be recovered from the data.  Only ~15 distinct labels are used but
each recurs across many rows, and every row sharing a label starts with the same leading
segment.  The longest common prefix inside a label group is therefore the signal peptide,
and the first position where the group diverges is the cleavage site.

This is inference, not ground truth: a group needs at least two members that disagree
somewhere, otherwise the "prefix" is just the whole sequence.  Anything that fails that
test is returned as unresolved rather than guessed at.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

MIN_SP_LENGTH = 12
MAX_SP_LENGTH = 60


@dataclass
class SignalPeptide:
    label: str
    sequence: str
    n_members: int
    resolved: bool
    reason: str = ""


def normalise_label(label: str) -> str:
    """Table S2 labels arrive with PDF line-wrap damage: 'endoxy lanase' -> 'endoxylanase'."""
    cleaned = re.sub(r"\s+", " ", label.strip().lower())
    return cleaned


def _longest_common_prefix(sequences: list[str]) -> str:
    sequences = sorted(sequences)
    first, last = sequences[0], sequences[-1]
    i = 0
    while i < len(first) and i < len(last) and first[i] == last[i]:
        i += 1
    return first[:i]


def infer_signal_peptides(rows: list[dict]) -> dict[str, SignalPeptide]:
    """Map normalised label -> inferred signal peptide, from (label, sequence) rows."""
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        label = normalise_label(row["signal_peptide"])
        sequence = row["sequence"].strip().upper()
        if label and sequence:
            groups[label].append(sequence)

    inferred: dict[str, SignalPeptide] = {}
    for label, sequences in groups.items():
        if len(sequences) < 2:
            inferred[label] = SignalPeptide(label, "", len(sequences), False, "singleton group")
            continue
        if len(set(sequences)) == 1:
            inferred[label] = SignalPeptide(
                label, "", len(sequences), False, "identical sequences, no divergence point"
            )
            continue
        prefix = _longest_common_prefix(sequences)
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
    """Split a fused sequence into (signal peptide, mature protein)."""
    sequence = sequence.strip().upper()
    if not sp_sequence or not sequence.startswith(sp_sequence):
        return None
    return sp_sequence, sequence[len(sp_sequence) :]
