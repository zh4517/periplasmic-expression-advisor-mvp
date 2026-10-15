"""Tier A: composition features.  Counting operations, no external data required.

The paper treats occurrence and occurrence frequency as two feature groups.  They are
informationally redundant given the sequence length (count = frequency * length), so both
are emitted for fidelity to PERISCOPE-Opt but the redundancy is flagged in the report.
"""

from __future__ import annotations

from collections import Counter

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
DIPEPTIDES = [a + b for a in AMINO_ACIDS for b in AMINO_ACIDS]

# Conventional residue groupings.  The paper names these groups but does not define their
# membership, so the membership used here is recorded explicitly.
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
    seq = sequence.upper().replace(" ", "").replace("\n", "")
    unknown = set(seq) - set(AMINO_ACIDS)
    if unknown:
        raise ValueError(f"non-standard residues {sorted(unknown)!r} in sequence")
    return seq


def amino_acid_counts(sequence: str) -> dict[str, int]:
    counts = Counter(normalise(sequence))
    return {f"Occ_{a}": counts.get(a, 0) for a in AMINO_ACIDS}


def amino_acid_frequencies(sequence: str) -> dict[str, float]:
    seq = normalise(sequence)
    counts = Counter(seq)
    return {f"OF_{a}": counts.get(a, 0) / len(seq) for a in AMINO_ACIDS}


def max_runs(sequence: str) -> dict[str, int]:
    """Longest consecutive stretch of each residue (paper feature MNC_X)."""
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
    seq = normalise(sequence)
    runs = max_runs(seq)
    return {f"OF_{k}": v / len(seq) for k, v in runs.items()}


def dipeptide_counts(sequence: str) -> dict[str, int]:
    seq = normalise(sequence)
    pairs = Counter(seq[i : i + 2] for i in range(len(seq) - 1))
    return {f"OccD_{d}": pairs.get(d, 0) for d in DIPEPTIDES}


def dipeptide_frequencies(sequence: str) -> dict[str, float]:
    seq = normalise(sequence)
    n = len(seq) - 1
    if n <= 0:
        return {f"OFD_{d}": 0.0 for d in DIPEPTIDES}
    pairs = Counter(seq[i : i + 2] for i in range(n))
    return {f"OFD_{d}": pairs.get(d, 0) / n for d in DIPEPTIDES}


def group_frequencies(sequence: str) -> dict[str, float]:
    seq = normalise(sequence)
    out = {}
    for name, residues in RESIDUE_GROUPS.items():
        out[f"OF_{name}"] = sum(seq.count(r) for r in residues) / len(seq)
    return out


def derived_composition(sequence: str) -> dict[str, float]:
    """Named differences and sums the paper uses that are not plain group counts."""
    seq = normalise(sequence)
    freq = {a: seq.count(a) / len(seq) for a in AMINO_ACIDS}
    return {
        # Frequency of aspartate minus glutamate; the paper treats the two acidic residues
        # as non-interchangeable and keeps the signed difference.
        "OF_DmE": freq["D"] - freq["E"],
        "OF_Hphil_ACDC": sum(freq[a] for a in "NQDEKRHPTY"),
    }
