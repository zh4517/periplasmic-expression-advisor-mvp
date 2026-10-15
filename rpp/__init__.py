"""Sequence feature extraction for the periplasmic expression task.

Tier A (composition) and Tier B (physicochemistry) only.  See NOT_REPRODUCED for the
PERISCOPE-Opt features this module deliberately does not emit and why.
"""

from rpp.composition import (
    AMINO_ACIDS,
    RESIDUE_GROUPS,
    amino_acid_counts,
    amino_acid_frequencies,
    derived_composition,
    dipeptide_counts,
    dipeptide_frequencies,
    group_frequencies,
    max_run_frequencies,
    max_runs,
)
from rpp.features import extract_features
from rpp.physicochemistry import NOT_REPRODUCED, physicochemical_features
from rpp.signal_peptides import infer_signal_peptides, split_signal_peptide

__all__ = [
    "AMINO_ACIDS",
    "RESIDUE_GROUPS",
    "amino_acid_counts",
    "amino_acid_frequencies",
    "derived_composition",
    "dipeptide_counts",
    "dipeptide_frequencies",
    "group_frequencies",
    "max_run_frequencies",
    "max_runs",
    "extract_features",
    "physicochemical_features",
    "infer_signal_peptides",
    "split_signal_peptide",
    "NOT_REPRODUCED",
]
