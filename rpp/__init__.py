"""用于周质表达任务的序列特征提取与严格评估。

``composition`` / ``physicochemistry`` / ``features`` hold Tier A (counts) and Tier B (published
scales).  ``dataset`` / ``cv`` / ``modeling`` / ``experiments`` hold the evaluation layer and need
pandas and scikit-learn; they are deliberately not re-exported here, so building the feature matrix
only requires Biopython.  See NOT_REPRODUCED for the PERISCOPE-Opt features this package does not
emit and why.
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
