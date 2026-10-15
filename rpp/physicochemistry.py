"""Tier B: physicochemical features.

Every scale is read from tables shipped with Biopython rather than retyped here, so the
numbers carry Biopython's provenance instead of ours.

Two honest limitations to know about before trusting a column:

1. Average-of-a-scale features are linear combinations of the Tier A frequency vector.
   They add no information beyond the 20 amino-acid frequencies; their value is that they
   hand the model a physically meaningful direction instead of forcing it to learn 20
   weights from ~100 observations.
2. Biopython's `secondary_structure_fraction` is NOT a secondary-structure propensity
   scale.  It is a residue-group membership count (helix=EMALK, turn=NPGSD,
   sheet=VIYFWLT, per the docstring, citing Haimov & Srebnik 2016, Hutchinson & Thornton
   1994, Kim & Berg 1993).  The paper's `*_PHD` features are Rost & Sander per-residue
   propensities, which Biopython does not ship, so those are listed as not reproduced.

pI depends on which pKa table the implementation uses.  On the same test sequence we
measured 5.271 (Biopython) against 5.826 (`peptides`), which is why PERISCOPE-Opt carries
both a plain `pI` and a `pI_Protpi` column: different tools really do return different
values.  We emit one, and name the implementation.
"""

from __future__ import annotations

import statistics
from functools import lru_cache

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils.ProtParamData import gravy_scales

from rpp.composition import normalise

# PERISCOPE-Opt features deliberately absent from this module.
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
    return dict(gravy_scales)


def mean_scale(sequence: str, scale: str) -> float:
    """Average a per-residue scale over the sequence: <f, phi> in composition space."""
    table = _scales()[scale]
    seq = normalise(sequence)
    return sum(table[r] for r in seq) / len(seq)


def fraction_below(sequence: str, scale: str, threshold: float = 0.0) -> float:
    """Share of residues under a threshold on a scale (the paper's hydrophilic grouping)."""
    table = _scales()[scale]
    seq = normalise(sequence)
    return sum(1 for r in seq if table[r] < threshold) / len(seq)


def physicochemical_features(sequence: str, prefix: str = "") -> dict[str, float]:
    seq = normalise(sequence)
    analysis = ProteinAnalysis(seq)
    helix, turn, sheet = analysis.secondary_structure_fraction()
    # Biopython returns (reduced, oxidized) cysteine-state coefficients, not a mapping.
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
        # flexibility() is a per-window list, not a scalar.
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
        features = {f"{prefix}_{k}": v for k, v in features.items()}
    return features
