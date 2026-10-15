"""Assemble Tier A + B into one flat feature row."""

from __future__ import annotations

from rpp.composition import (
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


def _composition_block(sequence: str) -> dict[str, float]:
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
    """One feature row for a fused SP-protein sequence.

    When the signal peptide is known, the same blocks are recomputed on the translocation
    signal and the mature protein separately.  Their effects on periplasmic yield are
    different mechanisms - secretion efficiency versus folding solubility - so pooling
    them into one vector, as PERISCOPE-Opt does, blurs the distinction.
    """
    seq = normalise(sequence)
    features = {"sp_resolved": signal_peptide is not None}
    features.update(_composition_block(seq))
    features.update(physicochemical_features(seq))
    if include_dipeptides:
        features.update(dipeptide_counts(seq))
        features.update(dipeptide_frequencies(seq))

    if signal_peptide:
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
