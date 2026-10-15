"""Hand-checked fixtures.  Every expected value below is computable by counting the
20-letter toy sequence directly, so the tests can be defended without trusting the code.

    MAILVFFFAAACCPGGWSMQ
    M A I L V F F F A A A C C P G G W S M Q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpp.composition import (  # noqa: E402
    amino_acid_counts,
    amino_acid_frequencies,
    derived_composition,
    dipeptide_counts,
    group_frequencies,
    max_runs,
)
from rpp.physicochemistry import mean_scale, physicochemical_features  # noqa: E402
from rpp.signal_peptides import infer_signal_peptides, split_signal_peptide  # noqa: E402

TOY = "MAILVFFFAAACCPGGWSMQ"


def test_amino_acid_counts():
    counts = amino_acid_counts(TOY)
    assert counts["Occ_A"] == 4
    assert counts["Occ_F"] == 3
    assert counts["Occ_C"] == 2
    assert counts["Occ_M"] == 2
    assert counts["Occ_I"] == 1
    assert counts["Occ_W"] == 1
    assert counts["Occ_K"] == 0
    assert sum(counts.values()) == len(TOY)


def test_frequencies_sum_to_one():
    freq = amino_acid_frequencies(TOY)
    assert abs(sum(freq.values()) - 1.0) < 1e-12
    assert freq["OF_A"] == 4 / 20


def test_max_runs_is_per_residue_not_global():
    runs = max_runs(TOY)
    assert runs["MNC_A"] == 3  # AAA
    assert runs["MNC_F"] == 3  # FFF
    assert runs["MNC_C"] == 2  # CC
    assert runs["MNC_M"] == 1  # two M but not adjacent
    assert runs["MNC_G"] == 2


def test_dipeptide_counts():
    pairs = dipeptide_counts(TOY)
    assert pairs["OccD_FF"] == 2
    assert pairs["OccD_AA"] == 2
    assert pairs["OccD_FA"] == 1
    assert pairs["OccD_MA"] == 1
    assert pairs["OccD_QM"] == 0  # only MQ exists, not QM
    assert sum(pairs.values()) == len(TOY) - 1


def test_derived_difference_feature():
    # D and E are both absent from the toy sequence.
    assert derived_composition(TOY)["OF_DmE"] == 0.0


def test_group_frequencies():
    groups = group_frequencies(TOY)
    assert groups["OF_Aromatic"] == 4 / 20  # F F F W
    assert groups["OF_Sulfur"] == 4 / 20  # C C M M


def test_gravy_matches_hand_calculation():
    # Sum of Kyte-Doolittle values over TOY is 29.3; 29.3 / 20 = 1.465.
    assert abs(mean_scale(TOY, "KyteDoolitle") - 1.465) < 1e-9
    assert abs(physicochemical_features(TOY)["gravy_kyte_doolittle"] - 1.465) < 1e-9


def test_physicochemical_shape():
    feats = physicochemical_features(TOY)
    assert feats["seq_length"] == 20
    assert feats["n_cysteine"] == 2
    assert feats["molecular_weight"] > 2000


def test_signal_peptide_inference_needs_divergence():
    sp = "MKYLLPTAAAGLLLLAAQPAMA"
    rows = [
        {"signal_peptide": "pelB", "sequence": sp + "MSEQKLISEEDL"},
        {"signal_peptide": "pelB", "sequence": sp + "QVNTEYMVNAITALY"},
        {"signal_peptide": "ompA", "sequence": sp + "MSEQKLISEEDL"},
        {"signal_peptide": "ompA", "sequence": sp + "MSEQKLISEEDL"},
    ]
    sps = infer_signal_peptides(rows)
    assert sps["pelb"].resolved
    assert sps["pelb"].sequence == sp
    assert split_signal_peptide(sps["pelb"].sequence + "MSEQKLISEEDL", sp) == (sp, "MSEQKLISEEDL")
    # A group of identical sequences contains no evidence of where the cut site is.
    assert not sps["ompa"].resolved
    assert sps["ompa"].reason == "identical sequences, no divergence point"


def test_inferred_signal_peptide_can_over_run():
    """Documented limitation: the prefix cannot stop at the cut site if the mature
    proteins happen to share residues immediately after it, so the inferred length is an
    upper bound.  Real data must be inspected, not trusted blindly."""
    sp = "MKYLLPTAAAGLLLLAAQPAMA"
    rows = [
        {"signal_peptide": "pelB", "sequence": sp + "MA" + "QSFDSNWIPL"},
        {"signal_peptide": "pelB", "sequence": sp + "MA" + "ITALYAGSGN"},
    ]
    inferred = infer_signal_peptides(rows)["pelb"].sequence
    assert inferred == sp + "MA", "prefix runs past the true cut site by the shared residues"


def test_rejects_non_standard_residues():
    try:
        amino_acid_counts("MAILXV")
    except ValueError:
        return
    raise AssertionError("expected ValueError for ambiguous residue X")


def run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  ok   {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
