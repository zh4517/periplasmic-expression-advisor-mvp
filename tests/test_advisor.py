"""Tests for what the advisor refuses, and for what it declines to rank.

    uv run --with pandas --with scikit-learn --with biopython==1.88 python tests/test_advisor.py

The advisor is the only place a wet-lab user meets this project, so these tests are not about the
predicted number.  They are about a query the model cannot answer coming back saying so.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from rpp.advisor import fit_advisor  # noqa: E402
from rpp.modeling import _expected_high_in_top  # noqa: E402

# The published pelB and ompA peptides, so the fixtures can be checked by eye.
PELB = "MKYLLPTAAAGLLLLAAQPAMA"
OMPA = "MKKTAIAIAVALAGFATVAQA"

ADVISOR = fit_advisor()


def _warnings(assessment) -> str:
    return " | ".join(assessment.warnings)


def test_a_sequence_that_is_not_protein_gets_no_number():
    assessment = ADVISOR.assess("seleno", PELB + "MUQC")
    assert not assessment.usable
    assert assessment.predicted_mgl is None
    assert "U" in _warnings(assessment) and "Q" not in _warnings(assessment)


def test_a_known_label_resolves_to_the_published_peptide():
    warnings: list[str] = []
    assert ADVISOR._resolve_signal_peptide("pelB", warnings) == PELB
    assert ADVISOR._resolve_signal_peptide("Pel B", warnings) == PELB
    assert ADVISOR._resolve_signal_peptide("ompA", warnings) == OMPA
    assert ADVISOR._resolve_signal_peptide(OMPA, warnings) == OMPA
    assert warnings == []


def test_an_unpublished_peptide_is_used_but_flagged_as_uncross_checked():
    warnings: list[str] = []
    for pasted in ("AAAAAAAACCCCAAAA", "MKKTAIAIAVALAGFATVAQK"):
        assert ADVISOR._resolve_signal_peptide(pasted, warnings) == pasted
    assert len(warnings) == 2 and all("never cross-checked" in w for w in warnings)


def test_every_bag_chooses_exactly_k_columns_and_the_tally_counts_only_those():
    """Catches a tally that ignores the mask: then every column looks equally chosen, and the
    stability figure the CLI prints is a lie."""
    bags = ADVISOR.pipeline.named_steps["bag"].models_
    chosen = [int(bag.named_steps["select"].get_support().sum()) for bag in bags]
    assert chosen == [ADVISOR.k] * ADVISOR.n_bags
    assert sum(ADVISOR.picked.values()) == ADVISOR.n_bags * ADVISOR.k
    assert max(ADVISOR.picked.values()) < ADVISOR.n_bags, "no selection instability to report"


def test_an_unresolved_label_is_said_so_rather_than_guessed():
    warnings: list[str] = []
    assert ADVISOR._resolve_signal_peptide("phoA", warnings) is None
    assert any("phoA" in warning for warning in warnings)


def test_a_signal_peptide_that_is_not_at_the_start_is_dropped_with_a_warning():
    assessment = ADVISOR.assess("wrong-sp", "AAAA" + PELB, signal_peptide="ompA")
    assert "is not the start" in _warnings(assessment)
    assert "no usable signal peptide" in _warnings(assessment)


def test_a_fragment_shorter_than_every_published_construct_is_extrapolation():
    assessment = ADVISOR.assess("fragment", OMPA + "AAAA", signal_peptide="ompA")
    assert "extrapolation" in _warnings(assessment)


def test_a_construct_already_in_the_table_reports_the_measured_yield():
    sequence, (row_no, yield_mgl) = next(iter(ADVISOR.seen.items()))
    assessment = ADVISOR.assess("known", sequence)
    assert assessment.observed_mgl == yield_mgl
    assert f"training row {row_no}" in _warnings(assessment)
    assert f"{yield_mgl:.3g} mg/L" in _warnings(assessment)


def test_the_bag_spread_is_reported_per_query_rather_than_as_one_global_number():
    spreads = [
        ADVISOR.assess(str(row[0]), sequence).spread_factor
        for sequence, row in list(ADVISOR.seen.items())[:30]
    ]
    assert len(set(spreads)) > 1, "every query carries the same spread, so it measures nothing"
    flagged = [
        assessment
        for assessment in (
            ADVISOR.assess(str(row[0]), sequence)
            for sequence, row in list(ADVISOR.seen.items())[:30]
        )
        if "disagree" in _warnings(assessment)
    ]
    assert flagged, "no published construct disagrees the bags, so the guard is untested"


def test_rank_numbers_answerable_queries_and_pushes_the_rest_last():
    ranked = ADVISOR.rank(
        [
            ("seleno", PELB + "MUQC", None),
            ("a", PELB + "AAAA" * 30, "pelB"),
            ("b", OMPA + "CCCC" * 30, "ompA"),
        ]
    )
    numbered = [a for a in ranked if a.rank]
    assert [a.rank for a in numbered] == list(range(1, len(numbered) + 1))
    assert [a.predicted_mgl for a in numbered] == sorted(a.predicted_mgl for a in numbered)[::-1]
    assert ranked[-1].name == "seleno" and ranked[-1].rank is None


def test_ties_are_counted_as_the_random_guess_they_are():
    """The top-k figure must not depend on which tied row the sort happened to reach first."""
    predicted = np.zeros(10)
    high_in_front = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    high_at_back = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    assert _expected_high_in_top(predicted, high_in_front, 2) == 0.6
    assert _expected_high_in_top(predicted, high_at_back, 2) == 0.6


def test_an_untied_ranking_is_counted_exactly():
    predicted = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
    high = np.array([1, 0, 1, 0, 0], dtype=bool)
    assert _expected_high_in_top(predicted, high, 1) == 1.0
    assert _expected_high_in_top(predicted, high, 2) == 1.0
    assert _expected_high_in_top(predicted, high, 5) == 2.0


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  ok   {test.__name__}")
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"  FAIL {test.__name__}: {type(error).__name__}: {error}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
