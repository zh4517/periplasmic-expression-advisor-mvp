"""Tests for the evaluation layer.  Each one pins a claim the deck will make, and the arithmetic
in them is checkable with a pencil: log10(1+99)=2, a decade of error is a factor of ten, and a
fold's selector cannot have seen more rows than the fold contains.

    uv run --with pandas --with scikit-learn --with biopython python tests/test_modelling.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rpp.modeling as modeling  # noqa: E402
from rpp.cv import assert_disjoint, iter_splits  # noqa: E402
from rpp.dataset import (  # noqa: E402
    YIELD_OFFSET,
    feature_columns,
    load_dataset,
    protein_key,
    signal_peptide_key,
)
from rpp.modeling import (  # noqa: E402
    build_pipeline,
    classes_from_log,
    cross_fit,
    leaked_columns,
    score,
    to_mg_per_l,
)

DF = load_dataset()


def _spy_anova(seen: list[int]):
    """Wrap the selector so we can count the rows it was handed."""
    original = modeling.SELECTORS["anova"]

    def spy(X, y):
        seen.append(len(y))
        return original(X, y)

    return original, spy


def test_grouped_splits_never_share_a_construct():
    for strategy in ("combo", "protein", "signal_peptide"):
        assert_disjoint(DF, strategy, list(iter_splits(DF, strategy)))


def test_random_split_does_share_constructs():
    """The trap this project exists to avoid: a random split re-uses the same protein on both sides."""
    train, test = next(iter_splits(DF, "random"))
    overlap = set(DF.iloc[train]["protein_group"]) & set(DF.iloc[test]["protein_group"])
    assert overlap, "no shared protein would mean the random split is not the thing we warn about"


def test_every_row_is_predicted_exactly_once():
    columns = feature_columns(DF, tiers=("A",))[:40]
    predictions = cross_fit(DF, columns, strategy="protein", model="ridge", k=5)
    assert len(predictions) == len(DF)
    assert not np.isnan(predictions).any()


def test_nested_selection_never_sees_the_whole_table():
    seen: list[int] = []
    original, spy = _spy_anova(seen)
    modeling.SELECTORS["anova"] = spy
    try:
        columns = feature_columns(DF, tiers=("A",))[:40]
        cross_fit(DF, columns, strategy="protein", model="ridge", k=5)
    finally:
        modeling.SELECTORS["anova"] = original

    assert len(seen) == 5, f"expected one selection per fold, saw {len(seen)}"
    assert max(seen) < len(DF), f"a fold selected on {max(seen)} of {len(DF)} rows"


def test_leaked_selection_is_fitted_on_every_label():
    seen: list[int] = []
    original, spy = _spy_anova(seen)
    modeling.SELECTORS["anova"] = spy
    try:
        kept = leaked_columns(DF, feature_columns(DF, tiers=("A",)), k=5)
    finally:
        modeling.SELECTORS["anova"] = original

    assert seen == [len(DF)], f"leaked selection should run once over all rows, saw {seen}"
    assert len(kept) == 5


def test_log_target_and_fold_error_arithmetic():
    assert YIELD_OFFSET == 1.0
    # 99 mg/L sits at log10(1 + 99) = 2, so the inverse must return to 99.
    assert abs(float(to_mg_per_l([2.0])[0]) - 99.0) < 1e-9
    assert abs(float(to_mg_per_l([0.0])[0])) < 1e-9, "a zero yield stays zero, not one"

    truth = np.array([0.5, 1.5, 2.5])
    frame = pd.DataFrame({"log10_yield": truth, "expression_class": ["L", "M", "H"]})
    perfect = score(frame, truth.copy())
    assert abs(perfect["median_fold_error"] - 1.0) < 1e-9
    assert abs(perfect["pearson_r_log"] - 1.0) < 1e-9

    off_by_decade = score(frame, truth + 1.0)
    assert abs(off_by_decade["median_fold_error"] - 10.0) < 1e-9


def test_class_edges_are_high_inclusive():
    logs = np.log10(YIELD_OFFSET + np.array([0.4, 0.5, 49.9, 50.0, 700.0]))
    assert list(classes_from_log(logs)) == ["L", "M", "M", "H", "H"], (
        "the supplement labels a 50.0 mg/L row as high, so the edge belongs to high"
    )


def test_grouping_keys_merge_spelling_variants():
    assert protein_key("Enhanced Green Flourescent protein (eGFP)") == "egfp"
    assert protein_key("Thioredoxin from E. coli (Trx)") == "trx"
    assert signal_peptide_key("PelB") == signal_peptide_key("pel B") == "pelb"
    assert DF["protein_group"].nunique() <= DF["protein"].nunique()
    assert DF["signal_peptide_group"].nunique() < DF["signal_peptide"].nunique()


def test_tiers_partition_the_feature_matrix():
    a = set(feature_columns(DF, tiers=("A",)))
    b = set(feature_columns(DF, tiers=("B",)))
    both = set(feature_columns(DF))
    assert a and b
    assert not (a & b), "a column cannot be counted and scale-weighted at once"
    assert a | b == both


def test_unknown_column_is_refused_not_ignored():
    renamed = DF.rename(columns={"Occ_A": "Occ_A_typo"})
    try:
        feature_columns(renamed)
    except ValueError:
        return
    raise AssertionError("a column no tier claims should stop the run, not vanish quietly")


def test_condition_rows_are_exactly_the_fully_numeric_cells():
    numeric = DF[["od_num", "iptg_num", "temperature_num", "time_num"]].notna().all(axis=1)
    assert (numeric.to_numpy() == DF["conditions_numeric"].to_numpy()).all()
    # 13 rows write "24h" in the OD column and 1 writes "10%" lactose; those are not OD and IPTG
    # values, so the honest count of fully stated conditions is 36 of 103, not the 50 a naive
    # read of the table would give.
    assert int(DF["conditions_numeric"].sum()) == 36


def test_target_is_defined_for_every_row():
    assert DF["log10_yield"].notna().all()
    assert set(DF["expression_class"]) == {"L", "M", "H"}


def test_forced_columns_survive_selection():
    """The four process conditions must reach the learner even when k is smaller than the matrix."""
    X = np.arange(180.0).reshape(20, 9)
    y = X[:, 4] + 2 * X[:, 5]
    pipeline = build_pipeline("ridge", k=2, forced=4)
    pipeline.fit(X, y)
    # Everything up to the learner is a transformer; the learner itself is not.
    assert pipeline[:-1].fit_transform(X, y).shape[1] == 6, "four forced columns plus two of the rest"


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
