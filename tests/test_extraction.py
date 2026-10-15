"""Tests for the cell-level reading rules of the PDF extractor.

    uv run --with pdfplumber python tests/test_extraction.py
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("extract_table_s2", ROOT / "scripts" / "extract_table_s2.py")
extract = importlib.util.module_from_spec(spec)
# @dataclass resolves annotations through sys.modules[cls.__module__], so the module has to be
# registered under the name it is given before it is executed.
sys.modules[spec.name] = extract
spec.loader.exec_module(extract)


def test_plain_numbers_are_read():
    assert extract.numbers_in("0.05") == [0.05]
    assert extract.numbers_in("24") == [24.0]


def test_ranges_and_lists_are_read_as_the_several_numbers_they_are():
    assert extract.numbers_in("0.4-0.5") == [0.4, 0.5]
    assert extract.numbers_in("22/37") == [22.0, 37.0]
    assert extract.numbers_in("0.1,0.5,1") == [0.1, 0.5, 1.0]


def test_typographic_dashes_are_ranges_too():
    # Row 61 writes 0.028–0.227 with an en dash; other rows use an ASCII hyphen.
    assert extract.numbers_in("0.028–0.227") == [0.028, 0.227]
    assert extract.numbers_in("12—16") == [12.0, 16.0]


def test_a_note_left_in_a_numeric_column_is_not_a_measurement():
    assert extract.numbers_in("24h") == [], "13 rows put an incubation time in the OD column"
    assert extract.numbers_in("10%") == [], "lactose percent is not 10 mM IPTG"
    assert extract.numbers_in("4.3-17.7%(Lac)") == []


def test_span_reports_width_as_well_as_ends():
    assert extract.span("17.8-45") == (17.8, 45.0, 2)
    assert extract.span("692.8") == (692.8, 692.8, 1)
    assert extract.span("24h") == (None, None, 0)


def test_class_boundaries():
    assert extract.classify(0.0) == "L"
    assert extract.classify(0.49) == "L"
    assert extract.classify(0.5) == "M"
    assert extract.classify(49.9) == "M"
    assert extract.classify(50.0) == "H", "the table labels a 50.0 row as high"
    assert extract.classify(700.0) == "H"
    assert extract.classify(None) == ""


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
