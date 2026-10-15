"""Extract Table S2 of the PERISCOPE-Opt supplementary material into a tidy CSV.

Source: Packiam K.A.R. et al., "PERISCOPE-Opt: Machine learning-based prediction of optimal
fermentation conditions and yields of recombinant periplasmic protein expressed in
Escherichia coli", Comput. Struct. Biotechnol. J. 20 (2022) 2909-2920,
DOI 10.1016/j.csbj.2022.06.006.  Table S2 lives on pp. 3-31 of the "*-mmc1.pdf" supplement.

Layout notes that drive this parser
-----------------------------------
* The PDF table carries no structural tags, so rows are recovered from word bounding
  boxes. The narrow left-hand "No" column holds exactly one integer per record and is
  used as the row anchor; wrapped continuation lines (and lines continued after a page
  break) carry no anchor and are appended to the record being built.
* Page footers print the bare page number in the horizontal middle of the page, which
  falls inside the sequence column. Those tokens are dropped by a bottom-band filter.
* Most cells are ranges or slash-separated alternatives ("0.5-0.6", "22/37",
  "17.8-45") rather than single values, because the paper tabulates one row per
  SP-protein combination and reports the span that was experimented with. Both the
  raw string and the parsed numeric span are kept so no information is lost.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

# Column name -> lower x0 bound. Boundaries are midpoints between observed column origins.
COLUMN_BOUNDS: list[tuple[str, float]] = [
    ("row_no", 76.0),
    ("protein", 90.0),
    ("signal_peptide", 180.0),
    ("sequence", 225.0),
    ("media", 436.0),
    ("od", 479.0),
    ("iptg", 508.0),
    ("temperature", 544.0),
    ("time", 612.0),
    ("yield_mgl", 662.0),
    ("level", 707.0),
    ("reference", 749.0),
]

FIRST_PAGE = 3
LAST_PAGE = 31

# Anything printed below this offset is the page-number footer, not table content.
FOOTER_TOP = 555.0

STANDARD_RESIDUES = set("ACDEFGHIKLMNPQRSTVWY")

# Columns whose wrapped fragments must be concatenated without a space, so that
# "0.5-" + "0.6" reassembles into "0.5-0.6" instead of "0.5- 0.6".
JOIN_DIRECT = {
    "sequence",
    "od",
    "iptg",
    "temperature",
    "time",
    "yield_mgl",
    "level",
    "media",
}

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
# A numeric cell is numbers joined by range or list separators, and nothing else.  The supplement
# mixes ASCII hyphens with typographic dashes inside the same column.
NUMERIC_CELL_RE = re.compile(r"\d*\.?\d+(?:\s*[-–—−/:,]\s*\d*\.?\d+)*")
LEVEL_RE = re.compile(r"\(?\b(L|M|H)\b\)?", re.IGNORECASE)


def column_for(x0: float) -> str:
    name = ""
    for candidate, bound in COLUMN_BOUNDS:
        if x0 >= bound:
            name = candidate
        else:
            break
    return name


@dataclass
class RawRecord:
    row_no: str
    columns: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))


def assemble_rows(pages) -> tuple[list[RawRecord], list[str]]:
    """Walk every word on every page, starting a new record at each "No" anchor."""
    records: list[RawRecord] = []
    stray: list[str] = []
    current: RawRecord | None = None

    for page_no, page in enumerate(pages, start=FIRST_PAGE):
        words = sorted(page.extract_words(), key=lambda w: (w["top"], w["x0"]))
        for word in words:
            if word["top"] > FOOTER_TOP:
                continue
            col = column_for(word["x0"])
            text = word["text"]
            if col == "row_no":
                if re.fullmatch(r"\d+", text):
                    current = RawRecord(row_no=text)
                    records.append(current)
                    continue
                if current is not None:
                    stray.append(f"p{page_no}: unparsed No-column token {text!r}")
                continue
            if current is None:
                continue  # table caption or header row above the first data record
            current.columns[col or "unassigned"].append(text)

    return records, stray


# Characters the PDF text layer drops in front of numbers (row 78 reads "`12-16" for the time
# range).  Scrubbed only from the condition and yield cells, so a marker inside a sequence stays.
PDF_JUNK = str.maketrans("", "", "`*^~")
SCRUBBED_COLUMNS = {"od", "iptg", "temperature", "time", "yield_mgl"}


def collapse(record: RawRecord, col: str) -> str:
    tokens = record.columns.get(col, [])
    if not tokens:
        return ""
    if col in JOIN_DIRECT:
        text = re.sub(r"\s*([-/])\s*", r"\1", "".join(tokens))
    else:
        text = re.sub(r"\s+", " ", " ".join(tokens))
    return (text.translate(PDF_JUNK) if col in SCRUBBED_COLUMNS else text).strip()


def numbers_in(cell: str) -> list[float]:
    """Numbers a cell genuinely reports as a measurement.

    A cell counts only if it is nothing but numbers joined by range or list separators.  The
    supplement otherwise drops notes into numeric columns: 13 rows carry "24h" in the OD column,
    which is a duration, and one carries "10%" for induction, which is a lactose level, not 10 mM
    IPTG.  Harvesting 24 and 10 as process conditions would invent values the paper never stated,
    so those cells keep their raw text and are counted as unspecified.
    """
    if not NUMERIC_CELL_RE.fullmatch(cell.strip()):
        return []
    return [float(n) for n in NUMBER_RE.findall(cell)]


def span(cell: str) -> tuple[float | None, float | None, int]:
    values = numbers_in(cell)
    if not values:
        return None, None, 0
    return min(values), max(values), len(values)


def classify(yield_mgl: float | None) -> str:
    """Paper thresholds. Boundary read as high-inclusive: the table labels 50.0 mg/L as H."""
    if yield_mgl is None:
        return ""
    if yield_mgl < 0.5:
        return "L"
    if yield_mgl < 50:
        return "M"
    return "H"


@dataclass
class Row:
    row_no: int
    protein: str
    signal_peptide: str
    sequence: str
    seq_length: int
    media: str
    od: str
    od_min: float | None
    od_max: float | None
    od_n: int
    iptg: str
    iptg_min: float | None
    iptg_max: float | None
    iptg_n: int
    temperature: str
    temperature_min: float | None
    temperature_max: float | None
    temperature_n: int
    time: str
    time_min: float | None
    time_max: float | None
    time_n: int
    yield_mgl: str
    yield_min: float | None
    yield_max: float | None
    yield_n: int
    level_reported: str
    level_codes: str
    level_ambiguous: bool
    rsm_optimized: bool
    level_from_yield_max: str
    level_disagrees: bool
    condition_fully_specified: bool
    nonstandard_residues: str
    reference: str


def to_row(record: RawRecord) -> Row:
    raw_yield = collapse(record, "yield_mgl")
    sequence = collapse(record, "sequence").upper()
    level_raw = collapse(record, "level")
    codes = sorted({c.group(1).upper() for c in LEVEL_RE.finditer(level_raw)})
    od_min, od_max, od_n = span(collapse(record, "od"))
    ip_min, ip_max, ip_n = span(collapse(record, "iptg"))
    t_min, t_max, t_n = span(collapse(record, "temperature"))
    h_min, h_max, h_n = span(collapse(record, "time"))
    y_min, y_max, y_n = span(raw_yield)
    derived = classify(y_max)

    return Row(
        row_no=int(record.row_no),
        protein=collapse(record, "protein"),
        signal_peptide=collapse(record, "signal_peptide"),
        sequence=sequence,
        seq_length=len(sequence),
        media=collapse(record, "media"),
        od=collapse(record, "od"),
        od_min=od_min,
        od_max=od_max,
        od_n=od_n,
        iptg=collapse(record, "iptg"),
        iptg_min=ip_min,
        iptg_max=ip_max,
        iptg_n=ip_n,
        temperature=collapse(record, "temperature"),
        temperature_min=t_min,
        temperature_max=t_max,
        temperature_n=t_n,
        time=collapse(record, "time"),
        time_min=h_min,
        time_max=h_max,
        time_n=h_n,
        yield_mgl=raw_yield,
        yield_min=y_min,
        yield_max=y_max,
        yield_n=y_n,
        level_reported=level_raw,
        level_codes="/".join(codes),
        level_ambiguous=len(codes) > 1,
        rsm_optimized="RSM" in level_raw.upper(),
        level_from_yield_max=derived,
        level_disagrees=bool(codes) and derived and derived not in codes,
        condition_fully_specified=all(
            n == 1 for n in (od_n, ip_n, t_n, h_n, y_n)
        ),
        nonstandard_residues="".join(sorted(set(sequence) - STANDARD_RESIDUES)),
        reference=collapse(record, "reference"),
    )


def report(rows: list[Row], stray: list[str]) -> list[str]:
    problems = list(stray)
    numbers = [r.row_no for r in rows]
    if numbers != list(range(1, len(numbers) + 1)):
        missing = sorted(set(range(1, max(numbers) + 1)) - set(numbers))
        problems.append(f"row numbers not contiguous; missing {missing[:40]}")
    for r in rows:
        if not r.sequence:
            problems.append(f"row {r.row_no}: empty sequence")
        elif r.nonstandard_residues:
            problems.append(
                f"row {r.row_no}: non-standard residues {r.nonstandard_residues!r}"
            )
        if r.yield_max is None:
            problems.append(f"row {r.row_no}: no numeric yield in {r.yield_mgl!r}")
        if not r.level_codes:
            problems.append(f"row {r.row_no}: no level code in {r.level_reported!r}")
        elif r.level_disagrees:
            problems.append(
                f"row {r.row_no}: reported {r.level_codes} but yield_max="
                f"{r.yield_max} implies {r.level_from_yield_max}"
            )
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", type=Path, required=True, help="supplementary PDF (mmc1)")
    ap.add_argument("--out", type=Path, required=True, help="destination CSV")
    args = ap.parse_args()

    with pdfplumber.open(args.pdf) as pdf:
        records, stray = assemble_rows(pdf.pages[FIRST_PAGE - 1 : LAST_PAGE])

    rows = [to_row(r) for r in records]
    problems = report(rows, stray)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(Row.__dataclass_fields__)
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    full = [r for r in rows if r.yield_max is not None]
    lengths = sorted(r.seq_length for r in rows if r.seq_length)
    yields = sorted(r.yield_max for r in full)
    specified = [r for r in rows if r.condition_fully_specified]

    print(f"rows extracted                 : {len(rows)}")
    print(f"distinct protein names         : {len({r.protein.lower() for r in rows})}")
    print(f"distinct SP-protein combos     : {len({(r.signal_peptide.lower(), r.protein.lower()) for r in rows})}")
    print(f"distinct signal peptides       : {len({r.signal_peptide.lower() for r in rows})}")
    print(f"sequence length                : min {lengths[0]} median {lengths[len(lengths)//2]} max {lengths[-1]}")
    print(f"yield_max mg/L                 : min {yields[0]} median {yields[len(yields)//2]} max {yields[-1]}")
    print(f"rows with a single yield value  : {sum(1 for r in full if r.yield_n == 1)} / {len(full)}")
    print(f"rows with a yield range         : {sum(1 for r in full if r.yield_n > 1)}")
    print(f"rows with an ambiguous level    : {sum(1 for r in rows if r.level_ambiguous)}")
    print(f"rows flagged RSM-optimized     : {sum(1 for r in rows if r.rsm_optimized)}")
    print(f"rows with fully specified single condition : {len(specified)}")
    print("level distribution (reported)   : "
          + ", ".join(f"{c}={sum(1 for r in rows if r.level_codes==c)}" for c in ("L", "M", "H", "ambiguous")))
    print(f"reported level vs yield_max disagreement   : {sum(1 for r in rows if r.level_disagrees)}")
    print(f"problem records                : {len(problems)}")
    for p in problems[:40]:
        print("  -", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
