# Recombinant protein expression condition recommender

MVP that helps a wet-lab scientist decide **which expression conditions to try first** when
cloning a new recombinant protein into the *E. coli* periplasm.

Starting point: [Packiam et al., *Comput. Struct. Biotechnol. J.* 20 (2022) 2909–2920,
DOI 10.1016/j.csbj.2022.06.006](https://doi.org/10.1016/j.csbj.2022.06.006) — PERISCOPE-Opt.

## Reproducibility audit of the published data

The paper describes **461 process-condition records** across 84 proteins and 103
signal-peptide–protein (SP-protein) combinations (Supplementary Table S1), and its model is
trained on one row per condition.

Table S2 of the supplement — the only per-record data published — contains **103 rows, not
461**. It reports one row *per SP-protein combination*, giving the **maximum** expression
achieved, and the condition cells are usually spans rather than the specific optimum:

| observed | value |
|---|---|
| rows in Table S2 | 103 |
| SP-protein combinations | 103 (matches Table S1) |
| distinct protein names | 83 (Table S1 states 84) |
| rows whose condition and yield are all single values | **50** |
| rows reporting a yield *range* | 20 |
| rows whose reported class straddles a threshold | 10 |
| rows flagged as response-surface-optimized | 14 |

So the per-condition response surface that PERISCOPE-Opt is trained on is **not recoverable
from the public supplementary material**. This project therefore states its target explicitly:

> **Task actually supported by public data:** given an SP-protein sequence, predict the
> *best achievable* periplasmic yield. Recommending *which* condition attains it is only
> learnable from the 50 fully specified rows and is treated as a separately evaluated,
> weaker claim rather than folded into a headline metric.

## Layout

```
scripts/extract_table_s2.py    parses the supplementary PDF into a tidy CSV
data/processed/table_s2.csv    103 rows, raw cell strings + parsed numeric spans
requirements.txt
```

Source PDFs are intentionally **not** committed (the article is CC BY-NC-ND). Re-fetch from
the DOI if needed.

## Run

```bash
pip install -r requirements.txt
python scripts/extract_table_s2.py \
    --pdf 1-s2.0-S2001037022002203-mmc1.pdf \
    --out data/processed/table_s2.csv
```

The script prints a validation report: row contiguity, sequence alphabet, yield parse
failures, and reported-vs-derived expression class. Expression classes are defined by the
paper as low < 0.5 mg/L, medium 0.5–50 mg/L, high ≥ 50 mg/L.

## Parsing approach

The PDF table has no embedded structure. Rows are recovered from word bounding boxes; the
narrow left-hand "No" column carries one integer per record and is used as the row anchor,
so wrapped sequence lines and records split across page breaks reassemble correctly. Page
footers print a bare page number in the horizontal middle of the page, which lands inside
the sequence column, and are removed by a bottom-band filter.
