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
rpp/composition.py             Tier A: counts, frequencies, per-residue longest runs, dipeptides, groups
rpp/physicochemistry.py        Tier B: scales averaged over the sequence, pI, MW, charge, stability
rpp/signal_peptides.py         recovers the signal peptide from fused sequences
rpp/features.py                assembles one flat feature row
scripts/extract_table_s2.py    parses the supplementary PDF into a tidy CSV
scripts/build_features.py      CSV in, feature matrix out
tests/test_features.py         hand-checkable fixtures, no test runner needed
data/processed/table_s2.csv    103 rows, raw cell strings + parsed numeric spans
data/processed/features_tier_ab.csv
```

Source PDFs are intentionally **not** committed (the article is CC BY-NC-ND). Re-fetch from
the DOI if needed.

## Sequence features

A feature is any function from a variable-length string over a 20-letter alphabet to a
number. Two families are implemented:

* **Tier A - composition.** Count letters: 20 amino-acid occurrences and frequencies, the
  longest consecutive run of each residue, 400 dipeptide occurrences and frequencies, and
  residue-group frequencies.
* **Tier B - physicochemistry.** Each residue carries published constants (hydropathy,
  molecular weight, ionisable groups). Averaging a scale over the sequence gives one
  number; `peptides` and Biopython both ship the tables, so nothing is retyped here.

Tier B adds no information over Tier A: an averaged scale is a linear combination of the
20 frequency columns, which live on a 19-dimensional simplex. Its value is that it hands a
~100-observation model a physically meaningful projection instead of making it learn 20
weights itself. That is regularisation from prior knowledge, not feature discovery.

The paper's 11,985 initial features decompose exactly as
FC1 149 + FC2 800 (400 dipeptides x count and frequency) + FC3 11,026 (= 149 choose 2, all
pairwise products of FC1) + FC4 10 (= 4 process variables + 6 pairwise products). FC3 is a
degree-2 expansion of an already rank-~22 block and is not reproduced; gradient boosting
finds interactions itself.

### Width against sample size

| | |
|---|---|
| rows | 103 |
| feature columns | 1163 (363 excluding dipeptides) |
| observations per feature | 0.089 |
| PERISCOPE-Opt's own ratio at full width | 461 / 11,985 = 0.038 |

The matrix is emitted wide on purpose. Selection is a modelling step and must live inside
cross-validation; reducing columns here would silently bake the whole dataset into every
split.

### Signal peptides

Table S2 gives the signal peptide as a free-text label plus one fused sequence, so the
cleavage point has to be recovered. Only ~15 distinct labels recur across the 103 rows, and
all rows sharing a label begin with the same segment, so the **longest common prefix within
a label group is the signal peptide**.

* 11 of 36 labels resolved, covering **67 / 103** rows.
* Every resolved sequence matches the canonical published signal peptide (ompA
  `MKKTAIAIAVALAGFATVAQA`, pelB `MKYLLPTAAAGLLLLAAQPAMA`, malE, torA, lpp, ompC), which
  independently validates the inference.
* 25 labels unresolved: `native` (n=7, by definition variable), 16 singletons, and
  `sspad` whose shared prefix is a 62-aa domain fusion rather than a signal peptide.
* Known limitation: the prefix cannot stop at the cut site when two mature proteins happen
  to share residues immediately after it, so inferred length is an **upper bound**.
  `tests/test_features.py` pins this behaviour rather than hiding it.

Because translocation efficiency and mature-protein solubility are different mechanisms,
the SP and the mature chain get their own feature blocks instead of one pooled vector as
in the paper.

### Not reproduced, and why

| feature | reason |
|---|---|
| `Helix_PHD`, `Sheet_PHD`, `Coil_PHD`, `Helix_to_Sheet_PHD` | needs Rost & Sander per-residue propensities, which Biopython does not ship. Biopython's `secondary_structure_fraction` is a residue-group membership count (helix = EMALK, turn = NPGSD, sheet = VIYFWLT), emitted as `ss_*` and **not** equivalent. |
| `Expno_AA_TM` | needs the Ulmschneider transmembrane propensity table. |
| `Pred_Sol`, folding rate | outputs of separate predictors; feeding one model's output into another inherits its bias and caps accuracy at its level. |
| `pI_Protpi` | separate implementation with its own pKa set. |

The pI caveat is measurable: on one test sequence Biopython returns 5.271 and `peptides`
returns 5.826. That is why the paper carries both a plain `pI` and a `pI_Protpi` column -
the same physical quantity really does differ between tools. We emit one and name the
implementation.

## Run

```bash
pip install -r requirements.txt
python scripts/extract_table_s2.py \
    --pdf 1-s2.0-S2001037022002203-mmc1.pdf \
    --out data/processed/table_s2.csv
python scripts/build_features.py
python tests/test_features.py
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
