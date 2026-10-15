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
| rows whose condition and yield are all single values | **36** |
| rows reporting a yield *range* | 20 |
| rows whose reported class straddles a threshold | 10 |
| rows flagged as response-surface-optimized | 14 |

The count of fully specified conditions was **50** in the first pass of this extractor and is 36
now. A cell-by-cell re-read found 13 rows writing `24h` in the OD column and one writing `10%` in
the IPTG column: an incubation time and a lactose level, which a numeric regex happily read as
OD = 24 and IPTG = 10 mM. `numbers_in()` now accepts a cell only if it is numbers and range
separators and nothing else, and `tests/test_extraction.py` pins both the acceptance of
`0.028–0.227` (the supplement mixes ASCII hyphens with typographic dashes) and the rejection of
`24h`. `rpp/dataset.py` recomputes the flag from the parsed numbers and refuses to run if it
disagrees with the extractor's own column.

So the per-condition response surface that PERISCOPE-Opt is trained on is **not recoverable
from the public supplementary material**. This project therefore states its target explicitly:

> **Task actually supported by public data:** given an SP-protein sequence, predict the
> *best achievable* periplasmic yield. Recommending *which* condition attains it is only
> learnable from the 36 fully specified rows and is treated as a separately evaluated,
> weaker claim rather than folded into a headline metric.

## Layout

```
rpp/composition.py             Tier A: counts, frequencies, per-residue longest runs, dipeptides, groups
rpp/physicochemistry.py        Tier B: scales averaged over the sequence, pI, MW, charge, stability
rpp/signal_peptides.py         recovers the signal peptide from fused sequences
rpp/features.py                assembles one flat feature row; feature_tiers() says which tier made each column
rpp/dataset.py                 target, expression class, grouping keys, tier lookup, PDF-cell re-checks
rpp/cv.py                      split strategies, and the guard that a grouped split shares no construct
rpp/modeling.py                selection-inside-the-fold pipelines, cross-fitted predictions and scores
rpp/experiments.py             the four evidence tables
scripts/extract_table_s2.py    parses the supplementary PDF into a tidy CSV
scripts/build_features.py      CSV in, feature matrix out
scripts/run_model.py           runs the experiments, prints the tables, writes model_results.csv
tests/test_extraction.py       cell-level reading rules
tests/test_features.py         hand-checkable fixtures, no test runner needed
tests/test_modelling.py        split integrity, selection visibility, score arithmetic
data/processed/table_s2.csv    103 rows, raw cell strings + parsed numeric spans
data/processed/features_tier_ab.csv
data/processed/model_results.csv
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
| columns in the CSV | 1163 = 19 descriptive + 1144 feature |
| Tier A / Tier B | 1090 / 54 |
| feature columns excluding the 800 dipeptide columns | 344 |
| observations per feature | 103 / 1144 = 0.090 |
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

## Evaluation

**Target.** `log10(1 + best yield)`. The offset keeps the ten rows that report 0 mg/L finite, and the
log turns the error scale into multiples, which is what a 0.6-to-700 mg/L range deserves: a
prediction of 1.2 for a true 0.6 is a doubling, while 400 for a true 480 is 17% off and far
better. Reported as `typical x off`, the median absolute error converted back to a factor.

**Splits.** 103 rows describe 80 distinct target proteins and 36 distinct signal peptides, so a
random split puts the same construct on both sides of the fold. Three grouped strategies hold out,
respectively, a whole SP-protein combination, a whole target protein, and a whole signal peptide.
Feature selection is fitted inside every training fold and never on the table.

`r(log)` is the Pearson correlation on the log scale, `H in top10` the share of genuinely high-yield
rows (>50 mg/L) among the ten the model ranks highest, against a base rate of 0.24.

| held out | model | k | r(log) | typical x off | acc | MCC | H in top10 |
|---|---|---|---|---|---|---|---|
| nothing (random) | ridge | 5 | 0.222 | 3.73 | 0.605 | 0.211 | 0.60 |
| combination | ridge | 5 | 0.298 | 3.39 | 0.602 | 0.214 | 0.70 |
| target protein | ridge | 5 | 0.257 | 3.86 | 0.621 | 0.269 | 0.60 |
| target protein | ridge | 20 | 0.138 | 4.00 | 0.621 | 0.258 | 0.70 |
| target protein | random forest | 5 | 0.208 | 4.12 | 0.602 | 0.214 | 0.40 |
| signal peptide | ridge | 20 | 0.175 | 3.67 | 0.612 | 0.248 | 0.50 |
| anything | predict the fold median | - | ~0 | 4.4 | 0.573 | 0.000 | 0.10-0.30 |

Reading: the yield *number* is weak (r 0.14-0.30, typically off by a factor of 3.4-4), but the
*ranking* is not - picking the top ten puts 6-7 true high-yield rows in the shortlist against 2.4
by chance. Class accuracy of 0.62 sits only just above the 0.573 of "call everything medium", which
is the honest answer to the paper's 75-81%: at this sample size the useful output is a shortlist, not
a label. The `r(log)` of the constant baseline is meaningless by construction (each fold has its own
median) and is printed only so the other columns have a floor.

**Leakage - what selecting before splitting is worth.**

| held out | model | k | r(log) nested | r(log) leaked | H in top10 nested | leaked |
|---|---|---|---|---|---|---|
| target protein | ridge | 20 | 0.138 | 0.514 | 0.70 | 0.80 |
| target protein | random forest | 20 | 0.064 | 0.450 | 0.50 | 0.70 |
| signal peptide | ridge | 20 | 0.175 | 0.452 | 0.50 | 0.80 |
| nothing (random) | ridge | 20 | 0.120 | 0.481 | 0.50 | 0.70 |

Ranking the 1144 columns against all 103 labels before splitting multiplies the correlation by
2.6-7x, without touching the accuracy figure much (0.621 -> 0.631). A model report that leads with
accuracy can hide this completely; one that leads with correlation cannot.

**Ablation - Tier A against Tier B.** Tier B is 54 columns of weighted averages over Tier A's 20
frequencies, so it cannot carry information Tier A lacks. Selecting from A+B and selecting from A
give figures that agree to three decimals, because the scores never rank a scale column into the top
k. Tier B alone is *worse than the baseline*: ridge on a held-out protein gives r = -0.21, pI against
a log-yield target chosen from 54 scale columns. So the paper's 11,985-column input is packaging, and
cutting it costs nothing. This is the result that justifies shipping a 1144-column matrix and a
20-frequency explanation to a wet lab.

**Do the process conditions help?** On the 36 rows that state OD, IPTG, temperature and time
completely, with those four forced into every fold:

| model | k | features | r(log) | typical x off | MCC |
|---|---|---|---|---|---|
| ridge | 20 | sequence only | 0.372 | 5.91 | 0.211 |
| ridge | 20 | sequence + conditions | 0.385 | 5.48 | 0.296 |
| random forest | 20 | sequence only | 0.234 | 5.67 | 0.052 |
| random forest | 20 | sequence + conditions | 0.222 | 4.58 | 0.052 |

Conditions move MCC from 0.21 to 0.30 on 36 rows and one split seed. That is not enough to claim the
model knows which condition to pick; it is enough to say the published data cannot settle it. The
recommender therefore ships with an explicit uncertainty statement rather than a ranked condition
table, which is the next section.

## When not to trust this

The tool is a **triage shortlist**: it says "these constructs are more likely than those to give you
material in the periplasm". Everything below is a measured limit on that claim, not a disclaimer
written after the fact.

**Do not read the yield number.** Typical error is a factor of 3.4-4 (`typical x off` in the tables
above; the fold median scores 4.4). A prediction of 10 mg/L is consistent with anywhere from roughly
3 to 35 mg/L. It is usable for ordering a queue, never for forecasting titre, planning media volume,
or any downstream economics.

**Do not read the class label.** Accuracy is 0.60-0.62; predicting "medium" for every row scores
0.573. The margin is inside the noise of a 103-row test set, so L/M/H is displayed as the dataset's
own vocabulary, not as a verdict. The ten rows reporting 0 mg/L are the sharpest case: log10(1+0) = 0
and any positive prediction is wrong by construction.

**Trust the ranking only near the top.** `H in top10` of 0.60-0.70 against a base rate of 0.24 is the
one figure clearly better than chance. Nothing here establishes that rank 1 beats rank 5; the metric
is a set-level hit rate, and the shortlist is presented as a set.

**Stay inside the domain the folds cover.** Held-out groups are drawn from the same 80 target proteins,
36 signal peptides, one organism (*E. coli*), one destination (periplasm), one publication, and
sequences of 80-668 residues (median 217). A protein outside that length range, a non-*E. coli* host,
cytoplasmic or secreted routing, or a signal peptide absent from these 36 is extrapolation, and the
demo says so explicitly rather than returning a confident number.

**The signal peptide is inferred, not given.** 67 of 103 rows have a recovered SP; the other 36 rows
carry one of the 25 labels that could not be resolved (`native`, singletons, the `sspad` domain
fusion) and enter the matrix with the SP block empty. For those rows the model is only ever reasoning
about the mature chain.

**The condition recommender is unproven.** 36 of 103 rows state all four process variables as single
values; 20 more report a yield as a range and 10 straddle a class threshold. Adding OD, IPTG,
temperature and time raises MCC from 0.211 to 0.296 on those 36 rows with one seed. Treat any suggested
condition as "the paper's own optimum", never as the model's advice.

**Discount any number from a pipeline that selected before splitting.** The leakage table multiplies
`r(log)` by 2.6-7x while accuracy moves by 0.010. That is why correlation is the reported metric here:
the source paper's headline accuracy figures are compatible with a much weaker model than they suggest.

What would change these answers: the full 461-row per-condition matrix (only the authors hold it), more
than one publication, and a blind wet-lab test of the top-10 hit rate on fresh constructs.

## Run

```bash
pip install -r requirements.txt

# 1. supplement PDF -> tidy per-record table (needs the CC BY-NC-ND source, not in the repo)
python scripts/extract_table_s2.py \
    --pdf 1-s2.0-S2001037022002203-mmc1.pdf --out data/processed/table_s2.csv

# 2. table -> Tier A + Tier B feature matrix
python scripts/build_features.py

# 3. the four evidence tables
python scripts/run_model.py
# add --quick for one repeat instead of three, or --selector mi for mutual information

# 4. checks, no test runner needed
python tests/test_extraction.py
python tests/test_features.py
python tests/test_modelling.py
```

Or run any step without installing anything, e.g.
`uv run --with pandas --with scikit-learn --with biopython==1.88 python scripts/run_model.py`.
Steps 2-4 read the committed CSVs, so the pipeline runs without the PDF.

The extraction script prints a validation report: row contiguity, sequence alphabet, yield parse
failures, and reported-vs-derived expression class. Expression classes are defined by the
paper as low < 0.5 mg/L, medium 0.5-50 mg/L, high >= 50 mg/L. `run_model.py` prints how many columns
each tier contributes, the held-out fold sizes per split, and refuses to continue if its own reading
of the condition cells disagrees with the extractor's.

## Parsing approach

The PDF table has no embedded structure. Rows are recovered from word bounding boxes; the
narrow left-hand "No" column carries one integer per record and is used as the row anchor,
so wrapped sequence lines and records split across page breaks reassemble correctly. Page
footers print a bare page number in the horizontal middle of the page, which lands inside
the sequence column, and are removed by a bottom-band filter.
