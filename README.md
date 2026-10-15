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
rpp/experiments.py             the five evidence tables
rpp/advisor.py                 the shipped product path: rank, error-band, and what makes an answer refuse
scripts/extract_table_s2.py    parses the supplementary PDF into a tidy CSV
scripts/build_features.py      CSV in, feature matrix out
scripts/run_model.py           runs the experiments, prints the tables, writes model_results.csv
scripts/advise.py              FASTA in, ranked shortlist with warnings out
tests/test_extraction.py       cell-level reading rules
tests/test_features.py         hand-checkable fixtures, no test runner needed
tests/test_modelling.py        split integrity, selection visibility, score arithmetic
tests/test_advisor.py          what the recommender refuses, and what it declines to rank
data/processed/table_s2.csv    103 rows, raw cell strings + parsed numeric spans
data/processed/features_tier_ab.csv
data/processed/model_results.csv
data/examples/candidates.fa    six query constructs, two of them deliberately unanswerable
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

`r(log)` is the Pearson correlation on the log scale - how close the points are to a straight line.
`rho(log)` is the same test on order alone: it only asks whether higher-yield proteins were ranked
higher, and ignores how far apart the model put them. `H in top10` is the share of genuinely
high-yield rows (>50 mg/L) among the ten the model ranks highest, against a base rate of 0.24 (25 of
103 are high); rows that land on the same predicted number are counted as the coin flip they are, by
averaging over every way of breaking the tie. `distinct` and `biggest tie` are not accuracy measures:
they say how many different answers the model actually gave for 103 proteins, and how many rows share
the single most common one.

| held out | model | k | r(log) | rho(log) | typical x off | acc | MCC | H in top10 | distinct | biggest tie |
|---|---|---|---|---|---|---|---|---|---|---|
| nothing (random) | ridge | 5 | 0.222 | 0.097 | 3.73 | 0.605 | 0.211 | 0.60 | 48 | 0.14 |
| combination | ridge | 5 | 0.298 | 0.081 | 3.39 | 0.602 | 0.214 | 0.70 | 49 | 0.13 |
| target protein | ridge | 5 | 0.257 | 0.163 | 3.86 | 0.621 | 0.269 | 0.60 | 44 | 0.12 |
| target protein | ridge | 20 | 0.138 | -0.003 | 4.00 | 0.621 | 0.258 | 0.70 | 102 | 0.02 |
| target protein | random forest | 5 | 0.208 | 0.131 | 4.12 | 0.602 | 0.214 | 0.40 | 44 | 0.12 |
| target protein | **bagged ridge - shipped** | 5 | 0.156 | -0.036 | 4.01 | 0.573 | 0.043 | 0.60 | 102 | 0.02 |
| signal peptide | ridge | 20 | 0.175 | 0.081 | 3.67 | 0.612 | 0.248 | 0.50 | 98 | 0.04 |
| anything | predict the fold median | - | ~0 | ~0 | 4.4 | 0.573 | 0.000 | 0.14-0.29 | 4-5 | 0.41 |

Reading: the yield *number* is weak (r 0.14-0.30, typically off by a factor of 3.4-4), and `rho` is
lower again - the order is barely better than a shuffle. Class accuracy of 0.62 sits only just above
the 0.573 of "call everything medium", which is the honest answer to the paper's 75-81%: at this
sample size the useful output is a shortlist, not a label. The `r(log)` of the constant baseline is
meaningless by construction (each fold has its own median) and is printed only so the other columns
have a floor. The next section is about why the two right-hand columns are in this table at all.

### Which five columns is a coin flip

The single best-scoring configuration - ridge, five features selected by ANOVA, target protein held
out - gives r = 0.257. Fit that same pipeline once on all 103 rows and ask which five columns it
chose:

```
OccD_PM  OccD_RE  OccD_YH  OFD_PM  OFD_YH
```

three two-letter patterns (Pro-Met, Arg-Glu, Tyr-His), read as a count and as a frequency. That is
the whole model. Now check the data by hand, no code required: **41 of the 103 published constructs
contain at least one of those three dipeptides; 62 contain none of them.** For those 62, all five
columns the model reads have the same value - zero - so they must receive one identical number. The
fitted model prints 17 distinct predictions for 103 proteins, and the biggest group holds 62 of
them.

This is not a bug. It is what selecting 5 columns out of 1144 with 103 rows does: the ranking is
noise-dominated, so a handful of rare patterns win, and the answer depends on the draw rather than on
biology. Two things follow, and both are measured here rather than argued.

* **Cross-validation partly hides it.** Every fold re-selects its own five columns, so different
  folds give different rows different numbers - which is why the table above shows 44 distinct values
  rather than 17. The headline `H in top10` = 0.60 survives tie-aware counting, so it is not an
  artifact of tie-breaking; it is a draw from an unstable procedure.
* **Bagging exposes and fixes it.** The shipped learner refits the same select-then-fit sequence on
  20 bootstrap resamples of the training rows and averages their answers. Every bag selects its own
  columns, so a construct with none of the winning patterns is no longer pinned to one value:
  102-103 distinct predictions, biggest tie 0.02. Asking which columns the 20 draws picked is the
  stability measurement itself:

  ```
  20 draws of 5 columns each landed on 56 different columns, none of them in every draw;
  most-chosen: OccD_RE x13, OccD_PM x8, OFD_PM x7, OccD_YH x5, OccD_MQ x4, OccD_PE x4
  ```

  100 picks spread over 56 columns; the most reproducible feature in the whole 1144 is chosen by 13
  draws out of 20, and 41 columns are chosen by exactly one. The same Arg-Glu / Pro-Met / Tyr-His
  family still leads, but nothing survives a resample intact.

The cost is honest and it is the interesting number. Across four seeds, bagged ridge k = 5 gives

| seed | r(log) | rho(log) | typical x off | H in top10 |
|---|---|---|---|---|
| 0 | 0.156 | -0.036 | 4.01 | 0.60 |
| 1 | 0.139 | +0.076 | 4.27 | 0.40 |
| 2 | 0.020 | -0.069 | 4.43 | 0.30 |
| 3 | 0.118 | -0.007 | 3.94 | 0.50 |

The shortlist's hit rate - the one figure this project claimed was clearly better than chance - moves
between 0.30 and 0.60 on nothing but which bootstrap rows got drawn, against 0.24 by guessing. Note
the opposite asymmetry too: the *single* draw does not move at all across those seeds (r = 0.257 at
every one), because grouped protein folds and ANOVA scores are deterministic. The instability is not
in which rows are held out; it is in which columns get read, and only bagging is able to show it.
The shipped tool therefore reports the bags' own disagreement per query, and refuses to rank a
construct whose bags disagree more than the model's measured error.

**Where the correlation actually lives.** Because the winning columns are rare ones, the standard
"clean up sparse features" step deletes the entire result:

| feature pool | columns | k | r(log) | rho(log) | H in top10 |
|---|---|---|---|---|---|
| everything | 1144 | 5 | 0.257 | 0.163 | 0.60 |
| drop columns nonzero in under 5% of rows | 1108 | 5 | 0.140 | 0.042 | 0.40 |
| ... under 10% | 1058 | 5 | 0.129 | 0.004 | 0.50 |
| ... under 20% | 936 | 5 | 0.056 | -0.065 | 0.50 |
| ... under 50% | 506 | 5 | -0.106 | -0.249 | 0.20 |
| no dipeptide columns at all | 344 | 5 | -0.163 | -0.236 | 0.20 |
| everything, scored by mutual information | 1144 | 5 | -0.087 | -0.063 | 0.00 |

Two defensible preprocessing choices - ignoring mostly-zero columns, using a different scorer - turn
a positive correlation negative. Whatever r = 0.257 is, it is not a general rule about periplasmic
expression; it is this table, this scorer and this width. The product consequence is that the tool
must be sold as a triage queue whose top-ten hit rate is somewhere between "a bit better than
chance" and "clearly better", never as a calibrated predictor.

**Leakage - what selecting before splitting is worth.**

| held out | model | k | r nested | r leaked | rho nested | rho leaked | H top10 nested | leaked |
|---|---|---|---|---|---|---|---|---|
| target protein | ridge | 5 | 0.257 | 0.427 | 0.163 | 0.153 | 0.60 | 0.80 |
| target protein | ridge | 20 | 0.138 | 0.514 | -0.003 | 0.399 | 0.70 | 0.80 |
| target protein | random forest | 20 | 0.064 | 0.450 | -0.029 | 0.364 | 0.50 | 0.70 |
| signal peptide | ridge | 20 | 0.175 | 0.452 | 0.081 | 0.342 | 0.50 | 0.80 |

Ranking the 1144 columns against all 103 labels before splitting lifts `r(log)` by 1.7-7x and, more
tellingly, takes the order-only figure from *nothing* (-0.003: the ranking is a shuffle) to 0.399 -
while class accuracy moves 0.621 -> 0.631. A model report that leads with accuracy can hide this
completely; one that leads with correlation cannot.

**Ablation - Tier A against Tier B.** Tier B is 54 columns of weighted averages over Tier A's 20
frequencies, so it cannot carry information Tier A lacks. Selecting from A+B and selecting from A
agree to three decimals at k = 5 (r 0.257 and 0.257) and to 0.003 at k = 20, because the scores almost
never rank a scale column into the top k. Tier B alone is *worse than the baseline*: ridge on the 54
scale columns alone, protein held out, gives r = -0.212 and rho = -0.208 - on a 103-row table,
selecting from published hydropathy and charge numbers is worse than guessing. So the paper's
11,985-column input is packaging, and cutting it costs nothing. This is the result that justifies
shipping a 1144-column matrix and a 20-frequency explanation to a wet lab.

**Do the process conditions help?** On the 36 rows that state OD, IPTG, temperature and time
completely, with those four forced into every fold:

| model | k | features | r(log) | rho(log) | typical x off | acc | MCC |
|---|---|---|---|---|---|---|---|
| ridge | 20 | sequence only | 0.372 | 0.407 | 5.91 | 0.528 | 0.211 |
| ridge | 20 | sequence + conditions | 0.385 | 0.427 | 5.48 | 0.556 | 0.296 |
| random forest | 20 | sequence only | 0.234 | 0.274 | 5.67 | 0.472 | 0.052 |
| random forest | 20 | sequence + conditions | 0.222 | 0.302 | 4.58 | 0.472 | 0.052 |

These are the best numbers in the project - and the best ranking figure (rho 0.43) is on the arm that
gets to *see* the four process variables. Even so: conditions move MCC from 0.21 to 0.30 on 36 rows,
one split, folds of seven. That is not enough to claim the model knows which condition to pick; it is
enough to say the published data cannot settle it. The recommender therefore ships with an explicit
refusal rather than a ranked condition table, which is the next section.

## The shipped path

Everything above is a measurement. This is what a wet-lab user actually gets, from
`python scripts/advise.py --fasta data/examples/candidates.fa --signal-peptide pelB` - six constructs,
four of them published rows that the model was fitted on, deliberately, so that the refusals are on
screen rather than in a slide note:

```
20 draws of 5 columns each landed on 56 different columns, none of them in every draw;
most-chosen: OccD_RE x13, OccD_PM x8, OFD_PM x7, OccD_YH x5, ...
typical error is a factor of 4.01, rho(log) -0.036 on a held-out protein

 1. Creatinase-Pseudomonas    106 mg/L  [26.4 - 424]  class H  bags disagree 5.9x
    ! the 20 selection draws disagree by a factor of 5.9, more than the model's measured
      typical error of 4.0x: ... its position is essentially the table's middle
    ! this exact construct is training row 25, measured at 479 mg/L; use that number, the
      model adds nothing
 ...
 -. Seleno-protein-with-U     no prediction
    ! no features, no prediction: non-standard residues ['B', 'U'] in sequence

No condition is recommended.  On the 36 rows that state OD, IPTG, temperature and time
completely, adding those four numbers moves MCC from 0.211 to 0.296 on one split seed -
too little to claim the model knows which flask to set up.
```

Per query the advisor checks, in this order: is the string protein at all (else refuse); did the
named signal peptide resolve against the table, and is it actually the start of the sequence (else
say which half of the feature vector is now missing); is the length inside the 80-668 residues every
scored construct falls in (else call it extrapolation); do the 20 selection draws agree, and if not,
is their disagreement worse than the model's own measured error (then say the construct is unrankable
rather than ranking it); is the answer outside the yield range the table was ever shown; and is this
exact sequence already measured, in which case quote the number the paper paid for. Output is a
ranked queue with a multiplication band, not a yield forecast.

## When not to trust this

The tool is a **triage shortlist**: it says "these constructs are more likely than those to give you
material in the periplasm". Everything below is a measured limit on that claim, not a disclaimer
written after the fact - and each one is enforced in `rpp/advisor.py`, so the demo refuses out loud
instead of printing a confident number.

**Do not read the yield number.** Typical error is a factor of 3.7-4.4 on the shipped learner (the
fold median scores 4.5). A prediction of 10 mg/L is consistent with anywhere from roughly 3 to 40
mg/L. It is usable for ordering a queue, never for forecasting titre, planning media volume, or any
downstream economics.

**Do not read the class label.** The shipped bagged learner scores 0.573 class accuracy on a held-out
protein - which is *exactly* the score of "call everything medium", because the dataset is 59 of 103
medium. L/M/H is displayed as the dataset's own vocabulary, not as a verdict. The ten rows reporting
0 mg/L are the sharpest case: log10(1+0) = 0 and any positive prediction is wrong by construction.

**The shortlist is better than chance, by an amount that moves.** `H in top10` of 0.60 against a base
rate of 0.24 is the one figure that looks like a result. Under four bootstrap seeds the shipped learner
returns 0.60, 0.40, 0.30, 0.50. The honest sentence is "somewhere between a bit better than chance and
clearly better", and nothing here establishes that rank 1 beats rank 5: the metric is a set-level hit
rate and the shortlist is presented as a set.

**The selected features are not a biological finding.** One draw of "pick the five best of 1144
columns" on all 103 rows chooses Pro-Met, Arg-Glu and Tyr-His dipeptide counts, and 62 of the 103
published constructs contain none of them - so those 62 proteins are handed one identical answer.
Deleting mostly-zero columns, or scoring features with mutual information instead of ANOVA, takes the
headline correlation from +0.26 to -0.16 and -0.09. Bagging removes the collapse (102 distinct
answers out of 103) and removes most of the correlation with it - and shows how unstable the choice
of five really was, with 56 different columns appearing across 20 draws and the best of them surviving
13.

**Stay inside the domain the folds cover.** Held-out groups are drawn from the same 80 target proteins,
36 signal peptides, one organism (*E. coli*), one destination (periplasm), one publication, and
sequences of 80-668 residues (median 217). A protein outside that length range, a non-*E. coli* host,
cytoplasmic or secreted routing, or a signal peptide absent from these 36 is extrapolation. `scripts/advise.py`
checks each of these and says which one it hit; it also refuses a sequence containing letters no model
was fitted on, and refuses to invent features for a mature chain too short to average a scale over.

**Trust a measured number over this tool whenever you have one.** If the query is a construct that is
already in Table S2, the advisor names the row and its measured yield and says the model adds nothing.

**The signal peptide is inferred, not given.** 67 of 103 rows have a recovered SP; the other 36 rows
carry one of the 25 labels that could not be resolved (`native`, singletons, the `sspad` domain
fusion) and enter the matrix with the SP block empty. For those rows the model is only ever reasoning
about the mature chain.

**The condition recommender is unproven.** 36 of 103 rows state all four process variables as single
values; 20 more report a yield as a range and 10 straddle a class threshold. Adding OD, IPTG,
temperature and time raises MCC from 0.211 to 0.296 on those 36 rows with one seed. Treat any suggested
condition as "the paper's own optimum", never as the model's advice - which is why the CLI prints a
refusal to recommend a condition instead of a condition table.

**Discount any number from a pipeline that selected before splitting.** The leakage table lifts
`r(log)` 1.7-7x and takes rank correlation from -0.003 (no ordering at all) to 0.399, while accuracy
moves by 0.010. That is why correlation and rank order are the reported metrics here: the source
paper's headline accuracy figures are compatible with a much weaker model than they suggest.

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

# 3. the five evidence tables
python scripts/run_model.py
# add --quick for one repeat instead of three, or --selector mi for mutual information

# 4. the product path: rank a FASTA of candidates and show what to distrust
python scripts/advise.py --fasta data/examples/candidates.fa --signal-peptide pelB
# a header written as name|sp=ompA overrides --signal-peptide for that one record

# 5. checks, no test runner needed
python tests/test_extraction.py
python tests/test_features.py
python tests/test_modelling.py
python tests/test_advisor.py
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
