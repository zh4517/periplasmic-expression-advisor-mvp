"""Turn the extracted feature matrix into something a model can be fitted to.

Three decisions live here rather than in the modelling code, because they are statements about
the *data*, not about the algorithm:

* **Target.** Table S2 reports the best yield found for each SP-protein combination, sometimes as
  a span ("17.8-45").  The claim the published rows can support is "how high can this construct
  be pushed", so the target is the top of the span, on a log scale.
* **Grouping keys.** The protein and signal-peptide cells are free text from a PDF, with casing and
  line-wrap damage.  Grouped cross-validation depends on these keys being the same construct being
  spelled the same way, so the normalisation is written down here and tested.
* **Tier of each column.** Which columns are counts and which are weighted sums of counts decides
  the ablation, and it is read out of the code that generated them instead of guessed from names.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from rpp.features import META_COLUMNS, feature_tiers

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "data" / "processed" / "features_tier_ab.csv"

# Ten rows report a yield of exactly 0 mg/L and log10(0) is undefined, so every yield is shifted
# by this pseudo-count before the transform.  It also caps the fold error a zero row can produce.
YIELD_OFFSET = 1.0

# PERISCOPE-Opt classes: low < 0.5, medium 0.5-50, high > 50 mg/L.  The supplement labels a row of
# exactly 50.0 as high, so the upper edge is high-inclusive.
CLASS_EDGES = (0.5, 50.0)

CONDITION_COLUMNS = ("od", "iptg", "temperature", "time")

DERIVED_COLUMNS = (
    "yield_best_mgl",
    "yield_low_mgl",
    "yield_is_single_value",
    "yield_span_mgl",
    "log10_yield",
    "expression_class",
    "protein_group",
    "signal_peptide_group",
    "combo_group",
    "conditions_numeric",
)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def protein_key(text: str) -> str:
    """Short identity of the target protein, e.g. 'Enhanced Green Flourescent protein (eGFP)' -> 'egfp'.

    The parenthesised acronym is what recurs across the table; grouping on the full description
    would let 'eGFP' written two ways sit on both sides of a split.
    """
    matches = re.findall(r"\(([^()]+)\)", _clean(text))
    if matches:
        return re.sub(r"[^a-z0-9]", "", matches[-1].lower())
    return re.sub(r"[^a-z0-9]", "", _clean(text).lower())


def signal_peptide_key(text: str) -> str:
    """Case- and punctuation-insensitive signal-peptide label, so 'PelB' and 'pelB' are one group."""
    return re.sub(r"[^a-z0-9]", "", _clean(text).lower())


def _numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """Numeric value of every cell, NaN where the PDF wrote a range or left it blank."""
    return frame.apply(lambda column: pd.to_numeric(column, errors="coerce"))


def load_dataset(path: Path = FEATURES_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    conditions = _numeric(df[list(CONDITION_COLUMNS)])

    yield_low = pd.to_numeric(df["yield_min"], errors="coerce")
    yield_best = pd.to_numeric(df["yield_max"], errors="coerce")
    span = yield_best - yield_low

    derived = pd.DataFrame(
        {
            "yield_low_mgl": yield_low,
            "yield_best_mgl": yield_best,
            "yield_span_mgl": span,
            "yield_is_single_value": span.fillna(1.0) == 0.0,
            # A pseudo-count keeps the ten rows that report 0 mg/L finite on a log scale.
            "log10_yield": np.log10(YIELD_OFFSET + yield_best),
            "expression_class": np.select(
                [yield_best < CLASS_EDGES[0], yield_best < CLASS_EDGES[1]],
                ["L", "M"],
                default="H",
            ),
            "protein_group": df["protein"].map(protein_key),
            "signal_peptide_group": df["signal_peptide"].map(signal_peptide_key),
            "conditions_numeric": conditions.notna().all(axis=1),
            **{f"{c}_num": conditions[c] for c in CONDITION_COLUMNS},
        }
    )
    derived["combo_group"] = derived["signal_peptide_group"] + "__" + derived["protein_group"]

    df = pd.concat([df, derived], axis=1)
    df = df.sort_values("row_no").reset_index(drop=True)

    # The extractor already asserted which rows state all four conditions.  Recomputing it from the
    # numbers has to agree, otherwise one of the two readings of the PDF is wrong.
    claimed = df["condition_fully_specified"].astype(str).str.lower().str.startswith("true")
    disagreement = df.loc[claimed != df["conditions_numeric"], "row_no"].tolist()
    if disagreement:
        raise AssertionError(
            f"condition_fully_specified disagrees on rows {disagreement[:10]}"
        )
    return df


def feature_columns(
    df: pd.DataFrame,
    tiers: tuple[str, ...] = ("A", "B"),
    include_conditions: bool = False,
) -> list[str]:
    """Sequence feature columns belonging to the requested tiers.

    Anything in the CSV that is neither a documented meta column nor a known feature is raised
    instead of silently dropped - a renamed column should break the run, not quietly shrink it.
    """
    tiers_map = feature_tiers()
    known = set(META_COLUMNS) | set(DERIVED_COLUMNS) | set(f"{c}_num" for c in CONDITION_COLUMNS)
    unknown = [c for c in df.columns if c not in known and c not in tiers_map]
    if unknown:
        raise ValueError(f"{len(unknown)} column(s) with no tier, e.g. {unknown[:5]}")

    columns = [c for c in df.columns if tiers_map.get(c) in tiers]
    if include_conditions:
        columns += [f"{c}_num" for c in CONDITION_COLUMNS]
    return columns


def describe(df: pd.DataFrame) -> str:
    tiers_map = feature_tiers()
    n_a = len(feature_columns(df, tiers=("A",)))
    n_b = len(feature_columns(df, tiers=("B",)))
    lines = [
        f"rows                                  : {len(df)}",
        f"yield known as a single value         : {int(df['yield_is_single_value'].sum())}",
        f"yield reported as a span              : {int((~df['yield_is_single_value']).sum())}",
        f"all four process conditions specified : {int(df['conditions_numeric'].sum())}",
        f"classes (L/M/H)                       : {df['expression_class'].value_counts().to_dict()}",
        f"distinct target proteins              : {df['protein_group'].nunique()}",
        f"distinct signal peptides              : {df['signal_peptide_group'].nunique()}",
        f"distinct SP-protein combinations      : {df['combo_group'].nunique()}",
        f"largest protein group                 : "
        f"{df['protein_group'].value_counts().index[0]} x "
        f"{int(df['protein_group'].value_counts().iloc[0])}",
        f"largest signal-peptide group          : "
        f"{df['signal_peptide_group'].value_counts().index[0]} x "
        f"{int(df['signal_peptide_group'].value_counts().iloc[0])}",
        f"features: Tier A {n_a} / Tier B {n_b} / both {n_a + n_b} "
        f"({len(tiers_map)} generated names)",
        f"observations per feature (A+B)        : {len(df) / (n_a + n_b):.3f}",
    ]
    return "\n".join(lines)
