"""Rank candidate constructs, and say when a ranking is not owed an answer.

Every figure in `scripts/run_model.py` is a score on held-out proteins and held-out signal peptides
drawn from the 103 published rows.  A query outside that envelope is not covered by those numbers, so
this module returns a warning where a confident number would otherwise be printed.

The shipped learner averages twenty bootstrap draws of the whole select-then-fit sequence.  A single
draw over 1144 columns and 103 rows picks five rare dipeptide counts, and 62 of the 103 published
constructs contain none of them - so one draw answers 62 different proteins with one identical number.
Bagging removes that, and the disagreement between its bags is a per-query uncertainty that can be
reported instead of a made-up confidence.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from rpp.composition import AMINO_ACIDS, normalise
from rpp.dataset import feature_columns, load_dataset, signal_peptide_key
from rpp.features import extract_features
from rpp.modeling import build_pipeline, classes_from_log, cross_fit, score, to_mg_per_l
from rpp.signal_peptides import infer_signal_peptides

ROOT = Path(__file__).resolve().parents[1]
TABLE_CSV = ROOT / "data" / "processed" / "table_s2.csv"

PRODUCTION = {"model": "bagged", "k": 5, "selector": "anova", "strategy": "protein"}

CONDITIONS_NOTE = (
    "No condition is recommended.  On the 36 rows that state OD, IPTG, temperature and time "
    "completely, adding those four numbers moves MCC from 0.211 to 0.296 on one split seed - "
    "too little to claim the model knows which flask to set up."
)


@dataclass
class Assessment:
    name: str
    sequence: str
    predicted_mgl: float | None
    band_mgl: tuple[float, float] | None
    spread_factor: float | None
    predicted_class: str | None
    rank: int | None
    observed_mgl: float | None
    warnings: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.predicted_mgl is not None


@dataclass
class Advisor:
    pipeline: object
    columns: list[str]
    band: float
    metrics: dict[str, float]
    length_range: tuple[int, int]
    yield_range: tuple[float, float]
    sp_by_label: dict[str, str]
    n_rows_without_sp: int
    seen: dict[str, tuple[int, float]]
    picked: dict[str, int]
    n_bags: int
    k: int
    published_peptides: frozenset[str]

    def bag_predictions(self, X: np.ndarray) -> np.ndarray:
        """One column per bag, so a query's answer can be shown with its own disagreement."""
        bags = self.pipeline.named_steps["bag"].models_
        return np.column_stack([bag.predict(X) for bag in bags])

    def assess(
        self, name: str, sequence: str, signal_peptide: str | None = None
    ) -> Assessment:
        warnings: list[str] = []
        try:
            seq = normalise(sequence)
        except ValueError as error:
            return Assessment(
                name=name,
                sequence=sequence,
                predicted_mgl=None,
                band_mgl=None,
                spread_factor=None,
                predicted_class=None,
                rank=None,
                observed_mgl=None,
                warnings=[f"no features, no prediction: {error}"],
            )

        sp = self._resolve_signal_peptide(signal_peptide, warnings)
        if sp and not seq.startswith(sp):
            warnings.append(
                f"signal peptide {sp!r} is not the start of the pasted sequence, so it was "
                "dropped and the whole string was read as one chain"
            )
            sp = None
        if sp is None:
            warnings.append(
                "no usable signal peptide, so the translocation half of the feature vector is "
                f"empty - the model sees the chain alone, as it did for {self.n_rows_without_sp} "
                "of the 103 published rows"
            )

        if not self.length_range[0] <= len(seq) <= self.length_range[1]:
            warnings.append(
                f"{len(seq)} residues sits outside the "
                f"{self.length_range[0]}-{self.length_range[1]} range every scored construct falls "
                "in; this answer is extrapolation"
            )

        try:
            row = extract_features(seq, signal_peptide=sp)
        except (ValueError, ZeroDivisionError) as error:
            # A mature chain of a few residues has no sliding windows left, so the scale columns
            # it would average are empty.  The library raises; the user gets the reason instead.
            warnings.append(f"too short to describe: {type(error).__name__}: {error}")
            return Assessment(
                name=name,
                sequence=seq,
                predicted_mgl=None,
                band_mgl=None,
                spread_factor=None,
                predicted_class=None,
                rank=None,
                observed_mgl=None,
                warnings=warnings,
            )
        X = pd.DataFrame([[row.get(c, np.nan) for c in self.columns]], columns=self.columns)
        per_bag = self.bag_predictions(X.to_numpy(dtype=float))[0]
        log_prediction = float(np.mean(per_bag))
        # The bags' own spread, on the same multiplication scale as the reported band.
        spread_factor = float(10.0 ** np.std(per_bag))
        predicted = float(to_mg_per_l(np.array([log_prediction]))[0])

        if spread_factor > self.band:
            warnings.append(
                f"the {self.n_bags} selection draws disagree by a factor of {spread_factor:.1f}, "
                f"more than the model's measured typical error of {self.band:.1f}x: this construct "
                "is outside what the model can rank, and its position is essentially the table's "
                "middle"
            )

        if not self.yield_range[0] <= predicted <= self.yield_range[1]:
            warnings.append(
                f"{predicted:.3g} mg/L is beyond every yield in the table "
                f"({self.yield_range[0]:.3g}-{self.yield_range[1]:.3g} mg/L)"
            )

        observed = self.seen.get(seq)
        if observed is not None:
            warnings.append(
                f"this exact construct is training row {observed[0]}, measured at "
                f"{observed[1]:.3g} mg/L; use that number, the model adds nothing"
            )

        return Assessment(
            name=name,
            sequence=seq,
            predicted_mgl=predicted,
            band_mgl=(predicted / self.band, predicted * self.band),
            spread_factor=spread_factor,
            predicted_class=str(classes_from_log(np.array([log_prediction]))[0]),
            rank=None,
            observed_mgl=observed[1] if observed else None,
            warnings=warnings,
        )

    def rank(self, candidates) -> list[Assessment]:
        """Order constructs by predicted yield; anything unrankable goes last with its reason."""
        assessments = [self.assess(*tuple(candidate)[:3]) for candidate in candidates]
        ordered = sorted((a for a in assessments if a.usable), key=lambda a: -a.predicted_mgl)
        for position, assessment in enumerate(ordered, start=1):
            assessment.rank = position
        return ordered + [a for a in assessments if not a.usable]

    def _resolve_signal_peptide(self, label: str | None, warnings: list[str]) -> str | None:
        """Accept a published label ('pelB') or a literal peptide, and say which one was used."""
        if not label:
            return None
        # signal_peptide_key, not normalise_label: the table writes 'PelB' and 'pel B' for one peptide,
        # and a lookup that keeps the space would silently fall through to the unverified literal path.
        key = signal_peptide_key(label)
        if key in self.sp_by_label:
            return self.sp_by_label[key]
        text = label.strip().upper()
        if text in self.published_peptides:
            # Pasted the peptide itself rather than naming it - matched against the table, so trusted.
            return text
        if len(text) >= 12 and set(text) <= set(AMINO_ACIDS):
            warnings.append(
                f"{label!r} is neither one of the {len(self.sp_by_label)} labels the table resolves "
                "nor one of their sequences, so the literal peptide was used and never cross-checked"
            )
            return text
        warnings.append(
            f"{label!r} is neither a known label nor a peptide sequence, so no signal "
            "peptide features were computed"
        )
        return None


def fit_advisor(df: pd.DataFrame | None = None, table: Path = TABLE_CSV, seed: int = 0) -> Advisor:
    """Fit the shipped pipeline and read back the envelope a query has to sit inside."""
    df = load_dataset() if df is None else df
    columns = feature_columns(df)
    config = {**PRODUCTION, "seed": seed}

    pipeline = build_pipeline(config["model"], config["k"], config["selector"], seed)
    X = df[columns].to_numpy(dtype=float)
    y = df["log10_yield"].to_numpy(dtype=float)
    pipeline.fit(X, y)

    # The band is not a guess.  It is the median absolute error of the honest cross-fit under a
    # held-out protein, converted back into a multiplication factor.
    held_out = cross_fit(
        df,
        columns,
        strategy=config["strategy"],
        model=config["model"],
        k=config["k"],
        selector=config["selector"],
        seed=seed,
    )
    metrics = score(df, held_out)

    with table.open() as fh:
        rows = list(csv.DictReader(fh))
    sps = infer_signal_peptides(rows)
    sp_by_label = {
        signal_peptide_key(label): info.sequence for label, info in sps.items() if info.resolved
    }

    observed = dict(zip(df["row_no"].astype(int), df["yield_best_mgl"].astype(float)))
    seen = {
        normalise(row["sequence"]): (int(row["row_no"]), observed[int(row["row_no"])])
        for row in rows
    }

    bags = pipeline.named_steps["bag"].models_
    picked: dict[str, int] = {}
    for bag in bags:
        for name, keep in zip(columns, bag.named_steps["select"].get_support()):
            if keep:
                picked[name] = picked.get(name, 0) + 1

    lengths = pd.to_numeric(df["seq_length"], errors="coerce")
    yields = pd.to_numeric(df["yield_best_mgl"], errors="coerce")
    return Advisor(
        pipeline=pipeline,
        columns=columns,
        band=float(metrics["median_fold_error"]),
        metrics=metrics,
        length_range=(int(lengths.min()), int(lengths.max())),
        yield_range=(float(yields.min()), float(yields.max())),
        sp_by_label=sp_by_label,
        n_rows_without_sp=int(len(df) - df["sp_resolved"].sum()),
        seen=seen,
        picked=picked,
        n_bags=len(bags),
        k=config["k"],
        published_peptides=frozenset(sp_by_label.values()),
    )
