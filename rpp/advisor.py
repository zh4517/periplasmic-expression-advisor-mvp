"""对候选构建体排序，并在证据不足时明确拒绝给出确定结论。

`scripts/run_model.py` 中的指标来自 103 条公开记录上的留出蛋白和留出信号肽验证。
若新输入超出这个适用范围，本模块会给出警告，而不是输出貌似可靠的数字。

最终模型对完整的“选特征后拟合”流程进行 20 次 bootstrap 重采样并取平均。单次从
1,144 列和 103 条记录中筛选时，容易选中 5 个稀有二肽；103 个公开构建体中有 62 个
完全不含这些二肽，因而会得到相同预测。Bagging 可缓解该问题，并将子模型分歧作为
单个查询的不确定性提示，而非编造置信度。
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
    # 单个候选构建体的最终评估结果，包含预测、区间、排序与风险提示。
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
        # 只有成功生成预测的候选才允许进入排序列表。
        return self.predicted_mgl is not None


@dataclass
class Advisor:
    # 将已训练的 bagged 模型、特征列和适用范围封装成面向用户的排序器。
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
        """返回每个 bootstrap 子模型的预测，用于计算该候选自身的不确定性。"""
        bags = self.pipeline.named_steps["bag"].models_
        return np.column_stack([bag.predict(X) for bag in bags])

    def assess(
        self, name: str, sequence: str, signal_peptide: str | None = None
    ) -> Assessment:
        # 按“序列合法性 → 信号肽 → 长度 → 特征 → bag 分歧 → 已知实测值”评估候选。
        warnings: list[str] = []
        try:
            # normalise 会将序列转为大写并拒绝非标准氨基酸。
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

        # 超出训练数据长度范围时仍可给演示性预测，但必须明确标记为外推。
        if not self.length_range[0] <= len(seq) <= self.length_range[1]:
            warnings.append(
                f"{len(seq)} residues sits outside the "
                f"{self.length_range[0]}-{self.length_range[1]} range every scored construct falls "
                "in; this answer is extrapolation"
            )

        try:
            row = extract_features(seq, signal_peptide=sp)
        except (ValueError, ZeroDivisionError) as error:
            # 成熟蛋白只剩数个残基时没有可用的滑动窗口，相关尺度列会为空；将库报错转为用户可读原因。
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
        # 每个 bag 独立做过特征选择；它们的分歧是每个候选的局部不确定性。
        per_bag = self.bag_predictions(X.to_numpy(dtype=float))[0]
        log_prediction = float(np.mean(per_bag))
        # 在对数空间取标准差，再转回“相差几倍”的直观尺度。
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
        """按预测产量排序；无法预测的候选置底并保留拒答原因。"""
        assessments = [self.assess(*tuple(candidate)[:3]) for candidate in candidates]
        ordered = sorted((a for a in assessments if a.usable), key=lambda a: -a.predicted_mgl)
        for position, assessment in enumerate(ordered, start=1):
            assessment.rank = position
        return ordered + [a for a in assessments if not a.usable]

    def _resolve_signal_peptide(self, label: str | None, warnings: list[str]) -> str | None:
        """解析信号肽名称或字面序列，并为未知标签留下可见警告。"""
        if not label:
            return None
        # 使用 signal_peptide_key 而非 normalise_label：表内 PelB 和 pel B 指向同一肽，保留空格会静默走到未验证的字面序列分支。
        key = signal_peptide_key(label)
        if key in self.sp_by_label:
            return self.sp_by_label[key]
        text = label.strip().upper()
        if text in self.published_peptides:
            # 用户粘贴的是肽序列本身而不是标签，且它与表中已解析序列匹配，因此可直接使用。
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
    """训练最终 Bagged Ridge，并读取长度、产量、信号肽等适用范围。"""
    df = load_dataset() if df is None else df
    columns = feature_columns(df)
    config = {**PRODUCTION, "seed": seed}

    pipeline = build_pipeline(config["model"], config["k"], config["selector"], seed)
    X = df[columns].to_numpy(dtype=float)
    y = df["log10_yield"].to_numpy(dtype=float)
    # 这里用全表拟合是为了部署；性能估计仍由后面的交叉拟合单独完成。
    pipeline.fit(X, y)

    # 典型误差必须来自留出预测，不能使用全表拟合后的训练误差。
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
    # 统计各特征被 20 个 bag 选中的次数，用于解释稳定性。
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
