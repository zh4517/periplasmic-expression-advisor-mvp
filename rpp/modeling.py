"""模型拟合器、交叉拟合预测，以及用于报告的评估指标。

所有会查看标签的步骤只有两种路径：在训练折内完成（`selection="nested"`），或在切分前
对全表完成（`selection="leaked"`）。保留后者是有意为之：两者差异量化了“先选特征再验证”
带来的乐观偏差，值得单独报告。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import matthews_corrcoef
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rpp.cv import iter_splits
from rpp.dataset import CLASS_EDGES, YIELD_OFFSET

MODELS = ("ridge", "rf", "bagged")
SELECTORS = {"anova": f_regression, "mi": mutual_info_regression}
RIDGE_ALPHAS = np.logspace(-2, 4, 25)


class ConstantPredictor(BaseEstimator, RegressorMixin):
    """始终输出训练标签中位数的基线模型，用来检验复杂模型是否真的有增益。"""

    def fit(self, X, y):
        # 不使用 X；只记录训练集标签的中位数，避免受极端高产量影响。
        self.value_ = float(np.median(y))
        return self

    def predict(self, X):
        # 为每一个待预测样本复制同一个基线值。
        return np.full(len(X), self.value_)


class BaggedSelector(BaseEstimator, RegressorMixin):
    """对多个 bootstrap 子模型取平均，每个子模型各自进行特征选择。

    从 1,144 列中依据 103 个标签筛选特征具有很强不稳定性。若看似有效的信号只来自一次
    偶然抽样而非序列本身，多次抽样取平均可削弱这种偶然性。
    """

    def __init__(
        self,
        model: str = "ridge",
        k: int | None = 5,
        selector: str = "anova",
        n_bags: int = 20,
        seed: int = 0,
    ):
        # 保存 sklearn 估计器参数；真正训练发生在 fit() 中。
        self.model = model
        self.k = k
        self.selector = selector
        self.n_bags = n_bags
        self.seed = seed

    def fit(self, X, y):
        # 每个 bootstrap 子模型都要重新抽样、重新选特征、重新训练。
        from sklearn.utils import resample

        # 固定随机种子，让同一配置的实验可复现。
        rng = np.random.default_rng(self.seed)
        self.models_ = []
        for _ in range(self.n_bags):
            # 有放回抽样：某些训练行会重复、某些不会出现，用于估计选择稳定性。
            indices = resample(np.arange(len(y)), random_state=int(rng.integers(1 << 30)))
            pipeline = build_pipeline(self.model, self.k, self.selector, self.seed)
            pipeline.fit(X[indices], np.asarray(y)[indices])
            self.models_.append(pipeline)
        return self

    def predict(self, X):
        # 对 20 个子模型的预测求平均，降低单次特征选择的偶然性。
        return np.mean([model.predict(X) for model in self.models_], axis=0)


def build_pipeline(
    model: str,
    k: int | None,
    selector: str = "anova",
    seed: int = 0,
    forced: int = 0,
) -> Pipeline:
    """构造“填补缺失值 → 选择特征 → 标准化 → 学习器”的训练管线。

    ``forced`` 指定无论评分如何都保留在最前面的列数，用于公平比较四个工艺条件是否有用，
    避免它们被 1,144 个序列特征挤掉。
    """
    if model == "bagged":
        # 每个 bag 内部都要选择特征，因此 bag 本身就是完整训练管线。
        return Pipeline(
            [("bag", BaggedSelector(model="ridge", k=k, selector=selector, seed=seed))]
        )

    # 缺失的序列/工艺特征用当前训练折的中位数填补，不能借用测试折信息。
    steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
    if k:
        # ANOVA 或互信息只在训练折上对“特征—标签”关系打分。
        score_func = SELECTORS[selector]
        if score_func is mutual_info_regression:
            # 互信息估计带随机性；固定种子保证相同实验配置可复现。
            from functools import partial

            score_func = partial(score_func, random_state=seed)
        if forced:
            from sklearn.compose import ColumnTransformer

            selector_model = ColumnTransformer(
                [
                    ("forced", "passthrough", slice(0, forced)),
                    ("rest", SelectKBest(score_func=score_func, k=k), slice(forced, None)),
                ],
                # 强制输出稠密矩阵；本版 sklearn 通过 sparse_threshold=0 实现。
                sparse_threshold=0,
            )
        else:
            selector_model = SelectKBest(score_func=score_func, k=k)
        steps.append(("select", selector_model))
    # Ridge 对量纲敏感，先将被选中的特征标准化。
    steps.append(("scale", StandardScaler()))

    if model == "constant":
        steps.append(("learn", ConstantPredictor()))
    elif model == "ridge":
        # RidgeCV 在预设 alpha 中选择正则化强度，限制高维模型权重过大。
        steps.append(("learn", RidgeCV(alphas=RIDGE_ALPHAS)))
    elif model == "rf":
        steps.append(
            (
                "learn",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    random_state=seed,
                    n_jobs=-1,
                ),
            )
        )
    else:
        raise ValueError(f"unknown model {model!r}")

    return Pipeline(steps)


@dataclass
class Run:
    """一组交叉拟合预测及其验证策略、模型和特征选择配置。"""

    predictions: np.ndarray
    strategy: str
    model: str
    k: int | None
    selection: str
    feature_set: str
    repeat: int


def cross_fit(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    strategy: str,
    model: str,
    k: int | None = 20,
    selector: str = "anova",
    selection: str = "nested",
    n_splits: int = 5,
    seed: int = 0,
    forced: int = 0,
) -> np.ndarray:
    """交叉拟合，保证每一行预测都来自未见过该行的训练模型。

    当 ``selection="leaked"`` 时，传入列已利用整张表的标签选出，其中包括测试行标签；
    这正是该对照版本要刻意模拟的错误流程。
    """
    if selection not in ("nested", "leaked"):
        raise ValueError(f"unknown selection mode {selection!r}")
    if selection == "nested" and not feature_cols:
        raise ValueError("no feature columns requested")

    # 在进入循环前只取输入列；标签在每一折内部由 pipeline 使用。
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["log10_yield"].to_numpy(dtype=float)
    predictions = np.full(len(df), np.nan)

    # iter_splits 按蛋白、组合或信号肽分组，防止相似构建体泄漏。
    for train, test in iter_splits(df, strategy, n_splits=n_splits, seed=seed):
        pipeline = build_pipeline(
            model, k if selection == "nested" else None, selector, seed, forced
        )
        # 特征选择、填补和标准化都只在 train 上 fit。
        pipeline.fit(X[train], y[train])
        predictions[test] = pipeline.predict(X[test])

    uncovered = np.isnan(predictions)
    if uncovered.any():
        raise RuntimeError(
            f"{int(uncovered.sum())} rows were never in a test fold under {strategy} splits"
        )
    return predictions


def leaked_columns(
    df: pd.DataFrame,
    feature_cols: list[str],
    k: int,
    selector: str = "anova",
    seed: int = 0,
) -> list[str]:
    """故意制造“先全表选特征”的泄漏版本，仅用于量化错误流程的乐观偏差。

    这模拟了常见的错误顺序：先选择特征，再切分数据，最后报告结果。选择步骤已经使用了
    后续被声称为留出集的标签。
    """
    X = df[feature_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=float(np.nanmedian(X)))
    y = df["log10_yield"].to_numpy(dtype=float)
    score_func = SELECTORS[selector]
    if score_func is mutual_info_regression:
        from functools import partial

        score_func = partial(score_func, random_state=seed)
    selector_model = SelectKBest(score_func=score_func, k=min(k, len(feature_cols)))
    # 这里包含未来测试行的标签，因此绝不能作为生产流程。
    selector_model.fit(X, y)
    keep = selector_model.get_support()
    return [name for name, flag in zip(feature_cols, keep) if flag]


def to_mg_per_l(log_predictions: np.ndarray) -> np.ndarray:
    """将模型预测的 log10(1 + 产量) 反变换为 mg/L。"""
    # 四舍五入消除浮点误差，避免报告为 50.0 mg/L 的样本落在中高产阈值稍下方而被错分。
    return np.round(np.maximum(10.0**np.asarray(log_predictions, dtype=float) - YIELD_OFFSET, 0.0), 6)


def classes_from_log(log_values: np.ndarray) -> np.ndarray:
    """将反变换后的产量按论文阈值映射为低、中、高三类。"""
    return np.select(
        [to_mg_per_l(log_values) < CLASS_EDGES[0], to_mg_per_l(log_values) < CLASS_EDGES[1]],
        ["L", "M"],
        default="H",
    )


def _expected_high_in_top(
    predicted: np.ndarray, high: np.ndarray, top_k: int
) -> float:
    """计算前 top_k 中真实高产样本的期望数量，并公平处理预测并列。

    当 Ridge 只选择 k=5 列时，缺少这些特征的行可能得到同一分数。若按原始表格行号打破并列，
    前十比例会依赖行的位置。此处对所有并列打破方式取平均，得到候选池真正应得的指标。
    """
    # 先按预测值从高到低排序；stable 只保证可复现，不用表格原始顺序破并列。
    order = np.argsort(-predicted, kind="stable")
    sorted_values = predicted[order]
    sorted_high = high[order].astype(float)

    expected = 0.0
    start = 0
    n = len(sorted_values)
    while start < n and start < top_k:
        end = start
        while end < n and sorted_values[end] == sorted_values[start]:
            end += 1
        if end <= top_k:
            expected += float(sorted_high[start:end].sum())
        else:
            # 并列组跨过 top_k 边界时，用该并列组的高产比例计算期望值。
            slots = top_k - start
            expected += slots * float(sorted_high[start:end].mean())
        start = end
    return expected


def score(df: pd.DataFrame, predictions: np.ndarray, top_k: int = 10) -> dict[str, float]:
    """汇总交叉拟合预测，输出相关性、倍数误差、分类和 Top-k 指标。"""
    y = df["log10_yield"].to_numpy(dtype=float)
    truth_class = df["expression_class"].to_numpy(dtype=str)
    # 对数空间的绝对误差转回原空间后表示“相差几倍”。
    delta = np.abs(predictions - y)

    high = truth_class == "H"
    expected_high = _expected_high_in_top(predictions, high, top_k)
    distinct, counts = np.unique(np.round(predictions, 9), return_counts=True)

    predicted_class = classes_from_log(predictions)
    return {
        "n": float(len(df)),
        "pearson_r_log": float(pearsonr(y, predictions)[0]),
        "spearman_rho_log": float(spearmanr(y, predictions)[0]),
        # 例如 log 误差 0.60 转回约 10^0.60 ≈ 4 倍。
        "median_fold_error": float(10.0 ** np.median(delta)),
        "worst_fold_error": float(10.0 ** np.quantile(delta, 0.95)),
        "class_accuracy": float((predicted_class == truth_class).mean()),
        "mcc": float(matthews_corrcoef(truth_class, predicted_class)),
        f"high_share_in_top{top_k}": expected_high / top_k,
        f"high_recall_at_top{top_k}": expected_high / max(int(high.sum()), 1),
        "high_base_rate": float(high.mean()),
        "distinct_predictions": float(len(distinct)),
        "modal_share": float(counts.max() / len(predictions)),
    }
