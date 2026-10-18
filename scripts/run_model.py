"""运行五类证据实验，并打印各实验的结论。

    uv run --with pandas --with scikit-learn --with biopython python scripts/run_model.py --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpp.dataset import describe, load_dataset  # noqa: E402
from rpp.experiments import summarise, run_all  # noqa: E402

DISPLAY = {
    "pearson_r_log": "r(log)",
    "spearman_rho_log": "rho(log)",
    "median_fold_error": "typical x off",
    "class_accuracy": "acc",
    "mcc": "MCC",
    "high_share_in_top10": "H in top10",
    "high_recall_at_top10": "H caught",
    "distinct_predictions": "distinct",
    "modal_share": "biggest tie",
}
KEY_COLUMNS = {
    "benchmark": ["strategy", "model", "k", "n_features"],
    "leakage": ["strategy", "model", "k", "selection"],
    "ablation": ["strategy", "model", "feature_set", "k", "n_features"],
    "conditions": ["model", "feature_set", "k", "n_features", "n_rows"],
    "signal_location": ["model", "feature_set", "k", "selector", "n_features"],
}


def table(summary: pd.DataFrame, experiment: str) -> str:
    """把某组实验结果裁剪为适合终端阅读的关键列和指标名称。"""
    keys = KEY_COLUMNS[experiment]
    metrics = [c for c in DISPLAY if c in summary.columns]
    view = summary[keys + metrics].copy()
    view.columns = keys + [DISPLAY[c] for c in metrics]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        return view.to_string(index=False)


def main() -> int:
    """运行基准、泄漏、消融和条件信息等全部对照实验，并保存长表结果。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="one repeat instead of three; same configurations, coarser random-split numbers",
    )
    ap.add_argument("--selector", choices=("anova", "mi"), default="anova")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "processed" / "model_results.csv",
    )
    args = ap.parse_args()

    df = load_dataset()
    # 先打印建模实际使用的数据规模，避免把论文原始实验总量误当作本次训练样本数。
    print("== the table we are actually modelling ==")
    print(describe(df))

    # 每行是“一个模型配置 × 一个外层验证折”的结果，汇总前保留全部明细以便审计。
    results = run_all(df, selector=args.selector, quick=args.quick)
    results.to_csv(args.out, index=False)

    fold_sizes = results.attrs.get("fold_sizes", {})
    print("\n== held-out sizes per split strategy ==")
    for strategy, sizes in fold_sizes.items():
        print(f"  {strategy:<15} {sizes}")

    for experiment in ("benchmark", "leakage", "ablation", "conditions", "signal_location"):
        summary = summarise(results, experiment)
        print(f"\n== {experiment} ({len(summary)} configurations) ==")
        print(table(summary, experiment))

    print(f"\nfull results -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
