"""从 data/processed/table_s2.csv 构建 Tier A 与 Tier B 特征矩阵。

    uv run --with biopython python scripts/build_features.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpp.features import extract_features  # noqa: E402
from rpp.physicochemistry import NOT_REPRODUCED  # noqa: E402
from rpp.signal_peptides import infer_signal_peptides, normalise_label  # noqa: E402

KEEP_COLUMNS = [
    "row_no",
    "protein",
    "signal_peptide",
    "media",
    "od",
    "iptg",
    "temperature",
    "time",
    "yield_mgl",
    "yield_min",
    "yield_max",
    "level_codes",
    "level_from_yield_max",
    "condition_fully_specified",
    "rsm_optimized",
    "reference",
]


def main() -> int:
    """从清理后的 Table S2 生成可供模型训练的 Tier A 与 Tier B 特征矩阵。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", type=Path, default=ROOT / "data/processed/table_s2.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "data/processed/features_tier_ab.csv")
    args = ap.parse_args()

    with args.src.open() as fh:
        # CSV 保留每条融合序列及其原始实验元数据。
        rows = list(csv.DictReader(fh))

    # 表格只给 SP 的文本标签；根据同标签融合序列的公共前缀推断可用切点。
    sps = infer_signal_peptides(rows)
    resolved = {k: v for k, v in sps.items() if v.resolved}
    unresolved = {k: v for k, v in sps.items() if not v.resolved}

    feature_rows = []
    for row in rows:
        label = normalise_label(row["signal_peptide"])
        sp = resolved[label].sequence if label in resolved else None
        # 无法可靠切分的记录仍保留全序列特征，但不伪造 SP / 成熟蛋白分段特征。
        features = extract_features(row["sequence"], signal_peptide=sp)
        record = {k: row.get(k, "") for k in KEEP_COLUMNS}
        record["sp_inferred_length"] = len(sp) if sp else 0
        record["sp_label_resolved"] = label in resolved
        record.update(features)
        feature_rows.append(record)

    # 输出目录不存在时创建；写出列顺序以第一条完整特征记录为准。
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(feature_rows[0].keys()))
        writer.writeheader()
        writer.writerows(feature_rows)

    n_cols = len(feature_rows[0])
    print(f"rows                       : {len(feature_rows)}")
    print(f"feature columns            : {n_cols}")
    print(f"observations per feature   : {len(feature_rows) / n_cols:.3f}")
    print(f"SP labels seen             : {len(sps)}")
    print(f"SP labels resolved         : {len(resolved)}")
    print(f"SP labels unresolved       : {len(unresolved)}")
    for label, info in sorted(unresolved.items(), key=lambda kv: -kv[1].n_members):
        print(f"  ! {label[:34]:34s} n={info.n_members:2d}  {info.reason}")
    print(f"sequences split on SP      : {sum(1 for r in feature_rows if r['sp_resolved'])}")
    print()
    print("PERISCOPE-Opt features not reproduced:")
    for name, reason in NOT_REPRODUCED.items():
        print(f"  - {name}\n      {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
