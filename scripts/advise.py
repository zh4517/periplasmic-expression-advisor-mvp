"""对 FASTA 中的构建体排序，并指出哪些结果不应被信任。

    uv run --with pandas --with scikit-learn --with biopython==1.88 \
        python scripts/advise.py --fasta data/examples/candidates.fa --signal-peptide pelB

Sequence names are used as-is.  A header of the form ``name|sp=pelB`` overrides the file-wide
``--signal-peptide`` for that one record.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpp.advisor import CONDITIONS_NOTE, fit_advisor  # noqa: E402

HEADER_SP = re.compile(r"\bsp=([A-Za-z0-9_-]+)")


def read_fasta(path: Path) -> list[tuple[str, str, str | None]]:
    """读取 FASTA，返回（名称、氨基酸序列、可选的单条 SP 标签）。"""
    records: list[tuple[str, str, str | None]] = []
    name, letters, sp = None, [], None

    def flush():
        # 遇到下一条标题或文件结束时，提交正在累积的上一条序列。
        if name is not None:
            records.append((name, "".join(letters), sp))

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith(">"):
            # ``|sp=pelB`` 可覆盖命令行给所有记录设置的默认信号肽。
            flush()
            head = line[1:]
            matched = HEADER_SP.search(head)
            sp = matched.group(1) if matched else None
            head = HEADER_SP.sub("", head).strip(" |,-:")
            name, letters = head or f"record{len(records) + 1}", []
        else:
            letters.append(line)
    flush()
    return records


def main() -> int:
    """解析命令行候选序列，训练已选模型并打印按预测产量排序的建议。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fasta", type=Path, required=True)
    ap.add_argument(
        "--signal-peptide",
        default=None,
        help="published label (pelB, ompA, ...) or a literal peptide, applied to every record",
    )
    args = ap.parse_args()

    records = read_fasta(args.fasta)
    if not records:
        print(f"no records in {args.fasta}", file=sys.stderr)
        return 1

    advisor = fit_advisor()
    # 单条 FASTA 上标注的 SP 优先；否则才使用命令行的统一 SP。
    ranked = advisor.rank(
        [(name, seq, sp or args.signal_peptide) for name, seq, sp in records]
    )

    # 输出重采样中最常被选中的特征，便于审查模型是否过分依赖偶然特征。
    common = sorted(advisor.picked.items(), key=lambda kv: (-kv[1], kv[0]))[:6]
    print(
        f"{advisor.n_bags} draws of {advisor.k} columns each landed on {len(advisor.picked)} "
        f"different columns, none of them in every draw; most-chosen: "
        + ", ".join(f"{name} x{count}" for name, count in common)
    )
    print(
        f"typical error is a factor of {advisor.band:.2f}, "
        f"rho(log) {advisor.metrics['spearman_rho_log']:.3f} on a held-out protein\n"
    )
    for assessment in ranked:
        if assessment.usable:
            low, high = assessment.band_mgl
            print(
                f"{assessment.rank:>2}. {assessment.name:<30} "
                f"{assessment.predicted_mgl:>8.3g} mg/L  [{low:.3g} - {high:.3g}]  "
                f"class {assessment.predicted_class}  "
                f"bags disagree {assessment.spread_factor:.1f}x"
            )
        else:
            print(f" -. {assessment.name:<30} no prediction")
        for warning in assessment.warnings:
            print(f"     ! {warning}")
    print(f"\n{CONDITIONS_NOTE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
