"""将 PERISCOPE-Opt 补充材料中的 Table S2 提取为整洁的 CSV。

Source: Packiam K.A.R. et al., "PERISCOPE-Opt: Machine learning-based prediction of optimal
fermentation conditions and yields of recombinant periplasmic protein expressed in
Escherichia coli", Comput. Struct. Biotechnol. J. 20 (2022) 2909-2920,
DOI 10.1016/j.csbj.2022.06.006.  Table S2 lives on pp. 3-31 of the "*-mmc1.pdf" supplement.

影响解析器设计的版式说明
------------------------
* PDF 表格不含结构化标签，因此通过单词边界框恢复行。窄的左侧 “No” 列每条记录只有一个
  整数，用作行锚点；换行或跨页延续的片段没有锚点，追加到正在构建的记录中。
* 页面页脚在水平中部印出纯页码，位置落在 sequence 列内，因此通过底部区域过滤掉这些词元。
* 大多数单元格给出范围或斜杠分隔的备选值（如 “0.5-0.6”“22/37”“17.8-45”），而不是单值；
  论文为每个信号肽—蛋白组合列一行，并报告实验探索范围。原始字符串与解析后的数值范围均保留。
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

# 列名到 x0 下界的映射；边界取自观察到的各列起点的中点。
COLUMN_BOUNDS: list[tuple[str, float]] = [
    ("row_no", 76.0),
    ("protein", 90.0),
    ("signal_peptide", 180.0),
    ("sequence", 225.0),
    ("media", 436.0),
    ("od", 479.0),
    ("iptg", 508.0),
    ("temperature", 544.0),
    ("time", 612.0),
    ("yield_mgl", 662.0),
    ("level", 707.0),
    ("reference", 749.0),
]

FIRST_PAGE = 3
LAST_PAGE = 31

# 低于此纵坐标的内容属于页码页脚，不是表格正文。
FOOTER_TOP = 555.0

STANDARD_RESIDUES = set("ACDEFGHIKLMNPQRSTVWY")

# 这些列的换行碎片必须无空格拼接，使 “0.5-” + “0.6” 还原为 “0.5-0.6”，而不是 “0.5- 0.6”。
JOIN_DIRECT = {
    "sequence",
    "od",
    "iptg",
    "temperature",
    "time",
    "yield_mgl",
    "level",
    "media",
}

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
# 数值单元格只能由数字和范围或列表分隔符组成；补充材料在同一列混用 ASCII 连字符和排版短横线。
NUMERIC_CELL_RE = re.compile(r"\d*\.?\d+(?:\s*[-–—−/:,]\s*\d*\.?\d+)*")
LEVEL_RE = re.compile(r"\(?\b(L|M|H)\b\)?", re.IGNORECASE)


def column_for(x0: float) -> str:
    """依据 PDF 单词的横坐标，将其归入 Table S2 的对应列。"""
    name = ""
    for candidate, bound in COLUMN_BOUNDS:
        if x0 >= bound:
            name = candidate
        else:
            break
    return name


@dataclass
class RawRecord:
    """PDF 解析阶段的原始行；同一列可暂存多个换行碎片。"""
    row_no: str
    columns: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))


def assemble_rows(pages) -> tuple[list[RawRecord], list[str]]:
    """逐词扫描 PDF，以左侧 No 编号作为新记录锚点并拼接换行碎片。"""
    records: list[RawRecord] = []
    stray: list[str] = []
    current: RawRecord | None = None

    for page_no, page in enumerate(pages, start=FIRST_PAGE):
        # 先按纵坐标再按横坐标排序，恢复页面的阅读顺序。
        words = sorted(page.extract_words(), key=lambda w: (w["top"], w["x0"]))
        for word in words:
            if word["top"] > FOOTER_TOP:
                # 过滤页脚页码，防止落在 sequence 列的数字被误拼进序列。
                continue
            col = column_for(word["x0"])
            text = word["text"]
            if col == "row_no":
                if re.fullmatch(r"\d+", text):
                    current = RawRecord(row_no=text)
                    records.append(current)
                    continue
                if current is not None:
                    stray.append(f"p{page_no}: unparsed No-column token {text!r}")
                continue
            if current is None:
                continue  # 第一条数据记录之前的表格标题或表头行。
            current.columns[col or "unassigned"].append(text)

    return records, stray


# PDF 文本层会在数字前混入的字符（第 78 行时间范围读作 “`12-16”）。只在条件和产量列中清理，避免误改序列内部字符。
PDF_JUNK = str.maketrans("", "", "`*^~")
SCRUBBED_COLUMNS = {"od", "iptg", "temperature", "time", "yield_mgl"}


def collapse(record: RawRecord, col: str) -> str:
    """按列规则把 PDF 的多个文本碎片合回一个单元格字符串。"""
    tokens = record.columns.get(col, [])
    if not tokens:
        return ""
    if col in JOIN_DIRECT:
        # 序列或数值范围的断行不能插入空格，例如 0.5- 与 0.6 应重新拼为 0.5-0.6。
        text = re.sub(r"\s*([-/])\s*", r"\1", "".join(tokens))
    else:
        text = re.sub(r"\s+", " ", " ".join(tokens))
    return (text.translate(PDF_JUNK) if col in SCRUBBED_COLUMNS else text).strip()


def numbers_in(cell: str) -> list[float]:
    """返回一个单元格确实报告为测量值的数字。

    仅当单元格完全由数字及范围或列表分隔符组成时才提取。补充材料会将备注放入数值列：13 行
    在 OD 列写有 “24h”，这表示时长；另一行诱导列写有 “10%”，表示乳糖浓度而非 10 mM IPTG。
    若把 24 和 10 当作工艺条件，就会凭空制造论文没有报告的值，因此这类单元格只保留原文。
    """
    if not NUMERIC_CELL_RE.fullmatch(cell.strip()):
        # 非纯数值单元格保留原文但不强行提数，避免把“24h”等备注误作工艺条件。
        return []
    return [float(n) for n in NUMBER_RE.findall(cell)]


def span(cell: str) -> tuple[float | None, float | None, int]:
    """从可信数值单元格返回最小值、最大值和报告数值个数。"""
    values = numbers_in(cell)
    if not values:
        return None, None, 0
    return min(values), max(values), len(values)


def classify(yield_mgl: float | None) -> str:
    """按论文阈值把产量映射为 L/M/H；50 mg/L 归入 H。"""
    if yield_mgl is None:
        return ""
    if yield_mgl < 0.5:
        return "L"
    if yield_mgl < 50:
        return "M"
    return "H"


@dataclass
class Row:
    """清洗后写入 CSV 的一条结构化 Table S2 记录。"""
    row_no: int
    protein: str
    signal_peptide: str
    sequence: str
    seq_length: int
    media: str
    od: str
    od_min: float | None
    od_max: float | None
    od_n: int
    iptg: str
    iptg_min: float | None
    iptg_max: float | None
    iptg_n: int
    temperature: str
    temperature_min: float | None
    temperature_max: float | None
    temperature_n: int
    time: str
    time_min: float | None
    time_max: float | None
    time_n: int
    yield_mgl: str
    yield_min: float | None
    yield_max: float | None
    yield_n: int
    level_reported: str
    level_codes: str
    level_ambiguous: bool
    rsm_optimized: bool
    level_from_yield_max: str
    level_disagrees: bool
    condition_fully_specified: bool
    nonstandard_residues: str
    reference: str


def to_row(record: RawRecord) -> Row:
    """将 PDF 原始记录清洗、拆分数值范围并构造成一条训练候选记录。"""
    raw_yield = collapse(record, "yield_mgl")
    sequence = collapse(record, "sequence").upper()
    level_raw = collapse(record, "level")
    # 表中可能同时含多个等级代码；去重后保留以标记歧义，不能任选一个当标签。
    codes = sorted({c.group(1).upper() for c in LEVEL_RE.finditer(level_raw)})
    od_min, od_max, od_n = span(collapse(record, "od"))
    ip_min, ip_max, ip_n = span(collapse(record, "iptg"))
    t_min, t_max, t_n = span(collapse(record, "temperature"))
    h_min, h_max, h_n = span(collapse(record, "time"))
    y_min, y_max, y_n = span(raw_yield)
    # 以产量范围上界派生标签，和论文报告代码并存，后续可检查两者是否矛盾。
    derived = classify(y_max)

    return Row(
        row_no=int(record.row_no),
        protein=collapse(record, "protein"),
        signal_peptide=collapse(record, "signal_peptide"),
        sequence=sequence,
        seq_length=len(sequence),
        media=collapse(record, "media"),
        od=collapse(record, "od"),
        od_min=od_min,
        od_max=od_max,
        od_n=od_n,
        iptg=collapse(record, "iptg"),
        iptg_min=ip_min,
        iptg_max=ip_max,
        iptg_n=ip_n,
        temperature=collapse(record, "temperature"),
        temperature_min=t_min,
        temperature_max=t_max,
        temperature_n=t_n,
        time=collapse(record, "time"),
        time_min=h_min,
        time_max=h_max,
        time_n=h_n,
        yield_mgl=raw_yield,
        yield_min=y_min,
        yield_max=y_max,
        yield_n=y_n,
        level_reported=level_raw,
        level_codes="/".join(codes),
        level_ambiguous=len(codes) > 1,
        rsm_optimized="RSM" in level_raw.upper(),
        level_from_yield_max=derived,
        level_disagrees=bool(codes) and derived and derived not in codes,
        # 五项均只有一个明确值时，才可称为完整指定的一组工艺条件。
        condition_fully_specified=all(
            n == 1 for n in (od_n, ip_n, t_n, h_n, y_n)
        ),
        nonstandard_residues="".join(sorted(set(sequence) - STANDARD_RESIDUES)),
        reference=collapse(record, "reference"),
    )


def report(rows: list[Row], stray: list[str]) -> list[str]:
    """检查提取结果的编号、序列、产量和等级一致性，返回问题清单。"""
    problems = list(stray)
    numbers = [r.row_no for r in rows]
    # 连续编号是 PDF 行解析没有漏行或错行的基本完整性检查。
    if numbers != list(range(1, len(numbers) + 1)):
        missing = sorted(set(range(1, max(numbers) + 1)) - set(numbers))
        problems.append(f"row numbers not contiguous; missing {missing[:40]}")
    for r in rows:
        if not r.sequence:
            problems.append(f"row {r.row_no}: empty sequence")
        elif r.nonstandard_residues:
            problems.append(
                f"row {r.row_no}: non-standard residues {r.nonstandard_residues!r}"
            )
        if r.yield_max is None:
            problems.append(f"row {r.row_no}: no numeric yield in {r.yield_mgl!r}")
        if not r.level_codes:
            problems.append(f"row {r.row_no}: no level code in {r.level_reported!r}")
        elif r.level_disagrees:
            problems.append(
                f"row {r.row_no}: reported {r.level_codes} but yield_max="
                f"{r.yield_max} implies {r.level_from_yield_max}"
            )
    return problems


def main() -> int:
    """读取补充 PDF 的目标页，提取、校验并导出整洁 CSV。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", type=Path, required=True, help="supplementary PDF (mmc1)")
    ap.add_argument("--out", type=Path, required=True, help="destination CSV")
    args = ap.parse_args()

    with pdfplumber.open(args.pdf) as pdf:
        # 页码按人类页码计数，Python 列表则从 0 开始，因此起点减一。
        records, stray = assemble_rows(pdf.pages[FIRST_PAGE - 1 : LAST_PAGE])

    # 解析与质量报告分离：即使有可疑行也可完整导出，供人工复核而不是静默丢数据。
    rows = [to_row(r) for r in records]
    problems = report(rows, stray)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(Row.__dataclass_fields__)
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    full = [r for r in rows if r.yield_max is not None]
    lengths = sorted(r.seq_length for r in rows if r.seq_length)
    yields = sorted(r.yield_max for r in full)
    specified = [r for r in rows if r.condition_fully_specified]

    print(f"rows extracted                 : {len(rows)}")
    print(f"distinct protein names         : {len({r.protein.lower() for r in rows})}")
    print(f"distinct SP-protein combos     : {len({(r.signal_peptide.lower(), r.protein.lower()) for r in rows})}")
    print(f"distinct signal peptides       : {len({r.signal_peptide.lower() for r in rows})}")
    print(f"sequence length                : min {lengths[0]} median {lengths[len(lengths)//2]} max {lengths[-1]}")
    print(f"yield_max mg/L                 : min {yields[0]} median {yields[len(yields)//2]} max {yields[-1]}")
    print(f"rows with a single yield value  : {sum(1 for r in full if r.yield_n == 1)} / {len(full)}")
    print(f"rows with a yield range         : {sum(1 for r in full if r.yield_n > 1)}")
    print(f"rows with an ambiguous level    : {sum(1 for r in rows if r.level_ambiguous)}")
    print(f"rows flagged RSM-optimized     : {sum(1 for r in rows if r.rsm_optimized)}")
    print(f"rows with fully specified single condition : {len(specified)}")
    print("level distribution (reported)   : "
          + ", ".join(f"{c}={sum(1 for r in rows if r.level_codes==c)}" for c in ("L", "M", "H", "ambiguous")))
    print(f"reported level vs yield_max disagreement   : {sum(1 for r in rows if r.level_disagrees)}")
    print(f"problem records                : {len(problems)}")
    for p in problems[:40]:
        print("  -", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
