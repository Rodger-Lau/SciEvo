#!/usr/bin/env python3
"""Export Ablation v2 summaries to a dependency-free XLSX workbook."""

from __future__ import annotations

import csv
import datetime as dt
import math
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "csv_files" / "BrainAI" / "ablation_v2_results.xlsx"

DATASETS = {
    "NYC": {"target": "CROWDIN", "time": "00:00-04:00"},
    "CHI": {"target": "RISK", "time": "00:00-04:00"},
    "SIP": {"target": "FLOW", "time": "00:00-04:00"},
}

MODES = {
    "gradc": {
        "label": "w/o GradC",
        "detail": "Global Shuffle",
        "description": "Remove gradient-ranked curriculum and globally shuffle eligible source tasks.",
    },
    "dac": {
        "label": "w/o DAC",
        "detail": "Fixed p=p0",
        "description": "Keep gradient-ranked order but replace difficulty-aware dropout with fixed p0.",
    },
    "datase": {
        "label": "w/o DataSE",
        "detail": "Data-only State (mean/std)",
        "description": "Remove gradient state and retain only data mean/std encoding for the policy.",
    },
    "reinedit": {
        "label": "w/o ReinEDIT",
        "detail": "Direct Full Fine-tuning",
        "description": "Remove reinforced policy learning and train all forecasting-model parameters directly.",
    },
    "mgo": {
        "label": "w/o MGO",
        "detail": "rho=0 / Real Labels",
        "description": "Disable memory-guided soft labels by forcing rho=0 and using real labels.",
    },
    "arcon": {
        "label": "w/o ARCon",
        "detail": "Fixed Freeze Ratio",
        "description": "Monitor activation frequency without changing the policy-selected freeze ratio.",
    },
}


def col_name(index: int) -> str:
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def cell(ref: str, value, style: int = 0) -> str:
    style_attr = f' s="{style}"' if style else ""
    if value is None or value == "":
        return f'<c r="{ref}"{style_attr}/>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            value = ""
        else:
            return f'<c r="{ref}"{style_attr}><v>{value}</v></c>'
    text = escape(str(value))
    return f'<c r="{ref}" t="inlineStr"{style_attr}><is><t>{text}</t></is></c>'


def worksheet_xml(
    rows: list[list],
    widths: list[float],
    *,
    title: str | None = None,
    best_cells: set[tuple[int, int]] | None = None,
    header_row: int = 1,
    autofilter: bool = True,
) -> str:
    best_cells = best_cells or set()
    row_offset = 1 if title else 0
    xml_rows = []
    if title:
        xml_rows.append(f'<row r="1" ht="24" customHeight="1">{cell("A1", title, 3)}</row>')

    for row_index, values in enumerate(rows, start=1 + row_offset):
        cells = []
        for col_index, value in enumerate(values, start=1):
            if row_index == header_row + row_offset:
                style = 1
            elif (row_index - row_offset, col_index) in best_cells:
                style = 4
            elif isinstance(value, float):
                style = 2
            elif value == "Complete":
                style = 5
            else:
                style = 0
            cells.append(cell(f"{col_name(col_index)}{row_index}", value, style))
        xml_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    last_col = col_name(max(len(row) for row in rows))
    last_row = len(rows) + row_offset
    cols = "".join(
        f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>'
        for index, width in enumerate(widths, start=1)
    )
    merge = f'<mergeCells count="1"><mergeCell ref="A1:{last_col}1"/></mergeCells>' if title else ""
    filter_xml = ""
    if autofilter:
        filter_row = header_row + row_offset
        filter_xml = f'<autoFilter ref="A{filter_row}:{last_col}{last_row}"/>'
    freeze_row = header_row + row_offset + 1
    pane = (
        f'<sheetViews><sheetView workbookViewId="0"><pane ySplit="{freeze_row - 1}" '
        f'topLeftCell="A{freeze_row}" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'{pane}<sheetFormatPr defaultRowHeight="15"/><cols>{cols}</cols>'
        f'<sheetData>{"".join(xml_rows)}</sheetData>{merge}{filter_xml}</worksheet>'
    )


def load_results() -> list[dict]:
    results = []
    for dataset, dataset_info in DATASETS.items():
        for mode, mode_info in MODES.items():
            experiment = (
                f"ablation_v2_no_{mode}_{dataset}_{dataset_info['target']}_task0_seed2025"
            )
            summary_path = (
                REPO_ROOT / "csv_files" / "BrainAI" / dataset / experiment / "summary.csv"
            )
            log_path = (
                REPO_ROOT / "outputs" / "BrainAI" / dataset / "ablation_v2" / f"{experiment}.txt"
            )
            if not summary_path.is_file():
                raise FileNotFoundError(f"Missing completed result: {summary_path}")
            with summary_path.open(newline="", encoding="utf-8") as handle:
                summary = next(csv.DictReader(handle))
            results.append(
                {
                    "dataset": dataset,
                    "target": dataset_info["target"],
                    "time": dataset_info["time"],
                    "seed": int(summary["seed"]),
                    "mode": mode,
                    "label": mode_info["label"],
                    "detail": mode_info["detail"],
                    "mae": float(summary["mae_mean"]),
                    "rmse": float(summary["rmse_mean"]),
                    "mape": float(summary["mape_mean"]),
                    "status": "Complete",
                    "experiment": experiment,
                    "summary_path": str(summary_path.relative_to(REPO_ROOT)),
                    "log_path": str(log_path.relative_to(REPO_ROOT)),
                }
            )
    return results


def build_workbook(results: list[dict]) -> None:
    summary_headers = [
        "Dataset", "Target", "Time", "Seed", "Ablation", "Setting",
        "MAE", "RMSE", "MAPE", "Status", "Experiment", "Summary path", "Log path",
    ]
    summary_rows = [summary_headers]
    for item in results:
        summary_rows.append(
            [
                item["dataset"], item["target"], item["time"], item["seed"],
                item["label"], item["detail"], item["mae"], item["rmse"],
                item["mape"], item["status"], item["experiment"],
                item["summary_path"], item["log_path"],
            ]
        )

    best_summary_cells = set()
    for dataset in DATASETS:
        dataset_items = [item for item in results if item["dataset"] == dataset]
        for metric_offset, metric in enumerate(("mae", "rmse", "mape"), start=7):
            best = min(item[metric] for item in dataset_items)
            for result_index, item in enumerate(results, start=2):
                if item["dataset"] == dataset and item[metric] == best:
                    best_summary_cells.add((result_index, metric_offset))

    comparison_headers = ["Dataset"]
    for mode_info in MODES.values():
        comparison_headers.extend(
            [f'{mode_info["label"]} MAE', f'{mode_info["label"]} RMSE', f'{mode_info["label"]} MAPE']
        )
    comparison_rows = [comparison_headers]
    comparison_best = set()
    for row_number, dataset in enumerate(DATASETS, start=2):
        row = [dataset]
        dataset_items = {item["mode"]: item for item in results if item["dataset"] == dataset}
        for mode in MODES:
            item = dataset_items[mode]
            row.extend([item["mae"], item["rmse"], item["mape"]])
        comparison_rows.append(row)
        for metric_index in range(3):
            metric_columns = [2 + metric_index, 5 + metric_index, 8 + metric_index]
            best_value = min(row[column - 1] for column in metric_columns)
            for column in metric_columns:
                if row[column - 1] == best_value:
                    comparison_best.add((row_number, column))

    notes_rows = [
        ["Item", "Value", "Description"],
        ["Source script", "scripts/ablation_v2/run_all.sh", "6 component ablations x 3 datasets; MAX_PARALLEL=2 by default."],
        ["Target task", "task0", "00:00-04:00 for all datasets."],
        ["Seed", 2025, "One run per ablation/dataset in the current script."],
    ]
    for mode_info in MODES.values():
        notes_rows.append([mode_info["label"], mode_info["detail"], mode_info["description"]])
    notes_rows.append(["Generated", dt.datetime.now().astimezone().isoformat(timespec="seconds"), str(OUTPUT_PATH.relative_to(REPO_ROOT))])

    sheets = [
        worksheet_xml(
            summary_rows,
            [10, 12, 14, 8, 16, 34, 12, 12, 12, 12, 48, 70, 70],
            title="Ablation v2 Results (lower is better)",
            best_cells=best_summary_cells,
        ),
        worksheet_xml(
            comparison_rows,
            [12] + [18] * (3 * len(MODES)),
            title="Dataset-wise Comparison (lower is better)",
            best_cells=comparison_best,
        ),
        worksheet_xml(notes_rows, [20, 42, 90], title="Experiment Notes", autofilter=False),
    ]

    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/worksheets/sheet3.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>'''
    root_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''
    workbook = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="Results" sheetId="1" r:id="rId1"/><sheet name="Comparison" sheetId="2" r:id="rId2"/><sheet name="Notes" sheetId="3" r:id="rId3"/></sheets>
</workbook>'''
    workbook_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet3.xml"/>
  <Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''
    styles = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="3"><font><sz val="11"/><name val="Calibri"/></font><font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="14"/><name val="Calibri"/></font></fonts>
  <fills count="4"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF4472C4"/><bgColor indexed="64"/></patternFill></fill><fill><patternFill patternType="solid"><fgColor rgb="FFC6EFCE"/><bgColor indexed="64"/></patternFill></fill></fills>
  <borders count="2"><border><left/><right/><top/><bottom/><diagonal/></border><border><left style="thin"><color rgb="FFD9E2F3"/></left><right style="thin"><color rgb="FFD9E2F3"/></right><top style="thin"><color rgb="FFD9E2F3"/></top><bottom style="thin"><color rgb="FFD9E2F3"/></bottom><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="6"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf><xf numFmtId="4" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/><xf numFmtId="0" fontId="2" fillId="0" borderId="0" xfId="0"/><xf numFmtId="4" fontId="0" fillId="3" borderId="1" xfId="0" applyNumberFormat="1"/><xf numFmtId="0" fontId="0" fillId="3" borderId="1" xfId="0"/></cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    core = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>Ablation v2 Results</dc:title><dc:creator>Codex</dc:creator><dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified></cp:coreProperties>'''
    app = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>Codex</Application></Properties>'''

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/styles.xml", styles)
        for index, sheet in enumerate(sheets, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", sheet)
        archive.writestr("docProps/core.xml", core)
        archive.writestr("docProps/app.xml", app)


def main() -> None:
    results = load_results()
    build_workbook(results)
    print(f"Exported {len(results)} completed results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
