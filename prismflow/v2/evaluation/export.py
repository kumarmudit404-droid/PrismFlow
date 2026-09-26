"""Export the hand-authored workbook to the JSON the loader reads.

``data/v2/part24_labeled_dataset.xlsx`` is the human-editable source of truth.
``data/v2/evaluation_queries.json`` is derived from it, and is what
``load_evaluation_dataset`` and the rest of the pipeline read. This module is
the only thing that writes that JSON.

Run it after editing the workbook::

    python -m prismflow.v2.evaluation.export

WHAT IT SKIPS, AND WHAT IT DELIBERATELY DOES NOT
------------------------------------------------
Two kinds of row never reach the JSON:

* the ``EX`` template row, which is documented in the workbook's Instructions
  tab as a format example and is the author's to delete or keep;
* rows where every authored field is still blank -- the unfilled tail of the
  50-row sheet.

Everything else is exported verbatim, INCLUDING a row that is only half
filled. That is the point: the loader's job is to reject a row that is not
ground truth, and an exporter that quietly dropped incomplete rows would hide
exactly the defect the loader exists to catch. A partially filled row should
fail loudly at load time, named by its id, not vanish between two files.

It also does not repair values. No trimming a stray ``<``, no guessing a
missing date, no normalising an outcome to the nearest legal spelling. The
JSON is a faithful transcription of what a human typed.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from prismflow.v2.evaluation.dataset import AUTHORED_FIELDS, DEFAULT_DATASET_PATH

DEFAULT_WORKBOOK_PATH = "data/v2/part24_labeled_dataset.xlsx"
SHEET_NAME = "Dataset"

# The workbook's worked example. Documented in its Instructions tab as a
# format reference, not data; deleting it is the author's call, not ours.
TEMPLATE_ROW_ID = "EX"


def _cell_to_str(value: Any) -> str:
    """Render one cell as the JSON wants it, without interpreting it.

    Dates become ISO-8601 days because Excel hands back ``datetime``; every
    other value becomes its stripped string form. A blank cell becomes ``""``.
    """
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value).strip()


def read_workbook_rows(
    workbook_path: Union[str, Path] = DEFAULT_WORKBOOK_PATH,
) -> List[Dict[str, str]]:
    """Read the Dataset sheet into a list of authored-field dicts.

    Raises:
        FileNotFoundError: the workbook is not where it is expected.
        ValueError: the sheet's header does not match the authored schema.
    """
    # Imported here, not at module scope: openpyxl is not in requirements.txt,
    # and importing the loader must never require a spreadsheet library.
    import openpyxl

    path = Path(workbook_path)
    if not path.is_file():
        raise FileNotFoundError("workbook not found: %s" % path)

    workbook = openpyxl.load_workbook(path, data_only=True)
    if SHEET_NAME not in workbook.sheetnames:
        raise ValueError(
            "%s has no %r sheet (found %s)"
            % (path, SHEET_NAME, workbook.sheetnames))
    sheet = workbook[SHEET_NAME]

    header = [_cell_to_str(cell.value) for cell in sheet[1]]
    if tuple(header) != AUTHORED_FIELDS:
        raise ValueError(
            "%s header is %s, expected %s -- the workbook and "
            "docs/part24-dataset-schema.md have diverged"
            % (path, header, list(AUTHORED_FIELDS)))

    rows: List[Dict[str, str]] = []
    for excel_row in range(2, sheet.max_row + 1):
        record = {
            field: _cell_to_str(sheet.cell(row=excel_row, column=index).value)
            for index, field in enumerate(AUTHORED_FIELDS, start=1)
        }
        record["_excel_row"] = str(excel_row)
        rows.append(record)
    return rows


def select_exportable(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Drop the template row and the fully blank tail. Keep everything else."""
    kept: List[Dict[str, str]] = []
    for record in rows:
        if record["id"] == TEMPLATE_ROW_ID:
            continue
        authored = [record[field] for field in AUTHORED_FIELDS if field != "id"]
        if not any(authored):
            continue
        kept.append(record)
    return kept


def export_dataset(
    workbook_path: Union[str, Path] = DEFAULT_WORKBOOK_PATH,
    json_path: Union[str, Path] = DEFAULT_DATASET_PATH,
) -> List[Dict[str, str]]:
    """Write the JSON the loader reads. Returns the rows written."""
    rows = select_exportable(read_workbook_rows(workbook_path))
    payload = [
        {field: record[field] for field in AUTHORED_FIELDS} for record in rows
    ]

    destination = Path(json_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workbook", default=DEFAULT_WORKBOOK_PATH)
    parser.add_argument("--out", default=DEFAULT_DATASET_PATH)
    args = parser.parse_args(argv)

    written = export_dataset(args.workbook, args.out)
    print("exported %d rows -> %s" % (len(written), args.out))
    for record in written:
        print("  %s  %-16s %-10s conflict=%s"
              % (record["id"], record["domain"],
                 record["actual_outcome"], record["conflict_expected"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
