"""Generic XLSX table parser."""

from pathlib import Path
from dataclasses import dataclass
from typing import Any

import openpyxl


@dataclass
class ParsedTable:
    """A complete parsed table with all data."""
    filename: str
    title: str | None
    headers: list[str]
    rows: list[dict[str, Any]]


def parse_xlsx_file(filepath: Path) -> ParsedTable | None:
    """Parse a single xlsx file and return all data.

    Handles header-row detection (skips image-filename rows)
    and ignores columns without headers.
    """
    try:
        wb = openpyxl.load_workbook(filepath, data_only=True)
    except Exception:
        return None

    sheet = wb.active
    all_data = list(sheet.iter_rows(values_only=True))
    if not all_data:
        wb.close()
        return None

    header_row_idx = 0
    for idx, row in enumerate(all_data[:3]):
        non_empty = [str(v) for v in row if v]
        if non_empty:
            img_count = sum(1 for v in non_empty if v.endswith(('.png', '.jpg', '.jpeg', '.gif')))
            if img_count > len(non_empty) / 2:
                header_row_idx = idx + 1
            else:
                header_row_idx = idx
                break

    if header_row_idx >= len(all_data):
        wb.close()
        return None

    raw_headers = all_data[header_row_idx]
    headers, valid_cols = [], []
    for i, val in enumerate(raw_headers):
        if val:
            h = str(val).strip()
            if len(h) > 100:
                h = h[:100] + "..."
            headers.append(h)
            valid_cols.append(i)

    rows = []
    for row_data in all_data[header_row_idx + 1:]:
        if not any(row_data):
            continue
        row_dict = {}
        for hi, ci in enumerate(valid_cols):
            if ci < len(row_data) and row_data[ci] is not None:
                row_dict[headers[hi]] = row_data[ci]
        if row_dict:
            rows.append(row_dict)

    wb.close()
    return ParsedTable(filename=filepath.name, title=derive_title(filepath.name),
                       headers=headers, rows=rows)


def derive_title(filename: str) -> str:
    """Derive a readable title from a filename like ``048_idk_table.xlsx``."""
    name = filename.rsplit('.', 1)[0]
    parts = name.split('_')
    if parts and parts[0].isdigit():
        parts = parts[1:]
    return ' '.join(parts).replace('_', ' ').title()
