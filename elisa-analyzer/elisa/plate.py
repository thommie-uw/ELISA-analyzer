"""Core plate constants and helpers for a 96-well plate (8 rows x 12 columns)."""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np

ROWS = list("ABCDEFGH")
COLS = list(range(1, 13))
N_ROWS = len(ROWS)
N_COLS = len(COLS)
N_WELLS = N_ROWS * N_COLS

WELL_RE = re.compile(r"^\s*([A-Ha-h])\s*0?(\d{1,2})\s*$")


def well_id(row_idx: int, col_idx: int) -> str:
    """0-based indices -> 'A01' style well id."""
    return f"{ROWS[row_idx]}{col_idx + 1:02d}"


def all_well_ids() -> list[str]:
    return [well_id(r, c) for r in range(N_ROWS) for c in range(N_COLS)]


def parse_well_id(text: object) -> tuple[int, int] | None:
    """'A1', 'a01', 'H12' -> (row_idx, col_idx). Returns None if not a well id."""
    if text is None:
        return None
    m = WELL_RE.match(str(text))
    if not m:
        return None
    r = ROWS.index(m.group(1).upper())
    c = int(m.group(2)) - 1
    if not 0 <= c < N_COLS:
        return None
    return r, c


def empty_plate(fill: float = np.nan) -> np.ndarray:
    return np.full((N_ROWS, N_COLS), fill, dtype=float)


def plate_to_long(matrix: np.ndarray, value_name: str = "od") -> list[dict]:
    """8x12 matrix -> list of {well, row, col, <value_name>} dicts."""
    out = []
    for r in range(N_ROWS):
        for c in range(N_COLS):
            out.append(
                {
                    "well": well_id(r, c),
                    "row": ROWS[r],
                    "col": c + 1,
                    value_name: float(matrix[r, c]),
                }
            )
    return out


def wells_in_order(order: str = "column") -> Iterable[tuple[int, int]]:
    """Iterate (row_idx, col_idx). order='column' fills A1,B1,...H1,A2...; 'row' fills A1..A12,B1.."""
    if order == "row":
        for r in range(N_ROWS):
            for c in range(N_COLS):
                yield r, c
    else:
        for c in range(N_COLS):
            for r in range(N_ROWS):
                yield r, c
