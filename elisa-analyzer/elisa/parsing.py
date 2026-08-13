"""Flexible readers for plate-reader exports (txt / csv / tsv / xls / xlsx).

The strategy is deliberately format-agnostic: every input is first flattened into
one or more 2-D grids of raw cell values, then we scan those grids for anything
that looks like an 8 x 12 block of numbers. That single approach covers

  * plain 8x12 CSV / TSV dumps, with or without A-H / 1-12 headers
  * Molecular Devices SoftMax Pro tab-delimited exports (leading blank column,
    temperature column, multiple wavelengths side by side, ``~End`` markers)
  * BioTek Gen5 / Tecan Magellan exports with metadata rows above the block
  * multi-sheet Excel workbooks and multi-plate text files

A separate path handles "long" exports that list one well per row
(``Well, Value`` style).
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .plate import COLS, N_COLS, N_ROWS, ROWS, parse_well_id

# Strings a reader may emit instead of a number.
NON_NUMERIC_SENTINELS = {
    "", "-", "--", "n/a", "na", "nan", "none", "null", "empty", ".",
    "ovrflw", "overflow", "over", "ovr", "sat", "#sat", "*", "**", "***",
    "####", "out of range", "range?", "?????",
}
OVERFLOW_SENTINELS = {"ovrflw", "overflow", "over", "ovr", "sat", "#sat", "*", "**", "***"}

_NUM_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


@dataclass
class PlateRead:
    """One detected 8x12 absorbance block."""

    name: str
    values: np.ndarray  # (8, 12) float, NaN where unreadable
    origin: str = ""
    n_numeric: int = 0
    overflow_wells: list[str] = field(default_factory=list)
    has_row_labels: bool = False
    has_col_labels: bool = False

    @property
    def n_missing(self) -> int:
        return int(np.isnan(self.values).sum())

    def describe(self) -> str:
        bits = [self.name]
        if self.origin:
            bits.append(f"({self.origin})")
        if self.n_missing:
            bits.append(f"- {self.n_missing} empty well(s)")
        if self.overflow_wells:
            bits.append(f"- {len(self.overflow_wells)} overflow")
        return " ".join(bits)


# --------------------------------------------------------------------------
# cell-level helpers
# --------------------------------------------------------------------------
def _clean(cell: Any) -> str:
    if cell is None:
        return ""
    if isinstance(cell, float) and np.isnan(cell):
        return ""
    return str(cell).strip()


def to_number(cell: Any) -> tuple[float, bool]:
    """Return (value, is_overflow). value is NaN when the cell is not a number."""
    if isinstance(cell, (int, float, np.integer, np.floating)) and not isinstance(cell, bool):
        v = float(cell)
        return (np.nan, False) if np.isnan(v) else (v, False)

    s = _clean(cell)
    low = s.lower()
    if low in OVERFLOW_SENTINELS:
        return np.nan, True
    if low in NON_NUMERIC_SENTINELS:
        return np.nan, False

    s2 = s.replace("−", "-").replace(" ", "")
    if _NUM_RE.match(s2):
        return float(s2), False
    # European decimal comma, e.g. "0,123"
    if "," in s2 and "." not in s2 and _NUM_RE.match(s2.replace(",", ".")):
        return float(s2.replace(",", ".")), False
    return np.nan, False


def _is_blank(cell: Any) -> bool:
    return _clean(cell).lower() in {"", "-", "--", "n/a", "na", "nan", "none", "."}


# --------------------------------------------------------------------------
# file -> grids
# --------------------------------------------------------------------------
def _grid_from_rows(rows: Sequence[Sequence[Any]]) -> np.ndarray:
    width = max((len(r) for r in rows), default=0)
    if width == 0:
        return np.empty((0, 0), dtype=object)
    padded = [list(r) + [""] * (width - len(r)) for r in rows]
    return np.array(padded, dtype=object)


def _sniff_delimiter(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()][:200]
    if not lines:
        return "\t"
    best, best_score = "\t", -1.0
    for delim in ["\t", ",", ";", "|"]:
        counts = [ln.count(delim) for ln in lines]
        if max(counts, default=0) == 0:
            continue
        # reward many fields and consistent field counts
        score = np.mean(counts) - np.std(counts) * 0.5
        if score > best_score:
            best, best_score = delim, score
    if best_score < 0:
        return r"\s+"
    return best


def _text_to_grids(text: str, label: str = "file") -> list[tuple[str, np.ndarray]]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    delim = _sniff_delimiter(text)
    rows: list[list[str]] = []
    for line in text.split("\n"):
        if delim == r"\s+":
            rows.append(re.split(r"\s{1,}", line.strip()))
        else:
            rows.append(line.split(delim))
    return [(label, _grid_from_rows(rows))]


def _excel_to_grids(data: bytes, filename: str) -> list[tuple[str, np.ndarray]]:
    engine = "xlrd" if filename.lower().endswith(".xls") else "openpyxl"
    try:
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None, engine=engine)
    except Exception:
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None)
    grids = []
    for sheet_name, df in sheets.items():
        if df.empty:
            continue
        grids.append((str(sheet_name), df.to_numpy(dtype=object)))
    return grids


def load_grids(data: bytes, filename: str) -> list[tuple[str, np.ndarray]]:
    """Read any supported file into (label, 2-D object grid) pairs."""
    lower = filename.lower()
    stem = filename.rsplit("/", 1)[-1].rsplit(".", 1)[0] or "file"
    if lower.endswith((".xlsx", ".xlsm", ".xls", ".xltx")):
        return _excel_to_grids(data, filename)
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return _text_to_grids(data.decode(encoding), stem)
        except UnicodeDecodeError:
            continue
    return _text_to_grids(data.decode("utf-8", errors="replace"), stem)


# --------------------------------------------------------------------------
# grid -> plate blocks
# --------------------------------------------------------------------------
def _numeric_masks(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = grid.shape
    vals = np.full((h, w), np.nan)
    over = np.zeros((h, w), dtype=bool)
    blank = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            v, o = to_number(grid[r, c])
            vals[r, c] = v
            over[r, c] = o
            blank[r, c] = _is_blank(grid[r, c])
    return vals, over, blank


def _row_labels_at(grid: np.ndarray, r0: int, col: int) -> bool:
    if col < 0:
        return False
    got = [_clean(grid[r0 + i, col]).upper().rstrip(":") for i in range(N_ROWS)]
    return got == ROWS


def _col_labels_at(grid: np.ndarray, row: int, c0: int) -> bool:
    if row < 0:
        return False
    got = []
    for i in range(N_COLS):
        v, _ = to_number(grid[row, c0 + i])
        got.append(int(v) if not np.isnan(v) else None)
    return got == COLS


def _block_name(grid: np.ndarray, r0: int, c0: int, fallback: str) -> str:
    """Look a few rows above the block for a plate / wavelength label."""
    h, w = grid.shape
    for r in range(r0 - 1, max(-1, r0 - 7), -1):  # nearest row first
        texts = [_clean(x) for x in grid[r, : min(w, c0 + N_COLS + 2)] if _clean(x)]
        if not texts:
            continue
        joined = re.sub(r"\s+", " ", " ".join(texts))
        if re.fullmatch(r"\d{3}", joined):  # bare wavelength label, e.g. "450"
            return f"{joined} nm"
        if re.search(r"plate|wavelength|\bread\b|\b\d{3}\s?nm\b|\bA\d{3}\b", joined, re.I):
            return joined[:80]
    return fallback


def find_plate_blocks(
    grid: np.ndarray,
    label: str,
    min_numeric: int = 48,
) -> list[PlateRead]:
    """Scan a grid for 8x12 blocks that are (almost) entirely numeric."""
    if grid.size == 0:
        return []
    h, w = grid.shape
    if h < N_ROWS or w < N_COLS:
        return []

    vals, over, blank = _numeric_masks(grid)
    isnum = ~np.isnan(vals)
    ok_cell = isnum | blank | over  # blanks/overflows are allowed inside a block

    candidates: list[tuple[float, int, int, dict]] = []
    for r0 in range(h - N_ROWS + 1):
        for c0 in range(w - N_COLS + 1):
            win_ok = ok_cell[r0 : r0 + N_ROWS, c0 : c0 + N_COLS]
            if not win_ok.all():
                continue
            n_num = int(isnum[r0 : r0 + N_ROWS, c0 : c0 + N_COLS].sum())
            if n_num < min_numeric:
                continue
            has_rows = _row_labels_at(grid, r0, c0 - 1) or (
                c0 >= 2 and _row_labels_at(grid, r0, c0 - 2)
            )
            has_cols = _col_labels_at(grid, r0 - 1, c0) or (
                r0 >= 2 and _col_labels_at(grid, r0 - 2, c0)
            )
            score = n_num + 40 * has_rows + 40 * has_cols
            candidates.append(
                (score, r0, c0, {"n_num": n_num, "rows": has_rows, "cols": has_cols})
            )

    if not candidates:
        return []

    # greedy: best score first, skip anything overlapping an accepted block
    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))
    taken: list[tuple[int, int]] = []
    reads: list[PlateRead] = []
    for score, r0, c0, meta in candidates:
        if any(
            abs(r0 - tr) < N_ROWS and abs(c0 - tc) < N_COLS for tr, tc in taken
        ):
            continue
        taken.append((r0, c0))
        block = vals[r0 : r0 + N_ROWS, c0 : c0 + N_COLS].copy()
        over_block = over[r0 : r0 + N_ROWS, c0 : c0 + N_COLS]
        overflow_wells = [
            f"{ROWS[i]}{j + 1:02d}"
            for i in range(N_ROWS)
            for j in range(N_COLS)
            if over_block[i, j]
        ]
        name = _block_name(grid, r0, c0, f"{label} @ R{r0 + 1}C{c0 + 1}")
        reads.append(
            PlateRead(
                name=name,
                values=block,
                origin=f"{label}, cell R{r0 + 1}C{c0 + 1}",
                n_numeric=meta["n_num"],
                overflow_wells=overflow_wells,
                has_row_labels=meta["rows"],
                has_col_labels=meta["cols"],
            )
        )

    # a lone block on a grid doesn't need a positional name
    if len(reads) == 1 and reads[0].name.startswith(f"{label} @ R"):
        reads[0].name = label

    reads.sort(key=lambda pr: (pr.origin,))
    return reads


def find_long_format(grid: np.ndarray, label: str) -> list[PlateRead]:
    """Handle exports shaped like  Well | Value  (one row per well)."""
    if grid.size == 0:
        return []
    h, w = grid.shape
    value_hint = re.compile(
        r"abs|\bod\b|optical|value|result|raw|signal|mean|reading|450|read", re.I
    )
    best: tuple[float, int, int, int] | None = None  # (score, n_pairs, well_col, value_col)
    for wc in range(w):
        coords = [parse_well_id(grid[r, wc]) for r in range(h)]
        n_hits = sum(1 for x in coords if x is not None)
        if n_hits < 24:
            continue
        for vc in range(w):
            if vc == wc:
                continue
            vals = []
            for r in range(h):
                if coords[r] is None:
                    continue
                v, _ = to_number(grid[r, vc])
                if not np.isnan(v):
                    vals.append(v)
            if len(vals) < 24:
                continue
            # a column of constants (e.g. a wavelength) is not the reading
            n_unique = len(set(np.round(vals, 9)))
            header = " ".join(_clean(grid[r, vc]) for r in range(min(h, 6)))
            score = len(vals) + 2.0 * n_unique + (50 if value_hint.search(header) else 0)
            if best is None or score > best[0]:
                best = (score, len(vals), wc, vc)
    if best is None or best[1] < 24:
        return []

    _, _, wc, vc = best
    values = np.full((N_ROWS, N_COLS), np.nan)
    for r in range(h):
        coord = parse_well_id(grid[r, wc])
        if coord is None:
            continue
        v, _ = to_number(grid[r, vc])
        values[coord[0], coord[1]] = v
    n_num = int(np.isfinite(values).sum())
    return [
        PlateRead(
            name=f"{label} (well/value list)",
            values=values,
            origin=f"{label}, long format",
            n_numeric=n_num,
        )
    ]


def read_plates(data: bytes, filename: str) -> list[PlateRead]:
    """Top-level entry point: bytes -> every plate block we can find."""
    grids = load_grids(data, filename)
    reads: list[PlateRead] = []
    for label, grid in grids:
        found = find_plate_blocks(grid, label)
        if not found:
            found = find_long_format(grid, label)
        reads.extend(found)

    # de-duplicate identical matrices coming from overlapping detections
    unique: list[PlateRead] = []
    for pr in reads:
        if any(
            np.allclose(pr.values, u.values, equal_nan=True) for u in unique
        ):
            continue
        unique.append(pr)

    if len(unique) > 1:
        seen: dict[str, int] = {}
        for pr in unique:
            seen[pr.name] = seen.get(pr.name, 0) + 1
            if seen[pr.name] > 1:
                pr.name = f"{pr.name} #{seen[pr.name]}"
    return unique
