"""Plate layout: what each of the 96 wells contains, plus presets and templates.

A layout is an 8 x 12 array of short text codes typed straight into the grid
editor:

    ""  "-"  "x"      empty / unused well
    B   BLK  BLANK    blank (buffer / substrate only) - used for OD correction
    NSB               non-specific binding control (reported, not subtracted)
    S1 .. S12         standard level 1..12 (concentration set in the standards table)
    QC1  QC-High      quality control sample
    anything else     a sample; wells sharing the same text are replicates
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .plate import N_COLS, N_ROWS, ROWS, well_id, wells_in_order

EMPTY = "empty"
BLANK = "blank"
NSB = "nsb"
STANDARD = "standard"
SAMPLE = "sample"
CONTROL = "control"

_EMPTY_TOKENS = {"", "-", "--", "x", "none", "empty", "unused", "nan"}
_BLANK_TOKENS = {"b", "blk", "blank", "zero", "buffer"}
_NSB_TOKENS = {"nsb", "nsb1", "non-specific"}
_STD_RE = re.compile(r"^s(?:td)?[\s_-]*(\d{1,2})$", re.I)
_QC_RE = re.compile(r"^(qc|ctrl|control)[\s_-]*(.*)$", re.I)


@dataclass(frozen=True)
class WellRole:
    role: str
    label: str = ""      # sample / control name, or "S3" for a standard
    level: int = 0       # standard level number


def parse_code(code: Any) -> WellRole:
    """Turn one grid-editor cell into a WellRole."""
    text = "" if code is None else str(code).strip()
    low = text.lower()
    if low in _EMPTY_TOKENS:
        return WellRole(EMPTY)
    if low in _BLANK_TOKENS:
        return WellRole(BLANK, "Blank")
    if low in _NSB_TOKENS:
        return WellRole(NSB, "NSB")
    m = _STD_RE.match(text)
    if m:
        lvl = int(m.group(1))
        if 1 <= lvl <= 12:
            return WellRole(STANDARD, f"S{lvl}", lvl)
    m = _QC_RE.match(text)
    if m:
        suffix = m.group(2).strip()
        return WellRole(CONTROL, text.strip() if suffix else "QC")
    return WellRole(SAMPLE, text)


def blank_layout() -> np.ndarray:
    return np.full((N_ROWS, N_COLS), "", dtype=object)


def layout_to_frame(layout: np.ndarray):
    import pandas as pd

    return pd.DataFrame(
        np.asarray(layout, dtype=object),
        index=ROWS,
        columns=[str(c) for c in range(1, N_COLS + 1)],
    )


def frame_to_layout(df) -> np.ndarray:
    arr = df.to_numpy(dtype=object)
    out = blank_layout()
    for r in range(min(N_ROWS, arr.shape[0])):
        for c in range(min(N_COLS, arr.shape[1])):
            v = arr[r, c]
            out[r, c] = "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip()
    return out


# --------------------------------------------------------------------------
# presets
# --------------------------------------------------------------------------
def _fill_standards(layout, n_levels, n_reps, orientation, include_blank):
    if orientation == "columns":
        for rep in range(n_reps):
            for lvl in range(n_levels):
                layout[lvl, rep] = f"S{lvl + 1}"
        if include_blank:
            for rep in range(n_reps):
                layout[n_levels, rep] = "B"
    else:  # standards laid out along a row
        for rep in range(n_reps):
            for lvl in range(n_levels):
                layout[rep, lvl] = f"S{lvl + 1}"
        if include_blank:
            for rep in range(n_reps):
                layout[rep, n_levels] = "B"
    return layout


PRESETS: dict[str, dict] = {
    "Empty Plate": {},
    "Standards Col 1-2, 8 Levels (Duplicate)": dict(
        n_levels=8, n_reps=2, orientation="columns", include_blank=False
    ),
    "Standards Col 1-2, 7 Levels + Blank (Duplicate)": dict(
        n_levels=7, n_reps=2, orientation="columns", include_blank=True
    ),
    "Standards Col 1, 8 Levels (Single)": dict(
        n_levels=8, n_reps=1, orientation="columns", include_blank=False
    ),
    "Standards Col 1-3, 8 Levels (Triplicate)": dict(
        n_levels=8, n_reps=3, orientation="columns", include_blank=False
    ),
    "Standards Row A-B, 11 Levels + Blank (Duplicate)": dict(
        n_levels=11, n_reps=2, orientation="rows", include_blank=True
    ),
}


def preset_layout(name: str) -> np.ndarray:
    layout = blank_layout()
    spec = PRESETS.get(name) or {}
    if not spec:
        return layout
    return _fill_standards(layout, **spec)


def autofill_samples(
    layout: np.ndarray,
    replicates: int = 2,
    order: str = "column",
    prefix: str = "Sample",
    start_at: int = 1,
    overwrite: bool = False,
) -> np.ndarray:
    """Name every still-unassigned well as a sample, grouping replicates.

    ``order='column'`` walks A1,B1,...H1,A2,... so a duplicate pair sits in
    vertically adjacent wells; ``order='row'`` walks A1..A12 then B1..
    """
    out = np.array(layout, dtype=object, copy=True)
    n = max(1, int(replicates))
    idx = int(start_at)
    counter = 0
    current = f"{prefix}{idx:02d}"
    for r, c in wells_in_order(order):
        cell = str(out[r, c]).strip()
        if cell and not overwrite:
            continue
        if counter == n:
            counter = 0
            idx += 1
            current = f"{prefix}{idx:02d}"
        out[r, c] = current
        counter += 1
    return out


# --------------------------------------------------------------------------
# summary + templates
# --------------------------------------------------------------------------
@dataclass
class LayoutSummary:
    standard_levels: list[int] = field(default_factory=list)
    sample_labels: list[str] = field(default_factory=list)
    control_labels: list[str] = field(default_factory=list)
    n_blank: int = 0
    n_nsb: int = 0
    n_empty: int = 0
    replicate_counts: dict[str, int] = field(default_factory=dict)


def summarise(layout: np.ndarray) -> LayoutSummary:
    s = LayoutSummary()
    seen_std: set[int] = set()
    for r in range(N_ROWS):
        for c in range(N_COLS):
            role = parse_code(layout[r, c])
            if role.role == EMPTY:
                s.n_empty += 1
            elif role.role == BLANK:
                s.n_blank += 1
            elif role.role == NSB:
                s.n_nsb += 1
            elif role.role == STANDARD:
                seen_std.add(role.level)
                s.replicate_counts[role.label] = s.replicate_counts.get(role.label, 0) + 1
            elif role.role == CONTROL:
                if role.label not in s.control_labels:
                    s.control_labels.append(role.label)
                s.replicate_counts[role.label] = s.replicate_counts.get(role.label, 0) + 1
            else:
                if role.label not in s.sample_labels:
                    s.sample_labels.append(role.label)
                s.replicate_counts[role.label] = s.replicate_counts.get(role.label, 0) + 1
    s.standard_levels = sorted(seen_std)
    return s


def validate(layout: np.ndarray, standard_conc: dict[str, float]) -> list[str]:
    """Human-readable problems that would stop or skew the analysis."""
    problems: list[str] = []
    s = summarise(layout)
    if not s.standard_levels:
        problems.append("No standard wells assigned (use codes S1, S2, ... in the grid).")
    else:
        missing = [
            f"S{lvl}" for lvl in s.standard_levels
            if standard_conc.get(f"S{lvl}") is None
            or not np.isfinite(float(standard_conc.get(f"S{lvl}", np.nan)))
        ]
        if missing:
            problems.append(
                "Standard level(s) without a concentration: " + ", ".join(missing)
            )
        distinct = {
            float(standard_conc[f"S{lvl}"])
            for lvl in s.standard_levels
            if standard_conc.get(f"S{lvl}") is not None
        }
        if len(distinct) < 3:
            problems.append("Need at least 3 distinct standard concentrations to fit a curve.")
    if not s.sample_labels and not s.control_labels:
        problems.append("No sample wells assigned - results will only cover the standards.")
    return problems


def default_standard_table(layout: np.ndarray, top: float = 1000.0, factor: float = 2.0):
    """Seed the standards table with a serial dilution from the top standard."""
    import pandas as pd

    s = summarise(layout)
    levels = s.standard_levels or list(range(1, 9))
    rows = []
    for i, lvl in enumerate(sorted(levels)):
        rows.append({"Standard": f"S{lvl}", "Concentration": top / (factor**i)})
    return pd.DataFrame(rows)


def to_template(
    layout: np.ndarray,
    standard_conc: dict[str, float],
    units: str = "pg/mL",
    dilutions: dict[str, float] | None = None,
    name: str = "layout",
) -> str:
    payload = {
        "format": "elisa-analyzer-layout",
        "version": 1,
        "name": name,
        "units": units,
        "layout": [[str(layout[r, c]) for c in range(N_COLS)] for r in range(N_ROWS)],
        "standards": {k: float(v) for k, v in standard_conc.items() if v is not None},
        "dilutions": {k: float(v) for k, v in (dilutions or {}).items()},
    }
    return json.dumps(payload, indent=2)


def from_template(text: str | bytes) -> dict:
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    payload = json.loads(text)
    grid = payload.get("layout") or []
    layout = blank_layout()
    for r in range(min(N_ROWS, len(grid))):
        row = grid[r]
        for c in range(min(N_COLS, len(row))):
            layout[r, c] = str(row[c])
    return {
        "layout": layout,
        "standards": {str(k): float(v) for k, v in (payload.get("standards") or {}).items()},
        "dilutions": {str(k): float(v) for k, v in (payload.get("dilutions") or {}).items()},
        "units": payload.get("units", "pg/mL"),
        "name": payload.get("name", "layout"),
    }


def role_matrix(layout: np.ndarray) -> np.ndarray:
    """8x12 array of role names, handy for colouring the plate map."""
    out = np.empty((N_ROWS, N_COLS), dtype=object)
    for r in range(N_ROWS):
        for c in range(N_COLS):
            out[r, c] = parse_code(layout[r, c]).role
    return out


def well_labels(layout: np.ndarray) -> dict[str, WellRole]:
    return {
        well_id(r, c): parse_code(layout[r, c])
        for r in range(N_ROWS)
        for c in range(N_COLS)
    }
