"""Turn a plate + layout into standards, sample concentrations and QC statistics."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .fitting import FitResult, fit_standard_curve
from .layout import BLANK, CONTROL, EMPTY, NSB, SAMPLE, STANDARD, parse_code
from .plate import N_COLS, N_ROWS, ROWS, well_id


ROLE_DISPLAY = {
    STANDARD: "Standard",
    BLANK: "Blank",
    SAMPLE: "Sample",
    CONTROL: "Control",
    NSB: "NSB",
    EMPTY: "Empty",
}


@dataclass
class AnalysisOptions:
    model: str = "4PL"
    weighting: str = "None"
    subtract_blank: bool = True
    fit_on: str = "Individual Replicates"  # or "Replicate Means"
    cv_threshold: float = 20.0
    recovery_low: float = 80.0
    recovery_high: float = 120.0
    units: str = "pg/mL"
    dilutions: dict[str, float] = field(default_factory=dict)
    excluded_wells: set[str] = field(default_factory=set)
    excluded_standard_levels: set[int] = field(default_factory=set)
    extrapolate: bool = False  # report values outside the standard range


@dataclass
class AnalysisResult:
    fit: FitResult
    wells: pd.DataFrame
    standards: pd.DataFrame
    samples: pd.DataFrame
    controls: pd.DataFrame
    options: AnalysisOptions
    blank_mean: float = np.nan
    blank_sd: float = np.nan
    blank_cv: float = np.nan
    n_blank: int = 0
    lloq: float = np.nan
    uloq: float = np.nan
    warnings: list[str] = field(default_factory=list)

    @property
    def units(self) -> str:
        return self.options.units

    def qc_summary(self) -> pd.DataFrame:
        std_ok = self.standards["Recovery Flag"].eq("OK").sum() if len(self.standards) else 0
        rows = [
            ("Curve Model", self.fit.model),
            ("Weighting", self.fit.weighting),
            ("R²", round(self.fit.r_squared, 5)),
            ("Adjusted R²", round(self.fit.adj_r_squared, 5)),
            ("RMSE (OD)", round(self.fit.rmse, 5)),
            ("Standard Points Fitted", self.fit.n_points),
            ("Standard Levels Within Recovery Limits", f"{std_ok} / {len(self.standards)}"),
            (f"LLOQ ({self.units})", _fmt(self.lloq)),
            (f"ULOQ ({self.units})", _fmt(self.uloq)),
            ("Blank Wells", self.n_blank),
            ("Mean Blank OD", _fmt(self.blank_mean, 4)),
            ("Blank %CV", _fmt(self.blank_cv, 1)),
            ("Samples Analysed", len(self.samples)),
            (
                f"Samples with %CV > {self.options.cv_threshold:g}%",
                int(self.samples["CV Flag"].eq("High CV").sum()) if len(self.samples) else 0,
            ),
            (
                "Samples Outside the Standard Range",
                int(self.samples["Range Flag"].ne("In Range").sum()) if len(self.samples) else 0,
            ),
        ]
        return pd.DataFrame(rows, columns=["Metric", "Value"])


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return round(float(v), nd)


def _dilution_for(dilutions: dict[str, float], label: str) -> float:
    """Dilution factor for a sample, defaulting to 1 for blank/zero/missing entries."""
    try:
        value = float(dilutions.get(label, 1.0) or 1.0)
    except (TypeError, ValueError):
        return 1.0
    return value if np.isfinite(value) and value > 0 else 1.0


def _cv(values: np.ndarray) -> float:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if vals.size < 2:
        return np.nan
    mean = vals.mean()
    if mean == 0:
        return np.nan
    return float(vals.std(ddof=1) / abs(mean) * 100.0)


def build_well_table(
    values: np.ndarray,
    layout: np.ndarray,
    standard_conc: dict[str, float],
    excluded_wells: set[str] | None = None,
) -> pd.DataFrame:
    excluded_wells = excluded_wells or set()
    rows = []
    for r in range(N_ROWS):
        for c in range(N_COLS):
            wid = well_id(r, c)
            role = parse_code(layout[r, c])
            nominal = np.nan
            if role.role == STANDARD:
                nominal = float(standard_conc.get(role.label, np.nan))
            rows.append(
                {
                    "Well": wid,
                    "Row": ROWS[r],
                    "Col": c + 1,
                    "Role": role.role,
                    "Label": role.label,
                    "Level": role.level or np.nan,
                    "Nominal": nominal,
                    "OD Raw": float(values[r, c]),
                    "Excluded": wid in excluded_wells,
                }
            )
    return pd.DataFrame(rows)


def analyse(
    values: np.ndarray,
    layout: np.ndarray,
    standard_conc: dict[str, float],
    options: AnalysisOptions | None = None,
) -> AnalysisResult:
    opts = options or AnalysisOptions()
    warns: list[str] = []
    wells = build_well_table(values, layout, standard_conc, opts.excluded_wells)

    # ---------------- blank correction ----------------
    blank_mask = (wells["Role"] == BLANK) & (~wells["Excluded"])
    blank_vals = wells.loc[blank_mask, "OD Raw"].to_numpy(dtype=float)
    blank_vals = blank_vals[np.isfinite(blank_vals)]
    blank_mean = float(blank_vals.mean()) if blank_vals.size else np.nan
    blank_sd = float(blank_vals.std(ddof=1)) if blank_vals.size > 1 else np.nan
    blank_cv = _cv(blank_vals)

    offset = 0.0
    if opts.subtract_blank:
        if blank_vals.size:
            offset = blank_mean
        else:
            warns.append(
                "Blank subtraction was requested but no wells are marked B / BLANK - "
                "raw ODs were used unchanged."
            )
    wells["OD Corrected"] = wells["OD Raw"] - offset

    # ---------------- standards & fit ----------------
    std_mask = (
        (wells["Role"] == STANDARD)
        & (~wells["Excluded"])
        & wells["OD Corrected"].apply(np.isfinite)
        & wells["Nominal"].apply(np.isfinite)
    )
    if opts.excluded_standard_levels:
        std_mask &= ~wells["Level"].isin(list(opts.excluded_standard_levels))
    std_wells = wells[std_mask]
    if std_wells.empty:
        raise ValueError("No usable standard wells - assign S1, S2, ... and give them concentrations.")

    if opts.fit_on.strip().lower().startswith("replicate mean"):
        grouped = std_wells.groupby("Label", sort=False).agg(
            x=("Nominal", "mean"), y=("OD Corrected", "mean")
        )
        fit_x = grouped["x"].to_numpy()
        fit_y = grouped["y"].to_numpy()
    else:
        fit_x = std_wells["Nominal"].to_numpy(dtype=float)
        fit_y = std_wells["OD Corrected"].to_numpy(dtype=float)

    fit = fit_standard_curve(fit_x, fit_y, model=opts.model, weighting=opts.weighting)

    # ---------------- back-calculate every well ----------------
    wells["Conc"] = fit.inverse(wells["OD Corrected"].to_numpy(dtype=float))
    wells.loc[wells["Role"] == EMPTY, "Conc"] = np.nan

    # Dilution applies to unknowns only - standards and blanks are read neat.
    wells["Dilution"] = [
        _dilution_for(opts.dilutions, label) if role in (SAMPLE, CONTROL) else 1.0
        for role, label in zip(wells["Role"], wells["Label"])
    ]
    wells["Conc Final"] = wells["Conc"] * wells["Dilution"]

    # quantification limits = lowest / highest non-zero standard actually used
    used_conc = std_wells["Nominal"].to_numpy(dtype=float)
    nonzero = used_conc[used_conc > 0]
    lloq = float(nonzero.min()) if nonzero.size else np.nan
    uloq = float(used_conc.max()) if used_conc.size else np.nan

    # ---------------- standards table ----------------
    std_rows = []
    all_std = wells[(wells["Role"] == STANDARD)]
    for label, grp in all_std.groupby("Label", sort=False):
        active = grp[~grp["Excluded"]]
        level = int(grp["Level"].iloc[0])
        nominal = float(grp["Nominal"].iloc[0])
        ods = active["OD Corrected"].to_numpy(dtype=float)
        ods = ods[np.isfinite(ods)]
        mean_od = float(ods.mean()) if ods.size else np.nan
        back = float(fit.inverse(np.array([mean_od]))[0]) if np.isfinite(mean_od) else np.nan
        recovery = (back / nominal * 100.0) if nominal > 0 and np.isfinite(back) else np.nan
        if not np.isfinite(recovery):
            rec_flag = "n/a"
        elif opts.recovery_low <= recovery <= opts.recovery_high:
            rec_flag = "OK"
        else:
            rec_flag = "Out of Limits"
        excluded_level = level in opts.excluded_standard_levels or bool(grp["Excluded"].all())
        std_rows.append(
            {
                "Standard": label,
                "Level": level,
                f"Nominal ({opts.units})": nominal,
                "N": int(ods.size),
                "Wells": ", ".join(active["Well"].tolist()),
                "Mean OD": mean_od,
                "SD OD": float(ods.std(ddof=1)) if ods.size > 1 else np.nan,
                "%CV OD": _cv(ods),
                f"Back-Calculated ({opts.units})": back,
                "Recovery %": recovery,
                "Recovery Flag": rec_flag,
                "Used in Fit": not excluded_level,
            }
        )
    standards = pd.DataFrame(std_rows).sort_values("Level").reset_index(drop=True)
    if len(standards):
        standards["%CV Flag"] = np.where(
            standards["%CV OD"] > opts.cv_threshold, "High CV", "OK"
        )

    # ---------------- samples & controls ----------------
    samples = _summarise_unknowns(wells, fit, opts, SAMPLE, lloq, uloq)
    controls = _summarise_unknowns(wells, fit, opts, CONTROL, lloq, uloq)

    nsb = wells[(wells["Role"] == NSB) & (~wells["Excluded"])]
    if len(nsb):
        warns.append(
            f"NSB wells ({', '.join(nsb['Well'])}) mean OD = "
            f"{nsb['OD corrected'].mean():.4f} (reported only, not subtracted)."
        )
    if fit.r_squared < 0.98:
        warns.append(
            f"Standard curve R^2 is {fit.r_squared:.4f}. Check for outlying standard "
            "replicates or try the other model."
        )

    return AnalysisResult(
        fit=fit,
        wells=wells,
        standards=standards,
        samples=samples,
        controls=controls,
        options=opts,
        blank_mean=blank_mean,
        blank_sd=blank_sd,
        blank_cv=blank_cv,
        n_blank=int(blank_vals.size),
        lloq=lloq,
        uloq=uloq,
        warnings=warns,
    )


def _summarise_unknowns(
    wells: pd.DataFrame,
    fit: FitResult,
    opts: AnalysisOptions,
    role: str,
    lloq: float,
    uloq: float,
) -> pd.DataFrame:
    subset = wells[wells["Role"] == role]
    rows = []
    for label, grp in subset.groupby("Label", sort=False):
        active = grp[~grp["Excluded"]]
        ods = active["OD Corrected"].to_numpy(dtype=float)
        ods_ok = ods[np.isfinite(ods)]
        concs = active["Conc"].to_numpy(dtype=float)
        concs_ok = concs[np.isfinite(concs)]

        mean_od = float(ods_ok.mean()) if ods_ok.size else np.nan
        conc_from_mean_od = (
            float(fit.inverse(np.array([mean_od]))[0]) if np.isfinite(mean_od) else np.nan
        )
        mean_conc = float(concs_ok.mean()) if concs_ok.size else np.nan
        sd_conc = float(concs_ok.std(ddof=1)) if concs_ok.size > 1 else np.nan
        dil = _dilution_for(opts.dilutions, label)

        n_unquant = int(np.isfinite(ods).sum() - concs_ok.size)
        if not np.isfinite(mean_conc):
            if np.isfinite(mean_od) and _above_top(mean_od, fit):
                range_flag = "Above Curve (> ULOQ)"
            elif np.isfinite(mean_od):
                range_flag = "Below Curve (< LLOQ)"
            else:
                range_flag = "No Reading"
        elif np.isfinite(lloq) and mean_conc < lloq:
            range_flag = "< LLOQ"
        elif np.isfinite(uloq) and mean_conc > uloq:
            range_flag = "> ULOQ"
        else:
            range_flag = "In Range"

        cv_conc = _cv(concs_ok)
        cv_od = _cv(ods_ok)
        cv_flag = "High CV" if np.isfinite(cv_od) and cv_od > opts.cv_threshold else "OK"

        report_conc = mean_conc
        if range_flag not in ("In Range",) and not opts.extrapolate:
            if range_flag in ("< LLOQ", "> ULOQ"):
                pass  # value still meaningful, just outside the validated range
            else:
                report_conc = np.nan

        rows.append(
            {
                "Sample": label,
                "N": int(ods_ok.size),
                "Wells": ", ".join(active["Well"].tolist()),
                "Mean OD": mean_od,
                "SD OD": float(ods_ok.std(ddof=1)) if ods_ok.size > 1 else np.nan,
                "%CV OD": cv_od,
                f"Mean Conc ({opts.units})": mean_conc,
                f"SD Conc ({opts.units})": sd_conc,
                "%CV Conc": cv_conc,
                "Dilution": dil,
                f"Final Conc ({opts.units})": report_conc * dil
                if np.isfinite(report_conc)
                else np.nan,
                f"Conc from Mean OD ({opts.units})": conc_from_mean_od,
                "Replicates Off Curve": n_unquant,
                "Range Flag": range_flag,
                "CV Flag": cv_flag,
            }
        )
    cols = [
        "Sample", "N", "Wells", "Mean OD", "SD OD", "%CV OD",
        f"Mean Conc ({opts.units})", f"SD Conc ({opts.units})", "%CV Conc",
        "Dilution", f"Final Conc ({opts.units})", f"Conc from Mean OD ({opts.units})",
        "Replicates Off Curve", "Range Flag", "CV Flag",
    ]
    return pd.DataFrame(rows, columns=cols)


def _above_top(od: float, fit: FitResult) -> bool:
    """True when an OD sits beyond the high-concentration asymptote."""
    a, d = fit.params["a"], fit.params["d"]
    return od > max(a, d) if d > a else od < min(a, d)


def display_wells(wells: pd.DataFrame, units: str) -> pd.DataFrame:
    """Well table with presentation-ready role names and column headers."""
    out = wells.copy()
    out["Role"] = out["Role"].map(lambda r: ROLE_DISPLAY.get(r, str(r).title()))
    return out.rename(
        columns={
            "Nominal": f"Nominal ({units})",
            "Conc": f"Back-Calculated ({units})",
            "Conc Final": f"Calculated Conc ({units})",
        }
    )


def plate_matrix(wells: pd.DataFrame, column: str) -> np.ndarray:
    out = np.full((N_ROWS, N_COLS), np.nan)
    for _, row in wells.iterrows():
        r = ROWS.index(row["Row"])
        c = int(row["Col"]) - 1
        val = row[column]
        out[r, c] = float(val) if isinstance(val, (int, float, np.floating)) else np.nan
    return out
