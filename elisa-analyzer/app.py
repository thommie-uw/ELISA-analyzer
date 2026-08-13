"""ELISA Plate Analyzer - Streamlit front end.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from elisa.analysis import AnalysisOptions, analyse, display_wells, plate_matrix
from elisa.fitting import MODELS, WEIGHTS
from elisa.layout import (
    PRESETS,
    autofill_samples,
    default_standard_table,
    frame_to_layout,
    from_template,
    layout_to_frame,
    preset_layout,
    summarise,
    to_template,
    validate,
)
from elisa.parsing import read_plates
from elisa.plate import N_COLS, N_ROWS, well_id
from elisa.plotting import (
    figure_to_png,
    plate_heatmap_figure,
    residual_figure,
    standard_curve_figure,
)
from elisa.reporting import build_report

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO_PLATE = os.path.join(HERE, "sample_data", "plate_softmax_pro.txt")
DEMO_LAYOUT = os.path.join(HERE, "sample_data", "demo_layout.json")

st.set_page_config(page_title="ELISA Plate Analyzer", page_icon="🧪", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1500px;}
      div[data-testid="stMetricValue"] {font-size: 1.45rem;}
      .stDataFrame {font-size: 0.9rem;}
      .legend-chip {display:inline-block;padding:2px 10px;margin:2px 6px 2px 0;
                    border-radius:11px;font-size:0.78rem;border:1px solid #d6dbe0;}
    </style>
    """,
    unsafe_allow_html=True,
)


def rerun():
    getattr(st, "rerun", getattr(st, "experimental_rerun", lambda: None))()


def show(fig, use_container_width: bool = True):
    """Render a matplotlib figure and release it (Streamlit reruns a lot)."""
    st.pyplot(fig, use_container_width=use_container_width)
    plt.close(fig)


# --------------------------------------------------------------------------
# session state
# --------------------------------------------------------------------------
def init_state():
    ss = st.session_state
    ss.setdefault("layout", preset_layout("Standards Col 1-2, 7 Levels + Blank (Duplicate)"))
    ss.setdefault("standards", {})
    ss.setdefault("dilutions", {})
    ss.setdefault("units", "pg/mL")
    ss.setdefault("editor_version", 0)
    ss.setdefault("file_bytes", None)
    ss.setdefault("file_name", "")
    ss.setdefault("excluded_wells", [])
    ss.setdefault("excluded_levels", [])


init_state()


def bump_editor():
    st.session_state.editor_version += 1


@st.cache_data(show_spinner=False)
def cached_read(data: bytes, name: str):
    return read_plates(data, name)


def load_demo():
    with open(DEMO_PLATE, "rb") as fh:
        st.session_state.file_bytes = fh.read()
    st.session_state.file_name = os.path.basename(DEMO_PLATE)
    with open(DEMO_LAYOUT, "r", encoding="utf-8") as fh:
        tpl = from_template(fh.read())
    st.session_state.layout = tpl["layout"]
    st.session_state.standards = tpl["standards"]
    st.session_state.dilutions = tpl["dilutions"]
    st.session_state.units = tpl["units"]
    st.session_state.excluded_wells = []
    st.session_state.excluded_levels = []
    bump_editor()


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("🧪 ELISA Analyzer")
    st.caption("96-well absorbance → concentrations, with 4PL / 5PL standard curves.")

    uploaded = st.file_uploader(
        "Plate Reader File",
        type=["txt", "csv", "tsv", "xls", "xlsx", "xlsm", "dat", "prn"],
        help="Any txt / csv / Excel export containing an 8 × 12 block of absorbances, "
             "or a well-by-well list. SoftMax Pro, Gen5 and Magellan exports work as-is.",
    )
    if uploaded is not None:
        data = uploaded.getvalue()
        if data != st.session_state.file_bytes or uploaded.name != st.session_state.file_name:
            st.session_state.file_bytes = data
            st.session_state.file_name = uploaded.name

    if st.button("Load Demo Plate", use_container_width=True):
        load_demo()
        rerun()

    st.divider()
    st.subheader("Curve Fitting")
    model = st.radio("Model", MODELS, horizontal=True,
                     help="5PL adds an asymmetry term — usually a better fit when the "
                          "curve is lopsided, at the cost of one more parameter.")
    weighting = st.selectbox(
        "Weighting", WEIGHTS, index=0,
        help="1/Y or 1/Y² stop the high standards from dominating the fit, which "
             "improves accuracy at the low end.",
    )
    fit_on = st.selectbox("Fit On", ["Individual Replicates", "Replicate Means"], index=0)
    subtract_blank = st.checkbox("Subtract Mean Blank OD", value=True)

    st.divider()
    st.subheader("Acceptance Criteria")
    cv_threshold = st.number_input("Max Replicate %CV", 1.0, 100.0, 20.0, 1.0)
    rec_low, rec_high = st.slider("Standard Recovery Limits (%)", 50, 150, (80, 120), 5)

    st.divider()
    st.subheader("Run Details")
    units = st.text_input("Concentration Units", st.session_state.units)
    st.session_state.units = units
    assay_name = st.text_input("Assay / Kit Name", "")
    analyst = st.text_input("Analyst", "")

    st.divider()
    tpl_file = st.file_uploader("Load Layout Template (.json)", type=["json"], key="tpl")
    if tpl_file is not None:
        try:
            tpl = from_template(tpl_file.getvalue())
            st.session_state.layout = tpl["layout"]
            st.session_state.standards = tpl["standards"]
            st.session_state.dilutions = tpl["dilutions"]
            st.session_state.units = tpl["units"]
            bump_editor()
            st.success(f"Loaded layout “{tpl['name']}”.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read that template: {exc}")


# --------------------------------------------------------------------------
# no file yet
# --------------------------------------------------------------------------
if not st.session_state.file_bytes:
    st.title("ELISA Plate Analyzer")
    st.markdown(
        """
Upload a plate-reader export in the sidebar to get started, or press **Load Demo Plate**
to see the whole workflow with a synthetic IL-6 plate.

**What It Does**

1. Finds the 8 × 12 absorbance block in almost any txt / csv / Excel export.
2. Lets you paint the plate layout — standards, blanks, samples, QC controls.
3. Fits a **4PL** or **5PL** standard curve (optionally weighted).
4. Back-calculates every well, averages replicates, and reports %CV, recovery,
   dilution-corrected concentrations and out-of-range flags.
5. Exports a formatted multi-sheet Excel report plus the curve as a PNG.

**Layout Codes** — type these into the plate grid:

| Code | Meaning |
| --- | --- |
| `S1` … `S12` | standard level 1–12 |
| `B` | blank / zero well |
| `NSB` | non-specific binding control |
| `QC1`, `QC-High` | quality control sample |
| any other text | a sample; wells sharing the text are replicates |
| blank cell or `-` | unused well |
        """
    )
    st.stop()


# --------------------------------------------------------------------------
# parse
# --------------------------------------------------------------------------
try:
    reads = cached_read(st.session_state.file_bytes, st.session_state.file_name)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not read **{st.session_state.file_name}**: {exc}")
    st.stop()

if not reads:
    st.error(
        f"No 8 × 12 block of absorbance values was found in **{st.session_state.file_name}**. "
        "Check that the export contains the plate as a grid or as a well/value list."
    )
    st.stop()

if len(reads) > 1:
    names = [r.describe() for r in reads]
    st.caption(f"{len(reads)} plate reads were found in this file.")
    picked = st.selectbox("Plate Read", names, index=0)
    plate = reads[names.index(picked)]
else:
    plate = reads[0]

values = plate.values

st.title("ELISA Plate Analyzer")
head = st.columns([3, 1, 1, 1])
head[0].caption(f"**{st.session_state.file_name}** — {plate.name}")
finite = values[np.isfinite(values)]
head[1].metric("Wells Read", f"{finite.size} / 96")
head[2].metric("Min OD", f"{finite.min():.3f}" if finite.size else "–")
head[3].metric("Max OD", f"{finite.max():.3f}" if finite.size else "–")
if plate.overflow_wells:
    st.warning(
        "Reader reported overflow / saturation in: " + ", ".join(plate.overflow_wells)
        + ". Those wells are treated as missing."
    )

tab_layout, tab_curve, tab_results, tab_qc, tab_export = st.tabs(
    ["Plate & Layout", "Standard Curve", "Results", "QC", "Export"]
)


# --------------------------------------------------------------------------
# tab 1 - plate & layout
# --------------------------------------------------------------------------
with tab_layout:
    left, right = st.columns([1.15, 1])

    with left:
        st.subheader("Raw Absorbance")
        show(plate_heatmap_figure(values, title=f"{plate.name} — Raw OD"),
                  use_container_width=True)

    with right:
        st.subheader("Plate Layout")
        c1, c2 = st.columns([2, 1])
        preset = c1.selectbox("Preset", list(PRESETS.keys()), index=2)
        if c2.button("Apply Preset", use_container_width=True):
            st.session_state.layout = preset_layout(preset)
            bump_editor()
            rerun()

        st.markdown(
            '<span class="legend-chip" style="background:#dce9f4">S1…S12 standard</span>'
            '<span class="legend-chip" style="background:#edeff2">B blank</span>'
            '<span class="legend-chip" style="background:#fdf0dc">text = sample</span>'
            '<span class="legend-chip" style="background:#e7e2f5">QC… control</span>'
            '<span class="legend-chip" style="background:#f6e6e6">NSB</span>',
            unsafe_allow_html=True,
        )

        edited = st.data_editor(
            layout_to_frame(st.session_state.layout),
            key=f"layout_editor_{st.session_state.editor_version}",
            use_container_width=True,
            height=330,
            column_config={
                str(c): st.column_config.TextColumn(str(c), width="small")
                for c in range(1, N_COLS + 1)
            },
        )
        st.session_state.layout = frame_to_layout(edited)
        st.caption(
            "Type directly in the grid — copy/paste from a spreadsheet works too. "
            "Wells sharing the same sample text are treated as replicates."
        )

        with st.expander("Auto-Name Remaining Wells as Samples"):
            a1, a2, a3, a4 = st.columns(4)
            reps = a1.number_input("Replicates", 1, 8, 2)
            order = a2.selectbox("Fill Order", ["Down Columns", "Across Rows"], index=0)
            prefix = a3.text_input("Prefix", "Sample")
            start = a4.number_input("Start At", 1, 99, 1)
            if st.button("Fill Empty Wells"):
                st.session_state.layout = autofill_samples(
                    st.session_state.layout,
                    reps,
                    "row" if order == "Across Rows" else "column",
                    prefix,
                    int(start),
                )
                bump_editor()
                rerun()

layout = st.session_state.layout
info = summarise(layout)

with tab_layout:
    st.divider()
    s1, s2 = st.columns([1, 1.3])

    with s1:
        st.subheader(f"Standard Concentrations ({units})")
        if info.standard_levels:
            base = default_standard_table(layout)
            base["Concentration"] = [
                st.session_state.standards.get(row, base["Concentration"].iloc[i])
                for i, row in enumerate(base["Standard"])
            ]
            std_edit = st.data_editor(
                base,
                key=f"std_table_{st.session_state.editor_version}_{len(info.standard_levels)}",
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Standard": st.column_config.TextColumn(disabled=True),
                    "Concentration": st.column_config.NumberColumn(
                        f"Concentration ({units})", format="%.4g", min_value=0.0
                    ),
                },
            )
            st.session_state.standards = {
                str(r["Standard"]): float(r["Concentration"])
                for _, r in std_edit.iterrows()
                if pd.notna(r["Concentration"])
            }
            st.caption(
                "Values default to a 2-fold serial dilution from 1000 — replace them "
                "with the concentrations on your kit's standard vial."
            )
            g1, g2, g3 = st.columns([1, 1, 1])
            top = g1.number_input("Top Standard", value=1000.0, step=1.0, format="%g")
            factor = g2.number_input("Dilution Factor", value=2.0, step=0.5, format="%g")
            if g3.button("Apply Serial Dilution", use_container_width=True):
                st.session_state.standards = {
                    f"S{lvl}": top / (factor**i)
                    for i, lvl in enumerate(sorted(info.standard_levels))
                }
                bump_editor()
                rerun()
        else:
            st.info("Assign standard wells (S1, S2, …) in the grid above first.")

    with s2:
        st.subheader("Sample Dilution Factors")
        if info.sample_labels or info.control_labels:
            labels = info.sample_labels + info.control_labels
            dil = pd.DataFrame(
                {
                    "Sample": labels,
                    "Replicates": [info.replicate_counts.get(l, 0) for l in labels],
                    "Dilution": [float(st.session_state.dilutions.get(l, 1.0)) for l in labels],
                }
            )
            dil_edit = st.data_editor(
                dil,
                key=f"dil_table_{len(labels)}",
                use_container_width=True,
                hide_index=True,
                height=260,
                column_config={
                    "Sample": st.column_config.TextColumn(disabled=True),
                    "Replicates": st.column_config.NumberColumn(disabled=True, width="small"),
                    "Dilution": st.column_config.NumberColumn(
                        "Dilution Factor", min_value=0.0001, format="%g"
                    ),
                },
            )
            st.session_state.dilutions = {
                str(r["Sample"]): float(r["Dilution"])
                for _, r in dil_edit.iterrows()
                if pd.notna(r["Dilution"])
            }
            st.caption("Final concentration = mean back-calculated value × dilution factor.")
        else:
            st.info("No sample wells assigned yet.")

    st.divider()
    e1, e2 = st.columns(2)
    all_wells = [well_id(r, c) for r in range(N_ROWS) for c in range(N_COLS)]
    st.session_state.excluded_wells = e1.multiselect(
        "Exclude Individual Wells",
        all_wells,
        default=[w for w in st.session_state.excluded_wells if w in all_wells],
        help="Drop outliers, bubbles or pipetting errors from their replicate group.",
    )
    level_opts = [f"S{l}" for l in info.standard_levels]
    st.session_state.excluded_levels = e2.multiselect(
        "Exclude Standard Levels",
        level_opts,
        help="Remove an entire standard level from the curve fit.",
        default=[l for l in st.session_state.excluded_levels if l in level_opts],
    )


# --------------------------------------------------------------------------
# run the analysis
# --------------------------------------------------------------------------
problems = validate(layout, st.session_state.standards)
blocking = [p for p in problems if "will only cover" not in p]

options = AnalysisOptions(
    model=model,
    weighting=weighting,
    subtract_blank=subtract_blank,
    fit_on=fit_on,
    cv_threshold=float(cv_threshold),
    recovery_low=float(rec_low),
    recovery_high=float(rec_high),
    units=units,
    dilutions=st.session_state.dilutions,
    excluded_wells=set(st.session_state.excluded_wells),
    excluded_standard_levels={
        int(s[1:]) for s in st.session_state.excluded_levels if s[1:].isdigit()
    },
)

result = None
if blocking:
    for tab in (tab_curve, tab_results, tab_qc, tab_export):
        with tab:
            for p in blocking:
                st.warning(p)
            st.info("Finish the layout on the **Plate & layout** tab, then come back.")
else:
    try:
        result = analyse(values, layout, st.session_state.standards, options)
    except Exception as exc:  # noqa: BLE001
        for tab in (tab_curve, tab_results, tab_qc, tab_export):
            with tab:
                st.error(f"Analysis failed: {exc}")


# --------------------------------------------------------------------------
# tab 2 - standard curve
# --------------------------------------------------------------------------
if result is not None:
    fit = result.fit

    with tab_curve:
        c1, c2 = st.columns([2.1, 1])
        with c1:
            opt1, opt2 = st.columns(2)
            log_x = opt1.toggle("Log Concentration Axis", value=True)
            show_samples = opt2.toggle("Overlay Samples", value=True)
            curve_fig = standard_curve_figure(
                result, log_x=log_x, show_samples=show_samples,
                title=f"{assay_name or 'ELISA'} Standard Curve — {fit.model}",
            )
            show(curve_fig)
            with st.expander("Residuals"):
                show(residual_figure(result, log_x=log_x))

        with c2:
            m1, m2 = st.columns(2)
            m1.metric("R²", f"{fit.r_squared:.5f}")
            m2.metric("RMSE (OD)", f"{fit.rmse:.4f}")
            m3, m4 = st.columns(2)
            m3.metric(f"LLOQ ({units})", f"{result.lloq:,.4g}")
            m4.metric(f"ULOQ ({units})", f"{result.uloq:,.4g}")
            st.markdown("**Fit Parameters**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Parameter": n, "Value": v,
                         "Std. Error": fit.stderr.get(k, np.nan)}
                        for (n, v), k in zip(fit.summary_rows(), fit.param_order)
                    ]
                ),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Value": st.column_config.NumberColumn(format="%.5g"),
                    "Std. Error": st.column_config.NumberColumn(format="%.3g"),
                },
            )
            st.code(fit.equation(), language="text")

        st.divider()
        st.subheader("Standards")
        st.dataframe(
            result.standards,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Mean OD": st.column_config.NumberColumn(format="%.4f"),
                "SD OD": st.column_config.NumberColumn(format="%.4f"),
                "%CV OD": st.column_config.NumberColumn(format="%.1f"),
                "Recovery %": st.column_config.NumberColumn(format="%.1f"),
                f"Nominal ({units})": st.column_config.NumberColumn(format="%.4g"),
                f"Back-Calculated ({units})": st.column_config.NumberColumn(format="%.4g"),
            },
        )

    # ----------------------------------------------------------------------
    # tab 3 - results
    # ----------------------------------------------------------------------
    with tab_results:
        samples = result.samples
        f1, f2, f3 = st.columns([1, 1, 2])
        only_flagged = f1.toggle("Only Flagged Rows", value=False)
        hide_wells = f2.toggle("Hide Well List", value=False)
        search = f3.text_input("Filter by Sample Name", "")

        view = samples.copy()
        if only_flagged:
            view = view[(view["CV Flag"] != "OK") | (view["Range Flag"] != "In Range")]
        if search:
            view = view[view["Sample"].str.contains(search, case=False, na=False)]
        if hide_wells:
            view = view.drop(columns=["Wells"])

        def _style(df):
            def colour(row):
                out = [""] * len(row)
                if row.get("CV Flag") == "High CV":
                    out[list(df.columns).index("%CV OD")] = "background-color:#fce4e2"
                if row.get("Range Flag", "In Range") != "In Range":
                    out[list(df.columns).index("Range Flag")] = "background-color:#fff4da"
                return out

            return df.style.apply(colour, axis=1).format(
                {
                    "Mean OD": "{:.4f}", "SD OD": "{:.4f}", "%CV OD": "{:.1f}",
                    "%CV Conc": "{:.1f}",
                    f"Mean Conc ({units})": "{:,.4g}",
                    f"SD Conc ({units})": "{:,.4g}",
                    f"Final Conc ({units})": "{:,.4g}",
                    f"Conc from Mean OD ({units})": "{:,.4g}",
                    "Dilution": "{:g}",
                },
                na_rep="–",
            )

        st.dataframe(_style(view), hide_index=True, use_container_width=True, height=520)
        st.caption(
            f"{len(view)} of {len(samples)} samples shown. "
            "“Mean Conc” averages the concentrations of each replicate; "
            "“Conc from Mean OD” back-calculates once from the averaged OD."
        )

        if len(result.controls):
            st.subheader("Controls")
            st.dataframe(result.controls, hide_index=True, use_container_width=True)

        st.divider()
        st.subheader("Back-Calculated Concentrations Across the Plate")
        show(
            plate_heatmap_figure(
                plate_matrix(result.wells, "Conc"),
                title=f"Concentration per Well ({units})",
            ),
            use_container_width=True,
        )

    # ----------------------------------------------------------------------
    # tab 4 - QC
    # ----------------------------------------------------------------------
    with tab_qc:
        for w in result.warnings:
            st.warning(w)
        for p in problems:
            if p not in blocking:
                st.info(p)

        q1, q2 = st.columns([1, 1.4])
        with q1:
            st.subheader("Summary")
            st.dataframe(result.qc_summary(), hide_index=True, use_container_width=True)
        with q2:
            st.subheader("Replicate Precision")
            cvs = result.samples[["Sample", "%CV OD", "%CV Conc", "N"]].dropna(subset=["%CV OD"])
            if len(cvs):
                k1, k2, k3 = st.columns(3)
                k1.metric("Median %CV (OD)", f"{cvs['%CV OD'].median():.1f}%")
                k2.metric("Mean %CV (OD)", f"{cvs['%CV OD'].mean():.1f}%")
                k3.metric("Worst %CV (OD)", f"{cvs['%CV OD'].max():.1f}%")
                st.bar_chart(
                    cvs.set_index("Sample")["%CV OD"].sort_values(ascending=False).head(20),
                    height=240,
                )
            else:
                st.info("No replicated samples to compute a CV from.")

        st.divider()
        st.subheader("Blank-Corrected Absorbance")
        show(
            plate_heatmap_figure(
                plate_matrix(result.wells, "OD Corrected"), title="Corrected OD per Well"
            ),
            use_container_width=True,
        )

        st.subheader("Well-Level Data")
        st.dataframe(
            display_wells(result.wells, units),
            hide_index=True,
            use_container_width=True,
            height=420,
        )

    # ----------------------------------------------------------------------
    # tab 5 - export
    # ----------------------------------------------------------------------
    with tab_export:
        st.subheader("Downloads")
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        base = os.path.splitext(st.session_state.file_name)[0]

        curve_png = figure_to_png(
            standard_curve_figure(
                result, title=f"{assay_name or 'ELISA'} Standard Curve — {fit.model}"
            )
        )
        resid_png = figure_to_png(residual_figure(result))
        xlsx = build_report(
            result,
            curve_png=curve_png,
            residual_png=resid_png,
            meta={
                "source_file": st.session_state.file_name,
                "plate_name": plate.name,
                "assay_name": assay_name,
                "analyst": analyst,
            },
            layout=layout,
        )

        d1, d2, d3, d4 = st.columns(4)
        d1.download_button(
            "📊 Excel Report", xlsx, f"{base}_ELISA_{stamp}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        d2.download_button(
            "📈 Standard Curve (PNG)", curve_png, f"{base}_curve_{stamp}.png",
            "image/png", use_container_width=True,
        )
        d3.download_button(
            "📄 Results (CSV)", result.samples.to_csv(index=False).encode(),
            f"{base}_results_{stamp}.csv", "text/csv", use_container_width=True,
        )
        d4.download_button(
            "🧩 Layout Template (JSON)",
            to_template(layout, st.session_state.standards, units,
                        st.session_state.dilutions, assay_name or base),
            f"{base}_layout.json", "application/json", use_container_width=True,
        )

        st.caption(
            "The Excel report contains Summary (with the curve image and fit parameters), "
            "Sample results, Standards, Well data and colour-coded Plate maps. "
            "Save the layout template to reuse this plate map on the next run."
        )
        st.divider()
        st.subheader("Preview")
        st.image(curve_png, use_container_width=True)
