# ELISA Plate Analyzer

A Streamlit app that turns raw 96-well absorbance exports into concentrations:
4PL/5PL standard curve, back-calculated sample values, replicate %CV, standard
recovery, dilution handling, out-of-range flags, and a formatted Excel report.

---

## Quick Start

Requires **Python 3.10–3.14** (`python3 --version` to check).

**macOS / Linux**

```bash
cd elisa-analyzer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

**Windows (PowerShell)**

```powershell
cd elisa-analyzer
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>. Click **Load Demo Plate** in the sidebar
to walk through the whole workflow on a synthetic IL-6 plate before using your own data.

`Ctrl+C` in the terminal stops the server. Next time, reactivate the environment
(`source .venv/bin/activate`) and `streamlit run app.py` — no need to reinstall.

### Deploying to Streamlit Community Cloud

To share it with colleagues instead of running it locally:

1. Push this folder to a GitHub repo (`requirements.txt` is already set up).
2. Sign in at <https://share.streamlit.io> with GitHub and authorise it.
3. **Create app → Deploy a public app from GitHub**, pick the repo/branch, set the
   main file to `app.py`, and deploy.

Note that free apps are public — anyone with the link can open them and upload
plates. The free tier allows **one private app at a time** (App settings →
Sharing → "Only specific people can view this app", then invite by email). If
your plate data is confidential, running locally is the safer option.

---

## Workflow

1. **Upload** your plate reader file in the sidebar.
2. **Plate & Layout tab** — pick a preset, then edit the 8 × 12 grid to match your
   plate. Enter the standard concentrations and any sample dilution factors.
3. **Standard Curve tab** — choose 4PL or 5PL in the sidebar and inspect the fit,
   R², parameters, residuals and standard recovery.
4. **Results tab** — per-sample concentrations with %CV and range flags.
5. **QC tab** — blank statistics, precision summary, well-level data.
6. **Export tab** — download the Excel report, curve PNG, results CSV and the
   layout template for reuse.

---

## Input Formats

The parser flattens any file into a grid and then hunts for an 8 × 12 block of
numbers, so it is not tied to one instrument. Verified against:

| Format | Notes |
| --- | --- |
| Plain CSV / TSV / TXT | with or without `A`–`H` and `1`–`12` headers |
| Molecular Devices **SoftMax Pro** | tab-delimited `Plate:` / `~End` blocks, temperature column, multiple wavelengths |
| BioTek **Gen5** | Excel or text with metadata rows above the plate |
| Tecan **Magellan** | same shape as Gen5 |
| Well/value lists | a `Well` column (`A1`, `A01`, …) plus a value column |
| Excel workbooks | every sheet is scanned; multi-plate files are supported |

If a file contains several plates or wavelengths, a selector appears above the
tabs. Reader sentinels such as `OVRFLW` and `***` are treated as missing wells
and reported as saturation warnings. European decimal commas are handled.

**If your export isn't detected**, the fastest fix is to paste the 8 × 12 block
into a fresh CSV. That said, send me the file and the parser can be tuned to it —
that's what the flexible scanner is for.

---

## Plate Layout Codes

Type these directly into the grid editor (copy/paste from a spreadsheet works):

| Code | Meaning |
| --- | --- |
| `S1` … `S12` | Standard level 1–12 (`Std 3` and `s3` also work) |
| `B`, `BLK`, `BLANK` | Blank / zero well — used for the OD correction |
| `NSB` | Non-specific binding control (reported, never subtracted) |
| `QC1`, `QC-High`, `Ctrl2` | Quality control sample, summarised separately |
| any other text | A sample. Wells sharing the same text are replicates |
| empty cell, `-`, `x` | Unused well |

Save the layout as a JSON template from the Export tab and reload it next run.

---

## The Calculations

**Blank correction.** Every well has the mean of the `B` wells subtracted before
anything else. Turn it off in the sidebar if your kit's protocol says not to.

**Curve models.** Both are fitted by non-linear least squares (`scipy.curve_fit`)
from multiple starting points, keeping the lowest residual sum of squares:

```
4PL:  y = d + (a − d) / (1 + (x/c)^b)
5PL:  y = d + (a − d) / (1 + (x/c)^b)^g
```

- `a` — response as x → 0 (bottom)
- `d` — response as x → ∞ (top)
- `c` — inflection point / EC50
- `b` — Hill slope
- `g` — asymmetry (5PL only; `g = 1` reduces the 5PL to the 4PL)

The parameterisation is monotonic for `b > 0`, so it fits sandwich assays
(increasing) and competitive assays (decreasing) without any extra switch.

**Weighting.** `None`, `1/Y` or `1/Y²`. Absorbance error is usually proportional
to signal, so unweighted fits let the top standards dominate and lose accuracy at
the bottom of the curve. `1/Y²` is the usual choice when the low end matters.

**Back-calculation** uses the analytic inverse, so it is exact rather than
iterative:

```
4PL:  x = c · ((a − d)/(y − d) − 1)^(1/b)
5PL:  x = c · (((a − d)/(y − d))^(1/g) − 1)^(1/b)
```

An OD outside the fitted asymptotes has no real solution and is returned as
"off curve" rather than extrapolated.

**Sample values.** Each replicate is back-calculated separately and then averaged
(`Mean Conc`), which is what the replicate %CV is computed from. The alternative
convention — average the ODs first, then back-calculate once — is reported
alongside as `Conc from Mean OD`. `Final Conc = Mean Conc × Dilution Factor`.

**Dilution.** The factor is applied per well, so the *Calculated Concentration*
heat map on the Results tab and the matching Plate Maps grid in the Excel report
show the same dilution-corrected numbers as the results table — no mental
arithmetic between views. Standards and blanks are always read neat. The map
colours on a log scale when the plate spans more than two orders of magnitude,
which it usually does once dilutions are in play.

**Precision.** %CV = SD / mean × 100 with the sample standard deviation
(`ddof = 1`), reported on both OD and concentration. Groups above the threshold
are flagged.

**Accuracy.** Each standard level's mean OD is put back through the curve;
recovery % = back-calculated / nominal × 100. Levels outside the acceptance
window (default 80–120%) are flagged — this is the single best indicator that a
curve is trustworthy.

**LLOQ / ULOQ** are the lowest non-zero and highest standard concentrations
actually used in the fit. Samples below, above, or entirely off the curve are
flagged rather than silently reported.

**Excluding data.** Individual wells (bubbles, pipetting misses) and whole
standard levels can be dropped at the bottom of the Plate & Layout tab; the curve
refits immediately and excluded points show as red crosses on the plot.

---

## Excel Report

| Sheet | Contents |
| --- | --- |
| Summary | Run metadata, QC table, fit parameters ± std. errors, equation, curve and residual images |
| Sample Results | Per-sample ODs, concentrations, %CV, dilution, flags |
| Controls | QC samples, when present |
| Standards | Nominal vs back-calculated, recovery %, per-level %CV |
| Well Data | All 96 wells: role, raw OD, corrected OD, back-calculated value |
| Plate Maps | Layout, raw OD, corrected OD and concentration grids, colour-coded by role |

---

## Project Layout

```
elisa-analyzer/
├── app.py                  Streamlit UI
├── elisa/
│   ├── plate.py            96-well constants and well-ID helpers
│   ├── parsing.py          instrument export readers
│   ├── layout.py           layout codes, presets, JSON templates
│   ├── fitting.py          4PL / 5PL models, fitting, analytic inverses
│   ├── analysis.py         blank correction, CVs, recovery, flags
│   ├── plotting.py         standard curve, residuals, plate heat maps
│   └── reporting.py        Excel report builder
├── sample_data/            synthetic plates in four export formats
└── tests/                  unit tests + a stubbed-Streamlit smoke test
```

All the science lives in `elisa/` with no Streamlit imports, so it can be reused
in a notebook or a batch script:

```python
from elisa import read_plates, analyse, AnalysisOptions
from elisa.layout import from_template

plate = read_plates(open("run1.txt", "rb").read(), "run1.txt")[0]
tpl = from_template(open("my_layout.json").read())
res = analyse(plate.values, tpl["layout"], tpl["standards"],
              AnalysisOptions(model="5PL", weighting="1/Y²", units="pg/mL"))
print(res.samples)
```

---

## Tests

```bash
python tests/test_pipeline.py     # parsing, fitting, layout, analysis  (24 tests)
python tests/test_app_smoke.py    # runs app.py against a stubbed Streamlit
```

`pytest tests/` works too. The fitting tests check that known 4PL/5PL parameters
are recovered from simulated data and that the analytic inverse round-trips
exactly; the analysis tests check blank subtraction, %CV arithmetic, dilution
handling, exclusions and range flags against hand calculations.

Regenerate the synthetic plates with `python sample_data/make_sample_data.py`.
