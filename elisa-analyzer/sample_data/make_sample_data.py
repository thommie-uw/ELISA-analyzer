"""Generate synthetic ELISA plates in several instrument export formats.

Run:  python sample_data/make_sample_data.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elisa.fitting import four_pl  # noqa: E402
from elisa.plate import ROWS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

TRUE = dict(a=0.048, b=1.15, c=42.0, d=2.72)
STD_CONC = [1000.0, 500.0, 250.0, 125.0, 62.5, 31.25, 15.625]  # S1..S7, pg/mL


def build_plate(seed: int = 7):
    rng = np.random.default_rng(seed)
    od = np.full((8, 12), np.nan)
    layout = np.full((8, 12), "", dtype=object)
    truth: dict[str, float] = {}

    # standards S1..S7 duplicate in columns 1-2, blank in H1/H2
    for i, conc in enumerate(STD_CONC):
        for col in (0, 1):
            layout[i, col] = f"S{i + 1}"
            od[i, col] = four_pl(conc, **TRUE) + rng.normal(0, 0.012 + 0.008 * conc / 1000)
    for col in (0, 1):
        layout[7, col] = "B"
        od[7, col] = rng.normal(0.045, 0.004)

    # samples: duplicate pairs down each of columns 3-12
    n = 0
    for col in range(2, 12):
        for pair in range(4):
            n += 1
            name = f"Sample{n:02d}"
            conc = float(10 ** rng.uniform(np.log10(8), np.log10(1400)))
            if n == 3:
                conc = 4200.0        # above the top standard
            if n == 7:
                conc = 1.5           # below the bottom standard
            if n == 11:
                conc = 45.0          # deliberate bad-pipetting pair, see below
            truth[name] = conc
            base_od = four_pl(conc, **TRUE)
            for k in range(2):
                r = pair * 2 + k
                layout[r, col] = name
                od[r, col] = base_od + rng.normal(0, 0.012)
                if n == 11:          # one replicate short-pipetted by ~35%
                    od[r, col] = base_od * (1.35 if k == 0 else 0.72) + rng.normal(0, 0.012)
    od = np.clip(od, 0.001, 4.0)
    return od, layout, truth


def write_generic_csv(od, path):
    df = pd.DataFrame(np.round(od, 4), index=ROWS, columns=range(1, 13))
    df.to_csv(path, index_label="")


def write_softmax_txt(od, path):
    lines = [
        "##BLOCKS= 1",
        "Plate:\tPlate1\t1.3\tPlateFormat\tEndpoint\tAbsorbance\tRaw\tFALSE\t1\t\t"
        "\t1\t12\t96\t1\t8\tNone\t2\t450\t620\t1\t12\t96\t1\t8",
        "\tTemperature(¡C)\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12",
    ]
    for i, row in enumerate(ROWS):
        temp = "24.5" if i == 0 else ""
        vals = "\t".join(f"{v:.4f}" for v in od[i])
        lines.append(f"{row}\t{temp}\t{vals}")
    lines += ["", "~End", "Original Filename: demo_run.pda; Date Last Saved: 8/13/2026 9:12:41 AM"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def write_gen5_xlsx(od, path):
    meta = [
        ["Software Version", "3.11.19"],
        ["Experiment File Path:", "C:/Experiments/ELISA_demo.xpt"],
        ["Protocol File Path:", "C:/Protocols/human_IL6.prt"],
        [],
        ["Plate Number", "Plate 1"],
        ["Date", "8/13/2026"],
        ["Reader Type:", "Synergy H1"],
        ["Procedure Details"],
        ["Read", "Absorbance Endpoint", "Wavelengths: 450, 570"],
        [],
        ["450"],
    ]
    rows = [r + [""] * (14 - len(r)) for r in meta]
    rows.append([""] + list(range(1, 13)) + [""])
    for i, row in enumerate(ROWS):
        rows.append([row] + [round(float(v), 4) for v in od[i]] + [""])
    pd.DataFrame(rows).to_excel(path, header=False, index=False)


def write_long_csv(od, path):
    recs = []
    for i, row in enumerate(ROWS):
        for j in range(12):
            recs.append({"Well": f"{row}{j + 1}", "Wavelength": 450,
                         "Absorbance": round(float(od[i, j]), 4)})
    pd.DataFrame(recs).to_csv(path, index=False)


def main():
    od, layout, truth = build_plate()
    write_generic_csv(od, os.path.join(HERE, "plate_generic.csv"))
    write_softmax_txt(od, os.path.join(HERE, "plate_softmax_pro.txt"))
    write_gen5_xlsx(od, os.path.join(HERE, "plate_gen5.xlsx"))
    write_long_csv(od, os.path.join(HERE, "plate_well_list.csv"))

    pd.DataFrame(
        [{"Sample": k, "True concentration (pg/mL)": round(v, 2)} for k, v in truth.items()]
    ).to_csv(os.path.join(HERE, "_expected_concentrations.csv"), index=False)

    import json
    from elisa.layout import to_template
    conc = {f"S{i + 1}": c for i, c in enumerate(STD_CONC)}
    with open(os.path.join(HERE, "demo_layout.json"), "w", encoding="utf-8") as fh:
        fh.write(to_template(layout, conc, units="pg/mL", name="Demo IL-6 plate"))
    print("Wrote sample files to", HERE)


if __name__ == "__main__":
    main()
