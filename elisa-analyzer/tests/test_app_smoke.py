"""Run app.py end-to-end against a stubbed Streamlit to catch wiring errors."""
from __future__ import annotations

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import streamlit_stub as stub  # noqa: E402

stub.install()

from elisa.layout import from_template  # noqa: E402


def run_app(with_data: bool = True, model: str = "4PL"):
    stub.CALLS.clear()
    stub.session_state.clear()
    stub.BUTTON_PRESSES.clear()

    if with_data:
        with open(os.path.join(ROOT, "sample_data", "plate_softmax_pro.txt"), "rb") as fh:
            stub.session_state["file_bytes"] = fh.read()
        stub.session_state["file_name"] = "plate_softmax_pro.txt"
        with open(os.path.join(ROOT, "sample_data", "demo_layout.json"), encoding="utf-8") as fh:
            tpl = from_template(fh.read())
        stub.session_state["layout"] = tpl["layout"]
        stub.session_state["standards"] = tpl["standards"]
        stub.session_state["dilutions"] = {"Sample01": 2.0}
        stub.session_state["units"] = tpl["units"]
        stub.session_state["editor_version"] = 0
        stub.session_state["excluded_wells"] = ["C05"]
        stub.session_state["excluded_levels"] = []

    # pick the model by ordering the radio options
    original_radio = stub.radio

    def radio(label, options, index=0, **kwargs):
        opts = list(options)
        if label == "Model" and model in opts:
            return model
        return original_radio(label, options, index=index, **kwargs)

    stub.radio = radio
    try:
        runpy.run_path(os.path.join(ROOT, "app.py"), run_name="__main__")
    except stub._StopApp:
        pass
    finally:
        stub.radio = original_radio
    return list(stub.CALLS)


def names(calls):
    return [c[0] for c in calls]


def test_landing_page_without_data():
    calls = run_app(with_data=False)
    assert ("stop", (), {}) in calls, "app should stop when no file is loaded"
    assert "file_uploader" in names(calls)


def test_full_run_4pl():
    calls = run_app(with_data=True, model="4PL")
    n = names(calls)
    assert "stop" not in n, "app stopped unexpectedly"
    errors = [c for c in calls if c[0] == "error"]
    assert not errors, f"app reported errors: {errors}"
    assert n.count("pyplot") >= 4, "expected several figures to render"
    downloads = [c for c in calls if c[0] == "download_button"]
    assert len(downloads) == 4, f"expected 4 downloads, got {len(downloads)}"
    for label, filename, size in (c[1] for c in downloads):
        assert size > 0, f"empty download: {label}"
    xlsx = next(c for c in downloads if "Excel" in c[1][0])
    assert xlsx[1][2] > 20_000, "Excel report looks too small"


def test_full_run_5pl():
    calls = run_app(with_data=True, model="5PL")
    assert not [c for c in calls if c[0] == "error"]
    assert "stop" not in names(calls)


def test_every_button_path_runs():
    """Press each action button so those branches are exercised too."""
    for label in (
        "Load Demo Plate",
        "Apply Preset",
        "Fill Empty Wells",
        "Apply Serial Dilution",
    ):
        stub.CALLS.clear()
        stub.session_state.clear()
        stub.BUTTON_PRESSES.clear()
        stub.BUTTON_PRESSES.add(label)
        try:
            runpy.run_path(os.path.join(ROOT, "app.py"), run_name="__main__")
        except stub._StopApp:
            pass
        errors = [c for c in stub.CALLS if c[0] == "error"]
        assert not errors, f"pressing “{label}” raised: {errors}"
    stub.BUTTON_PRESSES.clear()


def test_layout_template_upload():
    """Uploading a JSON template must repopulate layout, standards and units."""
    class _Upload:
        def __init__(self, blob, name):
            self._blob, self.name = blob, name

        def getvalue(self):
            return self._blob

    stub.CALLS.clear()
    stub.session_state.clear()
    stub.BUTTON_PRESSES.clear()
    with open(os.path.join(ROOT, "sample_data", "plate_generic.csv"), "rb") as fh:
        plate_blob = fh.read()
    with open(os.path.join(ROOT, "sample_data", "demo_layout.json"), "rb") as fh:
        tpl_blob = fh.read()
    stub.UPLOADS.clear()
    stub.UPLOADS["Plate Reader File"] = _Upload(plate_blob, "plate_generic.csv")
    stub.UPLOADS["tpl"] = _Upload(tpl_blob, "demo_layout.json")
    try:
        runpy.run_path(os.path.join(ROOT, "app.py"), run_name="__main__")
    except stub._StopApp:
        pass
    finally:
        stub.UPLOADS.clear()
    assert not [c for c in stub.CALLS if c[0] == "error"]
    assert stub.session_state["standards"], "standards should come from the template"
    assert len(stub.session_state["standards"]) == 7
    downloads = [c for c in stub.CALLS if c[0] == "download_button"]
    assert len(downloads) == 4


def test_layout_without_standards_is_reported():
    """A plate with only samples must be refused, not silently fitted."""
    from elisa.layout import blank_layout

    stub.CALLS.clear()
    stub.session_state.clear()
    with open(os.path.join(ROOT, "sample_data", "plate_softmax_pro.txt"), "rb") as fh:
        stub.session_state["file_bytes"] = fh.read()
    stub.session_state["file_name"] = "plate_softmax_pro.txt"
    lay = blank_layout()
    lay[:, :] = "Sample01"                        # <- no S1..Sn anywhere
    stub.session_state["layout"] = lay
    stub.session_state["standards"] = {}
    stub.session_state["dilutions"] = {}
    stub.session_state["units"] = "pg/mL"
    stub.session_state["editor_version"] = 0
    stub.session_state["excluded_wells"] = []
    stub.session_state["excluded_levels"] = []
    try:
        runpy.run_path(os.path.join(ROOT, "app.py"), run_name="__main__")
    except stub._StopApp:
        pass
    warned = [c for c in stub.CALLS if c[0] in ("warning", "error")]
    assert warned, "app should warn when no standard wells are assigned"
    assert not [c for c in stub.CALLS if c[0] == "download_button"], (
        "no report should be offered without a usable curve"
    )


if __name__ == "__main__":
    for fn in (
        test_landing_page_without_data,
        test_full_run_4pl,
        test_full_run_5pl,
        test_every_button_path_runs,
        test_layout_template_upload,
        test_layout_without_standards_is_reported,
    ):
        fn()
        print(f"  ok  {fn.__name__}")
    print("app smoke tests passed")
