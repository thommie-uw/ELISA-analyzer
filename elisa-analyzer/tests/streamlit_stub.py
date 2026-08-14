"""A minimal fake `streamlit` module so app.py can be exercised without a browser.

Widgets return their default value, containers are no-op context managers, and
every call is recorded so the harness can assert the app actually rendered
something. This is a smoke-test aid, not a Streamlit reimplementation.
"""
from __future__ import annotations

import sys
import types
from contextlib import contextmanager

CALLS: list[tuple[str, tuple, dict]] = []
UPLOADS: dict[str, object] = {}
BUTTON_PRESSES: set[str] = set()


class _SessionState(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as exc:
            raise AttributeError(k) from exc

    def __setattr__(self, k, v):
        self[k] = v


session_state = _SessionState()


class _Ctx:
    """Behaves as a container, a context manager and a delegating proxy."""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        return globals()[name] if name in globals() else _record(name)


def _record(name):
    def fn(*args, **kwargs):
        CALLS.append((name, args, kwargs))
        return None

    return fn


def _passthrough(name, value=None):
    def fn(*args, **kwargs):
        CALLS.append((name, args, kwargs))
        return value

    return fn


# ---- layout containers -----------------------------------------------------
def columns(spec, **kwargs):
    CALLS.append(("columns", (spec,), kwargs))
    n = spec if isinstance(spec, int) else len(spec)
    return [_Ctx() for _ in range(n)]


def tabs(labels, **kwargs):
    CALLS.append(("tabs", (labels,), kwargs))
    return [_Ctx() for _ in labels]


def expander(label, **kwargs):
    CALLS.append(("expander", (label,), kwargs))
    return _Ctx()


def container(**kwargs):
    return _Ctx()


@contextmanager
def spinner(*a, **k):
    yield


sidebar = _Ctx()


# ---- widgets ---------------------------------------------------------------
def file_uploader(label, **kwargs):
    CALLS.append(("file_uploader", (label,), kwargs))
    return UPLOADS.get(kwargs.get("key") or label)


def button(label, **kwargs):
    CALLS.append(("button", (label,), kwargs))
    return label in BUTTON_PRESSES


def radio(label, options, index=0, **kwargs):
    CALLS.append(("radio", (label, options), kwargs))
    return list(options)[index or 0]


def selectbox(label, options, index=0, **kwargs):
    CALLS.append(("selectbox", (label, options), kwargs))
    opts = list(options)
    return opts[index or 0] if opts else None


def multiselect(label, options, default=None, **kwargs):
    CALLS.append(("multiselect", (label, options), kwargs))
    return list(default or [])


def checkbox(label, value=False, **kwargs):
    CALLS.append(("checkbox", (label,), kwargs))
    return value


def toggle(label, value=False, **kwargs):
    CALLS.append(("toggle", (label,), kwargs))
    return value


def text_input(label, value="", **kwargs):
    CALLS.append(("text_input", (label,), kwargs))
    return value


def number_input(label, *args, **kwargs):
    CALLS.append(("number_input", (label,) + args, kwargs))
    if "value" in kwargs:
        return kwargs["value"]
    return args[2] if len(args) >= 3 else (args[0] if args else 0)


def slider(label, min_value=None, max_value=None, value=None, step=None, **kwargs):
    CALLS.append(("slider", (label,), kwargs))
    return value


def data_editor(data, **kwargs):
    CALLS.append(("data_editor", (), kwargs))
    return data


def download_button(label, data, file_name=None, mime=None, **kwargs):
    CALLS.append(("download_button", (label, file_name, len(data) if data is not None else 0), kwargs))
    return False


# ---- output ----------------------------------------------------------------
for _name in (
    "title", "header", "subheader", "caption", "markdown", "write", "text",
    "code", "info", "success", "warning", "error", "divider", "pyplot",
    "dataframe", "table", "metric", "image", "bar_chart", "line_chart",
    "json", "set_page_config", "rerun", "experimental_rerun", "toast",
    "progress", "empty", "altair_chart", "plotly_chart",
):
    globals()[_name] = _record(_name)


class _StopApp(Exception):
    pass


def stop():
    CALLS.append(("stop", (), {}))
    raise _StopApp()


# ---- cache -----------------------------------------------------------------
def cache_data(func=None, **kwargs):
    def wrap(f):
        return f

    return wrap(func) if callable(func) else wrap


cache_resource = cache_data


# ---- column_config ---------------------------------------------------------
class _Col:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs


column_config = types.SimpleNamespace(
    TextColumn=_Col, NumberColumn=_Col, CheckboxColumn=_Col,
    SelectboxColumn=_Col, Column=_Col, ListColumn=_Col,
)


def install():
    """Register this module as `streamlit` in sys.modules."""
    mod = sys.modules[__name__]
    sys.modules["streamlit"] = mod
    return mod
