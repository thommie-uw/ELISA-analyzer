"""4-parameter and 5-parameter logistic curve fitting for ELISA standards.

Model conventions (both models share a, b, c, d):

    4PL:  y = d + (a - d) / (1 + (x/c)**b)
    5PL:  y = d + (a - d) / (1 + (x/c)**b)**g

    a : response as x -> 0        (bottom for a sandwich ELISA)
    d : response as x -> infinity (top)
    c : inflection point / EC50 (same units as x)
    b : Hill slope
    g : asymmetry factor (5PL only; g = 1 collapses to the 4PL)

Both are monotonic for b > 0, so the same parameterisation covers increasing
(sandwich) and decreasing (competitive) assays -- a and d simply swap ranks.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit

MODELS = ("4PL", "5PL")
WEIGHTS = ("None", "1/Y", "1/Y²")


# --------------------------------------------------------------------------
# model functions
# --------------------------------------------------------------------------
def _safe_ratio_pow(x: np.ndarray, c: float, b: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    c = max(float(c), 1e-12)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = np.power(np.clip(x / c, 0.0, None), b)
    return np.nan_to_num(out, nan=0.0, posinf=1e300, neginf=0.0)


def four_pl(x, a, b, c, d):
    return d + (a - d) / (1.0 + _safe_ratio_pow(x, c, b))


def five_pl(x, a, b, c, d, g):
    base = 1.0 + _safe_ratio_pow(x, c, b)
    with np.errstate(over="ignore", invalid="ignore"):
        denom = np.power(base, g)
    denom = np.where(np.isfinite(denom) & (denom > 0), denom, 1e300)
    return d + (a - d) / denom


def four_pl_inverse(y, a, b, c, d):
    """Solve the 4PL for x. Returns NaN outside the asymptotes."""
    y = np.asarray(y, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (a - d) / (y - d)
        inner = ratio - 1.0
        x = c * np.power(inner, 1.0 / b)
    x = np.where(np.isfinite(x) & (inner >= 0), x, np.nan)
    return x


def five_pl_inverse(y, a, b, c, d, g):
    """Solve the 5PL for x. Returns NaN outside the asymptotes."""
    y = np.asarray(y, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (a - d) / (y - d)
        ratio = np.where(ratio > 0, ratio, np.nan)
        inner = np.power(ratio, 1.0 / g) - 1.0
        x = c * np.power(inner, 1.0 / b)
    x = np.where(np.isfinite(x) & (inner >= 0), x, np.nan)
    return x


# --------------------------------------------------------------------------
# result container
# --------------------------------------------------------------------------
@dataclass
class FitResult:
    model: str
    params: dict[str, float]
    stderr: dict[str, float]
    r_squared: float
    adj_r_squared: float
    rmse: float
    weighting: str
    n_points: int
    x: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    y: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    converged: bool = True
    message: str = ""

    @property
    def param_order(self) -> list[str]:
        return ["a", "b", "c", "d"] + (["g"] if self.model == "5PL" else [])

    def predict(self, x) -> np.ndarray:
        p = self.params
        if self.model == "5PL":
            return five_pl(x, p["a"], p["b"], p["c"], p["d"], p["g"])
        return four_pl(x, p["a"], p["b"], p["c"], p["d"])

    def inverse(self, y) -> np.ndarray:
        p = self.params
        if self.model == "5PL":
            return five_pl_inverse(y, p["a"], p["b"], p["c"], p["d"], p["g"])
        return four_pl_inverse(y, p["a"], p["b"], p["c"], p["d"])

    def residuals(self) -> np.ndarray:
        return self.y - self.predict(self.x)

    def equation(self) -> str:
        p = self.params
        if self.model == "5PL":
            return (
                f"y = {p['d']:.4g} + ({p['a']:.4g} - {p['d']:.4g}) / "
                f"(1 + (x/{p['c']:.4g})^{p['b']:.4g})^{p['g']:.4g}"
            )
        return (
            f"y = {p['d']:.4g} + ({p['a']:.4g} - {p['d']:.4g}) / "
            f"(1 + (x/{p['c']:.4g})^{p['b']:.4g})"
        )

    def summary_rows(self) -> list[tuple[str, float]]:
        labels = {
            "a": "a  (Response at Zero)",
            "b": "b  (Hill Slope)",
            "c": "c  (Inflection / EC50)",
            "d": "d  (Response at Infinity)",
            "g": "g  (Asymmetry)",
        }
        return [(labels[k], self.params[k]) for k in self.param_order]


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------
def _sigma_for(y: np.ndarray, weighting: str) -> np.ndarray | None:
    """curve_fit uses sigma as a std-dev; weight = 1/sigma**2.

    Accepts "1/Y^2" and "1/Y²" spellings, and is case-insensitive.
    """
    key = str(weighting).strip().lower().replace("^", "").replace("²", "2")
    if key in ("", "none"):
        return None
    mag = np.abs(y)
    floor = max(np.nanmax(mag) * 1e-3, 1e-6)
    mag = np.clip(mag, floor, None)
    if key == "1/y":
        return np.sqrt(mag)
    if key == "1/y2":
        return mag
    return None


def _initial_guesses(x: np.ndarray, y: np.ndarray, model: str) -> list[list[float]]:
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    lo_n = max(1, len(ys) // 6)
    a0 = float(np.mean(ys[:lo_n]))
    d0 = float(np.mean(ys[-lo_n:]))
    mid = (a0 + d0) / 2.0
    pos = xs[xs > 0]
    if pos.size:
        idx = int(np.argmin(np.abs(ys - mid)))
        c0 = float(xs[idx]) if xs[idx] > 0 else float(np.exp(np.mean(np.log(pos))))
        c0 = max(c0, float(pos.min()) * 1e-3)
    else:
        c0 = 1.0
    guesses = []
    for b0 in (1.0, 0.7, 1.5, 2.5, 0.4):
        for cm in (1.0, 0.3, 3.0):
            base = [a0, b0, c0 * cm, d0]
            guesses.append(base + [1.0] if model == "5PL" else base)
    if model == "5PL":
        guesses.append([a0, 1.0, c0, d0, 0.5])
        guesses.append([a0, 1.0, c0, d0, 2.0])
    return guesses


def _bounds(x: np.ndarray, y: np.ndarray, model: str):
    yr = float(np.nanmax(y) - np.nanmin(y)) or 1.0
    ylo = float(np.nanmin(y)) - 3 * yr
    yhi = float(np.nanmax(y)) + 3 * yr
    pos = x[x > 0]
    xlo = float(pos.min()) * 1e-4 if pos.size else 1e-9
    xhi = float(pos.max()) * 1e4 if pos.size else 1e9
    lower = [ylo, 1e-3, xlo, ylo]
    upper = [yhi, 50.0, xhi, yhi]
    if model == "5PL":
        lower.append(1e-3)
        upper.append(50.0)
    return lower, upper


def fit_standard_curve(
    x,
    y,
    model: str = "4PL",
    weighting: str = "None",
) -> FitResult:
    """Fit standards (x = concentration, y = corrected OD) with a 4PL or 5PL."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]

    model = model.upper()
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    n_params = 5 if model == "5PL" else 4
    if x.size < n_params:
        raise ValueError(
            f"A {model} fit needs at least {n_params} standard points; got {x.size}."
        )
    if np.unique(x).size < 3:
        raise ValueError("Standards must span at least 3 distinct concentrations.")

    func = five_pl if model == "5PL" else four_pl
    sigma = _sigma_for(y, weighting)
    lower, upper = _bounds(x, y, model)

    best = None
    last_err = ""
    for p0 in _initial_guesses(x, y, model):
        p0 = [float(np.clip(v, lo, hi)) for v, lo, hi in zip(p0, lower, upper)]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = curve_fit(
                    func,
                    x,
                    y,
                    p0=p0,
                    sigma=sigma,
                    absolute_sigma=False,
                    bounds=(lower, upper),
                    maxfev=200_000,
                )
        except Exception as exc:  # noqa: BLE001 - try the next start
            last_err = str(exc)
            continue
        resid = y - func(x, *popt)
        sse = float(np.sum(resid**2))
        if not np.isfinite(sse):
            continue
        if best is None or sse < best[0]:
            best = (sse, popt, pcov)

    if best is None:
        raise RuntimeError(
            f"The {model} fit did not converge. Check the standard "
            f"concentrations and ODs. ({last_err})"
        )

    sse, popt, pcov = best
    names = ["a", "b", "c", "d"] + (["g"] if model == "5PL" else [])
    params = {k: float(v) for k, v in zip(names, popt)}
    with np.errstate(invalid="ignore"):
        errs = np.sqrt(np.diag(pcov)) if pcov is not None else np.full(len(popt), np.nan)
    stderr = {k: float(v) for k, v in zip(names, errs)}

    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    dof = x.size - len(popt)
    adj = 1.0 - (1.0 - r2) * (x.size - 1) / dof if dof > 0 and np.isfinite(r2) else np.nan

    return FitResult(
        model=model,
        params=params,
        stderr=stderr,
        r_squared=float(r2),
        adj_r_squared=float(adj),
        rmse=float(np.sqrt(sse / x.size)),
        weighting=weighting,
        n_points=int(x.size),
        x=x,
        y=y,
        converged=True,
    )
