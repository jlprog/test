"""Interpolation strategies for the repo curve."""

from __future__ import annotations

import math
from enum import Enum


class Interpolation(Enum):
    LINEAR = "linear"           # Linear interpolation on rates
    LOG_LINEAR = "log_linear"   # Linear interpolation on log(DF) — flat-forward
    CUBIC_SPLINE = "cubic_spline"


def interpolate_rate(
    tenors: list[float],
    rates: list[float],
    tau: float,
    method: Interpolation,
) -> float:
    """
    Interpolate a repo rate at year-fraction *tau* from pillar (tenor, rate) pairs.

    Flat extrapolation is applied outside the pillar range.
    """
    if tau <= tenors[0]:
        return rates[0]
    if tau >= tenors[-1]:
        return rates[-1]

    idx = _search(tenors, tau)
    t0, t1 = tenors[idx], tenors[idx + 1]
    r0, r1 = rates[idx], rates[idx + 1]

    if method == Interpolation.LINEAR:
        return _linear(t0, r0, t1, r1, tau)

    if method == Interpolation.LOG_LINEAR:
        # Convert to discount factors, interpolate log-linearly, convert back.
        df0 = _simple_df(r0, t0)
        df1 = _simple_df(r1, t1)
        log_df = _linear(t0, math.log(df0), t1, math.log(df1), tau)
        df = math.exp(log_df)
        return _rate_from_simple_df(df, tau)

    if method == Interpolation.CUBIC_SPLINE:
        return _cubic_spline(tenors, rates, tau)

    raise ValueError(f"Unknown interpolation method: {method}")


def interpolate_log_df(
    tenors: list[float],
    log_dfs: list[float],
    tau: float,
) -> float:
    """
    Interpolate log(discount_factor) at *tau* using piecewise linear (flat-forward).

    Flat extrapolation on the short end; constant forward on the long end.
    """
    if tau <= tenors[0]:
        # Flat rate from first pillar: log_df scales linearly with tau
        return (tau / tenors[0]) * log_dfs[0]
    if tau >= tenors[-1]:
        # Constant forward rate beyond last pillar
        if len(tenors) == 1:
            return (tau / tenors[0]) * log_dfs[0]
        t0, t1 = tenors[-2], tenors[-1]
        ldf0, ldf1 = log_dfs[-2], log_dfs[-1]
        slope = (ldf1 - ldf0) / (t1 - t0)
        return ldf1 + slope * (tau - t1)

    idx = _search(tenors, tau)
    return _linear(tenors[idx], log_dfs[idx], tenors[idx + 1], log_dfs[idx + 1], tau)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear(x0: float, y0: float, x1: float, y1: float, x: float) -> float:
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _simple_df(rate: float, tau: float) -> float:
    return 1.0 / (1.0 + rate * tau)


def _rate_from_simple_df(df: float, tau: float) -> float:
    return (1.0 / df - 1.0) / tau


def _search(xs: list[float], x: float) -> int:
    """Return index i such that xs[i] <= x < xs[i+1]."""
    lo, hi = 0, len(xs) - 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _cubic_spline(xs: list[float], ys: list[float], x: float) -> float:
    """Natural cubic spline evaluated at x (pure-Python, no scipy dependency)."""
    n = len(xs) - 1
    h = [xs[i + 1] - xs[i] for i in range(n)]

    # Build tridiagonal system for second derivatives (natural spline: m[0]=m[n]=0)
    alpha = [0.0] * (n + 1)
    for i in range(1, n):
        alpha[i] = (3 / h[i] * (ys[i + 1] - ys[i]) - 3 / h[i - 1] * (ys[i] - ys[i - 1]))

    l = [1.0] + [0.0] * n
    mu = [0.0] * (n + 1)
    z = [0.0] * (n + 1)

    for i in range(1, n):
        l[i] = 2 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    l[n] = 1.0
    c = [0.0] * (n + 1)
    b = [0.0] * n
    d = [0.0] * n

    for j in range(n - 1, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (ys[j + 1] - ys[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    idx = _search(xs, x)
    dx = x - xs[idx]
    return ys[idx] + b[idx] * dx + c[idx] * dx ** 2 + d[idx] * dx ** 3
