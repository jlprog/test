"""
Bootstrap par Treasury rates to zero rates and discount factors.

Handles:
  - Treasury bills (discount-basis, maturities <= 12M)
  - Treasury notes and bonds (semi-annual coupon, maturities 2Y–30Y)

All rates are in decimal (e.g. 0.045 for 4.5%).
Times are year fractions measured from the curve anchor date.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .day_count import DayCount, add_months, generate_coupon_dates, year_fraction


# ---------------------------------------------------------------------------
# Input data model
# ---------------------------------------------------------------------------

@dataclass
class ParInstrument:
    """A single par-rate instrument for curve construction."""

    tenor: str          # human-readable label, e.g. "3M", "2Y"
    par_rate: float     # annualised par/coupon rate in decimal
    maturity: date      # exact maturity date
    is_bill: bool = False   # True => discount-basis bill, False => coupon bond
    frequency: int = 2  # coupon payments per year (2 = semi-annual)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BootstrapResult:
    """Bootstrapped zero curve pillars."""

    times: np.ndarray            # year fractions from anchor, shape (N,)
    discount_factors: np.ndarray # DF at each pillar, shape (N,)
    zero_rates: np.ndarray       # continuously compounded zero rates, shape (N,)
    segment_forwards: np.ndarray # average forward rate for each segment [t_{i-1}, t_i], shape (N,)
    pillar_forwards: np.ndarray  # instantaneous forward rate estimate at each pillar, shape (N,)


# ---------------------------------------------------------------------------
# Bootstrap implementation
# ---------------------------------------------------------------------------

class Bootstrapper:
    """
    Bootstraps a Treasury par curve into a zero / discount factor curve.

    Usage
    -----
    result = Bootstrapper(anchor_date, instruments).run()
    """

    def __init__(self, anchor: date, instruments: Sequence[ParInstrument]) -> None:
        self.anchor = anchor
        # Sort by maturity to ensure sequential bootstrapping
        self.instruments = sorted(instruments, key=lambda x: x.maturity)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> BootstrapResult:
        """Execute the bootstrap and return pillar arrays."""
        times: List[float] = []
        dfs: List[float] = []

        for inst in self.instruments:
            t = year_fraction(self.anchor, inst.maturity, DayCount.ACT_365)
            if t <= 0:
                raise ValueError(f"Instrument '{inst.tenor}' has non-positive time to maturity.")

            if inst.is_bill:
                df = self._bootstrap_bill(inst)
            else:
                df = self._bootstrap_coupon_bond(inst, times, dfs)

            times.append(t)
            dfs.append(df)

        t_arr = np.array(times, dtype=float)
        df_arr = np.array(dfs, dtype=float)

        # Ensure monotonically increasing times
        if np.any(np.diff(t_arr) <= 0):
            raise ValueError("Instrument maturities must be strictly increasing.")

        zero_rates = -np.log(df_arr) / t_arr

        # Prepend t=0 with DF=1 for segment forward calculation
        t_full = np.concatenate([[0.0], t_arr])
        log_df_full = np.concatenate([[0.0], np.log(df_arr)])

        # Segment forward rate F_i for interval [t_{i-1}, t_i]:
        # F_i = (z_i * t_i - z_{i-1} * t_{i-1}) / (t_i - t_{i-1})
        # equivalently: -(log_df_i - log_df_{i-1}) / (t_i - t_{i-1})
        seg_fwd = -(np.diff(log_df_full)) / np.diff(t_full)

        # Instantaneous forward estimates at pillars using Hagan-West formula
        pillar_fwd = self._estimate_pillar_forwards(t_full, seg_fwd)

        return BootstrapResult(
            times=t_arr,
            discount_factors=df_arr,
            zero_rates=zero_rates,
            segment_forwards=seg_fwd,
            pillar_forwards=pillar_fwd,
        )

    # ------------------------------------------------------------------
    # Private: bill bootstrap
    # ------------------------------------------------------------------

    def _bootstrap_bill(self, inst: ParInstrument) -> float:
        """
        Convert a Treasury bill discount rate to a discount factor.

        T-bills are quoted on an ACT/360 bank-discount basis:
            Price = 1 - d * (n / 360)
        where d is the discount rate and n is days to maturity.

        The discount factor equals the price (per unit of face value).
        """
        n = (inst.maturity - self.anchor).days
        price = 1.0 - inst.par_rate * n / 360.0
        if price <= 0:
            raise ValueError(
                f"Bill '{inst.tenor}' discount rate implies non-positive price."
            )
        return price

    # ------------------------------------------------------------------
    # Private: coupon bond bootstrap
    # ------------------------------------------------------------------

    def _bootstrap_coupon_bond(
        self,
        inst: ParInstrument,
        times: List[float],
        dfs: List[float],
    ) -> float:
        """
        Bootstrap the discount factor at the bond's maturity.

        For a par bond with coupon rate c (semi-annual), the price equals par:
            1 = (c/f) * sum(DF(t_i), i=1..n-1) + (1 + c/f) * DF(t_n)

        Solving for DF(t_n):
            DF(t_n) = (1 - (c/f) * sum(DF(t_i), i=1..n-1)) / (1 + c/f)

        Intermediate discount factors are obtained by log-linear interpolation
        (equivalent to linear interpolation of zero rates).
        """
        coupon_dates = generate_coupon_dates(self.anchor, inst.maturity, inst.frequency)
        if not coupon_dates:
            raise ValueError(f"No coupon dates generated for '{inst.tenor}'.")

        c_per_period = inst.par_rate / inst.frequency
        pv_intermediate = 0.0

        # Sum PV of all cashflows except the last (final principal + coupon)
        for cd in coupon_dates[:-1]:
            t = year_fraction(self.anchor, cd, DayCount.ACT_365)
            df = self._interpolate_df(t, times, dfs)
            pv_intermediate += c_per_period * df

        # Solve algebraically for the terminal discount factor
        df_terminal = (1.0 - pv_intermediate) / (1.0 + c_per_period)

        if df_terminal <= 0:
            raise ValueError(
                f"Bootstrap produced non-positive discount factor for '{inst.tenor}'. "
                "Check that par rates are reasonable and instruments are in maturity order."
            )

        return df_terminal

    # ------------------------------------------------------------------
    # Private: interpolation helpers
    # ------------------------------------------------------------------

    def _interpolate_df(
        self,
        t: float,
        times: List[float],
        dfs: List[float],
    ) -> float:
        """
        Log-linear interpolation of the discount factor at time t.

        Uses the bootstrapped pillars built so far. Extrapolates flat
        beyond the last pillar (should not occur in normal usage).
        """
        if not times:
            # No pillars yet; use flat extrapolation at rate 0 => DF = 1
            return 1.0

        t_arr = np.array(times)
        df_arr = np.array(dfs)

        if t <= 0:
            return 1.0

        if t >= t_arr[-1]:
            # Flat extrapolation of zero rate beyond last pillar
            z_last = -math.log(df_arr[-1]) / t_arr[-1]
            return math.exp(-z_last * t)

        if t <= t_arr[0]:
            # Interpolate between t=0 (DF=1) and first pillar
            t0, df0 = 0.0, 1.0
            t1, df1 = t_arr[0], df_arr[0]
        else:
            idx = np.searchsorted(t_arr, t, side="right") - 1
            t0, df0 = t_arr[idx], df_arr[idx]
            t1, df1 = t_arr[idx + 1], df_arr[idx + 1]

        # Log-linear: log(DF) is linear in t
        alpha = (t - t0) / (t1 - t0)
        log_df = (1 - alpha) * math.log(df0) + alpha * math.log(df1)
        return math.exp(log_df)

    # ------------------------------------------------------------------
    # Private: Hagan-West instantaneous forward estimates at pillars
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_pillar_forwards(
        t_full: np.ndarray,   # [t_0=0, t_1, ..., t_N]
        seg_fwd: np.ndarray,  # F_1, ..., F_N  (length N)
    ) -> np.ndarray:
        """
        Estimate instantaneous forward rates at pillar dates t_1 .. t_N
        using the Hagan-West weighted-average formula.

        Returns array of length N (one value per pillar).
        """
        N = len(seg_fwd)
        g = np.empty(N + 1)  # g[0] .. g[N]  (at t_0 through t_N)

        if N == 1:
            # Only one segment: flat forward
            g[0] = seg_fwd[0]
            g[1] = seg_fwd[0]
            return g[1:]  # return g at pillars t_1..t_N

        # Interior pillars: weighted average of adjacent segment forwards
        for i in range(1, N):
            h_prev = t_full[i] - t_full[i - 1]   # width of segment i
            h_next = t_full[i + 1] - t_full[i]    # width of segment i+1
            g[i] = (seg_fwd[i - 1] * h_next + seg_fwd[i] * h_prev) / (h_prev + h_next)

        # Boundary conditions (Hagan-West default)
        g[0] = seg_fwd[0] - (g[1] - seg_fwd[0]) / 2.0
        g[N] = seg_fwd[N - 1] - (g[N - 1] - seg_fwd[N - 1]) / 2.0

        # g[0] is at t=0; return only g[1..N] which are at the N pillars
        return g[1:]
