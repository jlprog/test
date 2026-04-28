"""
sofr_curve.py - SOFR OIS Discount Curve Construction

Bootstraps a SOFR discount curve from market OIS swap rates and deposits.
Instruments:
  - O/N deposit  (act/360)
  - Short OIS    1W, 2W, 1M, 2M, 3M, 6M, 9M, 12M  (act/365 fixed, single period)
  - Long OIS     2Y, 3Y, 5Y, 7Y, 10Y, 15Y, 20Y, 30Y (act/365 fixed, annual payments)
Interpolation: log-linear on discount factors (piecewise-flat instantaneous forward).
"""

import calendar
import math
from datetime import date, timedelta

import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def add_tenor(base: date, tenor: str) -> date:
    """Add a tenor string (e.g. '3M', '2Y', '1W', '10D') to a date."""
    n, unit = int(tenor[:-1]), tenor[-1].upper()
    if unit == 'D':
        return base + timedelta(days=n)
    if unit == 'W':
        return base + timedelta(weeks=n)
    if unit == 'M':
        month = base.month + n
        year  = base.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        day   = min(base.day, calendar.monthrange(year, month)[1])
        return date(year, month, day)
    if unit == 'Y':
        try:
            return date(base.year + n, base.month, base.day)
        except ValueError:
            return date(base.year + n, base.month,
                        calendar.monthrange(base.year + n, base.month)[1])
    raise ValueError(f"Unknown tenor unit: {unit!r}")


def dcf_act360(d1: date, d2: date) -> float:
    return (d2 - d1).days / 360.0


def dcf_act365(d1: date, d2: date) -> float:
    return (d2 - d1).days / 365.0


def years(val: date, d: date) -> float:
    """Act/365.25 year fraction from valuation date."""
    return (d - val).days / 365.25


# ---------------------------------------------------------------------------
# SOFR curve
# ---------------------------------------------------------------------------

class SOFRCurve:
    """
    SOFR OIS discount curve built by sequential bootstrapping.

    Parameters
    ----------
    valuation_date : date
    instruments    : list of dicts with keys
        tenor  – e.g. '1D', '3M', '5Y'
        rate   – quoted rate as a decimal (e.g. 0.043)
        type   – 'deposit' | 'ois'   (default 'ois')
    """

    def __init__(self, valuation_date: date, instruments: list[dict]):
        self.valuation_date = valuation_date
        # Pillar arrays include t=0, df=1
        self._times: list[float] = [0.0]
        self._log_dfs: list[float] = [0.0]
        self._bootstrap(instruments)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def discount_factor(self, d: date) -> float:
        """Log-linearly interpolated discount factor P(0, t)."""
        t = years(self.valuation_date, d)
        if t <= 0.0:
            return 1.0
        return math.exp(self._interp_log_df(t))

    def zero_rate(self, d: date) -> float:
        """Continuously compounded zero rate (act/365.25)."""
        t = years(self.valuation_date, d)
        if t <= 0.0:
            return 0.0
        return -self._interp_log_df(t) / t

    def forward_rate(self, d1: date, d2: date) -> float:
        """Simply-compounded forward rate between two dates."""
        df1 = self.discount_factor(d1)
        df2 = self.discount_factor(d2)
        dcf = dcf_act360(d1, d2)
        if dcf <= 0 or df2 <= 0:
            return 0.0
        return (df1 / df2 - 1.0) / dcf

    def par_swap_rate(self, maturity: date) -> float:
        """OIS par swap rate for given maturity."""
        annuity = self._fixed_annuity(self.valuation_date, maturity)
        if annuity == 0.0:
            return 0.0
        return (1.0 - self.discount_factor(maturity)) / annuity

    # ------------------------------------------------------------------
    # Private bootstrap
    # ------------------------------------------------------------------

    def _interp_log_df(self, t: float) -> float:
        """Log-linear (flat-forward) interpolation / extrapolation."""
        ts = self._times
        lds = self._log_dfs
        if t >= ts[-1]:
            # Flat instantaneous forward beyond last pillar
            if len(ts) < 2:
                return 0.0
            slope = (lds[-1] - lds[-2]) / (ts[-1] - ts[-2])
            return lds[-1] + slope * (t - ts[-1])
        return float(np.interp(t, ts, lds))

    def _df(self, d: date) -> float:
        """Internal discount factor used during bootstrapping."""
        return self.discount_factor(d)

    def _fixed_annuity(self, start: date, end: date) -> float:
        """PV01 of annual fixed leg (act/365)."""
        annuity = 0.0
        cur = start
        while True:
            nxt = add_tenor(cur, '1Y')
            pay = end if nxt >= end else nxt
            annuity += dcf_act365(cur, pay) * self._df(pay)
            cur = nxt
            if nxt >= end:
                break
        return annuity

    def _ois_npv(self, maturity: date, fixed_rate: float, df_T: float) -> float:
        """
        OIS swap NPV with trial df_T at maturity.
        floating PV = 1 - df(T)
        fixed    PV = fixed_rate * annuity
        """
        # Temporarily register the trial pillar
        t_T = years(self.valuation_date, maturity)
        self._times.append(t_T)
        self._log_dfs.append(math.log(df_T))

        try:
            annuity = self._fixed_annuity(self.valuation_date, maturity)
            npv = fixed_rate * annuity - (1.0 - df_T)
        finally:
            self._times.pop()
            self._log_dfs.pop()

        return npv

    def _bootstrap(self, instruments: list[dict]):
        val = self.valuation_date

        for inst in instruments:
            tenor     = inst['tenor']
            rate      = inst['rate']
            inst_type = inst.get('type', 'ois')
            maturity  = add_tenor(val, tenor)
            t         = years(val, maturity)

            if inst_type == 'deposit':
                # Simple money-market: df = 1 / (1 + r * dcf_act360)
                df = 1.0 / (1.0 + rate * dcf_act360(val, maturity))

            else:  # OIS swap
                if t <= 1.0 + 2 / 365.0:
                    # Single-period: df = 1 / (1 + r * dcf_act365)
                    df = 1.0 / (1.0 + rate * dcf_act365(val, maturity))
                else:
                    # Multi-period: numerical root-find
                    lo = max(1e-6, math.exp(-rate * t * 3))
                    hi = min(1.0 - 1e-9, math.exp(-rate * t * 0.1))
                    try:
                        df = brentq(
                            lambda x: self._ois_npv(maturity, rate, x),
                            lo, hi, xtol=1e-12, maxiter=200,
                        )
                    except ValueError:
                        df = math.exp(-rate * t)   # fallback

            self._times.append(t)
            self._log_dfs.append(math.log(df) if df > 0 else -rate * t)
