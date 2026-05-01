"""Repo curve: term structure of repurchase-agreement financing rates."""

from __future__ import annotations

import math
from datetime import date
from enum import Enum
from typing import Sequence

from .day_count import DayCount, year_fraction
from .interpolation import Interpolation, interpolate_log_df


class Compounding(Enum):
    SIMPLE = "simple"           # D = 1 / (1 + r * tau)  — money-market convention
    CONTINUOUS = "continuous"   # D = exp(-r * tau)


class RepoCurve:
    """
    Term structure of repurchase-agreement (repo) rates.

    The curve is defined by a set of market pillars — (maturity, rate) pairs —
    and can produce:

    * discount factors          D(0, T)
    * spot repo rates           r(0, T)
    * forward discount factors  D(T1, T2) = D(0, T2) / D(0, T1)
    * forward repo rates        r(T1, T2)
    * continuously-compounded zero rates

    Interpolation is log-linear on discount factors (flat-forward), which is
    the most common choice in practice because it guarantees positive DFs and
    piecewise-constant instantaneous forward rates (no arbitrage between pillars).

    Parameters
    ----------
    settlement_date:
        Curve anchor / value date (T = 0).
    pillars:
        Sequence of (maturity_date, annualised_rate) pairs.  Rates must be
        expressed in decimal form (e.g. 0.05 for 5 %).  At least one pillar
        is required.
    day_count:
        Day-count convention used for all year-fraction calculations.
        Default: ACT/360 (standard for USD repo).
    interpolation:
        Interpolation method applied between pillars.
        Default: LOG_LINEAR (flat-forward).
    compounding:
        Compounding convention for converting between rates and discount factors.
        Default: SIMPLE (money-market convention).
    """

    def __init__(
        self,
        settlement_date: date,
        pillars: Sequence[tuple[date, float]],
        day_count: DayCount = DayCount.ACT_360,
        interpolation: Interpolation = Interpolation.LOG_LINEAR,
        compounding: Compounding = Compounding.SIMPLE,
    ) -> None:
        if not pillars:
            raise ValueError("At least one pillar is required to build a RepoCurve.")

        self._settlement = settlement_date
        self._day_count = day_count
        self._interpolation = interpolation
        self._compounding = compounding

        sorted_pillars = sorted(pillars, key=lambda p: p[0])
        self._dates: list[date] = []
        self._rates: list[float] = []

        for mat, rate in sorted_pillars:
            if mat <= settlement_date:
                raise ValueError(
                    f"Pillar maturity {mat} must be strictly after "
                    f"settlement date {settlement_date}."
                )
            if rate < 0:
                raise ValueError(f"Repo rate must be non-negative; got {rate}.")
            self._dates.append(mat)
            self._rates.append(rate)

        # Precompute tenors (year fractions) and log(DF) at pillars.
        self._tenors: list[float] = [
            year_fraction(settlement_date, d, day_count) for d in self._dates
        ]
        self._log_dfs: list[float] = [
            math.log(self._df_from_rate(r, tau))
            for r, tau in zip(self._rates, self._tenors)
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def settlement_date(self) -> date:
        return self._settlement

    @property
    def day_count(self) -> DayCount:
        return self._day_count

    @property
    def interpolation(self) -> Interpolation:
        return self._interpolation

    @property
    def compounding(self) -> Compounding:
        return self._compounding

    @property
    def pillar_dates(self) -> list[date]:
        return list(self._dates)

    @property
    def pillar_rates(self) -> list[float]:
        return list(self._rates)

    @property
    def pillar_discount_factors(self) -> list[float]:
        return [math.exp(ldf) for ldf in self._log_dfs]

    def discount_factor(self, target_date: date) -> float:
        """
        Return D(0, T) — the discount factor from settlement to *target_date*.

        Returns 1.0 at the settlement date; raises for dates before settlement.
        """
        if target_date == self._settlement:
            return 1.0
        if target_date < self._settlement:
            raise ValueError(
                f"target_date {target_date} is before settlement_date {self._settlement}."
            )
        tau = year_fraction(self._settlement, target_date, self._day_count)
        log_df = interpolate_log_df(self._tenors, self._log_dfs, tau)
        return math.exp(log_df)

    def repo_rate(self, target_date: date) -> float:
        """Return the spot repo rate r(0, T) to *target_date*."""
        if target_date == self._settlement:
            return self._rates[0]
        df = self.discount_factor(target_date)
        tau = year_fraction(self._settlement, target_date, self._day_count)
        return self._rate_from_df(df, tau)

    def zero_rate(self, target_date: date) -> float:
        """Return the continuously-compounded zero rate to *target_date*."""
        if target_date == self._settlement:
            return self._rate_from_df(math.exp(self._log_dfs[0]), self._tenors[0])
        df = self.discount_factor(target_date)
        tau = year_fraction(self._settlement, target_date, self._day_count)
        return -math.log(df) / tau

    def forward_discount_factor(self, start_date: date, end_date: date) -> float:
        """
        Return D(T1, T2) = D(0, T2) / D(0, T1).

        Both dates must be >= settlement_date.
        """
        if end_date <= start_date:
            raise ValueError("end_date must be strictly after start_date.")
        return self.discount_factor(end_date) / self.discount_factor(start_date)

    def forward_rate(self, start_date: date, end_date: date) -> float:
        """Return the forward repo rate r(T1, T2) between *start_date* and *end_date*."""
        fwd_df = self.forward_discount_factor(start_date, end_date)
        tau = year_fraction(start_date, end_date, self._day_count)
        return self._rate_from_df(fwd_df, tau)

    def par_repo_rate(self, start_date: date, end_date: date) -> float:
        """
        Return the par (break-even) repo rate for a term repo from
        *start_date* to *end_date*, consistent with the discount curve.

        This is identical to forward_rate for single-period repos.
        """
        return self.forward_rate(start_date, end_date)

    def accrual_factor(self, start_date: date, end_date: date) -> float:
        """
        Return the accrual factor (1 + r * tau) for a simple-rate repo
        from *start_date* to *end_date*.
        """
        rate = self.forward_rate(start_date, end_date)
        tau = year_fraction(start_date, end_date, self._day_count)
        if self._compounding == Compounding.SIMPLE:
            return 1.0 + rate * tau
        return math.exp(rate * tau)

    def implied_financing_cost(
        self, clean_price: float, dirty_price: float, start_date: date, end_date: date
    ) -> float:
        """
        Implied repo rate for a cash-and-carry trade.

        Forward dirty price is computed from the spot dirty price and the
        discount factor over the holding period.

        Parameters
        ----------
        clean_price:
            Spot clean price (per 100 face).
        dirty_price:
            Spot dirty price (= clean_price + accrued).
        start_date, end_date:
            Holding period.
        """
        accrual = self.accrual_factor(start_date, end_date)
        fwd_dirty = dirty_price * accrual
        tau = year_fraction(start_date, end_date, self._day_count)
        if self._compounding == Compounding.SIMPLE:
            return (fwd_dirty / dirty_price - 1.0) / tau
        return math.log(fwd_dirty / dirty_price) / tau

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _df_from_rate(self, rate: float, tau: float) -> float:
        if self._compounding == Compounding.SIMPLE:
            return 1.0 / (1.0 + rate * tau)
        return math.exp(-rate * tau)

    def _rate_from_df(self, df: float, tau: float) -> float:
        if tau <= 0:
            raise ValueError("Year fraction must be positive.")
        if self._compounding == Compounding.SIMPLE:
            return (1.0 / df - 1.0) / tau
        return -math.log(df) / tau

    def __repr__(self) -> str:
        return (
            f"RepoCurve(settlement={self._settlement}, "
            f"pillars={len(self._dates)}, "
            f"day_count={self._day_count.value}, "
            f"interpolation={self._interpolation.value}, "
            f"compounding={self._compounding.value})"
        )
