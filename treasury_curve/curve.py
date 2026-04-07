"""
TreasuryCurve: public interface for a bootstrapped, monotone-convex Treasury curve.

Usage example:

    from datetime import date
    from treasury_curve.curve import TreasuryCurve
    from treasury_curve.bootstrap import ParInstrument

    anchor = date(2024, 1, 2)
    instruments = [
        ParInstrument("3M",  0.0540, date(2024, 4,  2), is_bill=True),
        ParInstrument("6M",  0.0535, date(2024, 7,  2), is_bill=True),
        ParInstrument("1Y",  0.0520, date(2025, 1,  2), is_bill=True),
        ParInstrument("2Y",  0.0480, date(2026, 1,  2)),
        ParInstrument("3Y",  0.0460, date(2027, 1,  2)),
        ParInstrument("5Y",  0.0440, date(2029, 1,  2)),
        ParInstrument("7Y",  0.0435, date(2031, 1,  2)),
        ParInstrument("10Y", 0.0430, date(2034, 1,  2)),
        ParInstrument("20Y", 0.0440, date(2044, 1,  2)),
        ParInstrument("30Y", 0.0445, date(2054, 1,  2)),
    ]
    curve = TreasuryCurve(anchor, instruments)

    print(curve.zero_rate("5Y"))
    print(curve.discount_factor("10Y"))
    print(curve.forward_rate("2Y", "3Y"))
"""

from __future__ import annotations

import math
from datetime import date
from typing import List, Sequence, Union

import numpy as np

from .bootstrap import BootstrapResult, Bootstrapper, ParInstrument
from .day_count import DayCount, generate_coupon_dates, parse_tenor, year_fraction
from .monotone_convex import BoundaryCondition, MonotoneConvex


# Type alias for tenor/date inputs
TenorOrDate = Union[str, float, date]


def _to_time(anchor: date, x: TenorOrDate) -> float:
    """Convert a tenor string, year-fraction float, or date to a year fraction."""
    if isinstance(x, date):
        return year_fraction(anchor, x, DayCount.ACT_365)
    if isinstance(x, (int, float)):
        return float(x)
    return parse_tenor(str(x))


class TreasuryCurve:
    """
    A continuously compounded Treasury zero/discount/forward curve built by:

    1. Bootstrapping par Treasury instruments (bills + coupon bonds) to
       zero rates / discount factors at each pillar.
    2. Fitting a Hagan-West monotone convex interpolant to the resulting
       forward rate curve, ensuring no negative forward rates and minimal
       oscillation between pillars.

    Parameters
    ----------
    anchor : date
        The curve reference (valuation) date.
    instruments : sequence of ParInstrument
        Par-rate instruments sorted by maturity (or left unsorted; the
        bootstrapper will sort them internally).
    boundary : BoundaryCondition
        Forward-rate boundary condition at curve endpoints.

    Attributes
    ----------
    anchor : date
    bootstrap_result : BootstrapResult
    interpolator : MonotoneConvex
    """

    def __init__(
        self,
        anchor: date,
        instruments: Sequence[ParInstrument],
        boundary: BoundaryCondition = BoundaryCondition.HAGAN_WEST,
    ) -> None:
        self.anchor = anchor
        bootstrapper = Bootstrapper(anchor, instruments)
        self.bootstrap_result: BootstrapResult = bootstrapper.run()

        br = self.bootstrap_result
        self.interpolator = MonotoneConvex(
            times=br.times,
            segment_forwards=br.segment_forwards,
            pillar_forwards=br.pillar_forwards,
            boundary=boundary,
        )

    # ------------------------------------------------------------------
    # Primary query methods
    # ------------------------------------------------------------------

    def zero_rate(
        self,
        tenor: TenorOrDate,
        compounding: str = "continuous",
    ) -> float:
        """
        Zero (spot) rate at the given tenor.

        Parameters
        ----------
        tenor : str | float | date
            e.g. "5Y", 5.0, or a date object.
        compounding : str
            "continuous"   => r such that DF = exp(-r * t)
            "semi-annual"  => y such that DF = (1 + y/2)^(-2t)
            "annual"       => y such that DF = (1 + y)^(-t)

        Returns
        -------
        float : annualised rate in decimal form.
        """
        t = _to_time(self.anchor, tenor)
        z_cont = self.interpolator.zero_rate(t)

        if compounding == "continuous":
            return z_cont

        df = math.exp(-z_cont * t)

        if compounding == "semi-annual":
            # DF = (1 + y/2)^(-2t)  =>  y = 2 * (DF^(-1/(2t)) - 1)
            return 2.0 * (df ** (-1.0 / (2.0 * t)) - 1.0)

        if compounding == "annual":
            # DF = (1 + y)^(-t)  =>  y = DF^(-1/t) - 1
            return df ** (-1.0 / t) - 1.0

        raise ValueError(f"Unknown compounding convention: '{compounding}'")

    def discount_factor(self, tenor: TenorOrDate) -> float:
        """
        Discount factor DF(t) = exp(-z(t) * t).

        Parameters
        ----------
        tenor : str | float | date
        """
        t = _to_time(self.anchor, tenor)
        return self.interpolator.discount_factor(t)

    def forward_rate(
        self,
        tenor_start: TenorOrDate,
        tenor_end: TenorOrDate,
        compounding: str = "continuous",
    ) -> float:
        """
        Continuously or semi-annually compounded forward rate for [t1, t2].

        Parameters
        ----------
        tenor_start, tenor_end : str | float | date
        compounding : str
            "continuous"  => f such that DF(t1)/DF(t2) = exp(f * (t2-t1))
            "semi-annual" => f such that DF(t1)/DF(t2) = (1 + f/2)^(2*(t2-t1))
            "simple"      => f such that DF(t1)/DF(t2) = 1 + f * (t2-t1)
        """
        t1 = _to_time(self.anchor, tenor_start)
        t2 = _to_time(self.anchor, tenor_end)

        if t2 <= t1:
            raise ValueError("tenor_end must be later than tenor_start.")

        f_cont = self.interpolator.period_forward_rate(t1, t2)

        if compounding == "continuous":
            return f_cont

        tau = t2 - t1
        df_ratio = math.exp(f_cont * tau)  # DF(t1)/DF(t2)

        if compounding == "semi-annual":
            return 2.0 * (df_ratio ** (1.0 / (2.0 * tau)) - 1.0)

        if compounding == "simple":
            return (df_ratio - 1.0) / tau

        raise ValueError(f"Unknown compounding convention: '{compounding}'")

    def instantaneous_forward(self, tenor: TenorOrDate) -> float:
        """
        Instantaneous forward rate f(t) from the monotone convex interpolant.

        Useful for visualising the shape of the forward curve.
        """
        t = _to_time(self.anchor, tenor)
        return self.interpolator.forward_rate(t)

    def par_rate(self, tenor: TenorOrDate, frequency: int = 2) -> float:
        """
        Reconstruct the par coupon rate at an arbitrary tenor by solving for
        the coupon c such that the bond prices at par:

            1 = (c/f) * sum(DF(t_i)) + (1 + c/f) * DF(t_n)
            =>  c = f * (1 - DF(t_n)) / sum(DF(t_i))

        Parameters
        ----------
        tenor : str | float | date
            Maturity of the hypothetical par bond.
        frequency : int
            Coupon payments per year (default 2 for semi-annual).

        Returns
        -------
        float : annualised par rate in decimal.
        """
        if isinstance(tenor, date):
            maturity = tenor
        else:
            t = _to_time(self.anchor, tenor)
            from datetime import timedelta
            maturity = self.anchor + timedelta(days=round(t * 365))

        coupon_dates = generate_coupon_dates(self.anchor, maturity, frequency)
        if not coupon_dates:
            raise ValueError("Could not generate coupon dates for par_rate calculation.")

        df_terminal = self.discount_factor(
            year_fraction(self.anchor, coupon_dates[-1], DayCount.ACT_365)
        )
        annuity = sum(
            self.discount_factor(year_fraction(self.anchor, cd, DayCount.ACT_365))
            for cd in coupon_dates
        )
        par = frequency * (1.0 - df_terminal) / annuity
        return par

    # ------------------------------------------------------------------
    # Introspection / display helpers
    # ------------------------------------------------------------------

    def pillar_summary(self) -> str:
        """Return a formatted table of bootstrapped pillar values."""
        br = self.bootstrap_result
        lines = [
            f"{'Tenor':>8}  {'Time':>8}  {'DF':>10}  {'Zero Rate':>12}  {'Seg Fwd':>12}",
            "-" * 58,
        ]
        N = len(br.times)
        for i in range(N):
            t = br.times[i]
            df = br.discount_factors[i]
            z = br.zero_rates[i]
            f = br.segment_forwards[i]
            # Reverse-lookup a friendly tenor label
            lines.append(
                f"{t:>8.4f}  {t:>8.4f}  {df:>10.6f}  {z*100:>11.4f}%  {f*100:>11.4f}%"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        N = len(self.bootstrap_result.times)
        return (
            f"TreasuryCurve(anchor={self.anchor}, pillars={N}, "
            f"max_tenor={self.bootstrap_result.times[-1]:.2f}Y)"
        )
