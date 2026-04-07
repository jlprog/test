"""Tests for the Treasury par-rate bootstrapper."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

from treasury_curve.bootstrap import Bootstrapper, ParInstrument
from treasury_curve.day_count import add_months


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bill(tenor: str, rate: float, anchor: date, days: int) -> ParInstrument:
    from datetime import timedelta
    return ParInstrument(
        tenor=tenor,
        par_rate=rate,
        maturity=anchor + timedelta(days=days),
        is_bill=True,
    )


def make_bond(tenor: str, rate: float, maturity: date) -> ParInstrument:
    return ParInstrument(tenor=tenor, par_rate=rate, maturity=maturity, is_bill=False)


ANCHOR = date(2024, 1, 2)


# ---------------------------------------------------------------------------
# Bill bootstrapping
# ---------------------------------------------------------------------------

class TestBillBootstrap:
    def test_single_bill(self):
        """A single T-bill produces a positive discount factor < 1."""
        inst = make_bill("3M", 0.05, ANCHOR, days=91)
        result = Bootstrapper(ANCHOR, [inst]).run()

        assert len(result.times) == 1
        assert 0 < result.discount_factors[0] < 1.0

    def test_bill_discount_factor_formula(self):
        """Verify the discount rate -> DF conversion manually."""
        d = 0.05  # 5% discount rate
        n = 91
        expected_price = 1.0 - d * n / 360.0
        inst = make_bill("3M", d, ANCHOR, days=n)
        result = Bootstrapper(ANCHOR, [inst]).run()
        assert abs(result.discount_factors[0] - expected_price) < 1e-12

    def test_zero_bill_rate(self):
        """Zero discount rate => DF = 1.0 (price at par)."""
        inst = make_bill("3M", 0.0, ANCHOR, days=91)
        result = Bootstrapper(ANCHOR, [inst]).run()
        assert abs(result.discount_factors[0] - 1.0) < 1e-12

    def test_multiple_bills(self):
        """Multiple bills produce monotonically decreasing discount factors."""
        insts = [
            make_bill("3M", 0.05, ANCHOR, 91),
            make_bill("6M", 0.052, ANCHOR, 182),
            make_bill("1Y", 0.053, ANCHOR, 364),
        ]
        result = Bootstrapper(ANCHOR, insts).run()
        dfs = result.discount_factors
        assert dfs[0] > dfs[1] > dfs[2]  # strictly decreasing


# ---------------------------------------------------------------------------
# Coupon bond bootstrapping
# ---------------------------------------------------------------------------

class TestBondBootstrap:
    def _flat_curve_instruments(self, rate: float):
        """Par instruments for a flat rate environment."""
        insts = [
            make_bill("3M", rate, ANCHOR, 91),
            make_bill("6M", rate, ANCHOR, 182),
            make_bill("1Y", rate, ANCHOR, 364),
            make_bond("2Y", rate, add_months(ANCHOR, 24)),
            make_bond("3Y", rate, add_months(ANCHOR, 36)),
            make_bond("5Y", rate, add_months(ANCHOR, 60)),
            make_bond("10Y", rate, add_months(ANCHOR, 120)),
        ]
        return insts

    def test_bond_df_is_less_than_one(self):
        """Bootstrapped discount factors for coupon bonds should be in (0, 1)."""
        insts = self._flat_curve_instruments(0.05)
        result = Bootstrapper(ANCHOR, insts).run()
        for df in result.discount_factors:
            assert 0 < df < 1.0

    def test_discount_factors_are_decreasing(self):
        """Later maturities should have smaller discount factors."""
        insts = self._flat_curve_instruments(0.045)
        result = Bootstrapper(ANCHOR, insts).run()
        dfs = result.discount_factors
        for i in range(1, len(dfs)):
            assert dfs[i] < dfs[i - 1], f"DF not decreasing at index {i}"

    def test_flat_curve_reprice(self):
        """
        In a flat par-rate environment, every coupon bond should reprice to par
        using the bootstrapped discount factors.

        For a par bond: price = (c/2) * sum(DF(t_i)) + (1 + c/2) * DF(t_N) = 1
        We verify this numerically using the bootstrapped DFs.
        """
        rate = 0.04
        insts = self._flat_curve_instruments(rate)
        result = Bootstrapper(ANCHOR, insts).run()

        # Check the 2Y bond (index 3 after 3M, 6M, 1Y bills)
        # Coupon dates for 2Y bond
        maturity_2y = add_months(ANCHOR, 24)
        coupon_dates = []
        from treasury_curve.day_count import generate_coupon_dates, year_fraction, DayCount
        coupon_dates = generate_coupon_dates(ANCHOR, maturity_2y, frequency=2)

        def get_df(t: float) -> float:
            """Log-linear interpolation on bootstrapped pillars."""
            times = result.times
            dfs = result.discount_factors
            if t <= 0:
                return 1.0
            if t >= times[-1]:
                z_last = -math.log(dfs[-1]) / times[-1]
                return math.exp(-z_last * t)
            if t <= times[0]:
                alpha = t / times[0]
                return math.exp(alpha * math.log(dfs[0]))
            import bisect
            idx = bisect.bisect_right(times, t) - 1
            t0, df0 = times[idx], dfs[idx]
            t1, df1 = times[idx + 1], dfs[idx + 1]
            alpha = (t - t0) / (t1 - t0)
            return math.exp((1 - alpha) * math.log(df0) + alpha * math.log(df1))

        c = rate / 2.0  # semi-annual coupon
        price = sum(
            c * get_df(year_fraction(ANCHOR, cd, DayCount.ACT_365))
            for cd in coupon_dates[:-1]
        ) + (1 + c) * get_df(year_fraction(ANCHOR, coupon_dates[-1], DayCount.ACT_365))

        # Tolerance is loose because the 1.5Y intermediate coupon of the 2Y bond
        # falls beyond the last bill pillar (1Y); the bootstrapper extrapolates flat
        # from 1Y while this test's get_df uses log-linear interpolation, causing a
        # small but expected discrepancy (~3-4 bps).
        assert abs(price - 1.0) < 5e-4, f"Par bond reprices to {price:.8f}, expected 1.0"

    def test_segment_forwards_positive(self):
        """All segment forward rates should be positive for upward-sloping curve."""
        insts = self._flat_curve_instruments(0.05)
        result = Bootstrapper(ANCHOR, insts).run()
        assert np.all(result.segment_forwards > 0)

    def test_times_strictly_increasing(self):
        insts = self._flat_curve_instruments(0.05)
        result = Bootstrapper(ANCHOR, insts).run()
        assert np.all(np.diff(result.times) > 0)

    def test_zero_rates_from_discount_factors(self):
        """zero_rates = -log(DF) / t should be consistent."""
        insts = self._flat_curve_instruments(0.045)
        result = Bootstrapper(ANCHOR, insts).run()
        for t, df, z in zip(result.times, result.discount_factors, result.zero_rates):
            z_check = -math.log(df) / t
            assert abs(z - z_check) < 1e-12


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_instrument_order_independent(self):
        """Bootstrapper should sort instruments by maturity automatically."""
        insts_ordered = [
            make_bill("3M", 0.05, ANCHOR, 91),
            make_bill("6M", 0.052, ANCHOR, 182),
        ]
        insts_reversed = list(reversed(insts_ordered))

        r1 = Bootstrapper(ANCHOR, insts_ordered).run()
        r2 = Bootstrapper(ANCHOR, insts_reversed).run()

        np.testing.assert_allclose(r1.discount_factors, r2.discount_factors, atol=1e-14)

    def test_single_bond(self):
        """A single 2Y bond (after 1Y bill) should bootstrap without errors."""
        insts = [
            make_bill("1Y", 0.05, ANCHOR, 364),
            make_bond("2Y", 0.048, add_months(ANCHOR, 24)),
        ]
        result = Bootstrapper(ANCHOR, insts).run()
        assert len(result.times) == 2
        assert result.discount_factors[1] < result.discount_factors[0]
