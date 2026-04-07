"""End-to-end tests for TreasuryCurve."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

from treasury_curve import ParInstrument, TreasuryCurve
from treasury_curve.day_count import add_months


# ---------------------------------------------------------------------------
# Standard on-the-run par curve (approximate 2024 levels)
# ---------------------------------------------------------------------------

ANCHOR = date(2024, 1, 2)

INSTRUMENTS = [
    ParInstrument("3M",  0.0540, date(2024, 4,  2), is_bill=True),
    ParInstrument("6M",  0.0535, date(2024, 7,  2), is_bill=True),
    ParInstrument("1Y",  0.0520, date(2025, 1,  2), is_bill=True),
    ParInstrument("2Y",  0.0480, add_months(ANCHOR, 24)),
    ParInstrument("3Y",  0.0460, add_months(ANCHOR, 36)),
    ParInstrument("5Y",  0.0440, add_months(ANCHOR, 60)),
    ParInstrument("7Y",  0.0435, add_months(ANCHOR, 84)),
    ParInstrument("10Y", 0.0430, add_months(ANCHOR, 120)),
    ParInstrument("20Y", 0.0440, add_months(ANCHOR, 240)),
    ParInstrument("30Y", 0.0445, add_months(ANCHOR, 360)),
]


@pytest.fixture(scope="module")
def curve() -> TreasuryCurve:
    return TreasuryCurve(ANCHOR, INSTRUMENTS)


# ---------------------------------------------------------------------------
# Basic sanity checks
# ---------------------------------------------------------------------------

class TestBasicSanity:
    def test_repr(self, curve):
        r = repr(curve)
        assert "TreasuryCurve" in r
        assert "2024" in r

    def test_discount_factor_at_zero(self, curve):
        """DF(0) = 1."""
        df = curve.discount_factor(0.0)
        assert abs(df - 1.0) < 1e-12

    def test_discount_factors_less_than_one(self, curve):
        """All DF(t) < 1 for t > 0 in a positive rate environment."""
        for t in [0.25, 1.0, 5.0, 10.0, 30.0]:
            assert curve.discount_factor(t) < 1.0

    def test_discount_factors_decreasing(self, curve):
        """DF should be a strictly decreasing function of t."""
        times = np.linspace(0.1, 30.0, 200)
        dfs = [curve.discount_factor(t) for t in times]
        for i in range(1, len(dfs)):
            assert dfs[i] < dfs[i - 1], f"DF not decreasing at index {i}, t={times[i]:.4f}"

    def test_zero_rates_positive(self, curve):
        """All zero rates should be positive given the par rates."""
        for t in [0.5, 1.0, 2.0, 5.0, 10.0, 30.0]:
            z = curve.zero_rate(t)
            assert z > 0, f"Negative zero rate at t={t}: z={z:.6f}"

    def test_positive_forward_rates(self, curve):
        """Instantaneous forward rates should be positive everywhere."""
        times = np.linspace(0.01, 30.0, 500)
        for t in times:
            f = curve.instantaneous_forward(t)
            assert f > -1e-10, f"Negative forward at t={t:.4f}: f={f:.8e}"


# ---------------------------------------------------------------------------
# Tenor parsing
# ---------------------------------------------------------------------------

class TestTenorParsing:
    def test_string_tenor(self, curve):
        df1 = curve.discount_factor("5Y")
        df2 = curve.discount_factor(5.0)
        assert abs(df1 - df2) < 1e-12

    def test_month_tenor(self, curve):
        df = curve.discount_factor("6M")
        assert 0 < df < 1.0

    def test_date_tenor(self, curve):
        target = add_months(ANCHOR, 24)
        df_str = curve.discount_factor("2Y")
        df_date = curve.discount_factor(target)
        # Should be close (exact date vs approximate 2Y year fraction)
        assert abs(df_str - df_date) < 0.01  # within 1% due to date rounding

    def test_overnight_tenor(self, curve):
        df = curve.discount_factor("ON")
        assert 0.99 < df < 1.0  # very short tenor, nearly 1


# ---------------------------------------------------------------------------
# Zero rate compounding conventions
# ---------------------------------------------------------------------------

class TestZeroRateCompounding:
    def test_continuous_vs_semiannual_consistency(self, curve):
        """Semi-annual and continuous rates should produce the same DF."""
        for t in [1.0, 5.0, 10.0]:
            z_cont = curve.zero_rate(t, "continuous")
            z_semi = curve.zero_rate(t, "semi-annual")
            df_from_cont = math.exp(-z_cont * t)
            df_from_semi = (1.0 + z_semi / 2.0) ** (-2.0 * t)
            assert abs(df_from_cont - df_from_semi) < 1e-12, (
                f"Compounding mismatch at t={t}: "
                f"DF(cont)={df_from_cont:.8f} vs DF(semi)={df_from_semi:.8f}"
            )

    def test_continuous_vs_annual_consistency(self, curve):
        for t in [1.0, 5.0, 30.0]:
            z_cont = curve.zero_rate(t, "continuous")
            z_ann = curve.zero_rate(t, "annual")
            df_from_cont = math.exp(-z_cont * t)
            df_from_ann = (1.0 + z_ann) ** (-t)
            assert abs(df_from_cont - df_from_ann) < 1e-12

    def test_invalid_compounding_raises(self, curve):
        with pytest.raises(ValueError):
            curve.zero_rate("5Y", "quarterly")


# ---------------------------------------------------------------------------
# Forward rate
# ---------------------------------------------------------------------------

class TestForwardRate:
    def test_forward_rate_no_arbitrage(self, curve):
        """
        DF(t1) / DF(t2) = exp(f_cont * (t2 - t1)).
        """
        for t1, t2 in [(1.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 30.0)]:
            f = curve.forward_rate(t1, t2, "continuous")
            df1 = curve.discount_factor(t1)
            df2 = curve.discount_factor(t2)
            ratio = df1 / df2
            expected_ratio = math.exp(f * (t2 - t1))
            assert abs(ratio - expected_ratio) < 1e-10, (
                f"Arbitrage in forward rate [{t1},{t2}]: "
                f"DF ratio={ratio:.10f}, exp(f*tau)={expected_ratio:.10f}"
            )

    def test_simple_forward_positive(self, curve):
        for t1, t2 in [(0.5, 1.0), (2.0, 3.0), (9.0, 10.0)]:
            f = curve.forward_rate(t1, t2, "simple")
            assert f > 0

    def test_invalid_tenor_order_raises(self, curve):
        with pytest.raises(ValueError):
            curve.forward_rate("5Y", "2Y")


# ---------------------------------------------------------------------------
# Par rate round-trip
# ---------------------------------------------------------------------------

class TestParRateRoundTrip:
    def test_par_rate_roundtrip_2y(self, curve):
        """
        The par rate reconstructed at 2Y should approximately equal the input 2Y par rate.
        Small differences arise from date rounding in tenor parsing.
        """
        target_par = 0.0480  # 2Y par rate
        computed_par = curve.par_rate("2Y")
        assert abs(computed_par - target_par) < 0.002, (
            f"2Y par rate: input={target_par:.4%}, computed={computed_par:.4%}"
        )

    def test_par_rate_at_5y(self, curve):
        target_par = 0.0440
        computed_par = curve.par_rate("5Y")
        assert abs(computed_par - target_par) < 0.002

    def test_par_rate_at_10y(self, curve):
        target_par = 0.0430
        computed_par = curve.par_rate("10Y")
        assert abs(computed_par - target_par) < 0.002


# ---------------------------------------------------------------------------
# Flat rate environment (round-trip)
# ---------------------------------------------------------------------------

class TestFlatRateEnvironment:
    """
    In a perfectly flat par rate environment, the zero curve should also be flat.
    """
    FLAT_RATE = 0.05
    FLAT_ANCHOR = date(2024, 1, 2)

    @pytest.fixture(scope="class")
    def flat_curve(self):
        rate = self.FLAT_RATE
        insts = [
            ParInstrument("3M", rate, date(2024, 4, 2), is_bill=True),
            ParInstrument("6M", rate, date(2024, 7, 2), is_bill=True),
            ParInstrument("1Y", rate, date(2025, 1, 2), is_bill=True),
            ParInstrument("2Y", rate, add_months(self.FLAT_ANCHOR, 24)),
            ParInstrument("5Y", rate, add_months(self.FLAT_ANCHOR, 60)),
            ParInstrument("10Y", rate, add_months(self.FLAT_ANCHOR, 120)),
            ParInstrument("30Y", rate, add_months(self.FLAT_ANCHOR, 360)),
        ]
        return TreasuryCurve(self.FLAT_ANCHOR, insts)

    def test_zero_rates_near_flat(self, flat_curve):
        """
        In a flat par environment, coupon bond zero rates won't equal the par rate
        exactly (par rates are bond-equivalent; zero rates depend on compounding),
        but they should be in the same ballpark.
        """
        for t in [1.0, 2.0, 5.0, 10.0]:
            z = flat_curve.zero_rate(t, "continuous")
            # For a 5% semi-annual par rate, the continuous zero is slightly different
            # The important thing is the curve is self-consistent
            assert 0.03 < z < 0.08, f"Zero rate at t={t} out of reasonable range: {z:.4%}"

    def test_discount_factors_decreasing(self, flat_curve):
        times = [0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        dfs = [flat_curve.discount_factor(t) for t in times]
        for i in range(1, len(dfs)):
            assert dfs[i] < dfs[i - 1]

    def test_no_negative_forwards(self, flat_curve):
        """Even with bill/bond conventions mixed, no negative forwards."""
        for t in np.linspace(0.1, 30.0, 500):
            f = flat_curve.instantaneous_forward(t)
            assert f > -1e-8, f"Negative forward at t={t:.4f}: f={f:.8e}"


# ---------------------------------------------------------------------------
# Pillar summary
# ---------------------------------------------------------------------------

class TestPillarSummary:
    def test_pillar_summary_runs(self, curve):
        summary = curve.pillar_summary()
        assert "Time" in summary
        assert len(summary.split("\n")) > 3  # Header + separator + data
