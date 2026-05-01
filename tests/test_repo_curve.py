"""Tests for the repo curve model."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from repo_curve import Compounding, DayCount, Interpolation, RepoCurve, year_fraction
from repo_curve.interpolation import _cubic_spline, _search, interpolate_log_df


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SETTLE = date(2024, 1, 2)

PILLARS = [
    (date(2024, 2, 2), 0.0520),   # 1M
    (date(2024, 4, 2), 0.0530),   # 3M
    (date(2024, 7, 2), 0.0525),   # 6M
    (date(2025, 1, 2), 0.0515),   # 1Y
]


@pytest.fixture
def flat_curve() -> RepoCurve:
    """Flat 5 % curve — analytic results are easy to verify."""
    rate = 0.05
    return RepoCurve(
        SETTLE,
        [
            (SETTLE + timedelta(days=30), rate),
            (SETTLE + timedelta(days=90), rate),
            (SETTLE + timedelta(days=180), rate),
            (SETTLE + timedelta(days=365), rate),
        ],
    )


@pytest.fixture
def market_curve() -> RepoCurve:
    return RepoCurve(SETTLE, PILLARS)


# ---------------------------------------------------------------------------
# year_fraction
# ---------------------------------------------------------------------------

class TestYearFraction:
    def test_act360_same_date(self):
        assert year_fraction(SETTLE, SETTLE, DayCount.ACT_360) == 0.0

    def test_act360_one_year(self):
        # 2024 is a leap year: Jan 2 2024 -> Jan 2 2025 = 366 actual days
        d = date(2025, 1, 2)
        assert year_fraction(SETTLE, d, DayCount.ACT_360) == pytest.approx(366 / 360)

    def test_act365_one_year(self):
        # 2024 is a leap year: 366 actual days
        d = date(2025, 1, 2)
        assert year_fraction(SETTLE, d, DayCount.ACT_365) == pytest.approx(366 / 365)

    def test_act_act_non_leap(self):
        start = date(2023, 1, 1)
        end = date(2024, 1, 1)
        assert year_fraction(start, end, DayCount.ACT_ACT) == pytest.approx(1.0)

    def test_act_act_leap_year(self):
        start = date(2024, 1, 1)
        end = date(2025, 1, 1)
        assert year_fraction(start, end, DayCount.ACT_ACT) == pytest.approx(1.0)

    def test_act_act_cross_year(self):
        # 6 months straddling a year boundary
        start = date(2023, 7, 1)
        end = date(2024, 1, 1)
        frac = year_fraction(start, end, DayCount.ACT_ACT)
        assert 0.49 < frac < 0.51


# ---------------------------------------------------------------------------
# RepoCurve construction
# ---------------------------------------------------------------------------

class TestRepoCurveConstruction:
    def test_rejects_empty_pillars(self):
        with pytest.raises(ValueError, match="pillar"):
            RepoCurve(SETTLE, [])

    def test_rejects_past_maturity(self):
        with pytest.raises(ValueError, match="settlement"):
            RepoCurve(SETTLE, [(SETTLE - timedelta(days=1), 0.05)])

    def test_rejects_same_day_maturity(self):
        with pytest.raises(ValueError, match="settlement"):
            RepoCurve(SETTLE, [(SETTLE, 0.05)])

    def test_rejects_negative_rate(self):
        with pytest.raises(ValueError, match="non-negative"):
            RepoCurve(SETTLE, [(SETTLE + timedelta(days=90), -0.01)])

    def test_single_pillar(self):
        c = RepoCurve(SETTLE, [(SETTLE + timedelta(days=90), 0.05)])
        assert len(c.pillar_dates) == 1

    def test_pillars_sorted_by_maturity(self):
        shuffled = [PILLARS[2], PILLARS[0], PILLARS[3], PILLARS[1]]
        c = RepoCurve(SETTLE, shuffled)
        assert c.pillar_dates == sorted(d for d, _ in PILLARS)

    def test_repr(self, market_curve):
        r = repr(market_curve)
        assert "RepoCurve" in r
        assert "settlement" in r


# ---------------------------------------------------------------------------
# Discount factors
# ---------------------------------------------------------------------------

class TestDiscountFactor:
    def test_at_settlement(self, flat_curve):
        assert flat_curve.discount_factor(SETTLE) == pytest.approx(1.0)

    def test_before_settlement_raises(self, flat_curve):
        with pytest.raises(ValueError, match="before settlement"):
            flat_curve.discount_factor(SETTLE - timedelta(days=1))

    def test_flat_curve_simple_compounding(self, flat_curve):
        rate = 0.05
        for days in [30, 90, 180, 365]:
            target = SETTLE + timedelta(days=days)
            tau = year_fraction(SETTLE, target, DayCount.ACT_360)
            expected = 1.0 / (1.0 + rate * tau)
            assert flat_curve.discount_factor(target) == pytest.approx(expected, rel=1e-6)

    def test_discount_factor_monotone_decreasing(self, market_curve):
        dates = [SETTLE + timedelta(days=d) for d in [10, 30, 60, 90, 180, 270, 365]]
        dfs = [market_curve.discount_factor(d) for d in dates]
        for i in range(len(dfs) - 1):
            assert dfs[i] > dfs[i + 1]

    def test_pillar_discount_factors_consistent(self, market_curve):
        for mat, rate in PILLARS:
            tau = year_fraction(SETTLE, mat, DayCount.ACT_360)
            expected_df = 1.0 / (1.0 + rate * tau)
            assert market_curve.discount_factor(mat) == pytest.approx(expected_df, rel=1e-8)

    def test_continuous_compounding(self):
        rate = 0.05
        mat = SETTLE + timedelta(days=180)
        c = RepoCurve(
            SETTLE,
            [(mat, rate)],
            compounding=Compounding.CONTINUOUS,
        )
        tau = year_fraction(SETTLE, mat, DayCount.ACT_360)
        expected = math.exp(-rate * tau)
        assert c.discount_factor(mat) == pytest.approx(expected, rel=1e-8)

    def test_at_pillar_boundaries(self, market_curve):
        for mat, _ in PILLARS:
            df = market_curve.discount_factor(mat)
            assert 0 < df < 1

    def test_short_end_extrapolation(self, market_curve):
        # Date before first pillar — should use first pillar rate
        early = SETTLE + timedelta(days=5)
        df = market_curve.discount_factor(early)
        assert 0 < df < 1

    def test_long_end_extrapolation(self, market_curve):
        # Date after last pillar — flat-forward extrapolation
        late = SETTLE + timedelta(days=500)
        df = market_curve.discount_factor(late)
        assert 0 < df < 1


# ---------------------------------------------------------------------------
# Repo rate
# ---------------------------------------------------------------------------

class TestRepoRate:
    def test_flat_curve_repo_rate(self, flat_curve):
        for days in [30, 90, 180, 365]:
            target = SETTLE + timedelta(days=days)
            assert flat_curve.repo_rate(target) == pytest.approx(0.05, rel=1e-6)

    def test_repo_rate_at_pillars(self, market_curve):
        for mat, rate in PILLARS:
            assert market_curve.repo_rate(mat) == pytest.approx(rate, rel=1e-6)

    def test_repo_rate_between_pillars(self, market_curve):
        mid = SETTLE + timedelta(days=60)
        rate = market_curve.repo_rate(mid)
        assert 0.0 < rate < 0.10  # reasonable range

    def test_repo_rate_positive(self, market_curve):
        dates = [SETTLE + timedelta(days=d) for d in [7, 30, 90, 180, 270, 365, 400]]
        for d in dates:
            assert market_curve.repo_rate(d) > 0


# ---------------------------------------------------------------------------
# Zero rates
# ---------------------------------------------------------------------------

class TestZeroRate:
    def test_zero_rate_flat_curve(self, flat_curve):
        """For a flat simple-rate curve the zero rate differs from repo rate."""
        for days in [30, 90, 180, 365]:
            target = SETTLE + timedelta(days=days)
            df = flat_curve.discount_factor(target)
            tau = year_fraction(SETTLE, target, DayCount.ACT_360)
            expected_zero = -math.log(df) / tau
            assert flat_curve.zero_rate(target) == pytest.approx(expected_zero, rel=1e-8)

    def test_zero_rate_positive(self, market_curve):
        dates = [SETTLE + timedelta(days=d) for d in [30, 90, 180, 365]]
        for d in dates:
            assert market_curve.zero_rate(d) > 0


# ---------------------------------------------------------------------------
# Forward rates and discount factors
# ---------------------------------------------------------------------------

class TestForwardRates:
    def test_end_before_start_raises(self, flat_curve):
        t1 = SETTLE + timedelta(days=90)
        t2 = SETTLE + timedelta(days=30)
        with pytest.raises(ValueError, match="end_date"):
            flat_curve.forward_discount_factor(t1, t2)

    def test_forward_df_consistency(self, market_curve):
        """D(0,T2) == D(0,T1) * D(T1,T2)."""
        t1 = SETTLE + timedelta(days=90)
        t2 = SETTLE + timedelta(days=270)
        df1 = market_curve.discount_factor(t1)
        df2 = market_curve.discount_factor(t2)
        fwd_df = market_curve.forward_discount_factor(t1, t2)
        assert df1 * fwd_df == pytest.approx(df2, rel=1e-10)

    def test_forward_rate_positive(self, market_curve):
        t1 = SETTLE + timedelta(days=90)
        t2 = SETTLE + timedelta(days=180)
        assert market_curve.forward_rate(t1, t2) > 0

    def test_flat_forward_rate_equals_spot(self):
        """On a flat continuously-compounded curve every forward rate equals the spot rate."""
        rate = 0.05
        mat = SETTLE + timedelta(days=365)
        c = RepoCurve(SETTLE, [(mat, rate)], compounding=Compounding.CONTINUOUS)
        t1 = SETTLE + timedelta(days=30)
        t2 = SETTLE + timedelta(days=90)
        fwd = c.forward_rate(t1, t2)
        assert fwd == pytest.approx(rate, rel=1e-8)

    def test_same_start_end_raises(self, flat_curve):
        t = SETTLE + timedelta(days=90)
        with pytest.raises(ValueError):
            flat_curve.forward_discount_factor(t, t)

    def test_forward_rate_equals_par_repo_rate(self, market_curve):
        t1 = SETTLE + timedelta(days=90)
        t2 = SETTLE + timedelta(days=180)
        assert market_curve.forward_rate(t1, t2) == market_curve.par_repo_rate(t1, t2)


# ---------------------------------------------------------------------------
# Accrual factor
# ---------------------------------------------------------------------------

class TestAccrualFactor:
    def test_accrual_factor_simple(self, flat_curve):
        t1 = SETTLE + timedelta(days=30)
        t2 = SETTLE + timedelta(days=120)
        af = flat_curve.accrual_factor(t1, t2)
        rate = flat_curve.forward_rate(t1, t2)
        tau = year_fraction(t1, t2, DayCount.ACT_360)
        assert af == pytest.approx(1.0 + rate * tau, rel=1e-10)

    def test_accrual_factor_continuous(self):
        rate = 0.05
        mat = SETTLE + timedelta(days=180)
        c = RepoCurve(
            SETTLE,
            [(mat, rate)],
            compounding=Compounding.CONTINUOUS,
        )
        t1 = SETTLE + timedelta(days=30)
        t2 = SETTLE + timedelta(days=120)
        af = c.accrual_factor(t1, t2)
        fwd = c.forward_rate(t1, t2)
        tau = year_fraction(t1, t2, DayCount.ACT_360)
        assert af == pytest.approx(math.exp(fwd * tau), rel=1e-10)


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

class TestInterpolation:
    def test_log_linear_returns_pillar_dfs(self, market_curve):
        """Interpolated DF at a pillar must equal the pillar DF."""
        for mat, _ in PILLARS:
            tau = year_fraction(SETTLE, mat, DayCount.ACT_360)
            log_df_interp = interpolate_log_df(
                market_curve._tenors, market_curve._log_dfs, tau
            )
            expected = math.log(market_curve.discount_factor(mat))
            assert log_df_interp == pytest.approx(expected, rel=1e-10)

    def test_linear_interpolation(self):
        rate = 0.05
        pillars = [
            (SETTLE + timedelta(days=30), rate),
            (SETTLE + timedelta(days=180), rate),
            (SETTLE + timedelta(days=365), rate),
        ]
        c = RepoCurve(SETTLE, pillars, interpolation=Interpolation.LINEAR)
        mid = SETTLE + timedelta(days=90)
        assert c.discount_factor(mid) > 0

    def test_cubic_spline_interpolation(self):
        pillars = [
            (SETTLE + timedelta(days=30), 0.052),
            (SETTLE + timedelta(days=90), 0.053),
            (SETTLE + timedelta(days=180), 0.0525),
            (SETTLE + timedelta(days=365), 0.0515),
        ]
        c = RepoCurve(SETTLE, pillars, interpolation=Interpolation.CUBIC_SPLINE)
        mid = SETTLE + timedelta(days=60)
        df = c.discount_factor(mid)
        assert 0 < df < 1

    def test_cubic_spline_passes_through_pillars(self):
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [1.0, 4.0, 9.0, 16.0]
        for x, y in zip(xs, ys):
            assert _cubic_spline(xs, ys, x) == pytest.approx(y, abs=1e-9)

    def test_binary_search(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _search(xs, 1.5) == 0
        assert _search(xs, 2.0) == 1
        assert _search(xs, 4.9) == 3


# ---------------------------------------------------------------------------
# Day count conventions via RepoCurve
# ---------------------------------------------------------------------------

class TestDayCountConventions:
    @pytest.mark.parametrize("dc", [DayCount.ACT_360, DayCount.ACT_365, DayCount.ACT_ACT])
    def test_discount_factor_positive(self, dc):
        mat = SETTLE + timedelta(days=180)
        c = RepoCurve(SETTLE, [(mat, 0.05)], day_count=dc)
        df = c.discount_factor(mat)
        assert 0 < df < 1

    def test_act360_vs_act365_different_df(self):
        mat = SETTLE + timedelta(days=180)
        c360 = RepoCurve(SETTLE, [(mat, 0.05)], day_count=DayCount.ACT_360)
        c365 = RepoCurve(SETTLE, [(mat, 0.05)], day_count=DayCount.ACT_365)
        # ACT/360 gives a larger year fraction → smaller DF for the same rate
        assert c360.discount_factor(mat) < c365.discount_factor(mat)


# ---------------------------------------------------------------------------
# Implied financing cost
# ---------------------------------------------------------------------------

class TestImpliedFinancingCost:
    def test_implied_cost_equals_forward_rate(self, flat_curve):
        """For a bond with no coupon in the period, implied financing == forward rate."""
        t1 = SETTLE + timedelta(days=30)
        t2 = SETTLE + timedelta(days=90)
        fwd = flat_curve.forward_rate(t1, t2)
        af = flat_curve.accrual_factor(t1, t2)
        dirty_price = 100.0
        implied = flat_curve.implied_financing_cost(
            dirty_price, dirty_price, t1, t2
        )
        # implied cost = ((dirty * af) / dirty - 1) / tau = (af - 1) / tau
        tau = year_fraction(t1, t2, DayCount.ACT_360)
        assert implied == pytest.approx((af - 1.0) / tau, rel=1e-10)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_pillar_curve(self):
        mat = SETTLE + timedelta(days=90)
        c = RepoCurve(SETTLE, [(mat, 0.05)])
        # Short end
        assert c.discount_factor(SETTLE + timedelta(days=30)) > 0
        # At pillar
        assert c.discount_factor(mat) == pytest.approx(1.0 / (1.0 + 0.05 * 90 / 360), rel=1e-8)
        # Long end (flat-forward extrapolation)
        assert c.discount_factor(SETTLE + timedelta(days=180)) > 0

    def test_very_short_tenor(self):
        mat = SETTLE + timedelta(days=365)
        c = RepoCurve(SETTLE, [(mat, 0.05)])
        overnight = SETTLE + timedelta(days=1)
        assert c.discount_factor(overnight) > 0

    def test_all_interpolation_methods_agree_at_pillars(self):
        for method in Interpolation:
            c = RepoCurve(SETTLE, PILLARS, interpolation=method)
            for mat, rate in PILLARS:
                assert c.repo_rate(mat) == pytest.approx(rate, rel=1e-6), (
                    f"Method {method} failed at pillar {mat}"
                )

    def test_compounding_types_give_consistent_zero_rates(self):
        """Both compounding conventions should give similar zero rates."""
        mat = SETTLE + timedelta(days=180)
        rate = 0.05
        c_simple = RepoCurve(SETTLE, [(mat, rate)], compounding=Compounding.SIMPLE)
        c_cont = RepoCurve(SETTLE, [(mat, rate)], compounding=Compounding.CONTINUOUS)
        z_simple = c_simple.zero_rate(mat)
        z_cont = c_cont.zero_rate(mat)
        # Zero rates are continuously compounded in both cases
        assert abs(z_simple - z_cont) < 0.005  # within 50 bps (different input conventions)
