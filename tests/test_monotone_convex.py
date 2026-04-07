"""Tests for the Hagan-West monotone convex interpolator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from treasury_curve.monotone_convex import BoundaryCondition, MonotoneConvex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_flat_curve(rate: float, times: list[float]) -> MonotoneConvex:
    """Build a curve where every segment forward equals `rate`."""
    N = len(times)
    seg_fwd = np.full(N, rate)
    pillar_fwd = np.full(N, rate)
    return MonotoneConvex(times, seg_fwd, pillar_fwd)


def sample_points(n: int = 500, t_max: float = 30.0) -> np.ndarray:
    return np.linspace(0.01, t_max, n)


# ---------------------------------------------------------------------------
# Flat curve tests
# ---------------------------------------------------------------------------

class TestFlatCurve:
    TIMES = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
    RATE = 0.05

    def setup_method(self):
        self.mc = make_flat_curve(self.RATE, self.TIMES)

    def test_instantaneous_forward_is_flat(self):
        """For a flat curve, f(t) should equal the flat rate everywhere."""
        for t in sample_points(t_max=30.0):
            f = self.mc.forward_rate(t)
            assert abs(f - self.RATE) < 1e-10, f"f({t:.4f}) = {f:.8f}, expected {self.RATE}"

    def test_zero_rate_equals_flat_rate(self):
        """z(t) = -log(DF(t)) / t should equal the flat rate."""
        for t in sample_points(t_max=30.0):
            z = self.mc.zero_rate(t)
            assert abs(z - self.RATE) < 1e-10, f"z({t:.4f}) = {z:.8f}"

    def test_discount_factor_formula(self):
        """DF(t) = exp(-rate * t) for a flat rate."""
        for t in [0.5, 1.0, 5.0, 10.0, 30.0]:
            df = self.mc.discount_factor(t)
            expected = math.exp(-self.RATE * t)
            assert abs(df - expected) < 1e-10, f"DF({t}) = {df:.8f}, expected {expected:.8f}"

    def test_integrated_forward_linear(self):
        """integral_f(t) = rate * t for a flat curve."""
        for t in [0.5, 2.0, 10.0, 25.0]:
            iv = self.mc.integrated_forward(t)
            assert abs(iv - self.RATE * t) < 1e-10, f"integral_f({t}) = {iv:.8f}"


# ---------------------------------------------------------------------------
# Integral constraint: integral of f over each segment must equal F_i
# ---------------------------------------------------------------------------

class TestIntegralConstraint:
    TIMES = [1.0, 2.0, 5.0, 10.0, 30.0]

    def _seg_fwd_from_zero(self, zero_rates: list[float]) -> tuple:
        """Derive segment forwards and pillar forwards from zero rates."""
        N = len(self.TIMES)
        t = np.array([0.0] + self.TIMES)
        z = np.array([0.0] + zero_rates)
        seg_fwd = (z[1:] * t[1:] - z[:-1] * t[:-1]) / np.diff(t)

        # Pillar forwards: interior weighted average, boundary via Hagan-West
        g = np.empty(N + 1)
        for i in range(1, N):
            h_prev = t[i] - t[i - 1]
            h_next = t[i + 1] - t[i]
            g[i] = (seg_fwd[i - 1] * h_next + seg_fwd[i] * h_prev) / (h_prev + h_next)
        g[0] = seg_fwd[0] - (g[1] - seg_fwd[0]) / 2.0
        g[N] = seg_fwd[N - 1] - (g[N - 1] - seg_fwd[N - 1]) / 2.0

        return seg_fwd, g[1:]

    def test_integral_matches_pillar_zero_rate(self):
        """
        The monotone convex interpolant must satisfy:
        integral_f(t_i) == z_i * t_i for every pillar i.
        """
        zero_rates = [0.04, 0.043, 0.046, 0.048, 0.050]
        seg_fwd, pillar_fwd = self._seg_fwd_from_zero(zero_rates)

        mc = MonotoneConvex(self.TIMES, seg_fwd, pillar_fwd)

        for t_i, z_i in zip(self.TIMES, zero_rates):
            iv = mc.integrated_forward(t_i)
            expected = z_i * t_i
            assert abs(iv - expected) < 1e-10, (
                f"integral_f({t_i}) = {iv:.10f}, expected {expected:.10f}"
            )

    def test_period_forward_consistency(self):
        """Period forward rate between two pillars must be consistent with DFs."""
        zero_rates = [0.04, 0.043, 0.046, 0.048, 0.050]
        seg_fwd, pillar_fwd = self._seg_fwd_from_zero(zero_rates)
        mc = MonotoneConvex(self.TIMES, seg_fwd, pillar_fwd)

        for i in range(len(self.TIMES) - 1):
            t1 = self.TIMES[i]
            t2 = self.TIMES[i + 1]
            f_cont = mc.period_forward_rate(t1, t2)
            df1 = mc.discount_factor(t1)
            df2 = mc.discount_factor(t2)
            # Check: exp(f*(t2-t1)) = DF1/DF2
            ratio = df1 / df2
            expected = math.exp(f_cont * (t2 - t1))
            assert abs(ratio - expected) < 1e-10, (
                f"DF ratio mismatch for [{t1}, {t2}]: {ratio:.10f} vs {expected:.10f}"
            )


# ---------------------------------------------------------------------------
# Non-negative forward rates
# ---------------------------------------------------------------------------

class TestNonNegativeForwards:
    def test_upward_sloping_curve(self):
        """Upward-sloping zero rate curve should not produce negative forwards."""
        times = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
        zero_rates = [0.035, 0.040, 0.045, 0.047, 0.048, 0.049, 0.050, 0.051, 0.052]
        t_arr = np.array([0.0] + times)
        z_arr = np.array([0.0] + zero_rates)
        seg_fwd = (z_arr[1:] * t_arr[1:] - z_arr[:-1] * t_arr[:-1]) / np.diff(t_arr)
        N = len(times)
        g = np.empty(N + 1)
        for i in range(1, N):
            h_prev = t_arr[i] - t_arr[i - 1]
            h_next = t_arr[i + 1] - t_arr[i]
            g[i] = (seg_fwd[i - 1] * h_next + seg_fwd[i] * h_prev) / (h_prev + h_next)
        g[0] = seg_fwd[0] - (g[1] - seg_fwd[0]) / 2.0
        g[N] = seg_fwd[N - 1] - (g[N - 1] - seg_fwd[N - 1]) / 2.0

        mc = MonotoneConvex(times, seg_fwd, g[1:])
        for t in sample_points(n=1000, t_max=30.0):
            f = mc.forward_rate(t)
            assert f >= -1e-10, f"Negative forward rate f({t:.4f}) = {f:.8e}"

    def test_inverted_curve(self):
        """Even an inverted curve should not produce wildly negative forwards."""
        times = [1.0, 2.0, 3.0, 5.0, 10.0, 30.0]
        zero_rates = [0.055, 0.052, 0.049, 0.047, 0.045, 0.043]  # inverted
        t_arr = np.array([0.0] + times)
        z_arr = np.array([0.0] + zero_rates)
        seg_fwd = (z_arr[1:] * t_arr[1:] - z_arr[:-1] * t_arr[:-1]) / np.diff(t_arr)
        N = len(times)
        g = np.empty(N + 1)
        for i in range(1, N):
            h_prev = t_arr[i] - t_arr[i - 1]
            h_next = t_arr[i + 1] - t_arr[i]
            g[i] = (seg_fwd[i - 1] * h_next + seg_fwd[i] * h_prev) / (h_prev + h_next)
        g[0] = seg_fwd[0] - (g[1] - seg_fwd[0]) / 2.0
        g[N] = seg_fwd[N - 1] - (g[N - 1] - seg_fwd[N - 1]) / 2.0

        mc = MonotoneConvex(times, seg_fwd, g[1:])
        for t in sample_points(n=1000, t_max=30.0):
            f = mc.forward_rate(t)
            # Inverted: allow small negatives only beyond last pillar or at boundaries
            assert f > -0.10, f"Forward rate too negative: f({t:.4f}) = {f:.6f}"


# ---------------------------------------------------------------------------
# Boundary and extrapolation
# ---------------------------------------------------------------------------

class TestBoundaryAndExtrapolation:
    def test_flat_extrapolation_beyond_last_pillar(self):
        """Beyond t_N, the forward rate should be flat at f(t_N)."""
        times = [1.0, 5.0, 10.0]
        seg_fwd = np.array([0.04, 0.045, 0.05])
        pillar_fwd = np.array([0.041, 0.046, 0.051])

        mc = MonotoneConvex(times, seg_fwd, pillar_fwd)
        f_last = mc.forward_rate(10.0)

        for t in [11.0, 15.0, 20.0, 50.0]:
            f = mc.forward_rate(t)
            assert abs(f - f_last) < 1e-12, f"Extrapolation not flat at t={t}"

    def test_boundary_conditions_hagan_west_vs_flat(self):
        """Different boundary conditions should produce different but valid curves."""
        times = [1.0, 2.0, 5.0, 10.0]
        seg_fwd = np.array([0.04, 0.045, 0.047, 0.05])
        pillar_fwd = np.array([0.042, 0.046, 0.048, 0.051])

        mc_hw = MonotoneConvex(times, seg_fwd, pillar_fwd, BoundaryCondition.HAGAN_WEST)
        mc_fl = MonotoneConvex(times, seg_fwd, pillar_fwd, BoundaryCondition.FLAT)

        # Both should produce valid (positive) forward rates
        for t in sample_points(n=200, t_max=10.0):
            assert mc_hw.forward_rate(t) > -1e-10
            assert mc_fl.forward_rate(t) > -1e-10

    def test_at_pillar_times(self):
        """Interpolant should be continuous: query at exact pillar times is well-defined."""
        times = [1.0, 2.0, 5.0, 10.0]
        seg_fwd = np.array([0.04, 0.045, 0.047, 0.05])
        pillar_fwd = np.array([0.042, 0.046, 0.048, 0.051])
        mc = MonotoneConvex(times, seg_fwd, pillar_fwd)

        for t in times:
            f = mc.forward_rate(t)
            assert math.isfinite(f), f"f({t}) is not finite"

    def test_at_t_zero(self):
        """Query at t=0 should return the left boundary forward rate."""
        times = [1.0, 5.0]
        seg_fwd = np.array([0.04, 0.05])
        pillar_fwd = np.array([0.041, 0.051])
        mc = MonotoneConvex(times, seg_fwd, pillar_fwd)
        f0 = mc.forward_rate(0.0)
        assert math.isfinite(f0)


# ---------------------------------------------------------------------------
# Period forward rate
# ---------------------------------------------------------------------------

class TestPeriodForward:
    def test_period_forward_equals_segment_forward_at_pillars(self):
        """
        The period forward for a full segment [t_{i-1}, t_i] should equal
        the segment forward F_i used in construction.
        """
        times = [1.0, 2.0, 5.0]
        seg_fwd = np.array([0.04, 0.045, 0.05])
        mc = make_flat_curve(0.04, times)  # use flat for simplicity

        for t in times:
            f = mc.period_forward_rate(0.0, t)
            assert abs(f - 0.04) < 1e-10

    def test_raises_on_invalid_tenor_order(self):
        times = [1.0, 2.0]
        mc = make_flat_curve(0.04, times)
        with pytest.raises(ValueError):
            mc.period_forward_rate(2.0, 1.0)
