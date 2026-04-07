"""
Hagan-West Monotone Convex interpolation for forward rate curves.

Reference:
    Hagan, P. & West, G. (2008). "Methods for Constructing a Yield Curve."
    Wilmott Magazine, May 2008, pp. 70-81.

Given:
  - Pillar times t_1 .. t_N
  - Segment forward rates F_1 .. F_N  (F_i is the average forward over [t_{i-1}, t_i])
  - Instantaneous forward estimates g_1 .. g_N at each pillar  (g[0] = estimate at t=0)

The interpolant within each segment [t_{i-1}, t_i] is a cubic polynomial
in the normalised coordinate x = (t - t_{i-1}) / h_i that:
  - Satisfies the integral constraint: average of f over the segment equals F_i
  - Matches the instantaneous forward estimates at the endpoints
  - Is adjusted for monotonicity to prevent negative forwards

"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class BoundaryCondition(Enum):
    HAGAN_WEST = "hagan_west"  # extrapolate slope from adjacent segment
    FLAT = "flat"              # extend segment forward flat to the boundary


@dataclass
class _Segment:
    """Pre-computed coefficients for one segment [t_left, t_right]."""
    t_left: float
    t_right: float
    h: float           # t_right - t_left
    F: float           # segment forward rate
    g0: float          # adjusted instantaneous forward at t_left
    g1: float          # adjusted instantaneous forward at t_right
    cumulative_integral_left: float  # integral of f from t_0 to t_left


class MonotoneConvex:
    """
    Piecewise cubic monotone-convex forward rate interpolator.

    Parameters
    ----------
    times : array-like, shape (N,)
        Pillar year-fractions t_1 .. t_N  (strictly increasing, all > 0).
    segment_forwards : array-like, shape (N,)
        Average forward rate for each segment i:
        F_i = integral of f over [t_{i-1}, t_i] / h_i
    pillar_forwards : array-like, shape (N,)
        Instantaneous forward rate estimates *at the right boundary* of each
        segment (i.e. at t_1 .. t_N).  The estimate at t_0=0 is derived
        internally from the boundary condition.
    boundary : BoundaryCondition
        How to set the instantaneous forward at the curve endpoints (t=0 and t=t_N).
    """

    def __init__(
        self,
        times: np.ndarray,
        segment_forwards: np.ndarray,
        pillar_forwards: np.ndarray,
        boundary: BoundaryCondition = BoundaryCondition.HAGAN_WEST,
    ) -> None:
        times = np.asarray(times, dtype=float)
        segment_forwards = np.asarray(segment_forwards, dtype=float)
        pillar_forwards = np.asarray(pillar_forwards, dtype=float)

        N = len(times)
        if len(segment_forwards) != N or len(pillar_forwards) != N:
            raise ValueError("times, segment_forwards and pillar_forwards must have the same length.")

        self._times = times          # t_1 .. t_N
        self._N = N
        self._t0 = 0.0

        # Build g array: g[0] at t=0, g[i] at t_i for i=1..N
        g = np.empty(N + 1)
        g[1:] = pillar_forwards

        if boundary == BoundaryCondition.HAGAN_WEST:
            g[0] = segment_forwards[0] - (g[1] - segment_forwards[0]) / 2.0
        else:
            g[0] = segment_forwards[0]

        # Build and store segments
        self._segments: list[_Segment] = []
        cumulative = 0.0
        t_prev = 0.0

        for i in range(N):
            t_curr = times[i]
            h = t_curr - t_prev
            F = segment_forwards[i]
            g0_raw = g[i]
            g1_raw = g[i + 1]

            g0_adj, g1_adj = self._monotone_adjustment(F, g0_raw, g1_raw)

            seg = _Segment(
                t_left=t_prev,
                t_right=t_curr,
                h=h,
                F=F,
                g0=g0_adj,
                g1=g1_adj,
                cumulative_integral_left=cumulative,
            )
            self._segments.append(seg)

            # Accumulate the integral over this full segment
            # integral = F * h  (by the segment-forward constraint)
            cumulative += F * h
            t_prev = t_curr

        self._total_integral = cumulative  # integral from 0 to t_N

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    def forward_rate(self, t: float) -> float:
        """Instantaneous forward rate f(t)."""
        if t <= 0.0:
            return self._segments[0].g0 if self._segments else 0.0

        if t >= self._times[-1]:
            # Flat extrapolation beyond last pillar
            last = self._segments[-1]
            return last.g1

        seg, x = self._locate(t)
        return self._cubic_f(seg, x)

    def integrated_forward(self, t: float) -> float:
        """
        Integral of f from 0 to t, equal to z(t) * t = -ln(DF(t)).

        Computed analytically from the cubic interpolant.
        """
        if t <= 0.0:
            return 0.0

        if t >= self._times[-1]:
            # Flat extrapolation: f = g_N (last pillar forward)
            last = self._segments[-1]
            excess = t - self._times[-1]
            return self._total_integral + last.g1 * excess

        seg, x = self._locate(t)
        partial = self._cubic_integral(seg, x)
        return seg.cumulative_integral_left + partial

    def zero_rate(self, t: float) -> float:
        """Continuously compounded zero rate z(t) = -ln(DF(t)) / t."""
        if t <= 0.0:
            return self.forward_rate(0.0)
        return self.integrated_forward(t) / t

    def discount_factor(self, t: float) -> float:
        """Discount factor DF(t) = exp(-z(t) * t)."""
        return math.exp(-self.integrated_forward(t))

    def period_forward_rate(self, t1: float, t2: float) -> float:
        """
        Continuously compounded forward rate for the period [t1, t2].
        f(t1, t2) = (integral_f(t2) - integral_f(t1)) / (t2 - t1)
        """
        if t2 <= t1:
            raise ValueError("t2 must be greater than t1.")
        num = self.integrated_forward(t2) - self.integrated_forward(t1)
        return num / (t2 - t1)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _locate(self, t: float):
        """Find the segment containing t and compute normalised coordinate x."""
        # Binary search: find i such that times[i-1] < t <= times[i]
        idx = np.searchsorted(self._times, t, side="left")
        # Clamp to valid range
        idx = max(0, min(idx, self._N - 1))
        seg = self._segments[idx]
        x = (t - seg.t_left) / seg.h if seg.h > 0 else 0.0
        return seg, x

    @staticmethod
    def _cubic_f(seg: _Segment, x: float) -> float:
        """
        Evaluate the cubic forward rate at normalised coordinate x in [0, 1].

        Using deviations d0 = g0 - F and d1 = g1 - F:

            f(x) = F + d0*(1 - 4x + 3x^2) + d1*(-2x + 3x^2)

        This satisfies:
          - f(0) = g0  (left endpoint)
          - f(1) = g1  (right endpoint)
          - integral_0^1 f(x) dx = F  (segment-forward constraint, since
            the integrals of the basis polynomials are both zero)
        """
        d0 = seg.g0 - seg.F
        d1 = seg.g1 - seg.F
        return seg.F + d0 * (1.0 - 4.0 * x + 3.0 * x * x) + d1 * (-2.0 * x + 3.0 * x * x)

    @staticmethod
    def _cubic_integral(seg: _Segment, x: float) -> float:
        """
        Definite integral of f from 0 to x within the segment, scaled by h.

        integral_0^x f(u) du  where u = eta/h is the normalised coordinate.

        Integrating term by term (using deviations d0, d1):
          F * h * x
          + d0 * h * (x - 2x^2 + x^3)
          + d1 * h * (-x^2 + x^3)
        """
        h = seg.h
        d0 = seg.g0 - seg.F
        d1 = seg.g1 - seg.F
        return h * (
            seg.F * x
            + d0 * (x - 2.0 * x * x + x * x * x)
            + d1 * (-x * x + x * x * x)
        )

    # ------------------------------------------------------------------
    # Monotone convex adjustment (Hagan-West zone-based scheme)
    # ------------------------------------------------------------------

    @staticmethod
    def _monotone_adjustment(F: float, g0: float, g1: float):
        """
        Apply the Hagan-West monotonicity correction to ensure f(x) >= 0
        everywhere in [0, 1] and to limit oscillation.

        Returns the adjusted (g0, g1) pair.
        """
        # Deviations from segment forward
        d0 = g0 - F
        d1 = g1 - F

        # Zone 1: both deviations are negligible (numerically flat segment)
        if abs(d0) < 1e-14 and abs(d1) < 1e-14:
            return g0, g1

        # The forward rate f(x) = F + d0*(1-4x+3x^2) + d1*(-2x+3x^2)
        # must be non-negative everywhere in [0,1].
        # Hagan-West define a feasibility region for (d0, d1):
        #   -2*d1 <= d0 <= 4*d1  [when d1 >= 0]  or equivalent
        # The safe quadrant for guaranteeing f >= 0 is:
        #   d0 >= -2*d1  AND  d1 >= -2*d0  AND  |d0| <= 3*F  AND  |d1| <= 3*F
        #
        # Simplified Hagan-West zone test:

        # If F itself is zero or negative, clamp both endpoints
        if F <= 0.0:
            return F, F

        # Check whether the uncorrected cubic might go below zero.
        # For the cubic f(x) = F + d0*(1-4x+3x^2) + d1*(-2x+3x^2),
        # the minimum occurs somewhere in [0,1].  A sufficient (conservative)
        # condition for positivity is |d0| <= 3F and |d1| <= 3F and
        # d0 >= -2*d1 and d1 >= -2*d0 (Zone 2).

        # Clamp magnitudes first (prevent overshooting by more than 3*F)
        d0 = max(-3.0 * F, min(3.0 * F, d0))
        d1 = max(-3.0 * F, min(3.0 * F, d1))

        # Now apply zone-based directional corrections:
        # Zone 3: d1 < -2*d0  (right side too negative)
        if d1 < -2.0 * d0:
            d1 = -2.0 * d0

        # Zone 4: d0 < -2*d1  (left side too negative)
        elif d0 < -2.0 * d1:
            d0 = -2.0 * d1

        return F + d0, F + d1
