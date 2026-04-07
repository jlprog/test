"""
Example: Build a US Treasury interest rate curve using monotone convex interpolation.

This script demonstrates how to:
1. Define par Treasury instruments (bills + notes/bonds)
2. Bootstrap to a zero/discount curve
3. Query zero rates, discount factors, and forward rates at arbitrary tenors
4. Visualise the resulting curves

Run with:
    python examples/build_curve.py
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np

from treasury_curve import ParInstrument, TreasuryCurve
from treasury_curve.day_count import add_months


# ---------------------------------------------------------------------------
# 1. Define the par curve (approximate on-the-run levels as of early 2024)
# ---------------------------------------------------------------------------

ANCHOR = date(2024, 1, 2)  # curve reference date

instruments = [
    # Treasury bills (discount-basis)
    ParInstrument("1M",  0.0543, date(2024, 2,  1), is_bill=True),
    ParInstrument("3M",  0.0540, date(2024, 4,  2), is_bill=True),
    ParInstrument("6M",  0.0535, date(2024, 7,  2), is_bill=True),
    ParInstrument("1Y",  0.0520, date(2025, 1,  2), is_bill=True),
    # Treasury notes and bonds (semi-annual coupon)
    ParInstrument("2Y",  0.0480, add_months(ANCHOR, 24)),
    ParInstrument("3Y",  0.0460, add_months(ANCHOR, 36)),
    ParInstrument("5Y",  0.0440, add_months(ANCHOR, 60)),
    ParInstrument("7Y",  0.0435, add_months(ANCHOR, 84)),
    ParInstrument("10Y", 0.0430, add_months(ANCHOR, 120)),
    ParInstrument("20Y", 0.0440, add_months(ANCHOR, 240)),
    ParInstrument("30Y", 0.0445, add_months(ANCHOR, 360)),
]

# ---------------------------------------------------------------------------
# 2. Build the curve
# ---------------------------------------------------------------------------

curve = TreasuryCurve(ANCHOR, instruments)
print(f"Built: {curve}\n")

# ---------------------------------------------------------------------------
# 3. Display bootstrapped pillars
# ---------------------------------------------------------------------------

print("=" * 60)
print("Bootstrapped Pillar Values")
print("=" * 60)
print(curve.pillar_summary())
print()

# ---------------------------------------------------------------------------
# 4. Query zero rates at standard tenors
# ---------------------------------------------------------------------------

query_tenors = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]

print("=" * 60)
print(f"{'Tenor':>6}  {'Zero (cont)':>12}  {'Zero (s.a.)':>12}  {'DF':>10}")
print("-" * 46)
for tenor in query_tenors:
    z_cont = curve.zero_rate(tenor, "continuous")
    z_semi = curve.zero_rate(tenor, "semi-annual")
    df = curve.discount_factor(tenor)
    print(f"{tenor:>6}  {z_cont*100:>11.4f}%  {z_semi*100:>11.4f}%  {df:>10.6f}")
print()

# ---------------------------------------------------------------------------
# 5. Forward rates between pillar dates
# ---------------------------------------------------------------------------

print("=" * 60)
print(f"{'Period':>12}  {'Fwd (cont)':>12}  {'Fwd (s.a.)':>12}  {'Fwd (simple)':>14}")
print("-" * 54)
periods = [
    ("3M", "6M"),
    ("6M", "1Y"),
    ("1Y", "2Y"),
    ("2Y", "3Y"),
    ("3Y", "5Y"),
    ("5Y", "7Y"),
    ("7Y", "10Y"),
    ("10Y", "20Y"),
    ("20Y", "30Y"),
]
for t1, t2 in periods:
    f_cont = curve.forward_rate(t1, t2, "continuous")
    f_semi = curve.forward_rate(t1, t2, "semi-annual")
    f_simp = curve.forward_rate(t1, t2, "simple")
    label = f"{t1}-{t2}"
    print(f"{label:>12}  {f_cont*100:>11.4f}%  {f_semi*100:>11.4f}%  {f_simp*100:>13.4f}%")
print()

# ---------------------------------------------------------------------------
# 6. Instantaneous forward rate curve (sampled for display)
# ---------------------------------------------------------------------------

print("=" * 60)
print("Instantaneous Forward Rate f(t)")
print("-" * 40)
print(f"{'t (years)':>10}  {'f(t)':>10}")
print("-" * 24)
for t in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 30.0]:
    f = curve.instantaneous_forward(t)
    print(f"{t:>10.2f}  {f*100:>9.4f}%")
print()

# ---------------------------------------------------------------------------
# 7. Par rate reconstruction (round-trip check)
# ---------------------------------------------------------------------------

print("=" * 60)
print("Par Rate Round-trip Check (input vs reconstructed)")
print("-" * 52)
print(f"{'Tenor':>6}  {'Input Par':>10}  {'Computed Par':>14}  {'Diff (bp)':>10}")
print("-" * 46)
par_tenors = ["2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
for tenor in par_tenors:
    # Find input par rate
    t_map = {inst.tenor: inst.par_rate for inst in instruments}
    input_par = t_map.get(tenor, float("nan"))
    computed_par = curve.par_rate(tenor)
    diff_bp = (computed_par - input_par) * 10000
    print(f"{tenor:>6}  {input_par*100:>9.4f}%  {computed_par*100:>13.4f}%  {diff_bp:>9.2f}")
print()

# ---------------------------------------------------------------------------
# 8. Verify no-arbitrage: check DF(t) is monotonically decreasing
# ---------------------------------------------------------------------------

times = np.linspace(0.01, 30.0, 3000)
dfs = np.array([curve.discount_factor(t) for t in times])
fwds = np.array([curve.instantaneous_forward(t) for t in times])

if np.all(np.diff(dfs) < 0):
    print("✓ Discount factors are strictly decreasing (no arbitrage).")
else:
    print("✗ WARNING: discount factors are NOT strictly decreasing!")

if np.all(fwds > -1e-10):
    print(f"✓ All instantaneous forward rates are non-negative "
          f"(min f = {fwds.min()*100:.4f}%).")
else:
    n_neg = np.sum(fwds < -1e-10)
    print(f"✗ WARNING: {n_neg} negative instantaneous forward rates detected "
          f"(min f = {fwds.min()*100:.4f}%).")
