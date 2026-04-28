"""
build_sofr_curve.py
===================
Builds and visualises the SOFR OIS discount curve for each of the four
key dates in the last 3 months (28-Jan-2026 → 28-Apr-2026).

Outputs
-------
* Console table: zero rates and discount factors at standard tenors.
* sofr_curves.png  – Zero-rate curves for all four dates.
* sofr_forwards.png – 3-month instantaneous forward curves.
* sofr_evolution.png – Evolution of selected tenors over time.
"""

from datetime import date
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sofr_curve import SOFRCurve, add_tenor
from market_data import get_instruments, snapshot_dates, data_source_info

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PLOT_TENORS = [
    '1W', '1M', '3M', '6M', '12M',
    '2Y', '3Y', '5Y', '7Y', '10Y', '15Y', '20Y', '30Y',
]

PRINT_TENORS = [
    '1D', '1W', '1M', '3M', '6M', '12M',
    '2Y', '3Y', '5Y', '10Y', '20Y', '30Y',
]

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
LABELS = {
    date(2026, 1, 28): '28-Jan-2026',
    date(2026, 2, 28): '28-Feb-2026',
    date(2026, 3, 28): '28-Mar-2026',
    date(2026, 4, 28): '28-Apr-2026',
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tenor_to_years(tenor: str) -> float:
    n, unit = int(tenor[:-1]), tenor[-1].upper()
    return {'D': n / 365, 'W': n / 52, 'M': n / 12, 'Y': float(n)}[unit]


def build_curve(as_of: date) -> SOFRCurve:
    insts = get_instruments(as_of)
    return SOFRCurve(as_of, insts)


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_curve(curve: SOFRCurve, label: str):
    val = curve.valuation_date
    header = f"\n{'='*62}\n  SOFR OIS Curve  |  {label}\n{'='*62}"
    print(header)
    print(f"  {'Tenor':<8} {'Maturity':<14} {'Zero Rate':>10}  {'Disc. Factor':>14}")
    print(f"  {'-'*8} {'-'*14} {'-'*10}  {'-'*14}")
    for tenor in PRINT_TENORS:
        mat = add_tenor(val, tenor)
        zr  = curve.zero_rate(mat)
        df  = curve.discount_factor(mat)
        print(f"  {tenor:<8} {mat.isoformat():<14} {zr*100:>9.4f}%  {df:>14.8f}")
    print()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_zero_curves(curves: dict[date, SOFRCurve]):
    fig, ax = plt.subplots(figsize=(12, 6))

    for (val_date, curve), color in zip(curves.items(), COLORS):
        xs, ys = [], []
        for tenor in PLOT_TENORS:
            mat = add_tenor(val_date, tenor)
            xs.append(tenor_to_years(tenor))
            ys.append(curve.zero_rate(mat) * 100)
        ax.plot(xs, ys, marker='o', markersize=5, linewidth=2,
                color=color, label=LABELS[val_date])

    ax.set_title('SOFR OIS Zero-Rate Curve  (Jan – Apr 2026)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Tenor (years)')
    ax.set_ylabel('Continuously Compounded Zero Rate (%)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_xticks([1/52, 1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])
    ax.set_xticklabels(['1W', '1M', '3M', '6M', '1Y', '2Y', '3Y',
                        '5Y', '7Y', '10Y', '20Y', '30Y'], fontsize=8)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f%%'))
    fig.tight_layout()
    fig.savefig('sofr_curves.png', dpi=150)
    print("Saved: sofr_curves.png")
    plt.close(fig)


def plot_forward_curves(curves: dict[date, SOFRCurve]):
    """3-month simply-compounded forward rate curve."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fwd_tenor = '3M'

    for (val_date, curve), color in zip(curves.items(), COLORS):
        xs, ys = [], []
        for tenor in PLOT_TENORS:
            fwd_start = add_tenor(val_date, tenor)
            fwd_end   = add_tenor(fwd_start, fwd_tenor)
            t = tenor_to_years(tenor)
            fwd = curve.forward_rate(fwd_start, fwd_end)
            xs.append(t)
            ys.append(fwd * 100)
        ax.plot(xs, ys, marker='s', markersize=5, linewidth=2,
                color=color, label=LABELS[val_date])

    ax.set_title('SOFR 3M×3M Forward Curve  (Jan – Apr 2026)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Forward Start (years)')
    ax.set_ylabel('3M Forward Rate, Act/360 (%)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_xticks([1/52, 1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])
    ax.set_xticklabels(['1W', '1M', '3M', '6M', '1Y', '2Y', '3Y',
                        '5Y', '7Y', '10Y', '20Y', '30Y'], fontsize=8)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f%%'))
    fig.tight_layout()
    fig.savefig('sofr_forwards.png', dpi=150)
    print("Saved: sofr_forwards.png")
    plt.close(fig)


def plot_evolution(curves: dict[date, SOFRCurve]):
    """Show how selected tenor zero rates evolved over the period."""
    watch_tenors = ['3M', '1Y', '2Y', '5Y', '10Y']
    watch_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    val_dates = sorted(curves.keys())
    x = [d for d in val_dates]

    fig, ax = plt.subplots(figsize=(12, 6))

    for tenor, color in zip(watch_tenors, watch_colors):
        ys = []
        for val_date in val_dates:
            curve = curves[val_date]
            mat   = add_tenor(val_date, tenor)
            ys.append(curve.zero_rate(mat) * 100)
        ax.plot(x, ys, marker='o', markersize=8, linewidth=2,
                color=color, label=tenor)
        for xi, yi in zip(x, ys):
            ax.annotate(f'{yi:.3f}%', (xi, yi),
                        textcoords='offset points', xytext=(0, 8),
                        ha='center', fontsize=8, color=color)

    ax.set_title('SOFR Zero Rate Evolution by Tenor  (Jan – Apr 2026)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Continuously Compounded Zero Rate (%)')
    ax.legend(title='Tenor', loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
    fig.autofmt_xdate()
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f%%'))
    fig.tight_layout()
    fig.savefig('sofr_evolution.png', dpi=150)
    print("Saved: sofr_evolution.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Curve summary statistics
# ---------------------------------------------------------------------------

def print_summary(curves: dict[date, SOFRCurve]):
    print("\n" + "="*70)
    print("  SUMMARY: SOFR Curve Key Metrics (last 3 months)")
    print("="*70)
    print(f"  {'Date':<14} {'O/N':>7} {'3M':>7} {'2Y':>7} {'10Y':>7} "
          f"{'2s10s':>8} {'Curve Shape'}")
    print(f"  {'-'*14} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*15}")

    for val_date in sorted(curves.keys()):
        curve = curves[val_date]
        on    = curve.zero_rate(add_tenor(val_date, '1D'))  * 100
        t3m   = curve.zero_rate(add_tenor(val_date, '3M'))  * 100
        t2y   = curve.zero_rate(add_tenor(val_date, '2Y'))  * 100
        t10y  = curve.zero_rate(add_tenor(val_date, '10Y')) * 100
        spread_2s10s = (t10y - t2y) * 100   # convert % to bp
        shape = 'Steepening' if spread_2s10s > 25 else (
                'Flat'       if abs(spread_2s10s) < 10 else
                'Inverted'   if spread_2s10s < -10 else 'Mild slope')
        print(f"  {LABELS[val_date]:<14} {on:>6.3f}% {t3m:>6.3f}% "
              f"{t2y:>6.3f}% {t10y:>6.3f}% {spread_2s10s:>+7.1f}bp  {shape}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\nBuilding SOFR OIS curves for the last 3 months...")
    # trigger fetch (prints data-source banner)
    _ = get_instruments(snapshot_dates()[0])
    print(f"Data source : {data_source_info()}\n")

    curves: dict[date, SOFRCurve] = {}
    for val_date in snapshot_dates():
        curve = build_curve(val_date)
        curves[val_date] = curve
        print_curve(curve, LABELS[val_date])

    print_summary(curves)

    plot_zero_curves(curves)
    plot_forward_curves(curves)
    plot_evolution(curves)

    print("\nDone.")


if __name__ == '__main__':
    main()
