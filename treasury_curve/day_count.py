"""Day count conventions and date utilities for Treasury curve construction."""

from __future__ import annotations

import math
from datetime import date, timedelta
from enum import Enum
from typing import List


class DayCount(Enum):
    ACT_ACT_ICMA = "ACT/ACT (ICMA)"
    ACT_360 = "ACT/360"
    ACT_365 = "ACT/365"
    THIRTY_360 = "30/360"


def year_fraction(start: date, end: date, convention: DayCount = DayCount.ACT_365) -> float:
    """Return the year fraction between two dates under the given day count convention."""
    if start == end:
        return 0.0
    if start > end:
        return -year_fraction(end, start, convention)

    if convention == DayCount.ACT_365:
        return (end - start).days / 365.0

    if convention == DayCount.ACT_360:
        return (end - start).days / 360.0

    if convention == DayCount.THIRTY_360:
        d1, m1, y1 = start.day, start.month, start.year
        d2, m2, y2 = end.day, end.month, end.year
        d1 = min(d1, 30)
        if d1 == 30:
            d2 = min(d2, 30)
        days = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
        return days / 360.0

    if convention == DayCount.ACT_ACT_ICMA:
        # Simplified: approximate as ACT/365.25 for non-coupon contexts
        return (end - start).days / 365.25

    raise ValueError(f"Unsupported day count convention: {convention}")


def add_months(d: date, months: int) -> date:
    """Add a number of months to a date, adjusting for end-of-month."""
    month = d.month - 1 + months
    year = d.year + month // 12
    month = month % 12 + 1
    # Clamp to end of month
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    day = min(d.day, last_day)
    return date(year, month, day)


def generate_coupon_dates(
    anchor: date,
    maturity: date,
    frequency: int = 2,  # semi-annual
) -> List[date]:
    """
    Generate coupon payment dates by rolling backward from maturity in steps of 12/frequency months.
    The first coupon may be a short (stub) period.
    """
    months_per_period = 12 // frequency
    dates: List[date] = []
    current = maturity
    while current > anchor:
        dates.append(current)
        current = add_months(current, -months_per_period)
    dates.reverse()
    return dates


def parse_tenor(tenor: str) -> float:
    """
    Convert a tenor string to a year fraction.

    Supported formats:
      "ON" or "O/N"  -> 1/365
      "1W", "2W"     -> weeks
      "1M".."12M"    -> months (30 days each)
      "1Y".."30Y"    -> years
      "180D"         -> calendar days
    """
    tenor = tenor.strip().upper()

    if tenor in ("ON", "O/N", "OVERNIGHT"):
        return 1.0 / 365.0

    if tenor.endswith("D"):
        return int(tenor[:-1]) / 365.0

    if tenor.endswith("W"):
        return int(tenor[:-1]) * 7.0 / 365.0

    if tenor.endswith("M"):
        return int(tenor[:-1]) / 12.0

    if tenor.endswith("Y"):
        return float(tenor[:-1])

    raise ValueError(f"Cannot parse tenor: '{tenor}'")
