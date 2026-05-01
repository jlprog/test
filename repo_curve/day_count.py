"""Day count conventions for year-fraction calculations."""

from __future__ import annotations

import calendar
from datetime import date
from enum import Enum


class DayCount(Enum):
    ACT_360 = "ACT/360"
    ACT_365 = "ACT/365"
    ACT_ACT = "ACT/ACT"


def year_fraction(start: date, end: date, convention: DayCount) -> float:
    """Return the year fraction between two dates under the given convention."""
    if start == end:
        return 0.0
    days = (end - start).days
    if convention == DayCount.ACT_360:
        return days / 360.0
    if convention == DayCount.ACT_365:
        return days / 365.0
    if convention == DayCount.ACT_ACT:
        return _act_act_isda(start, end)
    raise ValueError(f"Unknown day count convention: {convention}")


def _act_act_isda(start: date, end: date) -> float:
    """ISDA ACT/ACT: sum actual days in each calendar year, divided by that year's length."""
    frac = 0.0
    current = start
    while current.year < end.year:
        next_year = date(current.year + 1, 1, 1)
        days_in_year = 366 if calendar.isleap(current.year) else 365
        frac += (next_year - current).days / days_in_year
        current = next_year
    days_in_year = 366 if calendar.isleap(end.year) else 365
    frac += (end - current).days / days_in_year
    return frac
