"""
market_data.py - Simulated SOFR OIS market quotes for the period
                 2026-01-28 to 2026-04-28.

Rates reflect a plausible 2026 environment where the Fed completed its
easing cycle and SOFR settled in the 4.05–4.35 % range.  The short end
(O/N to 3 M) is slightly above the belly (1–2 Y) because the market expects
no further cuts near-term, while the 5–30 Y sector is above the front-end
(modestly upward-sloping long end).

All rates are expressed as decimals.
"""

from datetime import date
import math


# ---------------------------------------------------------------------------
# Snapshots: (date, {tenor: rate})
# These are end-of-day mid-market OIS quotes.
# ---------------------------------------------------------------------------

# Tenor grid used across all dates
_TENORS = [
    ('1D',  'deposit'),
    ('1W',  'ois'),
    ('2W',  'ois'),
    ('1M',  'ois'),
    ('2M',  'ois'),
    ('3M',  'ois'),
    ('6M',  'ois'),
    ('9M',  'ois'),
    ('12M', 'ois'),
    ('2Y',  'ois'),
    ('3Y',  'ois'),
    ('5Y',  'ois'),
    ('7Y',  'ois'),
    ('10Y', 'ois'),
    ('15Y', 'ois'),
    ('20Y', 'ois'),
    ('30Y', 'ois'),
]

# Reference rate snapshots for four key dates
_SNAPSHOTS: dict[date, dict[str, float]] = {

    # --- 28-Jan-2026 --------------------------------------------------
    # FOMC held at 4.25-4.50 %; market prices 1 cut in H2-2026.
    # Slight front-end inversion: O/N > 2Y.
    date(2026, 1, 28): {
        '1D':  0.04330,
        '1W':  0.04325,
        '2W':  0.04318,
        '1M':  0.04305,
        '2M':  0.04295,
        '3M':  0.04280,
        '6M':  0.04230,
        '9M':  0.04185,
        '12M': 0.04150,
        '2Y':  0.04120,
        '3Y':  0.04155,
        '5Y':  0.04225,
        '7Y':  0.04295,
        '10Y': 0.04380,
        '15Y': 0.04440,
        '20Y': 0.04470,
        '30Y': 0.04490,
    },

    # --- 28-Feb-2026 --------------------------------------------------
    # Strong payrolls; market prices out the remaining cut.
    # Front end rises; long end slightly higher on term premium.
    date(2026, 2, 28): {
        '1D':  0.04330,
        '1W':  0.04327,
        '2W':  0.04322,
        '1M':  0.04315,
        '2M':  0.04308,
        '3M':  0.04295,
        '6M':  0.04255,
        '9M':  0.04220,
        '12M': 0.04195,
        '2Y':  0.04175,
        '3Y':  0.04200,
        '5Y':  0.04265,
        '7Y':  0.04330,
        '10Y': 0.04410,
        '15Y': 0.04465,
        '20Y': 0.04490,
        '30Y': 0.04505,
    },

    # --- 28-Mar-2026 --------------------------------------------------
    # Softer CPI; market re-prices one 25 bp cut for Sep-2026.
    # Front end eases; curve steepens further.
    date(2026, 3, 28): {
        '1D':  0.04330,
        '1W':  0.04322,
        '2W':  0.04312,
        '1M':  0.04295,
        '2M':  0.04270,
        '3M':  0.04245,
        '6M':  0.04185,
        '9M':  0.04135,
        '12M': 0.04095,
        '2Y':  0.04060,
        '3Y':  0.04090,
        '5Y':  0.04170,
        '7Y':  0.04255,
        '10Y': 0.04355,
        '15Y': 0.04420,
        '20Y': 0.04455,
        '30Y': 0.04475,
    },

    # --- 28-Apr-2026 --------------------------------------------------
    # Trade-tariff uncertainty widens credit spreads; flight-to-quality
    # bid flattens the long end; front end unchanged (Fed on hold).
    date(2026, 4, 28): {
        '1D':  0.04330,
        '1W':  0.04320,
        '2W':  0.04305,
        '1M':  0.04280,
        '2M':  0.04248,
        '3M':  0.04215,
        '6M':  0.04150,
        '9M':  0.04095,
        '12M': 0.04052,
        '2Y':  0.04010,
        '3Y':  0.04035,
        '5Y':  0.04110,
        '7Y':  0.04195,
        '10Y': 0.04295,
        '15Y': 0.04365,
        '20Y': 0.04400,
        '30Y': 0.04420,
    },
}


def get_instruments(as_of: date) -> list[dict]:
    """
    Return market-instrument list for the given date.
    Dates between snapshots are linearly interpolated.
    """
    snap_dates = sorted(_SNAPSHOTS.keys())

    if as_of <= snap_dates[0]:
        rates = _SNAPSHOTS[snap_dates[0]]
    elif as_of >= snap_dates[-1]:
        rates = _SNAPSHOTS[snap_dates[-1]]
    else:
        # Linear interpolation between neighbouring snapshots
        for i in range(len(snap_dates) - 1):
            d0, d1 = snap_dates[i], snap_dates[i + 1]
            if d0 <= as_of <= d1:
                w = (as_of - d0).days / (d1 - d0).days
                r0, r1 = _SNAPSHOTS[d0], _SNAPSHOTS[d1]
                rates = {t: r0[t] + w * (r1[t] - r0[t]) for t in r0}
                break

    return [{'tenor': t, 'rate': rates[t], 'type': tp} for t, tp in _TENORS]


def snapshot_dates() -> list[date]:
    return sorted(_SNAPSHOTS.keys())
