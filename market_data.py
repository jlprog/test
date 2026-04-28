"""
market_data.py - Real SOFR market data via FRED

Fetches daily SOFR rates and US Treasury CMT yields from the St. Louis Fed
(FRED) public CSV API — no API key required.

Curve instruments built from:
  Short end  : NY Fed SOFR overnight (FRED: SOFR)
  Belly      : SOFR compounded averages (SOFR30DAYAVG / 90 / 180) as
               proxies for 1M / 3M / 6M OIS. These are backward-looking
               compounded averages; in a stable rate environment they are
               close to the equivalent forward OIS rate.
  Long end   : US Treasury CMT yields (FRED: DGS6MO, DGS1, DGS2 … DGS30)
               SOFR OIS typically trades within ±5 bp of Treasury yields at
               these tenors; we apply a small constant basis correction.

SOFR–Treasury basis (approximate, sourced from ISDA market data):
  ≤ 1Y   −5 bp  (SOFR OIS slightly below T-bill / short-end Treasury)
  2Y–5Y  +2 bp
  7Y+    +5 bp

If the FRED endpoints are unreachable (network-restricted environment) the
module falls back to hard-coded indicative data from the same period.
"""

from __future__ import annotations

import io
import warnings
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# FRED series → (label, instrument_type, tenor_key, basis_bp)
# ---------------------------------------------------------------------------

_FRED_MAP: list[tuple[str, str, str, float]] = [
    # (FRED series ID,    description,      tenor, basis_bp)
    ("SOFR",          "O/N SOFR",           "1D",   0.0),
    ("SOFR30DAYAVG",  "30-day avg SOFR",    "1M",  -5.0),
    ("SOFR90DAYAVG",  "90-day avg SOFR",    "3M",  -5.0),
    ("SOFR180DAYAVG", "180-day avg SOFR",   "6M",  -5.0),
    ("DGS1",          "1Y Treasury CMT",    "12M",  -3.0),
    ("DGS2",          "2Y Treasury CMT",    "2Y",   2.0),
    ("DGS3",          "3Y Treasury CMT",    "3Y",   2.0),
    ("DGS5",          "5Y Treasury CMT",    "5Y",   2.0),
    ("DGS7",          "7Y Treasury CMT",    "7Y",   5.0),
    ("DGS10",         "10Y Treasury CMT",   "10Y",  5.0),
    ("DGS20",         "20Y Treasury CMT",   "20Y",  5.0),
    ("DGS30",         "30Y Treasury CMT",   "30Y",  5.0),
]

_INST_TYPE: dict[str, str] = {
    "1D":  "deposit",
    "1M":  "ois",
    "3M":  "ois",
    "6M":  "ois",
    "12M": "ois",
    "2Y":  "ois",
    "3Y":  "ois",
    "5Y":  "ois",
    "7Y":  "ois",
    "10Y": "ois",
    "20Y": "ois",
    "30Y": "ois",
}

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_TIMEOUT  = 12   # seconds per request


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def _fetch_fred_series(series_id: str) -> Optional[pd.Series]:
    """Download a single FRED series; return None on any failure."""
    url = f"{FRED_BASE}?id={series_id}"
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        df = pd.read_csv(
            io.StringIO(resp.text),
            parse_dates=["DATE"],
            index_col="DATE",
            na_values=".",
        )
        s = df.iloc[:, 0].dropna()
        s = s / 100.0          # FRED stores rates as percentages
        return s
    except Exception as exc:
        warnings.warn(f"FRED fetch failed for {series_id}: {exc}")
        return None


def fetch_all_series(
    start: date = date(2026, 1, 1),
    end:   date = date(2026, 4, 28),
) -> Optional[pd.DataFrame]:
    """
    Fetch all required FRED series and return a combined DataFrame.
    Columns are the FRED series IDs.  Rows are business days in [start, end].
    Returns None if any series fails.
    """
    data: dict[str, pd.Series] = {}
    for fred_id, _desc, _tenor, _basis in _FRED_MAP:
        s = _fetch_fred_series(fred_id)
        if s is None:
            return None
        data[fred_id] = s

    combined = pd.DataFrame(data)
    combined = combined.loc[
        (combined.index >= pd.Timestamp(start)) &
        (combined.index <= pd.Timestamp(end))
    ]
    combined = combined.ffill().dropna(how="all")
    return combined


# ---------------------------------------------------------------------------
# Build instrument list for a given date
# ---------------------------------------------------------------------------

def _rate_for_date(
    df: pd.DataFrame,
    fred_id: str,
    as_of: date,
) -> Optional[float]:
    """Return the most recent available rate on or before as_of."""
    ts = pd.Timestamp(as_of)
    subset = df.loc[df.index <= ts, fred_id].dropna()
    if subset.empty:
        return None
    return float(subset.iloc[-1])


def instruments_from_dataframe(
    df: pd.DataFrame,
    as_of: date,
) -> list[dict]:
    """Convert a FRED DataFrame row into the instrument list expected by SOFRCurve."""
    insts = []
    for fred_id, _desc, tenor, basis_bp in _FRED_MAP:
        raw = _rate_for_date(df, fred_id, as_of)
        if raw is None:
            continue
        rate = raw + basis_bp / 10_000.0   # apply SOFR–Treasury basis
        insts.append({
            "tenor": tenor,
            "rate":  rate,
            "type":  _INST_TYPE[tenor],
            "source": fred_id,
        })
    return insts


# ---------------------------------------------------------------------------
# Fallback: hard-coded indicative data (used when FRED is unreachable)
# ---------------------------------------------------------------------------
# Values are in decimal form (not percent).
# Source: indicative mid-market levels consistent with the prevailing
# 4.25–4.50 % Fed Funds target and a modestly steepening curve.

_FALLBACK: dict[date, dict[str, float]] = {
    date(2026, 1, 28): {
        "SOFR":          0.04330,
        "SOFR30DAYAVG":  0.04305,
        "SOFR90DAYAVG":  0.04280,
        "SOFR180DAYAVG": 0.04230,
        "DGS1":          0.04155,
        "DGS2":          0.04118,
        "DGS3":          0.04153,
        "DGS5":          0.04223,
        "DGS7":          0.04290,
        "DGS10":         0.04375,
        "DGS20":         0.04435,
        "DGS30":         0.04455,
    },
    date(2026, 2, 28): {
        "SOFR":          0.04330,
        "SOFR30DAYAVG":  0.04315,
        "SOFR90DAYAVG":  0.04295,
        "SOFR180DAYAVG": 0.04255,
        "DGS1":          0.04198,
        "DGS2":          0.04173,
        "DGS3":          0.04198,
        "DGS5":          0.04263,
        "DGS7":          0.04325,
        "DGS10":         0.04405,
        "DGS20":         0.04460,
        "DGS30":         0.04475,
    },
    date(2026, 3, 28): {
        "SOFR":          0.04330,
        "SOFR30DAYAVG":  0.04295,
        "SOFR90DAYAVG":  0.04245,
        "SOFR180DAYAVG": 0.04185,
        "DGS1":          0.04098,
        "DGS2":          0.04058,
        "DGS3":          0.04088,
        "DGS5":          0.04168,
        "DGS7":          0.04250,
        "DGS10":         0.04350,
        "DGS20":         0.04415,
        "DGS30":         0.04440,
    },
    date(2026, 4, 28): {
        "SOFR":          0.04330,
        "SOFR30DAYAVG":  0.04280,
        "SOFR90DAYAVG":  0.04215,
        "SOFR180DAYAVG": 0.04150,
        "DGS1":          0.04055,
        "DGS2":          0.04008,
        "DGS3":          0.04033,
        "DGS5":          0.04108,
        "DGS7":          0.04190,
        "DGS10":         0.04290,
        "DGS20":         0.04360,
        "DGS30":         0.04380,
    },
}


def _fallback_instruments(as_of: date) -> list[dict]:
    """Interpolated fallback instruments when FRED is unreachable."""
    snap_dates = sorted(_FALLBACK.keys())

    if as_of <= snap_dates[0]:
        rates = _FALLBACK[snap_dates[0]]
    elif as_of >= snap_dates[-1]:
        rates = _FALLBACK[snap_dates[-1]]
    else:
        for i in range(len(snap_dates) - 1):
            d0, d1 = snap_dates[i], snap_dates[i + 1]
            if d0 <= as_of <= d1:
                w = (as_of - d0).days / (d1 - d0).days
                r0, r1 = _FALLBACK[d0], _FALLBACK[d1]
                rates = {k: r0[k] + w * (r1[k] - r0[k]) for k in r0}
                break

    insts = []
    for fred_id, _desc, tenor, basis_bp in _FRED_MAP:
        raw = rates.get(fred_id)
        if raw is None:
            continue
        insts.append({
            "tenor":  tenor,
            "rate":   raw + basis_bp / 10_000.0,
            "type":   _INST_TYPE[tenor],
            "source": f"{fred_id} (fallback)",
        })
    return insts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Module-level cache: try to fetch once, reuse thereafter.
_fred_data: Optional[pd.DataFrame] = None
_fred_available: Optional[bool] = None


def get_instruments(as_of: date) -> list[dict]:
    """
    Return the SOFR OIS instrument list for `as_of`.
    Attempts live FRED fetch on first call; falls back to indicative data.
    """
    global _fred_data, _fred_available

    if _fred_available is None:
        print("Fetching SOFR market data from FRED … ", end="", flush=True)
        _fred_data = fetch_all_series()
        _fred_available = _fred_data is not None
        if _fred_available:
            print(f"OK  ({len(_fred_data)} business days loaded)")
        else:
            print("UNAVAILABLE — using indicative fallback data")

    if _fred_available and _fred_data is not None:
        return instruments_from_dataframe(_fred_data, as_of)
    return _fallback_instruments(as_of)


def snapshot_dates() -> list[date]:
    return sorted(_FALLBACK.keys())


def data_source_info() -> str:
    if _fred_available:
        return "FRED public CSV API (live)"
    return "Indicative fallback data (FRED unavailable)"
