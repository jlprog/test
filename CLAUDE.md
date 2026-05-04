# CLAUDE.md — jlprog/test

This repository is a quantitative-finance Python project for **interest rate curve
construction**. The `master` branch is nearly empty (only `.nojekyll` from a prior
GitHub Pages era). All substantive code lives on three independent feature branches,
each implementing a different rate curve model.

---

## Repository structure

```
master / gh-pages
└── .nojekyll                  (empty — legacy GitHub Pages artefact)

claude/build-sofr-curve-ZtVXf          ← SOFR OIS discount curve (script-based)
├── sofr_curve.py              SOFRCurve class
├── market_data.py             FRED API fetch + fallback fixtures
├── build_sofr_curve.py        Runner: prints tables, generates 3 PNG charts
├── sofr_curve_methodology.md  Full methodology paper
├── sofr_curves.png / sofr_forwards.png / sofr_evolution.png
└── .gitignore

claude/build-repo-curve-model-QFfOo    ← Repo-rate term structure (installable package)
├── pyproject.toml             Package metadata (name: repo-curve, Python ≥3.10)
├── repo_curve/
│   ├── __init__.py            Public API re-exports
│   ├── curve.py               RepoCurve class
│   ├── day_count.py           DayCount enum + year_fraction()
│   └── interpolation.py       Interpolation enum + interpolate_log_df()
├── tests/
│   ├── __init__.py
│   └── test_repo_curve.py     50 pytest unit tests
└── .gitignore

claude/treasury-rate-curve-QXSCb       ← Treasury yield curve (installable package)
├── requirements.txt           numpy>=1.24, pytest>=7.0
├── treasury_curve/
│   ├── __init__.py            Public API re-exports
│   ├── bootstrap.py           Bootstrapper + ParInstrument + BootstrapResult
│   ├── curve.py               TreasuryCurve public interface
│   ├── day_count.py           DayCount enum, date utils, parse_tenor()
│   └── monotone_convex.py     Hagan-West monotone convex interpolant
├── tests/
│   ├── __init__.py
│   ├── test_bootstrap.py
│   ├── test_curve.py
│   └── test_monotone_convex.py   49 pytest unit tests total
└── examples/
    └── build_curve.py         End-to-end usage example
```

---

## Branch purposes and status

| Branch | Model | Package? | Tests | Dependencies |
|--------|-------|----------|-------|--------------|
| `claude/build-sofr-curve-ZtVXf` | SOFR OIS discount curve | No (scripts) | None | numpy, scipy, matplotlib |
| `claude/build-repo-curve-model-QFfOo` | Repo financing rate curve | Yes (`repo-curve`) | 50 | stdlib only (math, enum) |
| `claude/treasury-rate-curve-QXSCb` | Treasury yield curve | Yes (`treasury_curve`) | 49 | numpy |

---

## Financial domain conventions

### Rates and units
- **All rates are expressed as decimals**, never percentages (e.g. `0.05` means 5%).
- **All maturities/tenors** are `datetime.date` objects or tenor strings (`"3M"`, `"2Y"`, `"1W"`, `"10D"`).
- **Discount factors** are in `[0, 1]` (present value of 1 unit paid at maturity).

### Day count conventions
Every branch defines a `DayCount` enum:

| Convention | Used in |
|-----------|---------|
| `ACT/360` | USD repo (default in `RepoCurve`), SOFR deposits |
| `ACT/365` | SOFR OIS fixed leg, Treasury bootstrapping |
| `ACT/ACT (ISDA)` | `repo_curve` general use |
| `ACT/ACT (ICMA)` | `treasury_curve` coupon bonds |
| `30/360` | `treasury_curve` utility |

### Interpolation
All three models default to **log-linear interpolation on discount factors**
(equivalent to piecewise-constant instantaneous forward rates, also called
"flat-forward"). This is the standard market practice because it:
- Guarantees positive discount factors between pillars.
- Prevents spurious arbitrage between adjacent tenors.

`repo_curve` also supports `LINEAR` (on rates) and `CUBIC_SPLINE` (natural spline).
`treasury_curve` uses the **Hagan-West (2008) monotone convex** method on top of
the bootstrapped pillars, which additionally guarantees non-negative forward rates.

### Compounding conventions
- **Simple** (money-market): `D = 1 / (1 + r * τ)` — default for repo/SOFR deposits.
- **Continuous**: `D = exp(−r * τ)` — default zero rates in `SOFRCurve` and `TreasuryCurve`.
- **Semi-annual / annual**: available via `compounding` parameter on `TreasuryCurve`.

### Bootstrap methodology
All three curves use **sequential bootstrapping** (pillar by pillar, in maturity order):

1. **Short end** (≤ 1Y or bills): closed-form inversion of the pricing formula.
   - Deposits / T-bills: direct formula, no root-finding needed.
   - Short OIS (single-period): `df = 1 / (1 + r * τ_act365)`.

2. **Long end** (> 1Y, multi-period):
   - **SOFR**: Brent root-find (`scipy.optimize.brentq`) for the discount factor
     that zeros the OIS NPV, using intermediate log-linear interpolation.
   - **Treasury**: closed-form algebra. Given the par coupon, all intermediate
     coupon cash-flow DFs come from the already-bootstrapped pillars; the terminal
     DF is solved directly.

---

## Module APIs

### `repo_curve` package

```python
from repo_curve import RepoCurve, DayCount, Interpolation, Compounding

curve = RepoCurve(
    settlement_date=date(2024, 1, 2),
    pillars=[
        (date(2024, 4,  2), 0.053),
        (date(2024, 7,  2), 0.051),
        (date(2025, 1,  2), 0.049),
    ],
    day_count=DayCount.ACT_360,          # default
    interpolation=Interpolation.LOG_LINEAR,  # default
    compounding=Compounding.SIMPLE,      # default
)

curve.discount_factor(date(2024, 6, 1))  # D(0, T)
curve.repo_rate(date(2024, 6, 1))        # spot rate r(0, T)
curve.zero_rate(date(2024, 6, 1))        # continuously-compounded zero
curve.forward_rate(start, end)           # r(T1, T2)
curve.forward_discount_factor(start, end)
curve.accrual_factor(start, end)
curve.implied_financing_cost(clean_price, dirty_price, start, end)
```

Key invariant: `pillar_dates` must be strictly after `settlement_date`; rates must
be non-negative. Pillars are sorted internally by maturity.

### `SOFRCurve` (script, no package)

```python
from sofr_curve import SOFRCurve, add_tenor

curve = SOFRCurve(
    valuation_date=date(2026, 1, 28),
    instruments=[
        {'tenor': '1D',  'rate': 0.043, 'type': 'deposit'},
        {'tenor': '3M',  'rate': 0.0428},   # type defaults to 'ois'
        {'tenor': '10Y', 'rate': 0.0435},
    ],
)

curve.discount_factor(d)    # log-linear interpolated DF
curve.zero_rate(d)          # continuously compounded (act/365.25)
curve.forward_rate(d1, d2)  # simply-compounded act/360
curve.par_swap_rate(d)      # OIS par rate
```

`market_data.py` provides `get_instruments(as_of_date)` which fetches from the
FRED public CSV API (no key required) and falls back to hardcoded fixtures when
offline.

### `treasury_curve` package

```python
from treasury_curve import TreasuryCurve, ParInstrument

instruments = [
    ParInstrument("3M",  0.054, date(2024, 4,  2), is_bill=True),
    ParInstrument("2Y",  0.048, date(2026, 1,  2)),
    ParInstrument("10Y", 0.043, date(2034, 1,  2)),
]
curve = TreasuryCurve(anchor=date(2024, 1, 2), instruments=instruments)

curve.zero_rate("5Y")                          # default: continuous
curve.zero_rate("5Y", compounding="semi-annual")
curve.discount_factor("10Y")
curve.forward_rate("2Y", "3Y")
curve.forward_rate("2Y", "3Y", compounding="simple")
curve.instantaneous_forward("5Y")
curve.par_rate("5Y")
curve.pillar_summary()                         # formatted debug table
```

`TenorOrDate` inputs accept tenor strings (`"5Y"`, `"3M"`), year-fraction floats,
or `datetime.date` objects. `parse_tenor()` converts strings to year fractions.

`ParInstrument.is_bill=True` triggers discount-basis pricing (`Price = 1 − d × n/360`).
Coupon bonds default to semi-annual frequency (`frequency=2`).

---

## Running tests

### `repo_curve` (pyproject.toml)
```bash
# From the repo-curve branch root
pip install -e ".[dev]"   # installs pytest
pytest                    # runs tests/test_repo_curve.py (50 tests)
```

### `treasury_curve` (requirements.txt)
```bash
# From the treasury-curve branch root
pip install -r requirements.txt
pytest                    # runs tests/ (49 tests across 3 files)
```

### SOFR scripts (no test suite)
```bash
pip install numpy scipy matplotlib
python build_sofr_curve.py   # prints tables + writes 3 PNGs
```

---

## Development workflow

### Branching convention
New work branches are named `claude/<feature-slug>-<short-id>`. Each branch is
self-contained — there are no merges back to `master`. Do not base new work on
`master` unless explicitly instructed; use the most relevant feature branch as a
starting point instead.

### Committing
Commit messages follow this structure:
```
<Short imperative summary>

<Bullet list of what was added/changed and why>

https://claude.ai/code/session_<id>
```

### Python version
`repo_curve` requires Python ≥ 3.10 (uses `list[dict]` type hints in signatures).
`treasury_curve` and the SOFR scripts are compatible with Python 3.9+.

### Code style
- Type hints on all public functions and class `__init__` signatures.
- Enums (`DayCount`, `Interpolation`, `Compounding`, `BoundaryCondition`) for all
  multi-valued options — never bare strings in the core math layer.
- No external dependencies in `repo_curve` core (pure stdlib + `math`).
- `numpy` is used in `treasury_curve` for array math; avoid it in `repo_curve`.
- Comments only where the WHY is non-obvious (algorithm references, hidden invariants).
- No docstrings on private helpers (`_bootstrap`, `_interp_log_df`, etc.).

### Adding a new instrument type
1. Extend the `DayCount` enum if a new day-count is needed.
2. Add a branch in the bootstrap loop / `_bootstrap_*` method for the new type.
3. Write a test that verifies: (a) the bootstrapped pillar DF is consistent with
   the quoted rate, and (b) `par_*_rate(maturity)` round-trips back to the input rate.

### Adding a new interpolation method
1. Add a value to the `Interpolation` enum.
2. Implement the branch in `interpolate_rate` / `interpolate_log_df`.
3. Cover with tests: monotonicity at nodes, correct value at pillar dates, extrapolation behaviour.

---

## Key financial invariants (must hold in tests)

- **DF at settlement = 1.0**: `curve.discount_factor(settlement_date) == 1.0`
- **Monotonically decreasing DFs**: longer tenors must have strictly smaller DFs (positive rates).
- **Par rate round-trip**: building a curve from par rates, then querying `par_rate(maturity)` must return the input rate within ≤ 1 bp.
- **Forward consistency**: `D(0,T2) / D(0,T1) == D(T1,T2)` by definition.
- **Non-negative forward rates**: enforced explicitly by the Hagan-West interpolant in `treasury_curve`.
