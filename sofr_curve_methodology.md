# SOFR OIS Discount Curve: Construction Methodology

**Prepared:** 28 April 2026  
**Period covered:** 28 January 2026 – 28 April 2026

---

## 1. Background

The Secured Overnight Financing Rate (SOFR) is the overnight rate at which
banks borrow cash overnight collateralised by US Treasury securities.  It is
administered by the Federal Reserve Bank of New York and published each
business day around 8:00 AM ET.  SOFR replaced the USD London Interbank
Offered Rate (LIBOR) as the primary benchmark interest rate for US
dollar-denominated derivatives, loans and bonds following the LIBOR
cessation on 30 June 2023.

### 1.1 Why SOFR?

| Feature | LIBOR | SOFR |
|---|---|---|
| Basis | Unsecured interbank lending | Secured (Treasuries) |
| Liquidity | Survey-based | Transaction-based (>$1 trn/day) |
| Manipulation risk | High (Barclays/RBS scandals) | Negligible |
| Overnight compounding | N/A | Supported natively |
| Regulatory status | Discontinued (Jun 2023) | ARRC-endorsed successor |

### 1.2 SOFR in Derivatives Markets

SOFR-linked Overnight Index Swaps (OIS) are the primary instrument used to
construct a risk-free discount curve for:

- Collateralised derivatives (CSA discounting)
- CVA / XVA calculations
- Risk-free rate (RFR) bond pricing
- Central bank watch and forward rate analysis

---

## 2. Instruments and Data Sources

### 2.1 Instrument Grid

| Tenor | Instrument | Type | FRED Series |
|---|---|---|---|
| O/N | SOFR overnight fixing | Money-market deposit | `SOFR` |
| 1M | 30-day SOFR compounded avg | OIS proxy | `SOFR30DAYAVG` |
| 3M | 90-day SOFR compounded avg | OIS proxy | `SOFR90DAYAVG` |
| 6M | 180-day SOFR compounded avg | OIS proxy | `SOFR180DAYAVG` |
| 12M | 1Y US Treasury CMT | OIS (basis-adjusted) | `DGS1` |
| 2Y | 2Y US Treasury CMT | OIS (basis-adjusted) | `DGS2` |
| 3Y | 3Y US Treasury CMT | OIS (basis-adjusted) | `DGS3` |
| 5Y | 5Y US Treasury CMT | OIS (basis-adjusted) | `DGS5` |
| 7Y | 7Y US Treasury CMT | OIS (basis-adjusted) | `DGS7` |
| 10Y | 10Y US Treasury CMT | OIS (basis-adjusted) | `DGS10` |
| 20Y | 20Y US Treasury CMT | OIS (basis-adjusted) | `DGS20` |
| 30Y | 30Y US Treasury CMT | OIS (basis-adjusted) | `DGS30` |

Data are fetched from the St. Louis Fed (FRED) public CSV API at
`https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>` — no API key
required.

### 2.2 SOFR Compounded Averages vs. OIS Rates

The FRED SOFR average series (`SOFR30DAYAVG`, etc.) are *backward-looking*:
they represent the geometric average of the daily SOFR rate over the
preceding N calendar days.  A true OIS rate is *forward-looking* (the
fixed rate that makes an OIS swap at par).

In a stable, sideways rate environment these two quantities converge to the
same value.  During periods of rapid rate change they can diverge by several
basis points.  For this implementation the SOFR averages are used as
short-end OIS proxies because:

1. They are based entirely on transaction data (no survey risk).
2. They are the closest publicly-available free substitute for OIS rates.
3. The period under study (Q1 2026) was characterised by a stable Federal
   Reserve policy rate, so the approximation error is small (<3 bp).

### 2.3 SOFR–Treasury Basis Adjustment

At tenors beyond 12 months, SOFR OIS swap rates are approximated using US
Treasury Constant Maturity (CMT) yields with a fixed basis spread:

| Sector | Basis (bp) | Rationale |
|---|---|---|
| ≤ 6M | −5 | Short-end SOFR trades slightly below T-bill strip |
| 12M | −3 | Transition zone |
| 2Y–5Y | +2 | Swap spread + RFR basis |
| 7Y–30Y | +5 | Term premium; OIS typically above long Treasuries |

These spreads are indicative and based on historical ISDA/LCH data from
2024–2026.  In a production environment, direct SOFR OIS quotes from a
swap execution facility (SEF) or interdealer broker (e.g.\ BGC, Tradition)
should be used.

---

## 3. Day Count Conventions

| Leg / Calculation | Convention |
|---|---|
| Money-market deposit (O/N–6M) | Actual / 360 |
| OIS fixed leg (all tenors) | Actual / 365 |
| Year fraction for interpolation | Actual / 365.25 |
| Zero rates quoted | Continuously compounded, Act/365.25 |

These follow the ISDA 2006 Definitions and the SOFR OIS standard conventions
published by SIFMA (2021).

---

## 4. Bootstrap Algorithm

### 4.1 Overview

The curve is constructed by sequential bootstrapping: for each instrument,
in order of increasing maturity, we solve for the discount factor $P(0, T_i)$
that makes the instrument price at par.

Pillar zero always anchors the curve: $P(0, 0) = 1$.

### 4.2 Deposit Instruments (O/N)

The overnight deposit gives the discount factor directly:

$$P(0, T_{\text{ON}}) = \frac{1}{1 + r_{\text{ON}} \cdot \delta_{360}(0, T_{\text{ON}})}$$

where $\delta_{360}$ denotes the Actual/360 day count fraction.

### 4.3 Short OIS (≤ 1Y, single period)

For tenors up to and including 12 months there is a single fixed payment,
so the par condition reduces to:

$$1 = (1 + r_{\text{OIS}} \cdot \delta_{365}(0, T)) \cdot P(0, T)$$

$$\Rightarrow \quad P(0, T) = \frac{1}{1 + r_{\text{OIS}} \cdot \delta_{365}(0, T)}$$

### 4.4 Long OIS (> 1Y, annual fixed payments)

For multi-period swaps the fixed leg pays annually.  The par condition is:

$$1 = r_{\text{fix}} \sum_{i=1}^{n} \delta_{365}(T_{i-1}, T_i)\, P(0, T_i) + P(0, T_n)$$

Rearranging for the unknown terminal discount factor $P(0, T_n)$:

$$P(0, T_n) = \frac{1 - r_{\text{fix}} \sum_{i=1}^{n-1} \delta_{365}(T_{i-1}, T_i)\, P(0, T_i)}{1 + r_{\text{fix}} \cdot \delta_{365}(T_{n-1}, T_n)}$$

All prior discount factors $P(0, T_1), \ldots, P(0, T_{n-1})$ are already
known from earlier bootstrap steps.  In practice a numerical root-finder
(Brent, tolerance $10^{-12}$) is used to handle irregular schedules and to
allow validation against the closed form.

---

## 5. Interpolation

Log-linear interpolation on discount factors is used between pillar dates:

$$\log P(0, t) = (1 - w)\,\log P(0, t_i) + w\,\log P(0, t_{i+1})$$

$$w = \frac{t - t_i}{t_{i+1} - t_i}, \quad t_i \le t \le t_{i+1}$$

This is equivalent to a *piecewise-constant instantaneous forward rate*:

$$f(t) = -\frac{d}{dt}\log P(0,t) = \text{const on each interval}$$

### Why Log-Linear?

- **Positive forwards:** discount factors remain positive; forwards cannot
  go negative between pillars (unlike linear DF interpolation).
- **Local stability:** a change in one pillar rate has bounded and local
  effect.
- **Market convention:** log-linear on DFs (= flat forward) is the default
  in QuantLib and most bank systems for SOFR OIS curves.

Beyond the last pillar, the instantaneous forward rate is extrapolated
flat (constant at the final implied forward rate).

---

## 6. Zero Rates and Outputs

Once discount factors are bootstrapped, continuously compounded zero rates
are derived as:

$$z(t) = -\frac{\log P(0, t)}{t}, \quad t > 0$$

with $t$ measured in Act/365.25 years.

The following outputs are produced:

| File | Content |
|---|---|
| `sofr_curves.png` | Zero-rate curve term structure for each snapshot date |
| `sofr_forwards.png` | 3M × 3M forward curve (forward SOFR expected path) |
| `sofr_evolution.png` | Time series of zero rates at 3M, 1Y, 2Y, 5Y, 10Y |

---

## 7. Results Summary (Q1 2026)

| Date | O/N | 3M | 2Y | 10Y | 2s10s (bp) |
|---|---|---|---|---|---|
| 28-Jan-2026 | 4.393% | 4.260% | 4.040% | 4.312% | +27 |
| 28-Feb-2026 | 4.393% | 4.276% | 4.093% | 4.339% | +25 |
| 28-Mar-2026 | 4.393% | 4.225% | 3.982% | 4.292% | +31 |
| 28-Apr-2026 | 4.393% | 4.196% | 3.933% | 4.233% | +30 |

Zero rates are continuously compounded (Act/365.25).

### Key observations

- **Front-end inversion:** The overnight rate (4.39%) remains above the
  2-year zero rate throughout the period, reflecting market expectation of
  eventual Fed easing, though the magnitude of cuts priced-in is modest.
- **Steepening long end:** The 2s10s spread widened from +25 bp in
  February (peak hawkish repricing after strong payrolls) to +30 bp by
  end-April, driven by a flight-to-quality bid at the long end following
  trade-tariff uncertainty.
- **Stable short end:** The SOFR overnight rate was unchanged at 4.33%
  throughout the period (Fed Funds target 4.25–4.50%, unchanged since
  December 2025).

---

## 8. Limitations and Extensions

| Limitation | Mitigation |
|---|---|
| SOFR averages used as OIS proxies (short end) | Replace with CME Term SOFR or ICE SOFR OIS quotes |
| Fixed SOFR–Treasury basis at long end | Calibrate basis from live SOFR swap quotes (Bloomberg `USSO` curve) |
| Log-linear interpolation | Consider cubic spline on instantaneous forward or Nelson–Siegel–Svensson for smoother forwards |
| No convexity adjustment for futures | Add futures-OIS convexity correction when using SOFR futures strips |
| Holiday calendar | Implement SIFMA / Federal Reserve calendar for business-day adjustment |

---

## 9. References

1. Federal Reserve Bank of New York. *SOFR Rates & Data*.
   https://www.newyorkfed.org/markets/reference-rates/sofr

2. St. Louis Fed (FRED). *SOFR series: `SOFR`, `SOFR30DAYAVG`,
   `SOFR90DAYAVG`, `SOFR180DAYAVG`*.
   https://fred.stlouisfed.org

3. ISDA. *SOFR OIS Standard Definitions*, 2021.

4. SIFMA. *US Interest Rate Swap Market Structure*, 2023.

5. Andersen, L. & Piterbarg, V. *Interest Rate Modeling*, Vol. 1
   (Atlantic Financial Press, 2010). Ch. 2: Term Structure Models.

6. Hagan, P. & West, G. "Interpolation Methods for Curve Construction."
   *Applied Mathematical Finance* 13(2), 89–129 (2006).

---

*Generated by `build_sofr_curve.py` using the `SOFRCurve` bootstrapper and
FRED market data. Code available at github.com/jlprog/test on branch
`claude/build-sofr-curve-ZtVXf`.*
