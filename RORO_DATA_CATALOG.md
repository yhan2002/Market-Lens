# Market Lens RORO Data Catalog

Status: **SHADOW / NOT LIVE**  
Audit cutoff: **2026-08-12**  
Web score window: **2026-01-07 ~ 2026-08-12 (NQ 150 trading days)**

이 카탈로그는 RORO 점수에 사용하거나 향후 연결할 원천의 식별자, 시점 계약,
가용성, 결측 처리와 알려진 한계를 고정한다. 값이 없거나 역사 시점에 알 수 없었던
입력은 50점이나 proxy로 대체하지 않는다.

## 1. Market and macro inputs

| Block | Signal | Local security / field | Unit | Source | Full coverage | 2026 exact availability | PIT status | Freshness / transform | Status and action |
|---|---|---|---|---|---|---:|---|---|---|
| Market Trend | NQ price, momentum, MA, drawdown | `daily_prices / NQ=F / adj_close` | index points | Yahoo continuous-futures proxy | 2006-08-10~2026-08-12, 5,035 rows | 150/150 | `CONDITIONAL_PRICE_NO_VINTAGE` | exact NQ date; signal at close `t`, apply `t+1`; log return/MA200/DD60 | Active SHADOW input. No roll, collateral, margin or vendor-vintage history |
| Equity Volatility | VIX level | `daily_prices / ^VIX / adj_close` | index points | Yahoo | 2006-08-07~2026-08-12, 5,036 rows | 150/150 | `CONDITIONAL_PRICE_NO_VINTAGE` | exact NQ date; `log(VIX)` | Active SHADOW input |
| Equity Volatility | VIX curve | `^VIX`, `^VIX3M / adj_close` | log ratio | Cboe-first/Yahoo fallback | VIX3M 2006-08-10~2026-08-12, 5,032 rows | 150/150 | `CONDITIONAL_PRICE_NO_VINTAGE` | no fill; `log(VIX/VIX3M)` | Active SHADOW input; row-level provider is not retained |
| Rates Volatility | MOVE level | `daily_prices / ^MOVE / adj_close` | index points | Yahoo | 2006-08-10~2026-08-12, 4,926 rows | 133/150 | `CONDITIONAL_PRICE_NO_VINTAGE` | exact NQ date; no fill; `log(MOVE)` | Active when present. Missing 2026-07-20~08-11, resumed 08-12 |
| Credit | HY OAS level/change | `macro_observations / HY_OAS / value` | percentage points | ALFRED `BAMLH0A0HYM2`; later live FRED vintages | 2023-08-08~2026-08-11, 973 vintage rows / 790 dates | loader 150/150 | `PIT_CAUSAL` from recorded vintage | `available_at` after observation; next NQ session; max 5 calendar-day stale; log level and 20D change | Active SHADOW input. Pre-2025-12 live-FRED history is not retroactively causal |
| FX / Carry | Yen strength | legacy `daily_prices / JPY=X` is blocked; approved `roro_price_observations / JPY=X / close` | JPY per USD | `YAHOO_CHART_V1` | legacy 5,190 rows; approved store not populated | 0/150 in model | `DATA_BLOCKED` | preserve raw UTC timestamp and IANA timezone; use the first NQ 17:00 New York cutoff after availability | Do not use current Sunday/Friday-shifted dates or back-map Sunday to Friday |
| Safe Haven | GLD relative to NQ | `roro_price_observations / GLD / close` | USD per share; relative return in model | `YAHOO_CHART_V1` | approved store not populated | 0/150 | `DATA_BLOCKED` | same strict next-NQ-session rule; `GLD20D - NQ20D`; no fill | `GLD / close` is frozen; download and model integration remain pending |
| Market Internals | S&P 500 breadth | `breadth_metrics / breadth_200dma` | fraction | current public membership snapshot | 2026-08-12 only, 1 row | 0/150 | `DATA_BLOCKED` | requires point-in-time membership with `available_at <= decision_at`, exact coverage 100% | Current-membership history is prohibited for historical OOS |

### Approved FX·Gold observation contract (not populated)

- Contract: `yahoo-session-v1-jpyx-gld-close`; source: `YAHOO_CHART_V1`.
- Allowed rows: `JPY=X / close`, `GLD / close` only.
- Separate append-only columns:
  `contract_version, source, ticker, raw_timestamp_utc, source_timezone,`
  `observation_session_date, available_at, close, adj_close, fetched_at`.
- `source_timezone` is required IANA metadata. All stored timestamps are aware UTC.
- A value is usable only at the first NQ 17:00 New York cutoff after `available_at`. It appears
  on that date only; there is no fill. If observations collide
  after a holiday, the latest available observation wins.
- Existing `daily_prices` remains unchanged. Until approved rows are downloaded and the optional
  observation DB is supplied, both blocks remain `DATA_BLOCKED`. Populated historical Yahoo rows
  are `CONDITIONAL_PRICE_SESSION_CAUSAL`, not vendor-vintage strict PIT.

## 2. Portfolio internal inputs

Source contract: holdings event snapshots from the private database, carried forward only until
the next observed change, combined with adjusted-close price proxies and `NQ=F`. Individual
names, quantities and weights are never written to the public SHADOW payload.

| Signal | Aggregate field | Unit | 2026 availability | PIT / quality contract | Risk direction | Known limitations |
|---|---|---|---:|---|---|---|
| Weighted correlation 20D | `portfolio_correlation_20` | -1~1 | 150/150 | `CONDITIONAL_EVENT_PIT`; minimum 10 return observations; minimum 85% history weight coverage | higher is riskier | Historical holdings do not retain true `available_at` |
| Correlation trajectory | `portfolio_correlation_trajectory_1d/5d/20d` | change | derived where lag exists | exact NQ-row difference; no fill | rising is deterioration | first lag rows are naturally unavailable |
| NQ beta 60D | `nq_beta_60` | multiple | 150/150 | NQ benchmark only; each return observation requires at least 85% weight coverage | higher is riskier | not interchangeable with legacy SPY beta |
| Concentration | `concentration_hhi`, `effective_positions`, Top3/Top5 | fraction / count | 150/150 | proxy weights from quantity x adjusted close | higher HHI / lower effective N is riskier | two dates use adjusted-close proxy weights |
| Portfolio breadth 20D | `breadth_above_ma20` | fraction | 150/150 | weighted participation; required component | lower is riskier | short-history sleeve can be excluded only below the frozen 15% cap |
| Portfolio breadth 50D | `breadth_above_ma50` | fraction | 150/150 | diagnostic only in v1 | lower is riskier | not part of the five-component score |
| Portfolio breadth 200D | `breadth_above_ma200` | fraction | 79/150, starts 2026-04-21 | trend checklist only | lower is riskier | insufficient warm-up for the v1 score |
| Negative participation 20D | `negative_participation_20` | fraction | 150/150 | strict pair with MA20 breadth | higher is riskier | missing either member blocks the participation component |
| Portfolio drawdown 60D | `portfolio_drawdown_60` | return | 150/150 | weighted proxy portfolio index | deeper loss is riskier | proxy return, not audited fund NAV |
| Cross-sectional dispersion 20D | `dispersion_20` | annualized dispersion | 150/150 | diagnostic only | no frozen monotone direction | excluded from the risk score |

Quality range and exceptions for the 150-day window:

- Current-price weight coverage: **96.0%~100.0%**
- Score-relevant history weight coverage: **89.46%~100.0%**
- Short-history sleeve exclusion: **14 days**
- Causal carry-forward of an incomplete one-name snapshot: **3 days**
- Adjusted-close proxy weighting: **2 days**

## 3. Sector rotation inputs

| Signal | Securities / benchmark | Coverage | Role | PIT status / limitation |
|---|---|---|---|---|
| Sector adjusted prices | XLB, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY | each 2016-08-05~2026-08-12; 150/150 in web window | RRG position, 5D heading, relative return/volatility/drawdown | causal next-bar calculation, but no immutable price vintage |
| Communication Services | XLC / SPY | 2018-06-19~2026-08-12; 150/150 in web window | same as above | 11-sector common history cannot predate XLC availability |
| Relative benchmark | SPY | 2006-08-07~2026-08-12 | sector-relative calculations only | must not replace NQ as the Risk Management benchmark |
| Market context | immutable RORO SHADOW | 2026-01-07~2026-08-12, 150 rows | sector risk appetite / beta-rank fit | exact-date only; no HMM posterior dependency |

## 4. Output availability and fail-closed rules

- `RORO-3AXIS-v1` has 150/150 score rows because Market Trend and Portfolio Internals are
  mandatory and complete, while Macro may run `AS_AVAILABLE` with explicit completeness.
- Missing USDJPY, Gold or historical PIT breadth does not become a neutral 50. The associated
  block remains `DATA_BLOCKED`, reducing completeness and confidence.
- The historical portfolio rows are labelled `CONDITIONAL_EVENT_PIT`; they are not promoted to
  strict PIT merely because the calculation itself is causal.
- Frozen q20 readiness is **INSUFFICIENT_EVIDENCE**: 113 exact complete feature-label rows and
  2 independent drawdown episodes versus the required 252 rows and 20 episodes. Its probability
  remains `null` and fitting is blocked.
- Local immutable stores contain 150 daily 3-axis RORO rows, 153 diagnostic RORO-EW rows and
  363 Sector RORO rows across 33 decision dates. The RORO-EW series remains `NO_SCORE` because
  only three of seven strict blocks are currently available.
- The external website may receive aggregate-only fields after explicit approval. It must not
  receive names, identifiers, quantities, individual weights or holdings files.

## 5. Data remediation order

1. Populate the approved `JPY=X / close` store with auditable raw timestamps and session dates.
2. Populate the approved `GLD / close` store with the same source, timezone and availability metadata.
3. Repair the 17-day 2026 MOVE gap without forward-filling observations.
4. Obtain licensed/PIT S&P membership and reconstruct exact historical breadth.
5. Persist holdings fingerprint, `holdings_date` and true `available_at` prospectively.
6. Add provider and immutable ingest-vintage metadata before describing price data as strict PIT.

This document is the human-readable authoritative catalog. A tabular export must reproduce the
same contracts and must not weaken any `DATA_BLOCKED` or PIT classification.
