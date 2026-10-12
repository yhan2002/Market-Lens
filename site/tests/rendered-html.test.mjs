import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  applyBreadthRepairs,
  BREADTH_BASE_SERIES,
  BREADTH_REPAIR_SERIES,
  canonicalRoroProvenance,
  isRoroShadowSnapshot,
  roroHistoryFallbackSeries,
  roroProvenanceHash,
  trailingDifference,
  verifyRoroProvenance,
} from "../lib/roro.ts";
import {
  CURRENT_ATTENTION_THRESHOLD_MODEL,
  CURRENT_SECTOR_TURNING_MODEL,
  CURRENT_SECTOR_TURNING_SERIES,
  isSectorTurningSnapshot,
  sectorTurningProvenanceHash,
  verifySectorTurningProvenance,
} from "../lib/sector-turning.ts";

const testRoroComponents = [
  "credit", "rates_volatility", "equity_volatility", "fx_carry", "safe_haven",
  "equity_trend", "market_breadth", "portfolio_correlation", "portfolio_beta",
  "portfolio_concentration", "portfolio_participation", "portfolio_drawdown",
];

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the protected Risk Management entry page", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Market Lens Risk Management<\/title>/i);
  assert.match(html, /Market Lens Risk Management/);
  assert.match(html, /접근 코드/);
  assert.doesNotMatch(html, /Your site is taking shape|Starter Project/);
});

test("wires the three-axis RORO history without replacing policy history", async () => {
  const [dashboard, route, contract, schema] = await Promise.all([
    readFile(new URL("../app/DashboardApp.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/api/roro/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../lib/roro.ts", import.meta.url), "utf8"),
    readFile(new URL("../db/schema.ts", import.meta.url), "utf8"),
  ]);
  assert.match(dashboard, /Market Lens MARKET LENS/);
  assert.match(dashboard, /위험 점수/);
  assert.match(dashboard, /\/api\/roro\?date=/);
  assert.match(dashboard, /portfolio_correlation_20/);
  assert.match(dashboard, /isRoroShadowSnapshot\(roroCandidate\)/);
  assert.match(dashboard, /snapshot\.schema_version === 3 \|\| snapshot\.schema_version === 4 \? snapshot\.payload\.policy : null/);
  assert.match(dashboard, /상세 진단/);
  assert.doesNotMatch(dashboard, /20일 내 큰 하락 위험/);
  assert.match(dashboard, /위험 변화/);
  assert.match(dashboard, /판단 일치도/);
  assert.match(dashboard, /value\.axis_agreement/);
  assert.match(dashboard, /value\.turn_evidence/);
  assert.match(dashboard, /세부 신호/);
  assert.match(dashboard, /RORO_COMPONENTS\.map/);
  assert.match(dashboard, /score_trajectory_20d/);
  assert.match(route, /ACTIVE_RORO_SERIES/);
  assert.match(route, /immutable RORO history conflict/);
  assert.match(route, /candidate\.length === 1/);
  assert.match(route, /series override requires upload authorization/);
  assert.match(route, /payload\.schema_version !== 4/);
  assert.match(route, /await verifyRoroProvenance\(snapshot\)/);
  assert.match(route, /env\.DB\.batch\(prices\.map/);
  assert.match(contract, /schema_version: 1/);
  assert.match(contract, /schema_version: 2/);
  assert.match(contract, /schema_version: 3/);
  assert.match(contract, /schema_version: 4/);
  assert.match(contract, /const SERIES_BY_SCHEMA =/);
  assert.match(contract, /shadow-r3"/);
  assert.match(contract, /shadow-r4-q20"/);
  assert.match(contract, /shadow-r5-q20-policy"/);
  assert.match(contract, /shadow-r6-q20-policy-fxgold"/);
  assert.match(contract, /shadow-r7-credit-move"/);
  assert.match(contract, /value\.series_version === SERIES_BY_SCHEMA\[1\]/);
  assert.match(contract, /value\.series_version === SERIES_BY_SCHEMA\[2\]/);
  assert.match(contract, /value\.series_version === SERIES_BY_SCHEMA\[3\]/);
  assert.match(contract, /SERIES_BY_SCHEMA\[4\]\.includes\(value\.series_version/);
  assert.match(contract, /containsForbiddenIdentity\(value\.payload\)/);
  assert.match(contract, /!isoTimestamp\(value\.generated_at\)/);
  assert.match(contract, /FORBIDDEN_IDENTITY_KEYS\.has\(normalized\)/);
  assert.match(contract, /export type RoroShadowSnapshot = RoroShadowSnapshotV1 \| RoroShadowSnapshotV2 \| RoroShadowSnapshotV3 \| RoroShadowSnapshotV4/);
  assert.match(contract, /model_versions: RoroModelVersionsV1 & \{ q20: string \}/);
  assert.match(contract, /payload: RoroPayloadV1 & \{ q20: RoroQ20 \}/);
  assert.match(contract, /model_versions: RoroModelVersionsV1 & \{ q20: string; policy: string \}/);
  assert.match(contract, /payload: RoroPayloadV1 & \{ q20: RoroQ20; policy: RoroPolicy \}/);
  assert.match(contract, /price_observations: "yahoo-session-v1-jpyx-gld-close"/);
  assert.match(contract, /value\.model_versions\.price_observations === PRICE_OBSERVATIONS_CONTRACT/);
  assert.match(contract, /onlyKeys\(value, Q20_KEYS\)/);
  assert.match(contract, /value\.model_version !== "roro-q20-logit-v1-frozen8"/);
  assert.match(contract, /value\.horizon_trading_days !== 20/);
  assert.match(contract, /value\.drawdown_threshold !== -0\.05/);
  assert.match(contract, /value\.minimum_complete_labels !== 252/);
  assert.match(contract, /value\.minimum_independent_events !== 20/);
  assert.match(contract, /value\.probability !== null/);
  assert.match(contract, /value\.model_versions\.q20 === value\.payload\.q20\.model_version/);
  assert.match(contract, /onlyKeys\(value, POLICY_KEYS\)/);
  assert.match(contract, /value\.model_version !== "roro-policy-v1-score-fast3of8-confirm3-shadow"/);
  assert.match(contract, /integerBetween\(value\.score_based_long, 25, 100\)/);
  assert.match(contract, /integerBetween\(value\.raw_target_long, 25, 100\)/);
  assert.match(contract, /integerBetween\(value\.recommended_long, 25, 100\)/);
  assert.match(contract, /value\.fast_trigger_count !== active\.length/);
  assert.match(contract, /value\.fast_available_count !== FAST_CONDITION_KEYS\.length - unavailable\.length/);
  assert.match(contract, /active\.some\(\(key\) => unavailable\.includes\(key\)\)/);
  assert.match(contract, /Math\.max\(25, value\.score_based_long - 20\)/);
  assert.match(contract, /value\.payload\.policy\.score_based_long === scoreBasedLong\(value\.payload\.market_risk_score\)/);
  assert.match(contract, /value\.recommended_long <= value\.raw_target_long/);
  assert.match(contract, /value\.payload\.recommended_long === value\.payload\.policy\.recommended_long/);
  assert.match(contract, /shadow-r15-yahoo-usdjpy-fast/);
  assert.match(contract, /shadow-r17-exact-market-inputs/);
  assert.match(contract, /shadow-r18-history-from-20251008/);
  assert.match(contract, /shadow-r21-breadth-correction/);
  assert.match(contract, /unavailable_fast_conditions\.includes\("portfolio_correlation"\)/);
  assert.match(contract, /!value\.payload\.policy\.active_fast_conditions\.includes\("portfolio_correlation"\)/);
  assert.match(contract, /history_weight_coverage_60d/);
  assert.match(contract, /const COMPONENT_NAMES =/);
  assert.match(contract, /payload\.available_components/);
  assert.match(contract, /payload\[`\$\{component\}_contribution`\]/);
  assert.match(schema, /roroShadowSnapshots/);
  assert.match(schema, /uq_roro_shadow_series_date_v1/);
});

test("computes causal differences without bridging missing values", () => {
  const first = trailingDifference([0, 1, 4, 9]);
  assert.deepEqual(first, [null, 1, 3, 5]);
  assert.deepEqual(trailingDifference(first), [null, null, 2, 2]);
  assert.deepEqual(trailingDifference([0, 1, null, 9]), [null, 1, null, null]);
});

test("compares 40 Market Lens risk scores with NASDAQ levels and acceleration", async () => {
  const [dashboard, route, styles] = await Promise.all([
    readFile(new URL("../app/DashboardApp.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/api/roro/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);

  assert.match(dashboard, /function RiskScoreHistory/);
  assert.match(dashboard, /function RiskAccelerationHistory/);
  assert.match(dashboard, /\/api\/roro\?history=40/);
  assert.match(dashboard, /위험 점수 추이 vs 나스닥100/);
  assert.match(dashboard, /위험·나스닥100 변화 가속도/);
  assert.match(dashboard, /trailingDifference\(riskChanges\)/);
  assert.match(dashboard, /rawRiskAccelerations\.slice\(previousIndex \+ 1, index\)\.every\(finite\)/);
  assert.match(dashboard, /risk-history-transition/);
  assert.match(dashboard, /nasdaqAcceleration\(data\)/);
  assert.match(dashboard, /나스닥100 · 전일 대비/);
  assert.match(dashboard, /차트 기간 누적/);
  assert.doesNotMatch(dashboard, /NASDAQ · NQ/);
  assert.match(dashboard, /new Set\(versions\)\.size === 1/);
  assert.match(dashboard, /<RiskAccelerationHistory points=\{roroHistory\}/);
  assert.match(dashboard, /risk-history-score/);
  assert.match(dashboard, /risk-history-nasdaq/);
  assert.match(dashboard, /benchmark_close/);
  assert.match(dashboard, /benchmarkBase - 1/);
  assert.match(dashboard, /benchmarkFloor = -benchmarkCeiling/);
  assert.match(dashboard, /risk-history-midline/);
  assert.match(dashboard, /risk-history-layout/);
  assert.match(route, /historyLimit > 40/);
  assert.match(route, /rows\.reverse\(\)/);
  assert.match(route, /roroHistoryFallbackSeries/);
  assert.match(route, /new Map\(rows\.map/);
  assert.match(route, /benchmarkPrices/);
  assert.match(route, /benchmark_close/);
  assert.match(dashboard, /모델 개정 이력을 연결/);
  assert.match(dashboard, /risk-history-stats/);
  assert.match(dashboard, /risk-history-footer/);
  assert.match(styles, /\.risk-history-footer\s*\{[^}]*flex-direction:row/);
  assert.doesNotMatch(styles, /grid-template-columns:minmax\(0,1fr\) 128px/);
  assert.match(styles, /\.risk-history-panel/);
  assert.match(styles, /grid-template-columns:repeat\(2,minmax\(0,1fr\)\)/);
  assert.match(styles, /grid-column:1\/-1/);
  assert.match(styles, /#55bfff/);
  assert.match(styles, /#f0b84b/);
  assert.match(styles, /\.risk-acceleration-panel \.risk-history-transition/);
});

test("derives displayed risk changes from immutable web history", async () => {
  const dashboard = await readFile(new URL("../app/DashboardApp.tsx", import.meta.url), "utf8");

  assert.match(dashboard, /history\.findIndex\(\(item\) => item\.snapshot\.as_of === snapshot\.as_of\)/);
  assert.match(dashboard, /value\.market_risk_score - history\[historyIndex - lag\]\.snapshot\.payload\.market_risk_score/);
  assert.match(dashboard, /1일 \{point\(trajectory1\)\}/);
});

test("fills current Market Lens history from newest compatible series first", () => {
  const active = "roro-3axis-v1-w756-m252-p126-u20__roro-phase-v1__p-c85-isqc__shadow-r19-holdings-history-recalc";
  const fallback = roroHistoryFallbackSeries(active);
  assert.match(fallback[0], /shadow-r18-history-from-20251008$/);
  assert.match(fallback.at(-1), /shadow-r6-q20-policy-fxgold$/);
  assert.deepEqual(roroHistoryFallbackSeries("unknown"), []);
});

test("overlays late breadth only on matching dates without hiding a newer normal snapshot", async () => {
  const original = [{ asOfKey: 20260924, version: "base" }, { asOfKey: 20260923, version: "base" }];
  const repairs = [{ asOfKey: 20260923, version: "repair" }, { asOfKey: 20260921, version: "repair" }];
  assert.deepEqual(applyBreadthRepairs(original, repairs), [original[0], repairs[0]]);
  assert.equal(original[1].version, "base");
  const route = await readFile(new URL("../app/api/roro/route.ts", import.meta.url), "utf8");
  assert.match(route, /requestedSeries === null && series === BREADTH_BASE_SERIES/);
  assert.match(route, /breadth repair requires an original missing-breadth snapshot/);
  assert.ok(!roroHistoryFallbackSeries(BREADTH_BASE_SERIES).includes(BREADTH_REPAIR_SERIES));
});

test("uses the fixed cross-language RORO provenance encoding", () => {
  const vector = {
    int: 1,
    float: 1.0,
    negative_zero: -0.0,
    korean: "리스크",
    nested: [true, null, [2.5, "끝"]],
  };
  const expected = "o5{s5:666c6f6174d3ff0000000000000s3:696e74d3ff0000000000000" +
    "s6:6b6f7265616es9:eba6acec8aa4ed81acs13:6e656761746976655f7a65726f" +
    "d8000000000000000s6:6e6573746564a3[b1na2[d4004000000000000s3:eb819d]]}";
  assert.equal(canonicalRoroProvenance(vector), expected);
  assert.equal(canonicalRoroProvenance(1), canonicalRoroProvenance(1.0));
  assert.equal(canonicalRoroProvenance(-0.0), "d8000000000000000");
  assert.equal(createHash("sha256").update(expected).digest("hex"),
    "b64e38bcfa88f0e3d6df4f7a6aa48eff0f84d1ff8c59a2f3906b15f37e3cf51b");
});

test("uses the same fixed encoding for public Sector aggregate provenance", () => {
  const material = {
    series_version: "sector-series",
    date: "2026-01-07",
    input_cutoff: "2026-01-07",
    model_versions: { sector_roro: "sector-model", roro: "roro-series" },
    payload: {
      alpha: 10.0,
      rows: [{ sector: "XLK", score: -0.0 }],
      korean: "순환매",
    },
  };
  const actual = createHash("sha256").update(canonicalRoroProvenance(material)).digest("hex");
  assert.equal(actual, "bd779f950a535012441102bf5e1df25650a4a1f9bb497cb190a05b08ec6b93c6");
});

test("accepts only the frozen schema v4 price-observation contract", async () => {
  const snapshot = {
    schema_version: 4,
    series_version: "roro-3axis-v1-w756-m252-p126-u20__roro-phase-v1__p-c85-isqc__shadow-r6-q20-policy-fxgold",
    as_of: "2026-08-14",
    generated_at: "2026-08-14T00:00:00Z",
    input_cutoff: "2026-08-14",
    provenance_hash: "0".repeat(64),
    model_versions: {
      roro: "roro-ew-v1-w756-m252-u20",
      axis: "roro-3axis-v1-w756-m252-p126-u20",
      portfolio: "portfolio-internals-nq-v2-c85-incomplete-snapshot-qc",
      phase: "roro-phase-v1",
      q20: "roro-q20-logit-v1-frozen8",
      policy: "roro-policy-v1-score-fast3of8-confirm3-shadow",
      price_observations: "yahoo-session-v1-jpyx-gld-close",
    },
    payload: {
      credit_score: null,
      rates_volatility_score: null,
      equity_volatility_score: null,
      fx_carry_score: null,
      safe_haven_score: null,
      equity_trend_score: null,
      breadth_risk_percentile: null,
      portfolio_correlation_score: null,
      portfolio_beta_score: null,
      portfolio_concentration_score: null,
      portfolio_participation_score: null,
      portfolio_drawdown_score: null,
      ...Object.fromEntries(testRoroComponents.flatMap((component) => [
        [`${component}_contribution`, null],
        [`${component}_score_trajectory_1d`, null],
        [`${component}_score_trajectory_5d`, null],
        [`${component}_score_trajectory_20d`, null],
      ])),
      market_risk_score: 50,
      risk_regime: "YELLOW",
      recommended_long: 75,
      trajectory_1d: null,
      trajectory_5d: null,
      trajectory_20d: null,
      macro_risk_score: null,
      market_trend_score: 50,
      portfolio_internal_score: 50,
      confidence: 50,
      data_completeness: 50,
      axis_disagreement: 50,
      axis_agreement: 50,
      data_status: "PARTIAL",
      trend_phase: "TRANSITION",
      turn_evidence: [],
      checklist_available_count: 0,
      bottom_confirmation_count: 0,
      top_confirmation_count: 0,
      portfolio_correlation_20: null,
      portfolio_correlation_trajectory_5d: null,
      nq_beta_60: null,
      effective_positions: null,
      breadth_above_ma20: null,
      negative_participation_20: null,
      portfolio_drawdown_60: null,
      price_weight_coverage: null,
      history_weight_coverage_20d: null,
      history_weight_coverage_60d: null,
      data_quality: null,
      risk_drivers: [],
      relief_drivers: [],
      available_components: testRoroComponents,
      component_risk_drivers: ["credit"],
      component_relief_drivers: ["equity_trend"],
      q20: {
        model_version: "roro-q20-logit-v1-frozen8",
        status: "INSUFFICIENT_EVIDENCE",
        probability: null,
        fit_allowed: false,
        horizon_trading_days: 20,
        drawdown_threshold: -0.05,
        feature_rows: 100,
        complete_feature_labels: 80,
        independent_events: 5,
        minimum_complete_labels: 252,
        minimum_independent_events: 20,
      },
      policy: {
        model_version: "roro-policy-v1-score-fast3of8-confirm3-shadow",
        score_based_long: 75,
        raw_target_long: 75,
        recommended_long: 75,
        fast_status: "INCOMPLETE",
        fast_trigger_count: 0,
        fast_available_count: 6,
        active_fast_conditions: [],
        unavailable_fast_conditions: ["jpy_3d_drop", "portfolio_correlation"],
        phase: "AT_TARGET",
        episode_status: "INACTIVE",
        clear_streak: 0,
        primary_driver: "BALANCED SIGNALS",
      },
    },
  };
  assert.equal(isRoroShadowSnapshot(snapshot), true);
  assert.equal(await verifyRoroProvenance(snapshot), false);
  const verified = { ...snapshot, provenance_hash: await roroProvenanceHash(snapshot) };
  assert.equal(await verifyRoroProvenance(verified), true);
  const repair = {
    ...snapshot,
    series_version: BREADTH_REPAIR_SERIES,
    model_versions: { ...snapshot.model_versions, market_breadth: "external-breadth-v1-ndth-200dma" },
    payload: { ...snapshot.payload, breadth_risk_percentile: 50, market_data_status: "FULL" },
  };
  assert.equal(isRoroShadowSnapshot(repair), true);
  assert.equal(isRoroShadowSnapshot({ ...repair, payload: { ...repair.payload, breadth_risk_percentile: null } }), false);
  assert.equal(isRoroShadowSnapshot({ ...repair, payload: { ...repair.payload, market_data_status: "NQ_ONLY" } }), false);
  assert.equal(await verifyRoroProvenance({ ...repair, provenance_hash: await roroProvenanceHash(repair) }), true);
  assert.equal(isRoroShadowSnapshot({ ...snapshot, series_version: `${snapshot.series_version}-other` }), false);
  assert.equal(isRoroShadowSnapshot({
    ...snapshot,
    model_versions: { ...snapshot.model_versions, roro: "other" },
  }), false);
  assert.equal(isRoroShadowSnapshot({
    ...snapshot,
    model_versions: { ...snapshot.model_versions, price_observations: "other" },
  }), false);
  assert.equal(isRoroShadowSnapshot({
    ...snapshot,
    payload: {
      ...snapshot.payload,
      policy: {
        ...snapshot.payload.policy,
        fast_available_count: 7,
        unavailable_fast_conditions: ["portfolio_correlation"],
      },
    },
  }), true);
});

test("shows the schema v3 RORO policy fold while keeping prior schemas readable", async () => {
  const [dashboard, styles] = await Promise.all([
    readFile(new URL("../app/DashboardApp.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);
  assert.match(dashboard, /점수 기준 LONG/);
  assert.match(dashboard, /단기 위험 조정/);
  assert.match(dashboard, /조정 후 \{policy\.raw_target_long\}%/);
  assert.match(dashboard, /최종 LONG/);
  assert.match(dashboard, /발생 \{policy\.fast_trigger_count\}\/3/);
  assert.match(dashboard, /확인 가능 \{policy\.fast_available_count\}\/8/);
  assert.match(dashboard, /policy\.active_fast_conditions\.map\(\(key\) => FAST_SHORT_LABEL\[key\] \?\? key\)/);
  assert.match(dashboard, /policy\.unavailable_fast_conditions\.map\(\(key\) => FAST_SHORT_LABEL\[key\] \?\? key\)/);
  assert.match(dashboard, /안정 확인 \{policy\.clear_streak\}\/3일/);
  assert.match(styles, /\.roro-policy-console/);
  assert.match(styles, /\.roro-policy-fast\.triggered/);
  assert.match(styles, /\.roro-policy-fast\.incomplete/);
});

test("wires immutable aggregate-only Sector RORO history with legacy fallback", async () => {
  const [dashboard, route, contract, schema, environment] = await Promise.all([
    readFile(new URL("../app/DashboardApp.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/api/sector-roro/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../lib/sector-roro.ts", import.meta.url), "utf8"),
    readFile(new URL("../db/schema.ts", import.meta.url), "utf8"),
    readFile(new URL("../cloudflare-env.d.ts", import.meta.url), "utf8"),
  ]);
  assert.match(dashboard, /SECTOR MARKET LENS/);
  assert.match(dashboard, /\/api\/sector-roro\?date=/);
  assert.match(dashboard, /sectorRoroCandidate\.every\(isSectorRoroShadowSnapshot\)/);
  assert.match(dashboard, /const lastSectorRoro = sectorRoroHistory\.at\(-1\)/);
  assert.match(dashboard, /lastSectorRoro && lastSectorRoro\.as_of <= nextSnapshot\.as_of/);
  assert.match(dashboard, /AS OF \$\{sectorRoro\.as_of\}/);
  assert.match(dashboard, /sectorRoro \? <>/);
  assert.match(dashboard, /<SectorTable rows=\{snapshot\.sector\.rows\}/);
  assert.match(dashboard, /SELECT/);
  assert.match(dashboard, /WATCH/);
  assert.match(dashboard, /AVOID/);
  assert.match(route, /ACTIVE_SECTOR_RORO_SERIES/);
  assert.match(route, /isCurrentSectorRoroShadowSnapshot\(payload\)/);
  assert.match(route, /await verifySectorRoroProvenance\(snapshot\)/);
  assert.match(route, /lte\(sectorRoroShadowSnapshots\.asOfKey, requestedKey\)/);
  assert.match(route, /historyParam !== "2"/);
  assert.match(route, /limit\(historyParam === null \? 1 : 2\)/);
  assert.match(route, /immutable Sector RORO history conflict/);
  assert.match(route, /mode === "upload"/);
  assert.match(route, /candidate\.length === 1/);
  assert.match(route, /series override requires upload authorization/);
  assert.match(route, /dates: 1/);
  assert.doesNotMatch(route, /env\.DB\.batch/);
  assert.match(contract, /onlyKeys\(value/);
  assert.match(contract, /CURRENT_SECTOR_RORO_SCHEMA = 2/);
  assert.match(contract, /value\.series_version === CURRENT_SECTOR_RORO_SERIES/);
  assert.match(contract, /shadow-r21-breadth-correction/);
  assert.match(contract, /models\.sector_roro === CURRENT_SECTOR_RORO_MODEL/);
  assert.match(contract, /models\.roro === CURRENT_RORO_SERIES/);
  assert.match(contract, /payload\.alpha === 10 && payload\.top_k === 3/);
  assert.match(contract, /payload\.context_type === CURRENT_CONTEXT_TYPE/);
  assert.match(contract, /value\.schema_version === 1/);
  assert.match(contract, /canonicalRoroProvenance\(material\)/);
  assert.match(contract, /SECTOR_NAMES/);
  assert.match(contract, /payload\.training_end < value\.as_of/);
  assert.match(contract, /MODEL_TOP3/);
  assert.match(contract, /parsed\.toISOString\(\)\.slice\(0, 19\) === value\.slice\(0, 19\)/);
  assert.match(contract, /rows\.length === SECTOR_RORO_TICKERS\.length/);
  assert.match(contract, /new Set\(rows\.map\(\(row\) => row\.sector\)\)/);
  assert.match(contract, /value\.as_of >= "2026-01-07"/);
  assert.doesNotMatch(contract, /holding_name|quantity|isin|positions/);
  assert.match(schema, /sectorRoroShadowSnapshots/);
  assert.match(schema, /uq_sector_roro_shadow_series_date_v1/);
  assert.match(environment, /ACTIVE_SECTOR_RORO_SERIES/);
});

test("keeps the past-date UI and dated snapshot API wired together", async () => {
  const [dashboard, route] = await Promise.all([
    readFile(new URL("../app/DashboardApp.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/api/snapshot/route.ts", import.meta.url), "utf8"),
  ]);

  assert.match(dashboard, /type="date"/);
  assert.match(dashboard, /\/api\/snapshot\?date=/);
  assert.match(dashboard, /선택한 날짜 이전에 저장된 데이터가 없습니다/);
  assert.match(route, /snapshotKey/);
  assert.match(route, /ACTIVE_POLICY_SERIES/);
  assert.match(route, /LEGACY_POLICY_SERIES/);
  assert.match(route, /lte\(dashboardPolicySnapshots\.asOfKey, requestedKey\)/);
  assert.match(route, /orderBy\(desc\(dashboardPolicySnapshots\.asOfKey\)\)/);
  assert.match(route, /samePolicyState/);
  assert.match(route, /only the first series snapshot may use full replay/);
  assert.match(route, /immutable history: snapshot already exists/);
  assert.match(route, /onConflictDoNothing/);
  assert.doesNotMatch(route, /dashboardSnapshots/);
});

test("accepts versioned sequential policy v3 while preserving read compatibility types", async () => {
  const [dashboard, snapshotContract, styles] = await Promise.all([
    readFile(new URL("../app/DashboardApp.tsx", import.meta.url), "utf8"),
    readFile(new URL("../lib/snapshot.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);

  assert.match(snapshotContract, /schema_version: 1/);
  assert.match(snapshotContract, /schema_version: 2/);
  assert.match(snapshotContract, /schema_version: 3/);
  assert.match(snapshotContract, /risk_policy\?: LegacyRiskPolicySnapshot/);
  assert.match(snapshotContract, /risk_policy: RiskPolicySnapshotV2/);
  assert.match(snapshotContract, /risk_policy: RiskPolicySnapshotV3/);
  assert.match(snapshotContract, /series_version: string/);
  assert.match(snapshotContract, /policy_state: PolicyState \| null/);
  assert.match(snapshotContract, /recovery_clear_streak: number/);
  assert.match(snapshotContract, /policy\.policy_state\.exposure === policy\.recommended_exposure/);
  assert.match(snapshotContract, /state\.regime_model_version === snapshot\.regime\.model_version/);
  assert.match(snapshotContract, /"OK" \| "NO_DECISION"/);
  assert.match(snapshotContract, /slow_risk_score: number \| null/);
  assert.match(snapshotContract, /risk_score_delta\?: number \| null/);
  assert.match(snapshotContract, /risk_trend\?: "UP" \| "DOWN" \| "FLAT" \| null/);
  assert.match(snapshotContract, /fast_adjustment: number \| null/);
  assert.match(snapshotContract, /"TRIGGERED" \| "CLEAR" \| "INCOMPLETE"/);
  assert.match(snapshotContract, /data_quality: "EXACT" \| "PUBLIC_LIVE" \| "UNAVAILABLE"/);
  assert.doesNotMatch(snapshotContract, /data_quality: "EXACT" \| "PROXY"/);
  assert.match(snapshotContract, /value\.required !== 3/);
  assert.match(snapshotContract, /value\.total !== 8/);
  assert.match(snapshotContract, /value\.conditions\.length !== 8/);
  assert.match(snapshotContract, /isPercentGrid/);
  assert.match(snapshotContract, /isRiskPolicyV2\(value\.risk_policy\)/);
  assert.match(snapshotContract, /!isHistory\(value\.history\) \|\| !isRiskPolicyV3\(value\.risk_policy\)/);
  assert.match(snapshotContract, /isLegacyRiskPolicy\(value\.risk_policy\)/);
  assert.match(snapshotContract, /\[1, 2, 3\]\.includes/);
  assert.match(snapshotContract, /recent_peak_p_bear\?: number \| null/);
  assert.match(snapshotContract, /risk_off_peak_seen\?: boolean/);
  assert.match(snapshotContract, /isReentryPolicy\(value\.reentry\)/);
  assert.match(snapshotContract, /value\.advisory_only === true/);
  assert.match(snapshotContract, /hasOnlyKeys\(value, \[/);
  assert.match(snapshotContract, /"drawdown_onset", "risk_policy", "sector", "notices"/);
  assert.match(dashboard, /<RiskPolicyHero snapshot=\{snapshot\}/);
  assert.match(dashboard, /Risk Management/);
  assert.doesNotMatch(dashboard, /NQ HMM REGIME/);
  assert.match(dashboard, /RISK BAND/);
  assert.match(dashboard, /ROUTER RISK SCORE/);
  assert.doesNotMatch(dashboard, /HMM INTERNAL STATE PROBABILITY/);
  assert.match(dashboard, /v2\s*\?\s*policy\.slow_risk_score/);
  assert.match(dashboard, /v2 \? "ROUTER RISK SCORE" : "RISK SCORE"/);
  assert.match(dashboard, /참고 LONG 비중/);
  assert.match(dashboard, /DETERIORATING/);
  assert.match(dashboard, /IMPROVING/);
  assert.match(dashboard, /STABLE/);
  assert.match(dashboard, /PRIMARY DRIVER/);
  assert.match(dashboard, /FAST OVERLAY/);
  assert.match(dashboard, /RE-ENTRY/);
  assert.match(dashboard, /VIX 급등/);
  assert.match(dashboard, /NQ 5일선/);
  assert.match(dashboard, /일부 입력/);
  assert.match(dashboard, /필요 \{missingLabels\.join/);
  assert.doesNotMatch(dashboard, /className="probability-rail"/);
  assert.match(dashboard, /className="signal-console"/);
  assert.match(dashboard, /모델 기반 참고지표/);
  assert.doesNotMatch(dashboard, /Market Health Score/);
  assert.doesNotMatch(dashboard, /min\(기본/);
  assert.doesNotMatch(dashboard, /Shadow · 자문 only/);
  assert.match(styles, /\.command-center/);
  assert.match(styles, /\.band-green/);
  assert.match(styles, /\.band-yellow/);
  assert.match(styles, /\.band-red/);
  assert.match(styles, /\.signal-chip/);
  assert.match(styles, /@media \(max-width:540px\)/);
});

test("shows HMM only as a secondary diagnostic alongside RORO", async () => {
  const dashboard = await readFile(new URL("../app/DashboardApp.tsx", import.meta.url), "utf8");

  assert.match(dashboard, /MODEL PERSPECTIVES · DIAGNOSTIC/);
  assert.match(dashboard, /PRIMARY · Market Lens MARKET LENS/);
  assert.match(dashboard, /SECONDARY · HMM STATE LENS/);
  assert.match(dashboard, /HMM은 느린 상태 인식이며 전망 확률이 아닙니다/);
  assert.match(dashboard, /MODEL RELATIONSHIP/);
  assert.match(dashboard, /DIVERGENCE/);
  assert.doesNotMatch(dashboard, /NQ HMM REGIME/);
  assert.match(dashboard, /<ModelPerspectives snapshot=\{snapshot\} roro=\{roro\} \/>/);
});

test("polls only while the latest displayed view is active", async () => {
  const dashboard = await readFile(new URL("../app/DashboardApp.tsx", import.meta.url), "utf8");

  assert.match(dashboard, /AUTO_REFRESH_MS = 5 \* 60 \* 1000/);
  assert.match(dashboard, /displayAsOf === latestDate/);
  assert.match(dashboard, /window\.setInterval/);
  assert.match(dashboard, /window\.clearInterval/);
  assert.match(dashboard, /new AbortController\(\)/);
  assert.match(dashboard, /if \(requestRef\.current\) return/);
  assert.match(dashboard, /signal: controller\.signal/);
  assert.match(dashboard, /requestRef\.current\?\.abort\(\)/);
  assert.match(dashboard, /disabled=\{loading\}/);
  assert.match(dashboard, /과거 조회 · 자동 확인 일시정지/);
});

test("shows Market Lens history when the legacy policy snapshot is unavailable", async () => {
  const dashboard = await readFile(new URL("../app/DashboardApp.tsx", import.meta.url), "utf8");

  assert.match(dashboard, /else if \(requestedDate && historicalRoro\)/);
  assert.match(dashboard, /setRoroOnlyHistory\(true\)/);
  assert.match(dashboard, /Market Lens 이력만 보관되어 있습니다/);
  assert.match(dashboard, /historicalRoro\.as_of <= nextSnapshot\.as_of/);
});

test("queues an authenticated data refresh and polls it to completion", async () => {
  const [dashboard, route, schema] = await Promise.all([
    readFile(new URL("../app/DashboardApp.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/api/refresh/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../db/schema.ts", import.meta.url), "utf8"),
  ]);

  assert.match(dashboard, /fetch\("\/api\/refresh"/);
  assert.match(dashboard, /method: "POST"/);
  assert.match(dashboard, /REFRESH_POLL_MS = 5 \* 1000/);
  assert.match(dashboard, /\/api\/refresh\?id=/);
  assert.match(dashboard, /업데이트 요청됨/);
  assert.match(dashboard, /업데이트 중/);
  assert.match(dashboard, /업데이트 완료/);
  assert.match(dashboard, /await load\(token\)/);
  assert.match(route, /mode: "view" \| "upload"/);
  assert.match(route, /status IN \('PENDING', 'RUNNING'\)/);
  assert.match(route, /action === "claim"/);
  assert.match(route, /action === "finish"/);
  assert.match(route, /2 \* 60 \* 60 \* 1000/);
  assert.match(route, /UPDATE dashboard_refresh_requests/);
  assert.match(route, /RETURNING id, status/);
  assert.match(schema, /dashboardRefreshRequests/);
  assert.match(schema, /idx_dashboard_refresh_requests_status_id/);
});

test("keeps sector forecast status distinct from RRG alignment", async () => {
  const dashboard = await readFile(new URL("../app/DashboardApp.tsx", import.meta.url), "utf8");

  assert.match(dashboard, /20D RANK/);
  assert.match(dashboard, /MODEL STATUS/);
  assert.match(dashboard, /RRG ALIGN/);
  assert.match(dashboard, /row\.signal/);
  assert.match(dashboard, /quadrant_transition/);
  assert.match(dashboard, /SUPPORTIVE/);
  assert.match(dashboard, /EARLY/);
  assert.match(dashboard, /DIVERGENT/);
  assert.match(dashboard, /row\.roro_context === "FIT" \? "FIT" : "MISMATCH"/);
  assert.match(dashboard, /FIT CHANGE/);
  assert.match(dashboard, /MISMATCH → FIT/);
  assert.match(dashboard, /FIT → MISMATCH/);
  assert.match(dashboard, /history=2/);
  assert.doesNotMatch(dashboard, /row\.forecast_rank <= 3/);
});

test("validates and wires causal Sector Turning gauges with adaptive thresholds", async () => {
  const [dashboard, route, contract, schema, styles] = await Promise.all([
    readFile(new URL("../app/DashboardApp.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/api/sector-turning/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../lib/sector-turning.ts", import.meta.url), "utf8"),
    readFile(new URL("../db/schema.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);
  const reading = (score, threshold) => ({
    score, threshold, alert_margin: score - threshold, level: "WARNING",
    confirmation_score: 60, noise_score: 8, proxy_status: "DYNAMIC",
  });
  const snapshot = {
    schema_version: 1,
    series_version: CURRENT_SECTOR_TURNING_SERIES,
    as_of: "2026-09-03",
    generated_at: "2026-09-03T00:00:00Z",
    input_cutoff: "2026-09-03",
    provenance_hash: "0".repeat(64),
    model_versions: {
      turning_model: CURRENT_SECTOR_TURNING_MODEL,
      attention_threshold: CURRENT_ATTENTION_THRESHOLD_MODEL,
    },
    payload: {
      status: "RESEARCH_SHADOW_ONLY",
      pit_status: "PIT_PARTIAL/CURRENT_REVISED_HISTORY_ONLY",
      source_model: CURRENT_SECTOR_TURNING_MODEL,
      threshold_model: CURRENT_ATTENTION_THRESHOLD_MODEL,
      base_threshold: 90,
      priority_threshold: 95,
      threshold_floor: 85,
      threshold_ceiling: 95,
      feature_timing: "EOD_T_INPUT; ACTION_NO_EARLIER_THAN_T_PLUS_1",
      attention_proxy: "causal price and breadth proxies",
      rows: [
        { sector: "1010", sector_name: "Energy", attention_score: 20, attention_band: "LOW", rotation_out: reading(91, 90), rotation_in: null },
        { sector: "4510", sector_name: "Software", attention_score: 80, attention_band: "HIGH", rotation_out: null, rotation_in: reading(94, 93) },
      ],
    },
  };
  snapshot.provenance_hash = await sectorTurningProvenanceHash(snapshot);
  assert.equal(isSectorTurningSnapshot(snapshot), true);
  assert.equal(await verifySectorTurningProvenance(snapshot), true);
  assert.equal(isSectorTurningSnapshot({ ...snapshot, payload: { ...snapshot.payload, threshold_floor: 84 } }), false);
  assert.match(dashboard, /IN \/ OUT RPM Gauge/);
  assert.match(dashboard, /\/api\/sector-turning/);
  assert.match(dashboard, /DormantRpmGauge/);
  assert.match(route, /immutable Sector Turning history conflict/);
  assert.match(route, /await verifySectorTurningProvenance\(snapshot\)/);
  assert.match(contract, /expectedLevel/);
  assert.match(schema, /sectorTurningGaugeSnapshots/);
  assert.match(styles, /\.rpm-grid/);
  assert.match(styles, /\.attention-high/);
});

test("serves authenticated weekly Sector history with pre-portfolio N/A", async () => {
  const [dashboard, route, contract, schema, styles] = await Promise.all([
    readFile(new URL("../app/DashboardApp.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/api/sector-history/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../lib/sector-history.ts", import.meta.url), "utf8"),
    readFile(new URL("../db/schema.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);
  assert.match(dashboard, /\/api\/sector-history/);
  assert.match(dashboard, /ROTATION EVENT/);
  assert.match(dashboard, /rotationEvent\(row\)/);
  assert.match(dashboard, /HISTORICAL BACKFILL/);
  assert.match(dashboard, /ACTUAL 20D/);
  assert.match(dashboard, /row\.roro_context === "N\/A"/);
  assert.match(route, /isSectorHistory\(payload\)/);
  assert.match(route, /authorization/);
  assert.match(route, /ON CONFLICT\(id\) DO UPDATE/);
  assert.match(contract, /SECTOR_HISTORY_SERIES/);
  assert.match(contract, /"N\/A" \| "FIT" \| "RORO_MISMATCH"/);
  assert.match(contract, /group\.length === SECTOR_RORO_TICKERS\.length/);
  assert.match(schema, /sectorResearchHistory/);
  assert.match(styles, /\.sector-history-panel/);
});

test("serves long-run IT internals as an isolated XLK-relative research lens", async () => {
  const [dashboard, route, contract, schema, migration] = await Promise.all([
    readFile(new URL("../app/DashboardApp.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/api/it-internals/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../lib/it-internals.ts", import.meta.url), "utf8"),
    readFile(new URL("../db/schema.ts", import.meta.url), "utf8"),
    readFile(new URL("../drizzle/0007_far_dakota_north.sql", import.meta.url), "utf8"),
  ]);
  assert.match(dashboard, /IT INTERNALS · SHADOW/);
  assert.match(dashboard, /\/api\/it-internals/);
  assert.match(dashboard, /<ItInternalsPanel history=\{itInternalsHistory\}/);
  assert.match(dashboard, /XLK 내부 순환: Semiconductors vs Software/);
  assert.match(dashboard, /진짜 하드웨어 전체나 변동성 조정 신호는 아니며/);
  assert.match(route, /isItInternalsHistory\(payload\)/);
  assert.match(route, /ON CONFLICT\(id\) DO UPDATE/);
  assert.match(contract, /IT_INTERNALS_SCHEMA = 2/);
  assert.match(contract, /IT_INTERNALS_SERIES/);
  assert.match(contract, /benchmark: "XLK"/);
  assert.match(contract, /semiconductor: "SOXX"/);
  assert.match(contract, /software: "IGV"/);
  assert.match(contract, /cadence: "DAILY"/);
  assert.match(contract, /rows.length > 8_000/);
  assert.match(dashboard, /min=\{history\.start\} max=\{history\.as_of\}/);
  assert.match(dashboard, /20D LEADER CONTINUATION/);
  assert.match(dashboard, /OVERLAPPING/);
  assert.match(route, /observations: payload\.rows\.length/);
  assert.match(schema, /itInternalsHistory/);
  assert.match(migration, /CREATE TABLE `it_internals_history`/);
});

test("keeps Sector history visible and aligned during Market Lens-only lookback", async () => {
  const dashboard = await readFile(new URL("../app/DashboardApp.tsx", import.meta.url), "utf8");
  const parsedHistory = dashboard.indexOf("const sectorHistoryCandidate");
  const snapshotFailure = dashboard.indexOf("if (!response.ok)");

  assert.ok(parsedHistory >= 0 && parsedHistory < snapshotFailure);
  assert.match(dashboard, /setSectorHistory\(historicalSector\)/);
  assert.match(dashboard, /nearestDate\(selectedDate \|\| history\.as_of\)/);
  assert.match(dashboard, /<SectorHistoryPanel history=\{sectorHistory\} selectedDate=\{selectedDate\} \/>/);
});

test("allows Sector-only lookback before Market Lens history begins", async () => {
  const dashboard = await readFile(new URL("../app/DashboardApp.tsx", import.meta.url), "utf8");

  assert.match(dashboard, /historicalSector && requestedDate >= historicalSector\.start/);
  assert.match(dashboard, /historicalItInternals && requestedDate >= historicalItInternals\.start/);
  assert.match(dashboard, /const sectorOnlyHistory = roroOnlyHistory && !roro/);
  assert.match(dashboard, /이 날짜는 섹터·IT 연구 이력만 보관되어 있습니다/);
  assert.match(dashboard, /roro\?\.as_of \?\? selectedDate/);
});

test("persists the view token across shortcut-opened tabs until lock", async () => {
  const dashboard = await readFile(new URL("../app/DashboardApp.tsx", import.meta.url), "utf8");

  assert.match(dashboard, /localStorage\.setItem\("regime-view-token", nextToken\)/);
  assert.match(dashboard, /localStorage\.getItem\("regime-view-token"\)/);
  assert.match(dashboard, /localStorage\.removeItem\("regime-view-token"\)/);
  assert.doesNotMatch(dashboard, /sessionStorage/);
});

test("persists an accessible light and dark theme preference", async () => {
  const dashboard = await readFile(new URL("../app/DashboardApp.tsx", import.meta.url), "utf8");
  const styles = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");

  assert.match(dashboard, /localStorage\.getItem\("regime-theme"\)/);
  assert.match(dashboard, /document\.documentElement\.dataset\.theme = next/);
  assert.match(dashboard, /aria-label=.*모드로 전환/);
  assert.match(styles, /:root\[data-theme="light"\]/);
});
