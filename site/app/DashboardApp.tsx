"use client";
import { isRoroNarrative, type RoroNarrative } from "../lib/roro-narrative";
import { nasdaqClose, nasdaqDailyReturn, nasdaqAcceleration, type ChartBenchmark } from "../lib/chart-benchmark";

import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  DashboardSnapshot,
  FastOverlayCondition,
  ReentryCondition,
  RiskPolicySnapshot,
  RiskPolicySnapshotV2,
  SectorRow,
} from "../lib/snapshot";
import {
  BREADTH_REPAIR_SERIES,
  isRoroShadowSnapshot,
  trailingDifference,
  type RoroComponentName,
  type RoroPayloadV1,
  type RoroShadowSnapshot,
} from "../lib/roro";
import { isSectorRoroShadowSnapshot, rotationEvent, type SectorRoroShadowSnapshot } from "../lib/sector-roro";
import {
  isSectorTurningSnapshot,
  type SectorTurningRow,
  type SectorTurningSnapshot,
  type TurningAlertLevel,
  type TurningGaugeReading,
} from "../lib/sector-turning";
import { isSectorHistory, type SectorHistory } from "../lib/sector-history";
import { IT_STATES, isItInternalsHistory, type ItInternalsHistory, type ItInternalsRow, type ItState, type ItWinner } from "../lib/it-internals";
import ItSpreadChart from "./ItSpreadChart";
import SectorPairPanel from "./SectorPairPanel";
import RotationEntryPanel from "./RotationEntryPanel";
import SectorHybridPanel from "./SectorHybridPanel";
import { isHybridSnapshot, type HybridSnapshot } from "../lib/sector-hybrid";
import { isRotationEntry, type RotationEntry } from "../lib/rotation-entry";
import SectorTransitionPanel from "./SectorTransitionPanel";
import { isSectorTransitionSnapshot, type SectorTransitionSnapshot } from "../lib/sector-transition";

const AUTO_REFRESH_MS = 5 * 60 * 1000;
const REFRESH_POLL_MS = 5 * 1000;
const HMM_STATE_LABEL = { BULL: "RISK-ON", SIDEWAYS: "TRANSITION", BEAR: "RISK-OFF" } as const;
const QUADRANT_KO: Record<string, string> = {
  LEADING: "주도",
  WEAKENING: "둔화",
  LAGGING: "소외",
  IMPROVING: "회복",
};
const TRAJECTORY_LABEL: Record<string, string> = {
  NE_STRENGTHENING: "STRENGTHENING",
  NW_EARLY_RECOVERY: "EARLY RECOVERY",
  SE_LOSING_MOMENTUM: "LOSING MOMENTUM",
  SW_DETERIORATING: "DETERIORATING",
  WARMUP: "WARMUP",
};
const IT_STATE_LABEL: Record<ItState, string> = {
  BROAD_STRENGTH: "BROAD IT STRENGTH",
  BROAD_WEAKNESS: "BROAD IT WEAKNESS",
  SEMICONDUCTOR_LEAD: "SEMICONDUCTOR LEAD",
  SOFTWARE_LEAD: "SOFTWARE LEAD",
};

function pairWinner(value: number): ItWinner {
  return value > 0 ? "SEMICONDUCTOR" : value < 0 ? "SOFTWARE" : "TIE";
}

function decodeItInternalsRow(row: ItInternalsRow, prior?: ItInternalsRow) {
  const [decisionDate, state20Code, state60Code, soxx20, igv20, soxx60, igv60, pairSpread20, pairForward20] = row;
  const state20 = IT_STATES[state20Code];
  const priorState20 = prior ? IT_STATES[prior[1]] : null;
  return {
    decisionDate, state20, state60: IT_STATES[state60Code], soxx20, igv20, soxx60, igv60,
    pairSpread20, pairForward20,
    stateConfirmed: state20Code === state60Code,
    stateTransition: priorState20 === null ? "N/A" : priorState20 === state20 ? "—" : `${priorState20} → ${state20}`,
    currentPairLeader: pairWinner(pairSpread20),
    pairActualWinner: pairForward20 === null ? null : pairWinner(pairForward20),
  };
}
const FAST_SHORT_LABEL: Record<string, string> = {
  vix_1d_spike: "VIX 급등",
  vix_term_inversion: "VIX 곡선",
  nq_1d_drop: "NQ 급락",
  move_1d_spike: "MOVE 급등",
  hy_oas_widening: "HY 확대",
  jpy_3d_drop: "JPY 충격",
  breadth_1d_collapse: "시장폭 악화",
  portfolio_correlation: "상관 상승",
};
const REENTRY_SHORT_LABEL: Record<string, string> = {
  vix_term_structure: "VIX 곡선",
  hy_spread_stabilized: "HY 진정",
  nq_above_5dma: "NQ 5일선",
  breadth_divergence: "시장폭 반전",
};
const AXIS_LABEL: Record<string, string> = {
  macro_risk: "MACRO",
  market_trend: "NQ TREND",
  portfolio_internal: "PORTFOLIO",
};
const PHASE_LABEL: Record<string, string> = {
  BOTTOMING_CANDIDATE: "BOTTOMING",
  RECOVERY_CONFIRMING: "RECOVERY",
  TOPPING_CANDIDATE: "TOPPING",
  DOWNTREND: "DOWNTREND",
  UPTREND: "UPTREND",
  RISK_BUILDING: "RISK BUILDING",
  TRANSITION: "TRANSITION",
  INSUFFICIENT_EVIDENCE: "CHECKING",
};
const POLICY_PHASE_LABEL: Record<string, string> = {
  COLD_START: "첫 계산",
  RISK_REDUCTION: "즉시 축소",
  RECOVERY_HOLD: "재진입 대기",
  REENTRY_DIRECT: "재진입 확인",
  RECOVERY_CONFIRM: "안정 확인 중",
  AT_TARGET: "목표 비중",
};
const TURN_LABEL: Record<string, string> = {
  VIX_CURVE_5D_DOWN: "변동성 곡선 안정",
  VIX_CURVE_5D_UP: "변동성 곡선 악화",
  HY_OAS_20D_NON_WIDENING: "회사채 위험 안정",
  HY_OAS_20D_WIDENING: "회사채 위험 확대",
  PORTFOLIO_CORRELATION_5D_DOWN: "보유종목 동조화 완화",
  PORTFOLIO_CORRELATION_5D_UP: "보유종목 동조화 상승",
  MARKET_BREADTH_5D_UP: "시장 참여 폭 개선",
  MARKET_BREADTH_5D_DOWN: "시장 참여 폭 악화",
  PORTFOLIO_BREADTH_MA20_5D_UP: "보유종목 흐름 개선",
  PORTFOLIO_BREADTH_MA20_5D_DOWN: "보유종목 흐름 악화",
  NQ_ABOVE_MA5: "NQ 단기추세 상회",
  NQ_BELOW_MA5: "NQ 단기추세 하회",
  PRIOR_BOTTOMING_CANDIDATE_20D: "최근 바닥 후보",
  NQ_ABOVE_MA20: "NQ 중기추세 상회",
  RISK_5D_NOT_RISING: "위험 재상승 없음",
  RISK_SCORE_AT_LEAST_67: "위험점수 67 이상",
  RISK_SCORE_AT_MOST_33: "위험점수 33 이하",
  RISK_SCORE_5D_RISING: "위험점수 상승 중",
};
const RORO_COMPONENTS = [
  { key: "credit", group: "MACRO", label: "회사채 위험", score: "credit_score" },
  { key: "rates_volatility", group: "MACRO", label: "채권시장 불안", score: "rates_volatility_score" },
  { key: "equity_volatility", group: "MACRO", label: "주식 변동성", score: "equity_volatility_score" },
  { key: "fx_carry", group: "MACRO", label: "엔화 충격", score: "fx_carry_score" },
  { key: "safe_haven", group: "MACRO", label: "금 피난 수요", score: "safe_haven_score" },
  { key: "equity_trend", group: "MARKET", label: "NQ 추세", score: "equity_trend_score" },
  { key: "market_breadth", group: "MARKET", label: "시장 참여 폭", score: "breadth_risk_percentile" },
  { key: "portfolio_correlation", group: "PORTFOLIO", label: "종목 동조화", score: "portfolio_correlation_score" },
  { key: "portfolio_beta", group: "PORTFOLIO", label: "NQ 민감도", score: "portfolio_beta_score" },
  { key: "portfolio_concentration", group: "PORTFOLIO", label: "쏠림 정도", score: "portfolio_concentration_score" },
  { key: "portfolio_participation", group: "PORTFOLIO", label: "상승 참여도", score: "portfolio_participation_score" },
  { key: "portfolio_drawdown", group: "PORTFOLIO", label: "포트 하락폭", score: "portfolio_drawdown_score" },
] as const satisfies readonly {
  key: RoroComponentName;
  group: string;
  label: string;
  score: keyof RoroPayloadV1;
}[];

type RefreshRequest = {
  id: number;
  status: "PENDING" | "RUNNING" | "SUCCEEDED" | "FAILED";
  as_of: string | null;
  detail: string | null;
};

type Theme = "dark" | "light";

function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>("dark");
  useEffect(() => {
    const next: Theme = localStorage.getItem("regime-theme") === "light" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    setTheme(next);
  }, []);
  const toggle = () => {
    const next: Theme = theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("regime-theme", next);
    setTheme(next);
  };
  return <button type="button" className="theme-toggle" aria-label={`${theme === "dark" ? "라이트" : "다크"} 모드로 전환`} aria-pressed={theme === "light"} onClick={toggle}>{theme === "dark" ? "☀ LIGHT" : "◐ DARK"}</button>;
}

type RoroHistoryPoint = ChartBenchmark & {
  snapshot: RoroShadowSnapshot;
};

function isRoroHistoryPoint(value: unknown): value is RoroHistoryPoint {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const point = value as Record<string, unknown>;
  return isRoroShadowSnapshot(point.snapshot) &&
    (point.benchmark_close === null || typeof point.benchmark_close === "number" && Number.isFinite(point.benchmark_close) && point.benchmark_close > 0);
}

function refreshLabel(request: RefreshRequest | null, submitting: boolean) {
  if (submitting) return "업데이트 요청 중";
  if (request?.status === "PENDING") return "업데이트 요청됨";
  if (request?.status === "RUNNING") return "업데이트 중";
  if (request?.status === "SUCCEEDED") return `업데이트 완료${request.as_of ? ` · ${request.as_of}` : ""}`;
  if (request?.status === "FAILED") return "업데이트 실패";
  return null;
}

function transitionLabel(value: string) {
  const [from, to] = value.split("->");
  if (!from || !to) return value;
  return `${QUADRANT_KO[from] ?? from} → ${QUADRANT_KO[to] ?? to}`;
}
function modelStatus(row: SectorRow) {
  if (row.signal === "LEAD_CANDIDATE_SHADOW") {
    return row.predicted_excess_20d > 0 ? "MODEL_TOP3" : "NEGATIVE";
  }
  if (row.signal === "WATCH_SHADOW") {
    return row.predicted_excess_20d > 0 ? "POSITIVE_WATCH" : "NEGATIVE";
  }
  return row.signal || (row.predicted_excess_20d > 0 ? "POSITIVE_WATCH" : "NEGATIVE");
}

function rrgAlignment(row: SectorRow) {
  const positive = row.predicted_excess_20d > 0;
  const supportiveQuadrant = positive
    ? row.quadrant === "LEADING" || row.quadrant === "IMPROVING"
    : row.quadrant === "WEAKENING" || row.quadrant === "LAGGING";
  const supportiveTrajectory = positive
    ? row.tail_heading === "NE_STRENGTHENING" || row.tail_heading === "NW_EARLY_RECOVERY"
    : row.tail_heading === "SE_LOSING_MOMENTUM" || row.tail_heading === "SW_DETERIORATING";
  if (supportiveQuadrant && supportiveTrajectory) return "SUPPORTIVE";
  if (!supportiveQuadrant && supportiveTrajectory) return "EARLY";
  return "DIVERGENT";
}

function signedPercent(value: number) {
  return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(2)}%`;
}

function allocationPercent(value: number | null) {
  return value === null ? "—" : `${Math.round(value * 100)}%`;
}

function isV2Policy(policy: RiskPolicySnapshot): policy is RiskPolicySnapshotV2 {
  return "risk_score" in policy;
}

function scoreTrend(policy: RiskPolicySnapshotV2) {
  const trend = policy.risk_trend ?? (
    policy.risk_score_delta == null || policy.risk_score_delta === 0
      ? "FLAT"
      : policy.risk_score_delta > 0 ? "UP" : "DOWN"
  );
  return {
    trend,
    arrow: trend === "UP" ? "↑" : trend === "DOWN" ? "↓" : "→",
    label: trend === "UP" ? "DETERIORATING" : trend === "DOWN" ? "IMPROVING" : "STABLE",
    delta: policy.risk_score_delta == null
      ? ""
      : `${policy.risk_score_delta > 0 ? "+" : ""}${Math.round(policy.risk_score_delta)}`,
  };
}

function Login({ onUnlock, error }: { onUnlock: (token: string) => void; error: string }) {
  const [token, setToken] = useState("");
  function submit(event: FormEvent) {
    event.preventDefault();
    if (token.trim()) onUnlock(token.trim());
  }
  return (
    <main className="login-shell">
      <ThemeToggle />
      <section className="login-card">
        <div className="eyebrow">Market Lens · MARKET INTELLIGENCE</div>
        <div className="mark">R</div>
        <h1>Risk Management</h1>
        <p>Institutional risk regime dashboard</p>
        <form onSubmit={submit}>
          <label htmlFor="access-code">접근 코드</label>
          <input id="access-code" type="password" value={token} onChange={(event) => setToken(event.target.value)} />
          {error && <span className="form-error">{error}</span>}
          <button type="submit">대시보드 열기 <span>→</span></button>
        </form>
        <small>Authorized access only</small>
      </section>
    </main>
  );
}

function RiskCard({ title, data }: { title: string; data: DashboardSnapshot["crash_guard"] }) {
  const active = data && !["NORMAL_SHADOW", "DORMANT", "WARMUP"].includes(data.risk_band);
  return (
    <article className="risk-card">
      <div className="risk-card-head"><span className={`status-dot ${active ? "warn" : ""}`} /><b>{title}</b></div>
      <strong>{data?.risk_band.replaceAll("_SHADOW", "") ?? "N/A"}</strong>
      <span>{data?.score == null ? "—" : data.score.toFixed(3)}</span>
      <small>{data?.date ?? "—"}</small>
    </article>
  );
}

type SignalCondition = FastOverlayCondition | ReentryCondition;

function SignalChips({
  conditions,
  shortLabels,
}: {
  conditions: SignalCondition[] | null;
  shortLabels: Record<string, string>;
}) {
  const items = conditions ?? Object.entries(shortLabels).map(([key, label]) => ({
    key,
    label,
    status: "UNAVAILABLE",
    detail: "과거 스냅샷",
  }));
  return (
    <div className="signal-items" aria-label="정책 확인 항목">
      {items.map((condition) => {
        const label = shortLabels[condition.key] ?? condition.label;
        const description = `${condition.label}: ${condition.status}${condition.detail ? ` · ${condition.detail}` : ""}`;
        return <span
          key={condition.key}
          className={`signal-chip status-${condition.status.toLowerCase().replaceAll("_", "-")}`}
          title={description}
          aria-label={description}
        >{label}</span>;
      })}
    </div>
  );
}

function RiskPolicyHero({ snapshot }: { snapshot: DashboardSnapshot }) {
  const policy = snapshot.risk_policy ?? null;
  if (!policy || policy.decision_status !== "OK") {
    return (
      <section className="command-center band-neutral" aria-label="리스크 정책">
        <div className="command-topline"><span>RISK CONTROL SYSTEM</span><b>NO DECISION</b></div>
        <div className="command-grid command-empty">
          <div><span>STATUS</span><strong>{policy?.decision_status ?? "NO DECISION"}</strong></div>
          <div><span>ROUTER RISK SCORE</span><strong>—</strong></div>
          <div><span>참고 LONG 비중</span><strong>—</strong></div>
          <div><span>PRIMARY DRIVER</span><strong>{policy?.primary_driver || "필수 입력 누락"}</strong></div>
        </div>
      </section>
    );
  }

  const v2 = isV2Policy(policy);
  const band = v2 ? policy.risk_band : policy.health_band;
  const riskScore = v2
    ? policy.slow_risk_score
    : policy.market_health_score == null ? null : Math.round(100 - policy.market_health_score);
  const trend = v2 ? scoreTrend(policy) : null;
  const exposure = policy.recommended_exposure;
  const fastConditions = v2 ? policy.fast_overlay.conditions : null;
  const fastCount = v2 ? policy.fast_overlay.trigger_count : Math.min(policy.active_triggers.length, 8);
  const reentry = policy.reentry;
  const fastInputCount = fastConditions?.filter((item) => item.status !== "UNAVAILABLE").length ?? null;
  const reentryInputCount = reentry.conditions.filter((item) => item.status !== "UNAVAILABLE").length;
  const missingLabels = Array.from(new Set([
    ...(fastConditions ?? [])
      .filter((item) => item.status === "UNAVAILABLE")
      .map((item) => FAST_SHORT_LABEL[item.key] ?? item.label),
    ...reentry.conditions
      .filter((item) => item.status === "UNAVAILABLE")
      .map((item) => REENTRY_SHORT_LABEL[item.key] ?? item.label),
  ]));
  return (
    <>
      <section className={`command-center band-${band?.toLowerCase() ?? "neutral"}`} aria-label="리스크 정책 권고">
        <div className="command-topline">
          <span>RISK CONTROL SYSTEM</span>
          <b>{band ?? "N/A"} · {snapshot.as_of}</b>
        </div>
        <div className="command-grid">
          <div className="command-metric metric-regime">
            <span>RISK BAND</span>
            <strong>{band ?? "N/A"}</strong>
            <small>정책 참고 구간</small>
          </div>
          <div className="command-metric metric-score">
            <span>{v2 ? "ROUTER RISK SCORE" : "RISK SCORE"}</span>
            <strong>{riskScore ?? "—"}</strong>
            <small className={trend ? `trend-${trend.trend.toLowerCase()}` : ""}>
              {trend ? `${trend.label} ${trend.arrow}${trend.delta ? ` ${trend.delta}` : ""}` : "LEGACY"}
            </small>
          </div>
          <div className="command-metric metric-exposure">
            <span>참고 LONG 비중</span>
            <strong>{allocationPercent(exposure)}</strong>
            <div className="exposure-gauge" aria-label={`참고 Long 비중 ${allocationPercent(exposure)}`}>
              <i style={{ width: exposure === null ? "0%" : allocationPercent(exposure) }} />
            </div>
          </div>
          <div className="command-metric metric-driver">
            <span>PRIMARY DRIVER</span>
            <strong>{policy.primary_driver || "No dominant driver"}</strong>
          </div>
        </div>
      </section>

      <section className="signal-console" aria-label="정책 확인 신호">
        <div className="signal-group">
          <span>FAST OVERLAY</span>
          <b>{fastCount} 발생{fastInputCount === null ? "" : ` · ${fastInputCount}/8 입력`}</b>
          <SignalChips conditions={fastConditions} shortLabels={FAST_SHORT_LABEL} />
        </div>
        <div className="signal-divider" />
        <div className="signal-group">
          <span>RE-ENTRY</span>
          <b>{reentry.confirmed} 확인 · {reentryInputCount}/{reentry.total} 입력</b>
          <SignalChips conditions={reentry.conditions} shortLabels={REENTRY_SHORT_LABEL} />
        </div>
        <div className="signal-meta">
          <span>STATUS</span>
          <b>{v2 ? missingLabels.length ? "일부 입력" : "입력 완료" : "기존 화면"}</b>
          {missingLabels.length > 0 && <small>필요 {missingLabels.join(" · ")}</small>}
        </div>
      </section>
    </>
  );
}

function RoroHero({ snapshot, history, narrative }: { snapshot: RoroShadowSnapshot; history: RoroHistoryPoint[]; narrative: RoroNarrative | null }) {
  const value = snapshot.payload;
  const policy = snapshot.schema_version === 3 || snapshot.schema_version === 4 ? snapshot.payload.policy : null;
  const band = value.risk_regime.toLowerCase();
  const regime = value.risk_regime === "GREEN" ? "RISK-ON" : value.risk_regime === "RED" ? "RISK-OFF" : "TRANSITION";
  const axes = [
    ["MACRO", value.macro_risk_score],
    ["NQ TREND", value.market_trend_score],
    ["PORTFOLIO", value.portfolio_internal_score],
  ] as const;
  const axisDriver = value.risk_drivers.length
    ? value.risk_drivers.map((item) => AXIS_LABEL[item] ?? item).join(" + ")
    : value.relief_drivers.map((item) => AXIS_LABEL[item] ?? item).join(" + ") || "BALANCED SIGNALS";
  const driver = narrative?.provenance_hash === snapshot.provenance_hash && narrative.as_of === snapshot.as_of && narrative.series_version === snapshot.series_version
    ? narrative.summary : policy?.primary_driver ?? axisDriver;
  const metric = (label: string, raw: number | null, formatter: (number: number) => string) => (
    <span><b>{label}</b>{raw == null ? "—" : formatter(raw)}</span>
  );
  const point = (raw: number | null) => raw == null ? "—" : `${raw > 0 ? "+" : ""}${raw.toFixed(1)}`;
  const componentValue = (key: RoroComponentName, suffix: string) =>
    value[`${key}_${suffix}` as keyof RoroPayloadV1] as number | null;
  const turnEvidence = value.turn_evidence
    .slice(0, 3)
    .map((item) => TURN_LABEL[item] ?? item.replaceAll("_", " "));
  const historyIndex = history.findIndex((item) => item.snapshot.as_of === snapshot.as_of);
  const storedTrajectory = (lag: number) => historyIndex >= lag
    ? value.market_risk_score - history[historyIndex - lag].snapshot.payload.market_risk_score : null;
  const trajectory1 = storedTrajectory(1);
  const trajectory5 = storedTrajectory(5);
  const trajectory20 = storedTrajectory(20);
  return (
    <>
      <section className={`command-center band-${band}`} aria-label="Market Lens 멀티자산 리스크 지표">
        <div className="command-topline"><span>Market Lens MARKET LENS</span><b>{value.risk_regime} · {snapshot.as_of}</b></div>
        <div className="command-grid">
          <div className="command-metric metric-score"><span>위험 점수</span><strong>{Math.round(value.market_risk_score)}</strong><small className={trajectory5 != null && trajectory5 > 0 ? "trend-up" : trajectory5 != null && trajectory5 < 0 ? "trend-down" : "trend-flat"}>{point(trajectory5)} · 5일</small></div>
          <div className="command-metric metric-regime"><span>시장 흐름</span><strong>{regime}</strong><small>{PHASE_LABEL[value.trend_phase] ?? value.trend_phase}</small></div>
          <div className="command-metric metric-exposure"><span>{policy ? "최종 LONG 비중" : "참고 LONG 비중"}</span><strong>{Math.round(value.recommended_long)}%</strong><div className="exposure-gauge" aria-label={`Long 비중 ${Math.round(value.recommended_long)}%`}><i style={{ width: `${value.recommended_long}%` }} /></div></div>
          <div className="command-metric metric-driver"><span>주요 원인</span><strong>{driver}</strong><small>시장 형국을 만든 핵심 흐름 · 신뢰도 {Math.round(value.confidence)}</small></div>
        </div>
      </section>
      <details className="roro-diagnostic-details">
        <summary><span>상세 진단</span><small>신호 근거 · 포트폴리오 · 단기 위험</small></summary>
        <div className="roro-diagnostic-body">
        <div className="axis-rail" aria-label="3축 리스크 점수">
          {axes.map(([label, score]) => <div key={label}><span>{label}</span><b>{score == null ? "—" : Math.round(score)}</b><i><em style={{ width: score == null ? "0%" : `${score}%` }} /></i></div>)}
        </div>
      <section className="roro-context-strip" aria-label="Market Lens 방향과 판단 일치도">
        <div><span>위험 변화</span><b>1일 {point(trajectory1)} · 5일 {point(trajectory5)} · 20일 {point(trajectory20)}</b></div>
        <div><span>판단 일치도</span><b>{Math.round(value.axis_agreement)}%</b><small>축 간 차이 {value.axis_disagreement.toFixed(1)}</small></div>
        <div><span>전환 근거</span><b>{turnEvidence.join(" · ") || "뚜렷한 전환 근거 없음"}</b></div>
      </section>
      {policy && <section className="roro-policy-console" aria-label="Market Lens Long 정책 결정 과정">
        <div className="roro-policy-flow">
          <span><b>점수 기준 LONG</b><strong>{policy.score_based_long}%</strong></span>
          <i>→</i>
          <span><b>단기 위험 조정</b><strong>{policy.raw_target_long - policy.score_based_long > 0 ? "+" : ""}{policy.raw_target_long - policy.score_based_long}pp · 조정 후 {policy.raw_target_long}%</strong></span>
          <i>→</i>
          <span><b>최종 LONG</b><strong>{policy.recommended_long}%</strong></span>
        </div>
        <div className={`roro-policy-fast ${policy.fast_status.toLowerCase()}`}>
          <span>단기 위험 확인</span>
          <b>{policy.fast_status === "TRIGGERED" ? "위험 발생" : policy.fast_status === "CLEAR" ? "이상 없음" : "일부 미확인"} · 발생 {policy.fast_trigger_count}/3 · 확인 가능 {policy.fast_available_count}/8</b>
          <small>발생: {policy.active_fast_conditions.map((key) => FAST_SHORT_LABEL[key] ?? key).join(" · ") || "없음"} · 미확인: {policy.unavailable_fast_conditions.map((key) => FAST_SHORT_LABEL[key] ?? key).join(" · ") || "없음"}</small>
        </div>
        <div className="roro-policy-phase">
          <span>비중 조정 단계</span>
          <b>{POLICY_PHASE_LABEL[policy.phase] ?? policy.phase}</b>
          <small>{policy.episode_status === "ACTIVE" ? "위험 축소 구간" : "일반 구간"} · 안정 확인 {policy.clear_streak}/3일</small>
        </div>
      </section>}
      <section className="roro-console" aria-label="포트폴리오 내부 위험">
        <div className="roro-metrics">
          {metric("보유종목 동조화", value.portfolio_correlation_20, (number) => number.toFixed(2))}
          {metric("NQ 민감도", value.nq_beta_60, (number) => number.toFixed(2))}
          {metric("실질 분산 종목 수", value.effective_positions, (number) => number.toFixed(1))}
          {metric("단기 추세 상회", value.breadth_above_ma20, (number) => `${Math.round(number * 100)}%`)}
          {metric("최근 고점 대비", value.portfolio_drawdown_60, (number) => `${(number * 100).toFixed(1)}%`)}
        </div>
        <div className="roro-check"><span>바닥 근거</span><b>{value.bottom_confirmation_count}/{value.checklist_available_count}</b><span>고점 근거</span><b>{value.top_confirmation_count}/{value.checklist_available_count}</b></div>
        <div className="roro-quality"><span>데이터 상태</span><b>{value.data_status.replaceAll("_", " ")}</b><small>{value.data_quality?.replaceAll("_", " ") ?? "집계값만 사용"} · 이력 {value.history_weight_coverage_60d == null ? "—" : `${Math.round(value.history_weight_coverage_60d * 100)}%`}</small></div>
      </section>
      <details className="roro-block-details">
        <summary><span>세부 신호</span><b>{value.available_components.length}/{RORO_COMPONENTS.length} 입력</b><small>점수·변화·종합점수 기여도</small></summary>
        {!value.available_components.includes("market_breadth") && <p className="roro-quality" role="status">시장 참여 폭 입력 미완료 · 이 날짜의 시장 축은 NQ 추세만 반영합니다. 결측을 위험 0점으로 처리하지 않습니다.</p>}
        {snapshot.series_version === BREADTH_REPAIR_SERIES && <p className="roro-quality" role="status">참여 폭 보정본 · 최초 발표 이후 확보된 입력으로 재계산했습니다. 당시 발표값과 다를 수 있습니다.</p>}
        <div className="roro-block-table" role="table" aria-label="Market Lens 독립 정보 블록">
          <div className="roro-block-row roro-block-head" role="row"><span>구분</span><span>항목</span><span>점수</span><span>1D</span><span>5D</span><span>20D</span><span>기여</span><span>상태</span></div>
          {RORO_COMPONENTS.map((item) => {
            const score = value[item.score] as number | null;
            const available = value.available_components.includes(item.key);
            const contribution = componentValue(item.key, "contribution");
            return <div className={`roro-block-row ${available ? "available" : "missing"}`} role="row" key={item.key}>
              <span>{item.group}</span><b>{item.label}</b><strong>{score == null ? "—" : Math.round(score)}</strong>
              <span>{point(componentValue(item.key, "score_trajectory_1d"))}</span>
              <span>{point(componentValue(item.key, "score_trajectory_5d"))}</span>
              <span>{point(componentValue(item.key, "score_trajectory_20d"))}</span>
              <span className={contribution == null ? "" : contribution > 0 ? "risk" : contribution < 0 ? "relief" : ""}>{point(contribution)}</span>
              <small>{available ? "입력 있음" : item.key === "market_breadth" ? "미완료·점수 미반영" : "데이터 없음"}</small>
            </div>;
          })}
        </div>
      </details>
        </div>
      </details>
    </>
  );
}

function RiskScoreHistory({ points }: { points: RoroHistoryPoint[] }) {
  const data = points.slice(-40);
  if (!data.length) return null;
  const width = 760;
  const height = 280;
  const left = 58;
  const right = 100;
  const top = 22;
  const bottom = 38;
  const chartWidth = width - left - right;
  const chartHeight = height - top - bottom;
  const x = (index: number) => left + (data.length === 1 ? chartWidth : index * chartWidth / (data.length - 1));
  const y = (score: number) => top + (100 - score) * chartHeight / 100;
  const scores = data.map((item) => Math.round(item.snapshot.payload.market_risk_score));
  const latest = scores.at(-1) ?? 0;
  const periodChange = latest - scores[0];
  const joinedSeries = new Set(data.map((item) => item.snapshot.series_version)).size > 1;
  const benchmarkBase = nasdaqClose(data[0]);
  const benchmarkReturns = data.map((item) => {
    const close = nasdaqClose(item);
    return close !== null && benchmarkBase !== null ? (close / benchmarkBase - 1) * 100 : null;
  });
  const benchmarkValues = benchmarkReturns.filter((value): value is number => value !== null);
  const benchmarkReady = benchmarkValues.length > 0;
  const benchmarkMin = Math.min(0, ...benchmarkValues);
  const benchmarkMax = Math.max(0, ...benchmarkValues);
  const benchmarkExtent = Math.max(Math.abs(benchmarkMin), Math.abs(benchmarkMax));
  const benchmarkPadding = Math.max(benchmarkExtent * .08, .25);
  const benchmarkCeiling = Math.ceil((benchmarkExtent + benchmarkPadding) * 2) / 2;
  const benchmarkFloor = -benchmarkCeiling;
  const yBenchmark = (value: number) => top + (benchmarkCeiling - value) * chartHeight / (benchmarkCeiling - benchmarkFloor);
  const benchmarkLatest = benchmarkReturns.at(-1) ?? null;
  const latestPoint = data[data.length - 1];
  const benchmarkDaily = nasdaqDailyReturn(latestPoint, latestPoint.snapshot.as_of);
  const labelIndexes = [...new Set([0, Math.floor((data.length - 1) / 2), data.length - 1])];
  const signed = (value: number) => `${value > 0 ? "+" : ""}${value}`;
  const percent = (value: number) => `${value > 0 ? "+" : ""}${value.toFixed(2)}%`;

  return (
    <section className="risk-history-panel" aria-labelledby="risk-history-title">
      <div className="risk-history-head">
        <div><span>MARKET LENS HISTORY</span><h2 id="risk-history-title">위험 점수 추이 vs 나스닥100</h2></div>
        <time className="risk-history-asof" dateTime={latestPoint.snapshot.as_of}>{latestPoint.snapshot.as_of}</time>
      </div>
      <div className="risk-history-stats">
        <div className="risk-stat score"><span>위험 점수</span><strong>{latest}<small> / 100</small></strong><p className={periodChange > 0 ? "up" : periodChange < 0 ? "down" : "flat"}>기간 변화 {signed(periodChange)}pt</p></div>
        <div className="risk-stat nasdaq"><span>나스닥100 · 전일 대비</span><strong>{benchmarkDaily === null ? "—" : percent(benchmarkDaily)}</strong><p>{benchmarkDaily === null ? "데이터 없음" : "미국 종가 기준"}</p></div>
        <div className="risk-stat nasdaq"><span>차트 기간 누적</span><strong>{benchmarkLatest === null ? "—" : percent(benchmarkLatest)}</strong><p>{benchmarkLatest === null ? "데이터 없음" : `최근 ${data.length}거래일 구간`}</p></div>
      </div>
      <div className="risk-history-body">
        <div className="risk-history-chart">
          <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby="risk-history-title risk-history-description">
            <desc id="risk-history-description">위험점수와 차트 첫 거래일 대비 나스닥100 현물 지수(NDX) 누적 수익률입니다. 위험점수 50점과 지수 누적 수익률 0퍼센트를 중앙에 맞췄습니다. 전일 종가 대비 등락률은 상단에 별도로 표시합니다.</desc>
            <rect className="risk-band risk-band-red" x={left} y={y(100)} width={chartWidth} height={y(67) - y(100)} />
            <rect className="risk-band risk-band-yellow" x={left} y={y(67)} width={chartWidth} height={y(33) - y(67)} />
            <rect className="risk-band risk-band-green" x={left} y={y(33)} width={chartWidth} height={y(0) - y(33)} />
            {[100, 67, 33, 0].map((tick) => <g key={tick}><line className="risk-history-grid" x1={left} x2={width - right} y1={y(tick)} y2={y(tick)} /><text className="risk-history-y" x={left - 8} y={y(tick) + 3}>{tick}</text></g>)}
            <line className="risk-history-midline" x1={left} x2={width - right} y1={y(50)} y2={y(50)} />
            <text className="risk-history-y risk-history-midline-label" x={left - 8} y={y(50) + 3}>50</text>
            {data.slice(1).map((item, index) => <line key={item.snapshot.as_of} className="risk-history-score" x1={x(index)} y1={y(scores[index])} x2={x(index + 1)} y2={y(scores[index + 1])} />)}
            {data.map((item, index) => <circle className="risk-history-point score" key={item.snapshot.as_of} cx={x(index)} cy={y(scores[index])} r={index === data.length - 1 ? 3.5 : 2}><title>{`${item.snapshot.as_of} · 위험점수 ${scores[index]}${index ? ` · 전일 대비 ${signed(scores[index] - scores[index - 1])}` : ""}`}</title></circle>)}
            {benchmarkReturns.slice(1).map((value, index) => value !== null && benchmarkReturns[index] !== null ? <line key={data[index + 1].snapshot.as_of} className="risk-history-nasdaq" x1={x(index)} y1={yBenchmark(benchmarkReturns[index] as number)} x2={x(index + 1)} y2={yBenchmark(value)} /> : null)}
            {benchmarkReturns.map((value, index) => value === null ? null : <circle className="risk-history-point nasdaq" key={data[index].snapshot.as_of} cx={x(index)} cy={yBenchmark(value)} r={index === data.length - 1 ? 3.5 : 1.8}><title>{`${data[index].snapshot.as_of} · 나스닥100 기간 누적 ${percent(value)} · NDX ${nasdaqClose(data[index])?.toLocaleString()}`}</title></circle>)}
            {benchmarkReady && [benchmarkCeiling, 0, benchmarkFloor].map((tick) => <text className="risk-history-y risk-history-y-right" key={tick} x={width - right + 8} y={yBenchmark(tick) + 3}>{percent(tick)}</text>)}
            {labelIndexes.map((index) => <text className="risk-history-x" key={data[index].snapshot.as_of} x={x(index)} y={height - 10} textAnchor={index === 0 ? "start" : index === data.length - 1 ? "end" : "middle"}>{data[index].snapshot.as_of.slice(5)}</text>)}
          </svg>
        </div>
      </div>
      <footer className="risk-history-footer">
        <div className="risk-history-legend"><span><i className="score" />위험 점수 · 좌측</span><span><i className="nasdaq" />나스닥100 누적 · 우측</span></div>
        <details className="risk-history-method"><summary>계산 기준</summary><p>미국 종가 {latestPoint.snapshot.as_of} · 비교 종가 {latestPoint.benchmark_previous_as_of ?? "대기"}<br />누적 수익률 기준: {benchmarkReady ? `${data[0].snapshot.as_of} = 0%` : "NDX 시작일 종가 대기"}. 중앙선은 위험 50점 / 지수 0%입니다.<br />0은 안정, 100은 방어 필요. 위험모델 NQ 입력은 유지합니다.{joinedSeries ? " 모델 개정 이력을 연결했습니다." : ""}</p></details>
      </footer>
    </section>
  );
}

function RiskAccelerationHistory({ points }: { points: RoroHistoryPoint[] }) {
  const data = points.slice(-40);
  if (data.length < 3) return null;
  const width = 760;
  const height = 280;
  const left = 58;
  const right = 100;
  const top = 22;
  const bottom = 38;
  const chartWidth = width - left - right;
  const chartHeight = height - top - bottom;
  const x = (index: number) => left + index * chartWidth / (data.length - 1);
  const riskLevels = data.map((item) => item.snapshot.payload.market_risk_score);
  const riskChanges = trailingDifference(riskLevels);
  const rawRiskAccelerations = trailingDifference(riskChanges);
  const riskAccelerations = rawRiskAccelerations.map((value, index) => {
    if (index < 2 || value === null) return null;
    const versions = data.slice(index - 2, index + 1).map((item) => item.snapshot.series_version);
    return new Set(versions).size === 1 ? value : null;
  });
  const benchmarkAccelerations = nasdaqAcceleration(data);
  const finite = (value: number | null): value is number => value !== null && Number.isFinite(value);
  const riskValues = riskAccelerations.filter(finite);
  const benchmarkValues = benchmarkAccelerations.filter(finite);
  const riskCeiling = Math.max(1, Math.ceil(Math.max(0, ...riskValues.map(Math.abs)) * 10) / 10);
  const benchmarkCeiling = Math.max(.1, Math.ceil(Math.max(0, ...benchmarkValues.map(Math.abs)) * 10) / 10);
  const yRisk = (value: number) => top + (riskCeiling - value) * chartHeight / (riskCeiling * 2);
  const yBenchmark = (value: number) => top + (benchmarkCeiling - value) * chartHeight / (benchmarkCeiling * 2);
  const latestRisk = riskAccelerations.at(-1) ?? null;
  const latestBenchmark = benchmarkAccelerations.at(-1) ?? null;
  const labelIndexes = [...new Set([2, Math.floor((data.length - 1) / 2), data.length - 1])];
  const signed = (value: number, digits = 1) => `${value > 0 ? "+" : ""}${value.toFixed(digits)}`;

  return (
    <section className="risk-history-panel risk-acceleration-panel" aria-labelledby="risk-acceleration-title">
      <div className="risk-history-head">
        <div><span>SECOND DIFFERENCE · 40D</span><h2 id="risk-acceleration-title">위험·나스닥100 변화 가속도</h2></div>
        <time className="risk-history-asof" dateTime={data[data.length - 1].snapshot.as_of}>{data[data.length - 1].snapshot.as_of}</time>
      </div>
      <div className="risk-history-stats two-columns">
        <div className="risk-stat score"><span>위험 점수 Δ²</span><strong>{latestRisk === null ? "—" : signed(latestRisk)}<small>{latestRisk === null ? "" : " pt"}</small></strong><p>{latestRisk === null ? "데이터 없음" : "전일 점수 변화 대비"}</p></div>
        <div className="risk-stat nasdaq"><span>나스닥100 Δ수익률</span><strong>{latestBenchmark === null ? "—" : signed(latestBenchmark, 2)}<small>{latestBenchmark === null ? "" : " %p"}</small></strong><p>{latestBenchmark === null ? "데이터 없음" : "전일 일간수익률 대비"}</p></div>
      </div>
      <div className="risk-history-body">
        <div className="risk-history-chart">
          <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby="risk-acceleration-title risk-acceleration-description">
            <desc id="risk-acceleration-description">위험점수는 전일 점수 변화의 변화, 나스닥100 현물 지수는 전일 일간수익률 대비 변화입니다. 두 축의 0을 중앙에 맞추고 각각 대칭축으로 표시합니다.</desc>
            {[riskCeiling, 0, -riskCeiling].map((tick) => <g key={tick}><line className={tick === 0 ? "risk-history-midline" : "risk-history-grid"} x1={left} x2={width - right} y1={yRisk(tick)} y2={yRisk(tick)} /><text className="risk-history-y" x={left - 8} y={yRisk(tick) + 3}>{signed(tick)}</text></g>)}
            {riskAccelerations.map((value, index) => {
              if (value === null) return null;
              const previousIndex = riskAccelerations.slice(0, index).findLastIndex(finite);
              if (previousIndex < 0 || !rawRiskAccelerations.slice(previousIndex + 1, index).every(finite)) return null;
              return <line key={`risk-${data[index].snapshot.as_of}`} className={`risk-history-score${previousIndex === index - 1 ? "" : " risk-history-transition"}`} x1={x(previousIndex)} y1={yRisk(riskAccelerations[previousIndex] as number)} x2={x(index)} y2={yRisk(value)} />;
            })}
            {riskAccelerations.map((value, index) => value === null ? null : <circle className="risk-history-point score" key={`risk-point-${data[index].snapshot.as_of}`} cx={x(index)} cy={yRisk(value)} r={index === data.length - 1 ? 3.5 : 2}><title>{`${data[index].snapshot.as_of} · 위험 Δ² ${signed(value)}pt`}</title></circle>)}
            {benchmarkAccelerations.map((value, index) => index > 0 && value !== null && benchmarkAccelerations[index - 1] !== null ? <line key={`ndx-${data[index].snapshot.as_of}`} className="risk-history-nasdaq" x1={x(index - 1)} y1={yBenchmark(benchmarkAccelerations[index - 1] as number)} x2={x(index)} y2={yBenchmark(value)} /> : null)}
            {benchmarkAccelerations.map((value, index) => value === null ? null : <circle className="risk-history-point nasdaq" key={`ndx-point-${data[index].snapshot.as_of}`} cx={x(index)} cy={yBenchmark(value)} r={index === data.length - 1 ? 3.5 : 1.8}><title>{`${data[index].snapshot.as_of} · 나스닥100 Δ수익률 ${signed(value, 2)}%p`}</title></circle>)}
            {benchmarkValues.length > 0 && [benchmarkCeiling, 0, -benchmarkCeiling].map((tick) => <text className="risk-history-y risk-history-y-right" key={tick} x={width - right + 8} y={yBenchmark(tick) + 3}>{signed(tick, 1)}%p</text>)}
            {labelIndexes.map((index) => <text className="risk-history-x" key={data[index].snapshot.as_of} x={x(index)} y={height - 10} textAnchor={index === 2 ? "start" : index === data.length - 1 ? "end" : "middle"}>{data[index].snapshot.as_of.slice(5)}</text>)}
          </svg>
        </div>
      </div>
      <footer className="risk-history-footer">
        <div className="risk-history-legend"><span><i className="score" />위험 가속도 · 좌측</span><span><i className="nasdaq" />지수 가속도 · 우측</span></div>
        <details className="risk-history-method"><summary>해석 방법</summary><p>양수는 전일보다 기울기가 커졌다는 의미이며, 지수 상승 자체를 의미하지는 않습니다.<br />모델 전환은 옅은 점선으로 연결하고, 결측 구간은 연결하지 않습니다.</p></details>
      </footer>
    </section>
  );
}

function ModelPerspectives({ snapshot, roro }: { snapshot: DashboardSnapshot; roro: RoroShadowSnapshot | null }) {
  const probabilities = [
    ["BULL", snapshot.regime.p_bull, "bull"],
    ["SIDEWAYS", snapshot.regime.p_sideways, "sideways"],
    ["BEAR", snapshot.regime.p_bear, "bear"],
  ] as const;
  const activeHmm = probabilities.reduce((best, item) => item[1] > best[1] ? item : best);
  const hmmLevel = { BULL: 0, SIDEWAYS: 1, BEAR: 2 }[activeHmm[0]];
  const roroLevel = roro ? { GREEN: 0, YELLOW: 1, RED: 2 }[roro.payload.risk_regime] : null;
  const comparison = roroLevel == null
    ? "MARKET LENS 데이터 대기"
    : hmmLevel === roroLevel
      ? "두 모델의 위험 방향 일치"
      : hmmLevel > roroLevel
        ? "HMM이 더 방어적으로 해석"
        : "MARKET LENS가 더 방어적으로 해석";

  return (
    <section className="model-perspectives" aria-label="Market Lens와 HMM 모델 관점 비교">
      <div className="perspective-head"><div><span className="section-label">MODEL PERSPECTIVES · DIAGNOSTIC</span><h2>MARKET LENS vs HMM</h2></div><p>MARKET LENS는 Market Lens의 멀티자산 위험 판단 · HMM은 느린 상태 인식이며 전망 확률이 아닙니다.</p></div>
      <div className="perspective-grid">
        <article className="perspective-card roro-perspective">
          <span>PRIMARY · Market Lens MARKET LENS</span>
          <strong>{roro ? roro.payload.risk_regime : "—"}</strong>
          <b>{roro ? `위험점수 ${Math.round(roro.payload.market_risk_score)} · ${PHASE_LABEL[roro.payload.trend_phase] ?? roro.payload.trend_phase}` : "해당 날짜 MARKET LENS 없음"}</b>
        </article>
        <article className="perspective-card hmm-perspective">
          <span>SECONDARY · HMM STATE LENS</span>
          <strong>{HMM_STATE_LABEL[activeHmm[0]]}</strong>
          <b>내부 상태확률 · 전망 확률 아님</b>
          <div className="hmm-probabilities">
            {probabilities.map(([state, value, kind]) => <div key={state}><span><i className={kind} />{HMM_STATE_LABEL[state]}</span><b>{(value * 100).toFixed(1)}%</b><em><i className={kind} style={{ width: `${value * 100}%` }} /></em></div>)}
          </div>
        </article>
        <article className={`perspective-card comparison-perspective ${roroLevel === hmmLevel ? "aligned" : "divergent"}`}>
          <span>MODEL RELATIONSHIP</span>
          <strong>{roroLevel === hmmLevel ? "ALIGNED" : "DIVERGENCE"}</strong>
          <b>{comparison}</b>
        </article>
      </div>
    </section>
  );
}

const TURNING_LEVEL_KO: Record<TurningAlertLevel, string> = {
  NORMAL: "정상",
  WATCH: "관찰",
  WARNING: "경고",
  PRIORITY: "우선 경고",
};

function RpmGauge({ direction, row, reading }: {
  direction: "OUT" | "IN";
  row: SectorTurningRow;
  reading: TurningGaugeReading;
}) {
  return <article className={`rpm-gauge rpm-${direction.toLowerCase()} level-${reading.level.toLowerCase()}`}>
    <div className="rpm-gauge-title"><span>ROTATION {direction}</span><b>{row.sector_name}</b></div>
    <div className="rpm-dial" aria-label={`${row.sector_name} ${direction} 점수 ${reading.score.toFixed(0)}, 경고선 ${reading.threshold.toFixed(0)}`}>
      <svg viewBox="0 0 120 68" aria-hidden="true">
        <path className="rpm-track" pathLength="100" d="M 12 60 A 48 48 0 0 1 108 60" />
        <path className="rpm-value" pathLength="100" strokeDasharray={`${reading.score} 100`} d="M 12 60 A 48 48 0 0 1 108 60" />
      </svg>
      <strong>{reading.score.toFixed(0)}</strong><small>/ 100</small>
    </div>
    <div className="rpm-gauge-foot"><b>{TURNING_LEVEL_KO[reading.level]}</b><span>경고선 {reading.threshold.toFixed(0)}</span></div>
  </article>;
}

function DormantRpmGauge({ direction }: { direction: "OUT" | "IN" }) {
  return <article className={`rpm-gauge rpm-${direction.toLowerCase()} rpm-dormant`}>
    <div className="rpm-gauge-title"><span>ROTATION {direction}</span><b>활성 섹터 없음</b></div>
    <div className="rpm-dial" aria-label={`Rotation ${direction} 비활성`}><strong>—</strong></div>
    <div className="rpm-gauge-foot"><b>DORMANT</b><span>현재 계산 방향 아님</span></div>
  </article>;
}

function CompactGauge({ reading }: { reading: TurningGaugeReading | null }) {
  if (!reading) return <span className="turning-dormant">DORMANT</span>;
  return <div className={`turning-reading level-${reading.level.toLowerCase()}`}>
    <div><b>{reading.score.toFixed(0)}</b><span>/ {reading.threshold.toFixed(0)}</span><em>{TURNING_LEVEL_KO[reading.level]}</em></div>
    <i><span style={{ width: `${reading.score}%` }} /><u style={{ left: `${reading.threshold}%` }} /></i>
  </div>;
}

function SectorTurningPanel({ snapshot }: { snapshot: SectorTurningSnapshot }) {
  const ranked = [...snapshot.payload.rows].sort((left, right) => {
    const urgency = (row: SectorTurningRow) => Math.max(
      row.rotation_out?.alert_margin ?? -100,
      row.rotation_in?.alert_margin ?? -100,
    );
    return urgency(right) - urgency(left);
  });
  const hottest = (key: "rotation_out" | "rotation_in") => ranked
    .filter((row) => row[key] !== null)
    .sort((left, right) => (right[key]?.score ?? -1) - (left[key]?.score ?? -1))[0];
  const out = hottest("rotation_out");
  const incoming = hottest("rotation_in");
  return <section className="sector-turning-panel" aria-label="섹터 IN OUT 추세 전환 게이지">
    <div className="section-head turning-head">
      <div><div className="section-label">SECTOR TURNING · RESEARCH SHADOW</div><h2>IN / OUT RPM Gauge</h2></div>
      <p>관심·노이즈 반영 경고선 85–95 · AS OF {snapshot.as_of}</p>
    </div>
    <div className="rpm-grid">
      {out?.rotation_out ? <RpmGauge direction="OUT" row={out} reading={out.rotation_out} /> : <DormantRpmGauge direction="OUT" />}
      {incoming?.rotation_in ? <RpmGauge direction="IN" row={incoming} reading={incoming.rotation_in} /> : <DormantRpmGauge direction="IN" />}
      <div className="rpm-method">
        <span>THRESHOLD ENGINE</span><strong>점수는 고정 · 경고선만 조정</strong>
        <p>고관심·확인 부족이면 경고선을 높이고, breadth와 잔차가 같은 방향이면 낮춥니다.</p>
        <small>현재 관심도는 뉴스가 아닌 가격 잔차·변동성·가속도 프록시입니다.</small>
      </div>
    </div>
    <div className="table-wrap turning-table"><table>
      <thead><tr><th>SECTOR</th><th>ATTENTION</th><th>ROTATION OUT · 점수/경고선</th><th>ROTATION IN · 점수/경고선</th></tr></thead>
      <tbody>{ranked.map((row) => <tr key={row.sector}>
        <td><b>{row.sector}</b><span>{row.sector_name}</span></td>
        <td><b className={`attention-band attention-${row.attention_band.toLowerCase()}`}>{row.attention_score === null ? "N/A" : row.attention_score.toFixed(0)}</b><small>{row.attention_band}</small></td>
        <td><CompactGauge reading={row.rotation_out} /></td>
        <td><CompactGauge reading={row.rotation_in} /></td>
      </tr>)}</tbody>
    </table></div>
    <div className="legend"><span>85–95 동적 경고선</span><span>95+ 우선 경고</span><span>미활성 방향은 DORMANT</span><b>PIT PARTIAL · T+1 ACTION</b></div>
  </section>;
}

function SectorTable({ rows }: { rows: SectorRow[] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead><tr><th>20D RANK</th><th>SECTOR</th><th>RRG</th><th>TRAJECTORY</th><th>20D ALPHA</th><th>MODEL STATUS</th><th>RRG ALIGN</th></tr></thead>
        <tbody>{rows.map((row) => {
          const status = modelStatus(row);
          const alignment = rrgAlignment(row);
          return (
            <tr key={row.ticker} className={status === "MODEL_TOP3" ? "leader" : ""}>
              <td className="rank">{String(row.forecast_rank).padStart(2, "0")}</td>
              <td><b>{row.ticker}</b><span>{row.sector}</span></td>
              <td><i className={`quadrant q-${row.quadrant.toLowerCase()}`}>{QUADRANT_KO[row.quadrant] ?? row.quadrant}</i><small className="transition">{transitionLabel(row.quadrant_transition)}</small></td>
              <td><span className="heading">{TRAJECTORY_LABEL[row.tail_heading] ?? row.tail_heading.replaceAll("_", " ")}</span><small>Δ {row.delta_rs_ratio.toFixed(2)} / {row.delta_rs_momentum.toFixed(2)}</small></td>
              <td className={row.predicted_excess_20d > 0 ? "positive" : "negative"}>{signedPercent(row.predicted_excess_20d)}</td>
              <td><b className={`model-status status-${status.toLowerCase().replaceAll("_", "-")}`}>{status.replaceAll("_", " ")}</b></td>
              <td><b className={`rrg-align align-${alignment.toLowerCase()}`}>{alignment}</b></td>
            </tr>
          );
        })}</tbody>
      </table>
    </div>
  );
}

function SectorRoroTable({ snapshot, previous }: { snapshot: SectorRoroShadowSnapshot; previous: SectorRoroShadowSnapshot | null }) {
  const previousContext = new Map(previous?.payload.rows.map((row) => [row.sector, row.roro_context]));
  return (
    <div className="table-wrap">
      <table>
        <thead><tr><th>20D RANK</th><th>SECTOR</th><th>RRG</th><th>TRAJECTORY</th><th>ROTATION EVENT</th><th>20D ALPHA</th><th>MARKET FIT</th><th>FIT CHANGE</th><th>STATUS</th></tr></thead>
        <tbody>{snapshot.payload.rows.map((row) => {
          const before = previousContext.get(row.sector);
          const change = before === "RORO_MISMATCH" && row.roro_context === "FIT" ? "MISMATCH → FIT"
            : before === "FIT" && row.roro_context === "RORO_MISMATCH" ? "FIT → MISMATCH" : "—";
          return <tr key={row.sector} className={row.sector_status === "SELECT" ? "leader" : ""}>
            <td className="rank">{String(row.forecast_rank).padStart(2, "0")}</td>
            <td><b>{row.sector}</b><span>{row.sector_name}</span></td>
            <td><i className={`quadrant q-${row.quadrant.toLowerCase()}`}>{QUADRANT_KO[row.quadrant] ?? row.quadrant}</i><small className="transition">{transitionLabel(row.quadrant_transition)}</small></td>
            <td><span className="heading">{TRAJECTORY_LABEL[row.tail_heading] ?? row.tail_heading.replaceAll("_", " ")}</span><small>Δ {row.delta_rs_ratio.toFixed(2)} / {row.delta_rs_momentum.toFixed(2)}</small></td>
            <td><b className={`rrg-align ${rotationEvent(row) === "ROTATION IN" ? "align-supportive" : rotationEvent(row) === "ROTATION OUT" ? "align-divergent" : rotationEvent(row) === "EARLY BUILD" ? "align-early" : "align-na"}`}>{rotationEvent(row)}</b></td>
            <td className={row.predicted_excess_20d > 0 ? "positive" : "negative"}>{signedPercent(row.predicted_excess_20d)}</td>
            <td><b className={`rrg-align ${row.roro_context === "FIT" ? "align-supportive" : "align-divergent"}`}>{row.roro_context === "FIT" ? "FIT" : "MISMATCH"}</b><small className="sector-fit-score">{Math.round(row.roro_fit * 100)}</small></td>
            <td><b className={`rrg-align ${change.endsWith("FIT") ? "align-supportive" : change.endsWith("MISMATCH") ? "align-divergent" : ""}`}>{change}</b></td>
            <td><b className={`model-status status-${row.sector_status.toLowerCase()}`}>{row.sector_status}</b></td>
          </tr>;
        })}</tbody>
      </table>
    </div>
  );
}

function SectorHistoryPanel({ history, selectedDate }: { history: SectorHistory; selectedDate: string }) {
  const dates = useMemo(() => [...new Set(history.rows.map((row) => row.decision_date))].sort(), [history]);
  const nearestDate = useCallback((target: string) => dates.filter((day) => day <= target).at(-1) ?? dates[0] ?? "", [dates]);
  const [date, setDate] = useState(() => nearestDate(selectedDate || history.as_of));
  useEffect(() => setDate(nearestDate(selectedDate || history.as_of)), [history.as_of, nearestDate, selectedDate]);
  const rows = history.rows.filter((row) => row.decision_date === date).sort((a, b) => a.forecast_rank - b.forecast_rank);
  return <div className="sector-history-panel">
    <div className="sector-history-head">
      <div><span>HISTORICAL BACKFILL</span><h3>주간 신호 재현</h3><small>포트 이력 이전은 MARKET FIT N/A · 20D ALPHA/RRG는 유지</small></div>
      <label>판단일<select value={date} onChange={(event) => setDate(event.target.value)}>{dates.map((day) => <option key={day}>{day}</option>)}</select></label>
    </div>
    <div className="table-wrap">
      <table>
        <thead><tr><th>20D RANK</th><th>SECTOR</th><th>RRG</th><th>TRAJECTORY</th><th>ROTATION EVENT</th><th>20D ALPHA</th><th>ACTUAL 20D</th><th>MARKET FIT</th><th>FIT CHANGE</th><th>STATUS</th></tr></thead>
        <tbody>{rows.map((row) => <tr key={row.sector} className={row.sector_status === "SELECT" ? "leader" : ""}>
          <td className="rank">{String(row.forecast_rank).padStart(2, "0")}</td>
          <td><b>{row.sector}</b><span>{row.sector_name}</span></td>
          <td><i className={`quadrant q-${row.quadrant.toLowerCase()}`}>{QUADRANT_KO[row.quadrant] ?? row.quadrant}</i><small className="transition">{transitionLabel(row.quadrant_transition)}</small></td>
          <td><span className="heading">{TRAJECTORY_LABEL[row.tail_heading] ?? row.tail_heading.replaceAll("_", " ")}</span></td>
          <td><b className={`rrg-align ${rotationEvent(row) === "ROTATION IN" ? "align-supportive" : rotationEvent(row) === "ROTATION OUT" ? "align-divergent" : rotationEvent(row) === "EARLY BUILD" ? "align-early" : "align-na"}`}>{rotationEvent(row)}</b></td>
          <td className={row.predicted_excess_20d > 0 ? "positive" : "negative"}>{signedPercent(row.predicted_excess_20d)}</td>
          <td className={row.actual_excess_20d === null ? "pending" : row.actual_excess_20d > 0 ? "positive" : "negative"}>{row.actual_excess_20d === null ? "대기" : signedPercent(row.actual_excess_20d)}</td>
          <td>{row.roro_context === "N/A" ? <b className="rrg-align align-na">N/A</b> : <><b className={`rrg-align ${row.roro_context === "FIT" ? "align-supportive" : "align-divergent"}`}>{row.roro_context === "FIT" ? "FIT" : "MISMATCH"}</b><small className="sector-fit-score">{Math.round((row.roro_fit ?? 0) * 100)}</small></>}</td>
          <td><b className={`rrg-align ${row.fit_change.endsWith("→ FIT") ? "align-supportive" : row.fit_change.endsWith("MISMATCH") ? "align-divergent" : "align-na"}`}>{row.fit_change}</b></td>
          <td><b className={`model-status status-${row.sector_status.toLowerCase().replace("/", "-")}`}>{row.sector_status}</b></td>
        </tr>)}</tbody>
      </table>
    </div>
    <div className="legend"><span>{dates.length}주</span><span>{history.start} → {history.as_of}</span><b>RESEARCH · WEEKLY</b></div>
  </div>;
}

function ItInternalsPanel({ history, selectedDate }: { history: ItInternalsHistory; selectedDate: string }) {
  const rows = useMemo(() => history.rows.map((row, index) => decodeItInternalsRow(row, history.rows[index - 1])), [history]);
  const dates = useMemo(() => rows.map((row) => row.decisionDate), [rows]);
  const nearestDate = useCallback((target: string) => dates.filter((day) => day <= target).at(-1) ?? dates[0] ?? "", [dates]);
  const [date, setDate] = useState(() => nearestDate(selectedDate || history.as_of));
  useEffect(() => setDate(nearestDate(selectedDate || history.as_of)), [history.as_of, nearestDate, selectedDate]);
  const row = rows.find((candidate) => candidate.decisionDate === date) ?? rows[0]!;
  const matured = rows.filter((candidate) => candidate.pairActualWinner !== null);
  const hits = matured.filter((candidate) => candidate.currentPairLeader === candidate.pairActualWinner).length;
  const hitRate = matured.length ? hits / matured.length : null;
  const sleeves = [
    { name: "SEMICONDUCTORS", proxy: "SOXX", r20: row.soxx20, r60: row.soxx60, winner: "SEMICONDUCTOR" },
    { name: "SOFTWARE", proxy: "IGV", r20: row.igv20, r60: row.igv60, winner: "SOFTWARE" },
  ] as const;
  return <section className="sector-history-panel" aria-label="IT 내부 순환 연구 이력">
    <div className="sector-history-head">
      <div><span>IT INTERNALS · SHADOW</span><h3>XLK 내부 순환: Semiconductors vs Software</h3><small>SOXX와 IGV의 XLK 대비 상대수익률입니다. 진짜 하드웨어 전체나 변동성 조정 신호는 아니며, 공통 거래일 기준 일간 계산입니다.</small></div>
      <label>판단일<input type="date" min={history.start} max={history.as_of} value={date} onChange={(event) => setDate(nearestDate(event.target.value))} /></label>
    </div>
    <div className="table-wrap">
      <table>
        <thead><tr><th>SLEEVE</th><th>PROXY</th><th>20D vs XLK</th><th>60D vs XLK</th><th>20D LEADER</th><th>STATE</th><th>STATE CHANGE</th></tr></thead>
        <tbody>{sleeves.map((sleeve) => {
          const leading = row.currentPairLeader === sleeve.winner;
          return <tr key={sleeve.proxy} className={leading ? "leader" : ""}>
            <td><b>{sleeve.name}</b></td><td><b>{sleeve.proxy}</b></td>
            <td className={sleeve.r20 >= 0 ? "positive" : "negative"}>{signedPercent(sleeve.r20)}</td>
            <td className={sleeve.r60 >= 0 ? "positive" : "negative"}>{signedPercent(sleeve.r60)}</td>
            <td><b className={`model-status ${leading ? "status-select" : "status-watch"}`}>{leading ? "LEAD" : "TRAIL"}</b></td>
            <td><b className="rrg-align align-early">{IT_STATE_LABEL[row.state20]}</b><small className="sector-fit-score">{row.stateConfirmed ? "20D · 60D CONFIRMED" : "20D · 60D DIVERGENCE"}</small></td>
            <td><span className="heading">{row.stateTransition}</span></td>
          </tr>;
        })}</tbody>
      </table>
    </div>
    <div className="legend">
      <span>PAIR SPREAD 20D <b className={row.pairSpread20 >= 0 ? "positive" : "negative"}>{signedPercent(row.pairSpread20)}</b></span>
      <span>FORWARD 20D {row.pairForward20 !== null ? <b className={row.pairForward20 >= 0 ? "positive" : "negative"}>{signedPercent(row.pairForward20)}</b> : "대기"}</span>
      <span>20D LEADER CONTINUATION {hitRate === null ? "—" : `${(hitRate * 100).toFixed(1)}% · n=${matured.length} OVERLAPPING`}</span>
      <b>COMMON SESSION · DAILY · RESEARCH ONLY</b>
    </div>
    <ItSpreadChart rows={history.rows} date={date} prices={history.daily_pairs?.hw_sw} />
  </section>;
}

export default function Home() {
  const [token, setToken] = useState("");
  const [snapshot, setSnapshot] = useState<DashboardSnapshot | null>(null);
  const [roro, setRoro] = useState<RoroShadowSnapshot | null>(null);
  const [narrative, setNarrative] = useState<RoroNarrative | null>(null);
  const [roroHistory, setRoroHistory] = useState<RoroHistoryPoint[]>([]);
  const [roroOnlyHistory, setRoroOnlyHistory] = useState(false);
  const [rotationEntry, setRotationEntry] = useState<RotationEntry | null>(null);
  const [sectorHybrid, setSectorHybrid] = useState<HybridSnapshot | null>(null);
  const [sectorTransition, setSectorTransition] = useState<SectorTransitionSnapshot | null>(null);
  const [sectorRoro, setSectorRoro] = useState<SectorRoroShadowSnapshot | null>(null);
  const [previousSectorRoro, setPreviousSectorRoro] = useState<SectorRoroShadowSnapshot | null>(null);
  const [sectorTurning, setSectorTurning] = useState<SectorTurningSnapshot | null>(null);
  const [sectorHistory, setSectorHistory] = useState<SectorHistory | null>(null);
  const [itInternalsHistory, setItInternalsHistory] = useState<ItInternalsHistory | null>(null);
  const [error, setError] = useState("");
  const [selectedDate, setSelectedDate] = useState("");
  const [latestDate, setLatestDate] = useState("");
  const [loading, setLoading] = useState(false);
  const [refreshRequest, setRefreshRequest] = useState<RefreshRequest | null>(null);
  const [refreshSubmitting, setRefreshSubmitting] = useState(false);
  const requestRef = useRef<AbortController | null>(null);

  const load = useCallback(async (nextToken: string, requestedDate?: string) => {
    if (requestRef.current) return;
    const controller = new AbortController();
    requestRef.current = controller;
    setError("");
    setLoading(true);
    try {
      const endpoint = requestedDate
        ? `/api/snapshot?date=${encodeURIComponent(requestedDate)}`
        : "/api/snapshot";
      const roroEndpoint = requestedDate ? `/api/roro?include_narrative=1&date=${encodeURIComponent(requestedDate)}` : "/api/roro?include_narrative=1";
      const roroHistoryEndpoint = requestedDate ? `/api/roro?date=${encodeURIComponent(requestedDate)}&history=40` : "/api/roro?history=40";
      const sectorRoroEndpoint = requestedDate ? `/api/sector-roro?date=${encodeURIComponent(requestedDate)}&history=2` : "/api/sector-roro?history=2";
      const sectorTurningEndpoint = requestedDate ? `/api/sector-turning?date=${encodeURIComponent(requestedDate)}` : "/api/sector-turning";
      const [response, roroResponse, roroHistoryResponse, sectorRoroResponse, sectorTurningResponse, sectorHistoryResponse, itInternalsResponse, rotationEntryResponse, sectorTransitionResponse, hybridResponse] = await Promise.all([fetch(endpoint, {
        headers: { authorization: `Bearer ${nextToken}` },
        cache: "no-store",
        signal: controller.signal,
      }), fetch(roroEndpoint, {
        headers: { authorization: `Bearer ${nextToken}` },
        cache: "no-store",
        signal: controller.signal,
      }), fetch(roroHistoryEndpoint, {
        headers: { authorization: `Bearer ${nextToken}` },
        cache: "no-store",
        signal: controller.signal,
      }), fetch(sectorRoroEndpoint, {
        headers: { authorization: `Bearer ${nextToken}` },
        cache: "no-store",
        signal: controller.signal,
      }), fetch(sectorTurningEndpoint, {
        headers: { authorization: `Bearer ${nextToken}` },
        cache: "no-store",
        signal: controller.signal,
      }), fetch("/api/sector-history", {
        headers: { authorization: `Bearer ${nextToken}` },
        cache: "no-store",
        signal: controller.signal,
      }), fetch("/api/it-internals", {
        headers: { authorization: `Bearer ${nextToken}` },
        cache: "no-store",
        signal: controller.signal,
      }), fetch(requestedDate ? `/api/rotation-entry?date=${encodeURIComponent(requestedDate)}` : "/api/rotation-entry", {
        headers: { authorization: `Bearer ${nextToken}` }, cache: "no-store", signal: controller.signal,
      }), fetch(requestedDate ? `/api/sector-transition?date=${encodeURIComponent(requestedDate)}` : "/api/sector-transition", {
        headers: { authorization: `Bearer ${nextToken}` }, cache: "no-store", signal: controller.signal,
      }), fetch(requestedDate ? `/api/sector-hybrid?date=${encodeURIComponent(requestedDate)}` : "/api/sector-hybrid", {
        headers: { authorization: `Bearer ${nextToken}` }, cache: "no-store", signal: controller.signal,
      })]);
      const hybridCandidate: unknown = hybridResponse.ok ? await hybridResponse.json() : null;
      setSectorHybrid(isHybridSnapshot(hybridCandidate) ? hybridCandidate : null);
      const rotationCandidate: unknown = rotationEntryResponse.ok ? await rotationEntryResponse.json() : null;
      setRotationEntry(isRotationEntry(rotationCandidate) ? rotationCandidate : null);
      const sectorTransitionCandidate: unknown = sectorTransitionResponse.ok ? await sectorTransitionResponse.json() : null;
      setSectorTransition(isSectorTransitionSnapshot(sectorTransitionCandidate) ? sectorTransitionCandidate : null);
      const roroEnvelope = roroResponse.ok ? await roroResponse.json() : null;
      const roroCandidate: unknown = roroEnvelope?.snapshot ?? roroEnvelope;
      setNarrative(isRoroNarrative(roroEnvelope?.narrative) ? roroEnvelope.narrative : null);
      const historicalRoro = isRoroShadowSnapshot(roroCandidate) ? roroCandidate : null;
      const roroHistoryCandidate: unknown = roroHistoryResponse.ok ? await roroHistoryResponse.json() : null;
      const nextRoroHistory = Array.isArray(roroHistoryCandidate) && roroHistoryCandidate.every(isRoroHistoryPoint) ? roroHistoryCandidate : [];
      const sectorHistoryCandidate: unknown = sectorHistoryResponse.ok ? await sectorHistoryResponse.json() : null;
      const historicalSector = isSectorHistory(sectorHistoryCandidate) ? sectorHistoryCandidate : null;
      const itInternalsCandidate: unknown = itInternalsResponse.ok ? await itInternalsResponse.json() : null;
      const historicalItInternals = isItInternalsHistory(itInternalsCandidate) ? itInternalsCandidate : null;
      const sectorTurningCandidate: unknown = sectorTurningResponse.ok ? await sectorTurningResponse.json() : null;
      const historicalTurning = isSectorTurningSnapshot(sectorTurningCandidate) ? sectorTurningCandidate : null;
      if (!response.ok) {
        if (response.status === 401) {
          setSnapshot(null);
          setError("접근 코드를 확인하세요.");
        } else if (requestedDate && historicalRoro) {
          setRoro(historicalRoro);
          setRoroHistory(nextRoroHistory);
          setSectorRoro(null);
          setPreviousSectorRoro(null);
          setSectorTurning(historicalTurning);
          setSectorHistory(historicalSector);
          setItInternalsHistory(historicalItInternals);
          setRoroOnlyHistory(true);
          setSelectedDate(historicalRoro.as_of);
          setToken(nextToken);
          localStorage.setItem("regime-view-token", nextToken);
        } else if (requestedDate && (historicalTurning || (historicalSector && requestedDate >= historicalSector.start) || (historicalItInternals && requestedDate >= historicalItInternals.start))) {
          setRoro(null);
          setRoroHistory([]);
          setSectorRoro(null);
          setPreviousSectorRoro(null);
          setSectorTurning(historicalTurning);
          setSectorHistory(historicalSector);
          setItInternalsHistory(historicalItInternals);
          setRoroOnlyHistory(true);
          setSelectedDate(requestedDate);
          setToken(nextToken);
          localStorage.setItem("regime-view-token", nextToken);
        } else if (requestedDate) {
          setError("선택한 날짜 이전에 저장된 데이터가 없습니다.");
        } else {
          setError("아직 업로드된 데이터가 없습니다.");
        }
        return;
      }
      const nextSnapshot: DashboardSnapshot = await response.json();
      const nextRoro = historicalRoro && historicalRoro.as_of <= nextSnapshot.as_of ? historicalRoro : null;
      const sectorRoroCandidate: unknown = sectorRoroResponse.ok ? await sectorRoroResponse.json() : null;
      const sectorRoroHistory = Array.isArray(sectorRoroCandidate) && sectorRoroCandidate.every(isSectorRoroShadowSnapshot)
        ? sectorRoroCandidate : [];
      const lastSectorRoro = sectorRoroHistory.at(-1);
      const nextSectorRoro = lastSectorRoro && lastSectorRoro.as_of <= nextSnapshot.as_of ? lastSectorRoro : null;
      const nextPreviousSectorRoro = nextSectorRoro && sectorRoroHistory.length > 1 ? sectorRoroHistory.at(-2) ?? null : null;
      const nextSectorTurning = historicalTurning && historicalTurning.as_of <= nextSnapshot.as_of ? historicalTurning : null;
      setSnapshot(nextSnapshot);
      setRoro(nextRoro);
      setRoroHistory(nextRoroHistory);
      setRoroOnlyHistory(false);
      setSectorRoro(nextSectorRoro);
      setPreviousSectorRoro(nextPreviousSectorRoro);
      setSectorTurning(nextSectorTurning);
      setSectorHistory(historicalSector);
      setItInternalsHistory(historicalItInternals);
      const newestDate = isRotationEntry(rotationCandidate) && rotationCandidate.as_of > nextSnapshot.as_of ? rotationCandidate.as_of : nextSnapshot.as_of;
      setSelectedDate(requestedDate || newestDate);
      if (!requestedDate) setLatestDate(newestDate);
      setToken(nextToken);
      localStorage.setItem("regime-view-token", nextToken);
    } catch {
      if (!controller.signal.aborted) {
        setError("데이터를 불러오지 못했습니다. 잠시 후 다시 시도하세요.");
      }
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null;
        setLoading(false);
      }
    }
  }, []);

  const requestLatest = useCallback(async () => {
    if (!token) return;
    setError("");
    setRefreshSubmitting(true);
    try {
      const response = await fetch("/api/refresh", {
        method: "POST",
        headers: {
          authorization: `Bearer ${token}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({ action: "request" }),
      });
      if (!response.ok) throw new Error("refresh request failed");
      setRefreshRequest(await response.json() as RefreshRequest);
    } catch {
      setError("업데이트를 요청하지 못했습니다. 잠시 후 다시 시도하세요.");
    } finally {
      setRefreshSubmitting(false);
    }
  }, [token]);

  useEffect(() => {
    const saved = localStorage.getItem("regime-view-token");
    const timer = saved ? window.setTimeout(() => void load(saved), 0) : null;
    return () => {
      if (timer !== null) window.clearTimeout(timer);
      requestRef.current?.abort();
      requestRef.current = null;
    };
  }, [load]);

  const sectorOnlyHistory = roroOnlyHistory && !roro;
  const displayAsOf = roroOnlyHistory ? roro?.as_of ?? selectedDate : selectedDate;
  const showingLatest = Boolean(displayAsOf && latestDate && displayAsOf === latestDate);
  useEffect(() => {
    if (!token || !showingLatest) return;
    const timer = window.setInterval(() => void load(token), AUTO_REFRESH_MS);
    return () => window.clearInterval(timer);
  }, [load, showingLatest, token]);

  useEffect(() => {
    if (!token || !refreshRequest || !["PENDING", "RUNNING"].includes(refreshRequest.status)) return;
    const controller = new AbortController();
    let checking = false;
    const poll = async () => {
      if (checking) return;
      checking = true;
      try {
        const response = await fetch(`/api/refresh?id=${refreshRequest.id}`, {
          headers: { authorization: `Bearer ${token}` },
          cache: "no-store",
          signal: controller.signal,
        });
        if (!response.ok) throw new Error("refresh status failed");
        const next = await response.json() as RefreshRequest;
        setRefreshRequest(next);
        if (next.status === "SUCCEEDED") {
          await load(token);
        } else if (next.status === "FAILED") {
          setError(next.detail || "업데이트에 실패했습니다.");
        }
      } catch {
        if (!controller.signal.aborted) setError("업데이트 상태를 확인하지 못했습니다.");
      } finally {
        checking = false;
      }
    };
    const timer = window.setInterval(() => void poll(), REFRESH_POLL_MS);
    return () => {
      window.clearInterval(timer);
      controller.abort();
    };
  }, [load, refreshRequest?.id, refreshRequest?.status, token]);

  const refreshBusy = refreshSubmitting || refreshRequest?.status === "PENDING" || refreshRequest?.status === "RUNNING";
  const refreshStatus = refreshLabel(refreshRequest, refreshSubmitting);

  if (!snapshot) return <Login onUnlock={(value) => void load(value)} error={error} />;

  return (
    <main className="dashboard">
      <header className="dashboard-header">
        <div className="brand-block">
          <div className="eyebrow">Market Lens · MARKET INTELLIGENCE</div>
          <h1>Market Lens <span>시장 · 섹터 모니터</span></h1>
        </div>
        <div className="header-actions">
          <div className="as-of"><span>AS OF</span><b>{displayAsOf}</b><small>{showingLatest ? "LIVE" : "HISTORY"}</small></div>
          <form onSubmit={(event) => { event.preventDefault(); if (selectedDate) void load(token, selectedDate); }}>
            <label htmlFor="history-date">기준일</label>
            <input id="history-date" type="date" value={selectedDate} max={latestDate || undefined} disabled={loading} onChange={(event) => setSelectedDate(event.target.value)} />
            <button type="submit" disabled={loading || !selectedDate}>{loading ? "…" : "조회"}</button>
          </form>
          <button disabled={loading || refreshBusy} onClick={() => void requestLatest()}>최신</button>
          <ThemeToggle />
          <button className="quiet" onClick={() => { requestRef.current?.abort(); requestRef.current = null; setLoading(false); setRefreshRequest(null); localStorage.removeItem("regime-view-token"); setSnapshot(null); setRoro(null); setRoroHistory([]); setRoroOnlyHistory(false); setSectorRoro(null); setPreviousSectorRoro(null); setSectorTurning(null); setSectorTransition(null); setSectorHistory(null); setItInternalsHistory(null); setToken(""); }}>잠금</button>
          <span className="refresh-status" aria-live="polite">{refreshStatus ?? (showingLatest ? "5분 자동 확인" : "과거 조회 · 자동 확인 일시정지")}</span>
          {error && <span className="header-error" role="alert">{error}</span>}
        </div>
      </header>

      <nav className="workspace-nav" aria-label="대시보드 영역">
        {!sectorOnlyHistory && <a href="#market-overview">시장 요약</a>}
        <a href="#sector-rotation">섹터 순환</a>
        {itInternalsHistory && <a href="#relative-strength">상대강도 비교</a>}
        <a href="#research-reference">연구 참고</a>
      </nav>

      {!sectorOnlyHistory && <section id="market-overview" className="workspace-section" aria-labelledby="market-title">
        <div className="workspace-heading"><div><span className="section-label">MARKET OVERVIEW</span><h2 id="market-title">시장 요약</h2></div><p>위험 수준과 비중 판단</p></div>
        {roro ? <RoroHero snapshot={roro} history={roroHistory} narrative={narrative} /> : <RiskPolicyHero snapshot={snapshot} />}
        {roro && <details className="workspace-fold">
          <summary><span>위험 추이와 나스닥</span><small>최근 40개 관측 · 점수와 변화 속도</small></summary>
          <div className={`risk-history-layout${roroOnlyHistory ? " history-only" : ""}`}>
            <RiskScoreHistory points={roroHistory} />
            <RiskAccelerationHistory points={roroHistory} />
          </div>
        </details>}
      </section>}

      <section id="sector-rotation" className="workspace-section" aria-labelledby="rotation-title">
        <div className="workspace-heading"><div><span className="section-label">SECTOR ROTATION</span><h2 id="rotation-title">섹터 순환</h2></div><p>가격 기반 추세 전환 · 주간 판단</p></div>
        <SectorHybridPanel snapshot={sectorHybrid} date={selectedDate} />
      </section>

      {itInternalsHistory && <section id="relative-strength" className="workspace-section" aria-labelledby="relative-title">
        <div className="workspace-heading"><div><span className="section-label">RELATIVE STRENGTH</span><h2 id="relative-title">상대강도 비교</h2></div><p>일간 흐름과 장기 관계</p></div>
        <details className="workspace-fold" open>
          <summary><span>반도체 · 소프트웨어</span><small>SOXX / IGV · XLK 대비</small></summary>
          <ItInternalsPanel history={itInternalsHistory} selectedDate={selectedDate} />
        </details>
        <details className="workspace-fold">
          <summary><span>섹터 페어 비교</span><small>11개 섹터 선택 · |Z-score| 상위 2개</small></summary>
          <SectorPairPanel key={`${selectedDate}-${itInternalsHistory.as_of}`} prices={itInternalsHistory.daily_pairs?.comm_energy} sectors={itInternalsHistory.sector_prices} selectedDate={selectedDate} />
        </details>
      </section>}

      <section id="research-reference" className="workspace-section" aria-labelledby="research-title">
        <div className="workspace-heading"><div><span className="section-label">RESEARCH REFERENCE</span><h2 id="research-title">연구 참고</h2></div><p>보조 모델과 과거 검증</p></div>
        <details className="workspace-fold"><summary><span>기존 순위 기반 섹터 순환</span><small>20일 수익률 · 순위 개선 경보</small></summary><RotationEntryPanel snapshot={rotationEntry} date={selectedDate} /></details>
        {sectorTransition && sectorTransition.as_of <= selectedDate && <details className="workspace-fold"><summary><span>추세 전환 가능성</span><small>과거 대비 점수 · 연구 참고</small></summary><SectorTransitionPanel snapshot={sectorTransition} /></details>}
        {roroOnlyHistory && <p className="roro-history-note">{sectorOnlyHistory ? "이 날짜는 섹터·IT 연구 이력만 보관되어 있습니다. 가장 가까운 이전 주간 판단일을 표시합니다." : "이 날짜는 Market Lens 이력만 보관되어 있습니다. 정책·섹터 세부 카드는 표시하지 않습니다."}</p>}
        {!roroOnlyHistory && <details className="workspace-fold">
          <summary><span>시장 보조 모델</span><small>단기 위험 · 모델별 관점</small></summary>
          <section className="model-layer" aria-label="모델 세부 신호">
            <RiskCard title="FAST SIGNAL · CRASH GUARD" data={snapshot.crash_guard} />
            <RiskCard title="DRAWDOWN ONSET" data={snapshot.drawdown_onset} />
          </section>
          <ModelPerspectives snapshot={snapshot} roro={roro} />
        </details>}
        {sectorTurning && <details className="workspace-fold"><summary><span>기존 IN / OUT 게이지</span><small>추세 전환 연구</small></summary><SectorTurningPanel snapshot={sectorTurning} /></details>}
        {!roroOnlyHistory && <details className="workspace-fold"><summary><span>기존 RRG · 20일 수익률 모델</span><small>주간 판단 · 연구 참고</small></summary>
      <section className="sector-section">
        <div className="section-head">
          <div><div className="section-label">{sectorRoro ? "SECTOR MARKET LENS" : "WEEKLY LEADING FORECAST"}</div><h2>Sector Rotation</h2></div>
          <p>{sectorRoro ? `RRG · 20D alpha · market fit · AS OF ${sectorRoro.as_of}` : "RRG · 20D relative alpha"}</p>
        </div>
        {sectorRoro ? <>
          <SectorRoroTable snapshot={sectorRoro} previous={previousSectorRoro} />
          <div className="legend"><span><i className="dot green" /> SELECT</span><span><i className="dot blue" /> WATCH</span><span><i className="dot red" /> AVOID</span><b>{sectorRoro.payload.training_weeks}W TRAINING</b></div>
        </> : <>
          <SectorTable rows={snapshot.sector.rows} />
          <div className="legend"><span><i className="dot green" /> 주도</span><span><i className="dot blue" /> 회복</span><span><i className="dot amber" /> 둔화</span><span><i className="dot gray" /> 소외</span><b>{snapshot.sector.training_weeks}W TRAINING</b></div>
        </>}
      </section>
        </details>}
        {sectorHistory && <details className="workspace-fold"><summary><span>섹터 모델 과거 이력</span><small>저장된 판단과 실현 결과</small></summary><SectorHistoryPanel history={sectorHistory} selectedDate={selectedDate} /></details>}
      </section>

      <footer><p>모델 기반 참고지표</p><p>{new Date(sectorOnlyHistory && (sectorHistory || itInternalsHistory) ? (sectorHistory?.generated_at ?? itInternalsHistory!.generated_at) : roroOnlyHistory && roro ? roro.generated_at : snapshot.generated_at).toLocaleString("ko-KR")}</p></footer>
    </main>
  );
}
