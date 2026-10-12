"""One-line, fact-bound RORO narrative generation."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Callable


MODEL_VERSION = "google/gemini-2.5-flash-lite"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_OUTPUT_CHARS = 180
LOGGER = logging.getLogger(__name__)

_FACTOR_LABELS = {
    "credit": "회사채 위험",
    "rates_volatility": "채권시장 불안",
    "equity_volatility": "주식 변동성",
    "fx_carry": "엔화 충격",
    "safe_haven": "금 피난 수요",
    "equity_trend": "나스닥 추세",
    "market_breadth": "시장 참여 폭",
    "portfolio_correlation": "보유 종목 동조화",
    "portfolio_beta": "나스닥 민감도",
    "portfolio_concentration": "포트폴리오 쏠림",
    "portfolio_participation": "상승 참여도",
    "portfolio_drawdown": "포트폴리오 하락폭",
}

_FACTOR_SCORE_FIELDS = {
    **{name: f"{name}_score" for name in _FACTOR_LABELS},
    "market_breadth": "breadth_risk_percentile",
}

_CRASH_LABELS = {
    "vix_jump_1d": ("VIX 하루 상승률", "percent"),
    "vix_acceleration_1d": ("VIX 상승 가속도", "percent"),
    "nq_loss_3d": ("나스닥 3일 하락률", "percent"),
    "credit_loss_3d": ("회사채 3일 약세", "percent"),
    "vix_level": ("VIX 수준", "level"),
    "vix_change_5d": ("VIX 5일 상승률", "percent"),
    "nq_loss_5d": ("나스닥 5일 하락률", "percent"),
    "credit_loss_5d": ("회사채 5일 약세", "percent"),
    "rsp_weakness_5d": ("동일가중주 5일 상대약세", "percent"),
    "iwm_weakness_5d": ("중소형주 5일 상대약세", "percent"),
}

_EVIDENCE_LABELS = {
    "VIX_CURVE_5D_DOWN": "변동성 곡선 안정",
    "VIX_CURVE_5D_UP": "변동성 곡선 악화",
    "HY_OAS_20D_NON_WIDENING": "회사채 위험 안정",
    "HY_OAS_20D_WIDENING": "회사채 위험 확대",
    "PORTFOLIO_CORRELATION_5D_DOWN": "보유 종목 동조화 완화",
    "PORTFOLIO_CORRELATION_5D_UP": "보유 종목 동조화 상승",
    "MARKET_BREADTH_5D_UP": "시장 참여 폭 개선",
    "MARKET_BREADTH_5D_DOWN": "시장 참여 폭 악화",
    "PORTFOLIO_BREADTH_MA20_5D_UP": "보유 종목의 단기 추세 개선",
    "PORTFOLIO_BREADTH_MA20_5D_DOWN": "보유 종목의 단기 추세 악화",
    "NQ_ABOVE_MA5": "나스닥의 단기 추세 회복",
    "NQ_BELOW_MA5": "나스닥의 단기 추세 약화",
}


def load_api_key(env_path: str | Path) -> str | None:
    """Read only OPENROUTER_API_KEY from a local dotenv file; never log it."""
    value = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if value:
        return value
    path = Path(env_path)
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"\s*(?:export\s+)?OPENROUTER_API_KEY\s*=\s*['\"]?([^'\"\s#]+)['\"]?\s*", line)
        if match:
            return match.group(1)
    return None


def _number(value: Any, digits: int = 1) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if math.isfinite(number) else None


def _facts(
    payload: dict[str, Any], crash_guard: dict[str, Any] | None = None
) -> str:
    policy = payload["policy"]
    evidence = [_EVIDENCE_LABELS.get(str(item), str(item)) for item in payload.get("turn_evidence", [])]
    factors = []
    for name, label in _FACTOR_LABELS.items():
        factors.append({
            "팩터": label,
            "점수": _number(payload.get(_FACTOR_SCORE_FIELDS[name])),
            "1일변화": _number(payload.get(f"{name}_score_trajectory_1d")),
            "5일변화": _number(payload.get(f"{name}_score_trajectory_5d")),
            "20일변화": _number(payload.get(f"{name}_score_trajectory_20d")),
            "종합점수기여": _number(payload.get(f"{name}_contribution")),
        })
    short_term = None
    if crash_guard:
        components = []
        for name, (label, unit) in _CRASH_LABELS.items():
            item = crash_guard.get("components", {}).get(name, {})
            raw = _number(item.get("raw"), 4)
            percentile = _number(item.get("percentile"), 3)
            components.append({
                "요인": label,
                "원값": None if raw is None else round(raw * 100, 2) if unit == "percent" else raw,
                "단위": "%" if unit == "percent" else "지수",
                "과거백분위": None if percentile is None else round(percentile * 100, 1),
            })
        short_term = {
            "점수": _number(crash_guard.get("score"), 3),
            "1일점수": _number(crash_guard.get("score_1d"), 3),
            "5일점수": _number(crash_guard.get("score_5d"), 3),
            "취약성경고기준": crash_guard.get("vulnerable_cutoff"),
            "추가축소검토기준": crash_guard.get("crash_watch_cutoff"),
            "상태": crash_guard.get("risk_band"),
            "세부요인": components,
        }
    return json.dumps({
        "종합위험점수": _number(payload.get("market_risk_score")),
        "전일대비_위험점수_변화": _number(payload.get("trajectory_1d")),
        "최근_5일_위험점수_변화": _number(payload.get("trajectory_5d")),
        "최근_20일_위험점수_변화": _number(payload.get("trajectory_20d")),
        "거시_위험점수": _number(payload.get("macro_risk_score")),
        "나스닥_추세_위험점수": _number(payload.get("market_trend_score")),
        "포트폴리오_내부_위험점수": _number(payload.get("portfolio_internal_score")),
        "12개_세부팩터": factors,
        "관측된_전환_근거": evidence,
        "단기_위험_발생건수": policy["fast_trigger_count"],
        "단기_위험_확인상태": policy["fast_status"],
        "Crash_Guard": short_term,
    }, ensure_ascii=False, separators=(",", ":"))


def generate_summary(
    payload: dict[str, Any],
    api_key: str,
    *,
    crash_guard: dict[str, Any] | None = None,
    request: Callable[..., Any] = urllib.request.urlopen,
    diagnostics: dict[str, Any] | None = None,
) -> str | None:
    """Return one Korean sentence or None; the caller keeps its deterministic fallback."""
    diagnostic = diagnostics if diagnostics is not None else {}
    diagnostic.clear()
    diagnostic.update(model=MODEL_VERSION, status="started")
    def fail(reason: str, **details: Any) -> None:
        diagnostic.update(status="failed", reason=reason, **details)
        # Never log exception text, response bodies, prompts, or credentials.
        LOGGER.warning("RORO narrative %s", json.dumps(diagnostic, ensure_ascii=True))
    if not api_key:
        fail("missing_api_key")
        return None
    try:
        body = {
            "model": MODEL_VERSION,
            "max_tokens": 256,
            "temperature": 0.2,
            "provider": {"data_collection": "deny"},
            "messages": [
                {"role": "system", "content": "제공된 사실만 사용해 외부 투자자가 이해할 수 있는 한국어 한 문장(180자 이내)을 쓴다. 먼저 현재 시장 위험의 방향과 단기 경고 단계를 말하고, 점수 기여·변화·백분위가 큰 2~3개 요인만 골라 이유를 설명한다. 모든 팩터를 나열하지 않는다. 0.60 이상은 단기 취약성 경고, 0.80 이상은 추가 축소 검토 단계다. 손실률·약세 원값은 양수일수록 하락이 크다는 뜻이다. 모델명·상태코드·영문 약어는 자연어로 바꾸고 미래 예측·매매 권유·제공되지 않은 인과 추론은 금지한다."},
                {"role": "user", "content": _facts(payload, crash_guard)},
            ],
        }
        req = urllib.request.Request(
            API_URL,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with request(req, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
        choices = result.get("choices") or []
        choice = choices[0] if choices else {}
        finish_reason = choice.get("finish_reason")
        if not choice or finish_reason not in {None, "stop"}:
            fail("response_not_completed", incomplete_reason=finish_reason or "unknown")
            return None
        text = (choice.get("message") or {}).get("content") or ""
        text = " ".join(str(text).split())
        if not text or len(text) > MAX_OUTPUT_CHARS:
            fail("empty_output" if not text else "output_too_long")
            return None
        diagnostic.update(status="success", output_chars=len(text))
        LOGGER.info("RORO narrative success model=%s", MODEL_VERSION)
        return text
    except urllib.error.HTTPError as error:
        fail("http_error", http_status=error.code)
        return None
    except Exception as error:
        fail("request_or_input_error", exception_type=type(error).__name__)
        return None


def fallback_summary(payload: dict[str, Any]) -> str:
    score = _number(payload.get("market_risk_score"))
    change = _number(payload.get("trajectory_1d"))
    movement = "전일 비교 자료가 없으며" if change is None else f"전일 대비 {change:+g}점 변동했으며"
    return f"위험점수는 {score:g}점으로 {movement}, 세부 위험 신호를 함께 확인해야 합니다." if score is not None else "위험점수 입력이 부족해 시장 해설을 생성하지 못했습니다."


def regenerate_observation(observation: dict[str, Any], api_key: str, *, crash_guard=None, request=urllib.request.urlopen) -> dict[str, Any]:
    """Generate an overlay bound to an immutable observation; never mutate it."""
    from datetime import datetime, timezone
    diagnostics: dict[str, Any] = {}
    summary = generate_summary(observation["payload"], api_key, crash_guard=crash_guard,
                               request=request, diagnostics=diagnostics)
    return {"as_of": observation["date"], "series_version": observation["series_version"],
            "provenance_hash": observation["provenance_hash"], "model": MODEL_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary, "diagnostics": diagnostics}


def main() -> None:
    import argparse
    from roro_shadow import read_shadow_history
    from roro_site_publisher import DEFAULT_NARRATIVE_ENV, DEFAULT_CONFIG, DEFAULT_CONSENT, _headers
    from crash_guard import get_crash_guard_narrative_context
    parser = argparse.ArgumentParser(description="Regenerate only the narrative; no score rebuild or history mutation.")
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--shadow-database", type=Path, default=Path("outputs/roro_shadow/roro_shadow.db"))
    parser.add_argument("--publish-database", type=Path, default=Path("regime_router_publish.db"))
    parser.add_argument("--narrative-env", type=Path, default=DEFAULT_NARRATIVE_ENV)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--consent", type=Path, default=DEFAULT_CONSENT)
    parser.add_argument("--upload", action="store_true", help="Upload a successful narrative overlay (requires deployed endpoint).")
    parser.add_argument("--output", type=Path, default=Path("outputs/roro_shadow/narrative_result.json"))
    args = parser.parse_args()
    config = None
    if args.upload:
        consent = json.loads(args.consent.read_text(encoding="utf-8-sig"))
        if consent.get("allow_portfolio_aggregate_upload") is not True:
            parser.error("aggregate upload consent required")
        config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    observations = read_shadow_history(args.shadow_database, start=args.as_of, end=args.as_of)
    if len(observations) != 1:
        parser.error("exact-date snapshot is required")
    try:
        crash_guard = get_crash_guard_narrative_context(args.publish_database, args.as_of)
    except Exception as error:
        LOGGER.warning("RORO narrative crash context unavailable: %s", type(error).__name__)
        crash_guard = None
    result = regenerate_observation(observations[0], load_api_key(args.narrative_env) or "", crash_guard=crash_guard)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if not result["summary"]:
        print(json.dumps(result["diagnostics"]))
        raise SystemExit(1)
    if config:
        record = {key: value for key, value in result.items() if key != "diagnostics"}
        req = urllib.request.Request(config["site_url"].rstrip("/") + "/api/roro",
            data=json.dumps({"narrative": record}, ensure_ascii=False).encode("utf-8"),
            headers=_headers(config["upload_token"], config.get("sites_bypass_token")), method="POST")
        with urllib.request.urlopen(req, timeout=30) as response:
            uploaded = json.loads(response.read())
        if uploaded.get("ok") is not True:
            raise RuntimeError("narrative upload was not acknowledged")
    print(json.dumps({"as_of": args.as_of, "summary": result["summary"], "uploaded": bool(config)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
