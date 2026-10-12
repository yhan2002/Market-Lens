"""Publish aggregate-only RORO SHADOW history to the separate Sites endpoint."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
from contextlib import closing
from datetime import date
from pathlib import Path
from typing import Any

from roro_q20 import validate_q20_readiness_metadata
from roro_narrative import load_api_key
from roro_shadow import (
    BREADTH_REPAIR_SERIES_VERSION,
    SERIES_VERSION,
    build_shadow_series,
    model_versions_for_series,
    read_shadow_history,
    validate_policy_audit,
)


APP_DIR = Path(__file__).resolve().parent
DEFAULT_PUBLISH_DATABASE = APP_DIR / "regime_router_publish.db"
DEFAULT_PRIVATE_DATABASE = APP_DIR / "regime_router_shadow.db"
DEFAULT_SHADOW_DATABASE = APP_DIR / "outputs" / "roro_shadow" / "roro_shadow.db"
DEFAULT_CONFIG = APP_DIR / "site_upload_config.json"
DEFAULT_CONSENT = APP_DIR / "roro_publish_consent.json"
DEFAULT_OUTPUT = APP_DIR / "outputs" / "roro_shadow" / "site_payload.json"
DEFAULT_NARRATIVE_ENV = APP_DIR.parent / "telegram-news-pipeline" / ".env"
UPLOAD_BATCH_SIZE = 1
BENCHMARK_BATCH_SIZE = 40
CHART_BENCHMARK = "^NDX"


def _latest_nq_date(database: str | Path) -> str:
    path = Path(database)
    if not path.is_file():
        raise FileNotFoundError(path)
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        row = connection.execute(
            "SELECT MAX(date) FROM daily_prices WHERE ticker='NQ=F'"
        ).fetchone()
    if not row or not row[0]:
        raise ValueError("NQ=F 기준일이 없습니다.")
    return str(row[0])[:10]


def _nasdaq100_closes(database: str | Path, dates: list[str]) -> list[dict[str, Any]]:
    if not dates:
        return []
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        rows = connection.execute(
            """WITH sessions AS (
                   SELECT DISTINCT date FROM daily_prices WHERE ticker IN (?, 'SPY') AND date<=?
               ), prices AS (
                   SELECT s.date, p.close FROM sessions s LEFT JOIN daily_prices p
                   ON p.date=s.date AND p.ticker=?
               )
               SELECT date, close, LAG(date) OVER (ORDER BY date), LAG(close) OVER (ORDER BY date)
               FROM prices ORDER BY date""",
            (CHART_BENCHMARK, max(dates), CHART_BENCHMARK),
        ).fetchall()
    requested = set(dates)
    prices = []
    for day, close, previous_day, previous_close in rows:
        if day not in requested or close is None or not math.isfinite(close) or close <= 0:
            continue
        previous_valid = previous_close is not None and math.isfinite(previous_close) and previous_close > 0
        prices.append({"as_of": day, "close": float(close),
                       "previous_as_of": previous_day if previous_valid else None,
                       "previous_close": float(previous_close) if previous_valid else None})
    return prices


def web_snapshot(observation: dict[str, Any]) -> dict[str, Any]:
    """Wrap one immutable aggregate observation in the public r6 envelope."""
    payload = observation.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("RORO observation payload is missing")
    required = {
        "market_risk_score", "risk_regime", "recommended_long", "market_trend_score",
        "portfolio_internal_score", "confidence", "data_completeness", "data_status",
        "trend_phase", "turn_evidence", "checklist_available_count",
        "bottom_confirmation_count", "top_confirmation_count", "risk_drivers", "relief_drivers",
        "history_weight_coverage_20d", "history_weight_coverage_60d", "q20",
        "policy",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError("missing RORO web fields: " + ", ".join(missing))
    forbidden = {"ticker", "isin", "name", "quantity", "market_value", "daily_holdings"}
    if forbidden & payload.keys():
        raise ValueError("holding identity cannot be published")
    if str(observation.get("series_version")) not in {SERIES_VERSION, BREADTH_REPAIR_SERIES_VERSION}:
        raise ValueError("RORO publisher accepts only the current r6 series")
    day = str(observation["date"])
    if observation.get("input_cutoff") != day:
        raise ValueError("RORO input cutoff must equal observation date")
    validate_q20_readiness_metadata(payload["q20"])
    validate_policy_audit(
        payload["policy"],
        recommended_long=payload["recommended_long"],
        risk_score=payload["market_risk_score"],
        allow_session_jpy=True,
    )
    return {
        "schema_version": 4,
        "series_version": str(observation["series_version"]),
        "as_of": day,
        "generated_at": str(observation["created_at"]),
        "input_cutoff": day,
        "provenance_hash": str(observation["provenance_hash"]),
        "model_versions": model_versions_for_series(str(observation["series_version"])),
        "payload": payload,
    }


def _headers(token: str, sites_bypass_token: str | None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    if sites_bypass_token:
        headers["OAI-Sites-Authorization"] = f"Bearer {sites_bypass_token}"
    return headers


def upload_snapshots(
    snapshots: list[dict[str, Any]],
    site_url: str,
    token: str,
    sites_bypass_token: str | None = None,
) -> dict[str, Any]:
    if not snapshots:
        raise ValueError("no RORO snapshots to upload")
    endpoint = site_url.rstrip("/") + "/api/roro"
    totals = {"rows": 0, "inserted": 0, "idempotent": 0}
    for offset in range(0, len(snapshots), UPLOAD_BATCH_SIZE):
        batch = snapshots[offset : offset + UPLOAD_BATCH_SIZE]
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(batch, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8"),
            headers=_headers(token, sites_bypass_token),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"RORO 업로드 실패 HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(f"RORO 업로드 연결 실패: {error}") from error
        totals["rows"] += int(result.get("rows", len(batch)))
        totals["inserted"] += int(result.get("inserted", 0))
        totals["idempotent"] += int(result.get("idempotent", 0))
    totals.update({"series": snapshots[0]["series_version"], "as_of": snapshots[-1]["as_of"]})
    return totals


def upload_benchmark_prices(
    prices: list[dict[str, Any]],
    site_url: str,
    token: str,
    sites_bypass_token: str | None = None,
) -> dict[str, int]:
    if not prices:
        raise ValueError("no benchmark prices to upload")
    seen = set()
    def valid_close(value: Any) -> bool:
        return type(value) in (int, float) and math.isfinite(value) and 0 < value < 1e9

    for item in prices:
        try:
            day = date.fromisoformat(item["as_of"])
            previous = item["previous_as_of"]
            previous_close = item["previous_close"]
            valid = (set(item) == {"as_of", "close", "previous_as_of", "previous_close"}
                     and day.isoformat() == item["as_of"] and item["as_of"] not in seen
                     and valid_close(item["close"])
                     and ((previous is None and previous_close is None)
                          or (isinstance(previous, str) and date.fromisoformat(previous).isoformat() == previous
                              and previous < item["as_of"] and valid_close(previous_close))))
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError("invalid Nasdaq100 benchmark price")
        seen.add(item["as_of"])
    endpoint = site_url.rstrip("/") + "/api/roro"
    totals = {"rows": 0, "updated": 0}
    for offset in range(0, len(prices), BENCHMARK_BATCH_SIZE):
        batch = prices[offset : offset + BENCHMARK_BATCH_SIZE]
        request = urllib.request.Request(
            endpoint,
            data=json.dumps({"benchmark_ticker": CHART_BENCHMARK, "benchmark_prices": batch}, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8"),
            headers=_headers(token, sites_bypass_token),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"NASDAQ 종가 업로드 실패 HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(f"NASDAQ 종가 업로드 연결 실패: {error}") from error
        totals["rows"] += int(result.get("rows", len(batch)))
        totals["updated"] += int(result.get("updated", 0))
    return totals


def download_snapshot(
    site_url: str,
    token: str,
    *,
    series: str = SERIES_VERSION,
    as_of: str | None = None,
    sites_bypass_token: str | None = None,
) -> dict[str, Any] | None:
    query = {"series": series}
    if as_of is not None:
        query["date"] = as_of
    endpoint = site_url.rstrip("/") + "/api/roro?" + urllib.parse.urlencode(query)
    headers = _headers(token, sites_bypass_token)
    headers.pop("Content-Type", None)
    request = urllib.request.Request(endpoint, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"RORO 조회 실패 HTTP {error.code}: {detail}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"RORO 조회 연결 실패: {error}") from error


def build_and_publish(
    publish_database: str | Path,
    private_database: str | Path,
    shadow_database: str | Path,
    *,
    price_observation_database: str | Path,
    breadth_observation_database: str | Path | None = None,
    as_of: str | None = None,
    latest_only: bool = False,
    require_as_of: bool = False,
    config: dict[str, Any] | None = None,
    narrative_env: str | Path = DEFAULT_NARRATIVE_ENV,
    repair_missing_breadth: bool = False,
    no_narrative: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    cutoff = as_of or _latest_nq_date(publish_database)
    build = build_shadow_series(
        publish_database, private_database, shadow_database,
        price_observation_database=price_observation_database,
        breadth_observation_database=breadth_observation_database,
        as_of=cutoff, latest_only=latest_only,
        narrative_api_key=None if no_narrative else load_api_key(narrative_env),
        require_complete_breadth=require_as_of,
        repair_missing_breadth=repair_missing_breadth,
    )
    effective_cutoff = str(build.get("effective_as_of", build["as_of"]))
    if require_as_of and cutoff is not None and effective_cutoff != cutoff:
        raise RuntimeError(
            f"Market Lens 입력 미완료: 요청일 {cutoff}, 생성 가능일 {effective_cutoff}"
        )
    observations = read_shadow_history(
        shadow_database,
        series_version=str(build.get("series_version", SERIES_VERSION)),
        start=effective_cutoff if latest_only else None,
        end=effective_cutoff,
    )
    if latest_only and observations:
        observations = observations[-1:]
    if require_as_of and (
        not observations
        or observations[-1].get("payload", {}).get("macro_data_status") != "FULL"
    ):
        raise RuntimeError(f"Market Lens MACRO 입력 미완료: 요청일 {cutoff}")
    if require_as_of and observations[-1].get("payload", {}).get("breadth_risk_percentile") is None:
        raise RuntimeError(f"Market Lens 시장 참여 폭 입력 미완료: 요청일 {cutoff}; 게시하지 않았습니다.")
    snapshots = [web_snapshot(observation) for observation in observations]
    uploaded = None
    if config is not None:
        uploaded = upload_snapshots(
            snapshots,
            str(config["site_url"]),
            str(config["upload_token"]),
            config.get("sites_bypass_token"),
        )
        prices = _nasdaq100_closes(publish_database, [snapshot["as_of"] for snapshot in snapshots])
        uploaded["benchmark_rows"] = 0
        uploaded["benchmark_missing_dates"] = sorted({snapshot["as_of"] for snapshot in snapshots} - {price["as_of"] for price in prices})
        if prices:
            benchmark_uploaded = upload_benchmark_prices(
                prices, str(config["site_url"]), str(config["upload_token"]), config.get("sites_bypass_token"),
            )
            uploaded["benchmark_rows"] = benchmark_uploaded["rows"]
    return build, snapshots, uploaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish-database", type=Path, default=DEFAULT_PUBLISH_DATABASE)
    parser.add_argument("--private-database", type=Path, default=DEFAULT_PRIVATE_DATABASE)
    parser.add_argument("--shadow-database", type=Path, default=DEFAULT_SHADOW_DATABASE)
    parser.add_argument("--price-observation-database", type=Path)
    parser.add_argument("--breadth-observation-database", type=Path)
    parser.add_argument("--benchmark-only", action="store_true", help="위험모델·GPT 호출 없이 NDX 차트 가격만 게시")
    parser.add_argument("--refresh-benchmark", action="store_true", help="--benchmark-only: NDX 최근 2년 가격 최초 수집/증분 갱신")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--consent", type=Path, default=DEFAULT_CONSENT)
    parser.add_argument("--as-of")
    parser.add_argument("--latest-only", action="store_true")
    parser.add_argument("--repair-missing-breadth", action="store_true", help="입력이 도착한 기존 결측 날짜만 별도 불변 보정 버전으로 게시")
    parser.add_argument(
        "--require-as-of",
        action="store_true",
        help="요청일 행을 생성·게시하지 못하면 성공 처리하지 않음",
    )
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--no-narrative", action="store_true", help="유료 한줄평 API 호출 없이 규칙 기반 설명만 저장")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--narrative-env", type=Path, default=DEFAULT_NARRATIVE_ENV)
    args = parser.parse_args()
    if getattr(args, "refresh_benchmark", False) and not getattr(args, "benchmark_only", False):
        parser.error("--refresh-benchmark requires --benchmark-only")
    if not getattr(args, "benchmark_only", False) and (not args.price_observation_database or not args.breadth_observation_database):
        parser.error("model publishing requires --price-observation-database and --breadth-observation-database")
    if not args.no_upload and not args.config.is_file():
        raise FileNotFoundError(args.config)
    consent = (
        json.loads(args.consent.read_text(encoding="utf-8-sig"))
        if args.consent.is_file()
        else {}
    )
    upload_allowed = consent.get("allow_portfolio_aggregate_upload") is True
    config = (
        json.loads(args.config.read_text(encoding="utf-8-sig"))
        if not args.no_upload and upload_allowed
        else None
    )
    if getattr(args, "benchmark_only", False):
        if args.refresh_benchmark:
            from market_data import download_prices, register_universe
            register_universe(args.publish_database, (CHART_BENCHMARK,), group_name="DISPLAY_BENCHMARK")
            download_prices(args.publish_database, years=2, group_name="DISPLAY_BENCHMARK")
        cutoff = args.as_of or _latest_nq_date(args.publish_database)
        with closing(sqlite3.connect(args.publish_database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            dates = [row[0] for row in connection.execute(
                "SELECT date FROM daily_prices WHERE ticker=? AND date<=? ORDER BY date", (CHART_BENCHMARK, cutoff),
            )]
        prices = _nasdaq100_closes(args.publish_database, dates)
        if not prices:
            raise ValueError("NDX 종가가 없습니다. --benchmark-only --refresh-benchmark --no-upload로 먼저 수집하세요.")
        uploaded = upload_benchmark_prices(prices, str(config["site_url"]), str(config["upload_token"]), config.get("sites_bypass_token")) if config else None
        print(json.dumps({"benchmark": CHART_BENCHMARK, "rows": len(prices), "as_of": prices[-1]["as_of"],
                          "upload": uploaded, "upload_allowed": upload_allowed}, ensure_ascii=False))
        return
    build, snapshots, uploaded = build_and_publish(
        args.publish_database,
        args.private_database,
        args.shadow_database,
        price_observation_database=args.price_observation_database,
        breadth_observation_database=getattr(args, "breadth_observation_database", None),
        as_of=args.as_of,
        latest_only=args.latest_only,
        require_as_of=getattr(args, "require_as_of", False),
        config=config,
        narrative_env=getattr(args, "narrative_env", DEFAULT_NARRATIVE_ENV),
        repair_missing_breadth=getattr(args, "repair_missing_breadth", False),
        no_narrative=getattr(args, "no_narrative", False),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(snapshots, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    if not args.no_upload and not upload_allowed:
        print("RORO 외부 게시 보류: roro_publish_consent.json의 명시 동의가 필요합니다.")
    print(json.dumps({"build": build, "upload": uploaded}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
