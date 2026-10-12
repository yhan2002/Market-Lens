"""Poll one web refresh request and run the existing dashboard updater."""

from __future__ import annotations

import json
import os
import sys
import traceback
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

from macro_data import refresh_turning_macro_inputs
from market_data import probe_latest_completed_market_date
from refresh_pipeline import is_refresh_busy, refresh_lock, run_refresh_pipeline


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "site_upload_config.json"
PUBLISH_DATABASE = BASE_DIR / "regime_router_publish.db"
UPDATE_LOG = Path(os.environ.get("LOCALAPPDATA", BASE_DIR)) / "RegimeRouter" / "logs" / "site_update.log"


def load_config(path: Path = CONFIG_PATH) -> dict[str, str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = ("site_url", "upload_token")
    missing = [key for key in required if not str(config.get(key, "")).strip()]
    if missing:
        raise ValueError(f"site_upload_config.json 필수 항목 누락: {', '.join(missing)}")
    if not str(config["site_url"]).lower().startswith(("https://", "http://localhost")):
        raise ValueError("site_url은 HTTPS 주소여야 합니다.")
    return {key: str(value).strip() for key, value in config.items() if value is not None}


def post_refresh(config: dict[str, str], payload: dict[str, Any]) -> dict[str, Any] | None:
    endpoint = config["site_url"].rstrip("/") + "/api/refresh"
    headers = {
        "Authorization": f"Bearer {config['upload_token']}",
        "Content-Type": "application/json",
    }
    bypass = config.get("sites_bypass_token", "")
    if bypass and not bypass.startswith("REPLACE_"):
        headers["OAI-Sites-Authorization"] = f"Bearer {bypass}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            if getattr(response, "status", response.getcode()) == 204:
                return None
            raw = response.read()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"새로고침 API HTTP {error.code}") from error
    return json.loads(raw.decode("utf-8")) if raw else {}


def server_snapshot_as_of(config: dict[str, str]) -> str:
    return _server_as_of(config, "/api/snapshot", "서버 스냅샷")


def server_roro_as_of(config: dict[str, str]) -> str | None:
    try:
        return _server_as_of(config, "/api/roro", "Market Lens")
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise RuntimeError(f"Market Lens API HTTP {error.code}") from error


def _server_as_of(config: dict[str, str], path: str, label: str) -> str:
    endpoint = config["site_url"].rstrip("/") + path
    headers = {"Authorization": f"Bearer {config['upload_token']}"}
    bypass = config.get("sites_bypass_token", "")
    if bypass and not bypass.startswith("REPLACE_"):
        headers["OAI-Sites-Authorization"] = f"Bearer {bypass}"
    request = urllib.request.Request(endpoint, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        raise
    value = payload.get("as_of") if isinstance(payload, dict) else None
    if not isinstance(value, str):
        raise RuntimeError(f"{label} 기준일이 없습니다.")
    date.fromisoformat(value)
    return value


def _log_exception() -> None:
    try:
        UPDATE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with UPDATE_LOG.open("a", encoding="utf-8") as log:
            log.write("\n[웹 갱신 예외]\n")
            traceback.print_exc(file=log)
    except OSError:
        # Logging must never hide the actual refresh error.
        return


def main() -> int:
    try:
        # Do not claim a request while the scheduled/manual runner owns the lock.
        if is_refresh_busy():
            return 0
        config = load_config()
        response = post_refresh(config, {"action": "claim"})
        claim = response.get("request", response) if response else None
        if not isinstance(claim, dict) or claim.get("id") is None:
            return 0
        request_id = claim["id"]

        print(f"웹 새로고침 요청 {request_id} 처리 시작")
        as_of: str | None = None
        try:
            server_as_of = server_snapshot_as_of(config)
            roro_as_of = server_roro_as_of(config)
            probe_date = probe_latest_completed_market_date(
                tickers=("NQ=F",), fallback_database=PUBLISH_DATABASE
            )
            dashboard_current = (
                PUBLISH_DATABASE.is_file()
                and date.fromisoformat(server_as_of) >= probe_date
            )
            if (
                dashboard_current
                and roro_as_of is not None
                and date.fromisoformat(roro_as_of) >= probe_date
            ):
                with refresh_lock():
                    _counts, used_stored = refresh_turning_macro_inputs(
                        PUBLISH_DATABASE, probe_date.isoformat()
                    )
                succeeded = True
                as_of = server_as_of
                macro_detail = (
                    "전환 매크로 직전 유효값 유지"
                    if used_stored
                    else "전환 매크로 최신화 완료"
                )
                detail = (
                    f"새 거래일 없음 · {as_of} 유지 · {macro_detail}"
                    " · 보유내역 재확인 안 함"
                    " · 보유내역 반영은 다음 예약 갱신 또는 UPDATE_SITE.cmd 수동 갱신 필요"
                )
            else:
                result = run_refresh_pipeline(probe_date)
                refreshed_dashboard = server_snapshot_as_of(config)
                refreshed_roro = server_roro_as_of(config)
                if date.fromisoformat(refreshed_dashboard) < probe_date:
                    raise RuntimeError(
                        f"기본 대시보드 게시일 {refreshed_dashboard} < NQ {probe_date}"
                    )
                if refreshed_roro is None:
                    raise RuntimeError("Market Lens 게시 결과가 없습니다.")
                if date.fromisoformat(refreshed_roro) < probe_date:
                    raise RuntimeError(
                        f"Market Lens 게시일 {refreshed_roro} < NQ {probe_date}"
                    )
                succeeded = True
                as_of = refreshed_dashboard
                completion = "부분 완료" if result.warnings else "갱신 및 게시 완료"
                detail = f"{completion} · {probe_date.isoformat()}"
                if result.holdings_as_of:
                    detail += f" · 보유내역 기준일 {result.holdings_as_of}"
                if result.warnings:
                    detail += " · " + " ".join(result.warnings)
        except Exception as error:
            _log_exception()
            succeeded = False
            detail = f"갱신 실패: {error}"

        finish: dict[str, Any] = {
            "action": "finish",
            "id": request_id,
            "status": "SUCCEEDED" if succeeded else "FAILED",
            "detail": detail,
        }
        if as_of:
            finish["as_of"] = as_of
        post_refresh(config, finish)
        print(detail)
        return 0 if succeeded else 1
    except Exception as error:
        _log_exception()
        print(f"웹 새로고침 처리 실패: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
