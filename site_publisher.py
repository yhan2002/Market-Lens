from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import os
from typing import Any, Callable, Mapping

from crash_guard import get_crash_guard_status
from breadth_metrics import (
    EXACT_BREADTH_SOURCE,
    LIVE_BREADTH_SOURCE,
    SP500_BREADTH_GROUP,
    compute_breadth_metrics,
    current_korea_date,
    register_members_overlapping,
    sync_current_open_data_membership,
)
from macro_data import refresh_turning_macro_inputs
from market_data import download_prices, register_universe
from portfolio_risk import (
    EVENT_HOLDINGS_SOURCE,
    calculate_portfolio_risk_from_database,
    store_portfolio_risk_metric,
)
from regime_router import get_regime_status
from regime_router_app import HMM_MODEL_VERSION, latest_nq_date, update_everything
from research_drawdown_onset import get_shadow_hazard_status
from router_score_v2 import build_router_scores_from_database
from risk_policy import (
    POLICY_VERSION,
    RULE_SCORE_POLICY_VERSION,
    RULE_STRESS_SCORE_POLICY_VERSION,
    RULE_STRESS_SCORE_V3_LIVE_POLICY_VERSION,
    RULE_STRESS_SCORE_V3_POLICY_VERSION,
    evaluate_reentry,
    evaluate_risk_policy,
)
from risk_signals import (
    build_fast_signal_snapshot,
    load_breadth_history,
    load_hy_oas_history,
    observation_is_recent,
)
from sector_rotation import (
    SECTOR_GROUP,
    build_rrg_signals,
    latest_snapshot,
    load_price_matrix,
    register_sector_universe,
)
from sector_rotation_forecast import fit_leading_forecast, load_frame
from trading_log import import_trading_log, latest_long_tickers


APP_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DATABASE = APP_DIR / "regime_router_shadow.db"
DEFAULT_DATABASE = APP_DIR / "regime_router_publish.db"
DEFAULT_CONFIG = APP_DIR / "site_upload_config.json"
DEFAULT_HOLDINGS_WORKBOOK = Path(
    os.environ.get("REGIME_ROUTER_HOLDINGS_SOURCE", APP_DIR / "private" / "holdings-source.xlsx")
)
FORBIDDEN_TABLES = {"daily_holdings", "inferred_actions"}
PUBLIC_UNIVERSE_GROUPS = {
    "DISPLAY_BENCHMARK",
    "REGIME_BENCHMARK",
    "REGIME_CROSS_ASSET",
    "SP500_SECTOR_ETF",
    SP500_BREADTH_GROUP,
}
PUBLIC_METADATA_KEYS = {
    "last_price_download_at",
    "last_regime_fit_at",
    "last_public_sp500_membership_sync",
}
LEGACY_PUBLISH_TABLES = {"portfolio_nav", "drawdown_events"}
PUBLISH_TABLES = {
    "metadata",
    "universe",
    "daily_prices",
    "reviewed_move_closes",
    "regime_probabilities",
    "regime_model_probabilities",
    "drawdown_hazard_scores",
    "macro_observations",
    "breadth_metrics",
    "portfolio_risk_metrics",
    "sp500_membership",
}
ACTIVE_POLICY_VERSION = RULE_STRESS_SCORE_V3_LIVE_POLICY_VERSION
ROLLBACK_HMM_MODEL_VERSION = "hmm3-sticky-nq-core-v1-k25-m252-w1260-r20-s42-n3"
LEGACY_POLICY_SERIES_VERSION = (
    f"{ACTIVE_POLICY_VERSION}__{ROLLBACK_HMM_MODEL_VERSION}__r4"
)
POLICY_SERIES_VERSION = f"{ACTIVE_POLICY_VERSION}__{HMM_MODEL_VERSION}__r6-breadth-correction"


def ensure_sanitized_database(database: Path) -> None:
    if database.name != "regime_router_publish.db":
        raise ValueError("게시에는 regime_router_publish.db만 사용할 수 있습니다.")
    with closing(sqlite3.connect(database)) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        found = sorted(tables & FORBIDDEN_TABLES)
        if found:
            raise ValueError("민감 테이블이 포함된 DB는 게시할 수 없습니다: " + ", ".join(found))
        unexpected = sorted(
            table
            for table in tables - PUBLISH_TABLES
            if not table.startswith("sqlite_")
        )
        if unexpected:
            raise ValueError("게시 DB에 허용되지 않은 테이블이 있습니다: " + ", ".join(unexpected))
        if "metadata" in tables:
            forbidden_metadata = connection.execute(
                f"SELECT key FROM metadata WHERE key NOT IN ({','.join('?' for _ in PUBLIC_METADATA_KEYS)}) LIMIT 1",
                tuple(sorted(PUBLIC_METADATA_KEYS)),
            ).fetchone()
            if forbidden_metadata:
                raise ValueError("게시 DB에 허용되지 않은 metadata가 있습니다.")
        if "universe" in tables:
            forbidden_group = connection.execute(
                f"SELECT group_name FROM universe WHERE group_name NOT IN ({','.join('?' for _ in PUBLIC_UNIVERSE_GROUPS)}) LIMIT 1",
                tuple(sorted(PUBLIC_UNIVERSE_GROUPS)),
            ).fetchone()
            if forbidden_group:
                raise ValueError("게시 DB에 내부 종목 유니버스가 있습니다.")
        if "daily_prices" in tables and "universe" in tables:
            membership_clause = (
                "AND NOT EXISTS (SELECT 1 FROM sp500_membership m WHERE m.ticker=daily_prices.ticker)"
                if "sp500_membership" in tables
                else ""
            )
            unsafe_price = connection.execute(
                f"""
                SELECT ticker FROM daily_prices
                WHERE ticker NOT IN ('SPY', 'QQQ')
                  AND NOT EXISTS (
                    SELECT 1 FROM universe u
                    WHERE u.ticker=daily_prices.ticker
                      AND u.group_name IN ({','.join('?' for _ in PUBLIC_UNIVERSE_GROUPS)})
                  )
                  {membership_clause}
                LIMIT 1
                """,
                tuple(sorted(PUBLIC_UNIVERSE_GROUPS)),
            ).fetchone()
            if unsafe_price:
                raise ValueError("게시 DB에 내부 유니버스 가격행이 있습니다.")


def sanitize_publish_database(database: Path) -> None:
    """Remove legacy internal-universe traces from the derived publish database."""
    if database.name != "regime_router_publish.db":
        raise ValueError("게시에는 regime_router_publish.db만 사용할 수 있습니다.")
    with closing(sqlite3.connect(database)) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        found = sorted(tables & FORBIDDEN_TABLES)
        if found:
            raise ValueError("민감 테이블이 포함된 DB는 게시할 수 없습니다: " + ", ".join(found))
        unexpected = sorted(
            table
            for table in tables - PUBLISH_TABLES
            if not table.startswith("sqlite_")
        )
        unknown = sorted(set(unexpected) - LEGACY_PUBLISH_TABLES)
        if unknown:
            raise ValueError("게시 DB에 알 수 없는 테이블이 있습니다: " + ", ".join(unknown))
        for table in sorted(set(unexpected) & LEGACY_PUBLISH_TABLES):
            connection.execute(f'DROP TABLE "{table}"')
        if "universe" in tables:
            connection.execute(
                f"DELETE FROM universe WHERE group_name NOT IN ({','.join('?' for _ in PUBLIC_UNIVERSE_GROUPS)})",
                tuple(sorted(PUBLIC_UNIVERSE_GROUPS)),
            )
        if "metadata" in tables:
            connection.execute(
                f"DELETE FROM metadata WHERE key NOT IN ({','.join('?' for _ in PUBLIC_METADATA_KEYS)})",
                tuple(sorted(PUBLIC_METADATA_KEYS)),
            )
        if "daily_prices" in tables and "universe" in tables:
            membership_clause = (
                "AND NOT EXISTS (SELECT 1 FROM sp500_membership m WHERE m.ticker=daily_prices.ticker)"
                if "sp500_membership" in tables
                else ""
            )
            connection.execute(
                f"""
                DELETE FROM daily_prices
                WHERE ticker NOT IN ('SPY', 'QQQ')
                  AND NOT EXISTS (
                    SELECT 1 FROM universe u
                    WHERE u.ticker=daily_prices.ticker
                      AND u.group_name IN ({','.join('?' for _ in PUBLIC_UNIVERSE_GROUPS)})
                  )
                  {membership_clause}
                """,
                tuple(sorted(PUBLIC_UNIVERSE_GROUPS)),
            )
        connection.commit()
    ensure_sanitized_database(database)


def create_publish_database(source: Path, target: Path) -> None:
    """Create an allowlisted market-only database without touching the source."""
    if target.exists():
        raise FileExistsError(f"게시용 DB가 이미 있습니다: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(source)) as source_connection:
        tables = source_connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
        ).fetchall()
        indexes = source_connection.execute(
            "SELECT tbl_name, sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
        ).fetchall()
        available = {name for name, _ in tables}
        if "daily_prices" not in available:
            raise ValueError("원본 DB에 daily_prices가 없습니다.")
        with closing(sqlite3.connect(target)) as target_connection:
            target_connection.execute("ATTACH DATABASE ? AS source", (str(source.resolve()),))
            for name, create_sql in tables:
                if name not in PUBLISH_TABLES:
                    continue
                target_connection.execute(create_sql)
                if name == "metadata":
                    continue
                if name == "universe":
                    target_connection.execute(
                        f'INSERT INTO "universe" SELECT * FROM source."universe" '
                        f"WHERE group_name IN ({','.join('?' for _ in PUBLIC_UNIVERSE_GROUPS)})",
                        tuple(sorted(PUBLIC_UNIVERSE_GROUPS)),
                    )
                    continue
                if name == "daily_prices":
                    has_membership = "sp500_membership" in available
                    universe_clause = (
                        f"OR EXISTS ("
                        f'SELECT 1 FROM source."universe" u '
                        f"WHERE u.ticker=p.ticker "
                        f"AND u.group_name IN ({','.join('?' for _ in PUBLIC_UNIVERSE_GROUPS)})"
                        f")"
                        if "universe" in available
                        else ""
                    )
                    membership_clause = (
                        'OR EXISTS (SELECT 1 FROM source."sp500_membership" m WHERE m.ticker=p.ticker)'
                        if has_membership
                        else ""
                    )
                    target_connection.execute(
                        f"""
                        INSERT INTO "daily_prices"
                        SELECT p.* FROM source."daily_prices" p
                        WHERE p.ticker IN ('SPY', 'QQQ')
                           {universe_clause}
                           {membership_clause}
                        """,
                        (
                            tuple(sorted(PUBLIC_UNIVERSE_GROUPS))
                            if "universe" in available
                            else ()
                        ),
                    )
                    continue
                target_connection.execute(f'INSERT INTO "{name}" SELECT * FROM source."{name}"')
            for table_name, create_sql in indexes:
                if table_name in PUBLISH_TABLES:
                    target_connection.execute(create_sql)
            target_connection.commit()
    ensure_sanitized_database(target)


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def refresh_models(
    database: Path,
    progress: Callable[[str], None] = print,
    *,
    required_price_date: date | None = None,
) -> None:
    ensure_sanitized_database(database)
    update_everything(
        database,
        progress,
        include_equity_universe=False,
        required_price_date=required_price_date,
    )
    progress("섹터 ETF 유니버스 등록 중…")
    register_sector_universe(database)
    progress("섹터 ETF 당일 가격 갱신 중…")
    count, rows, _ = download_prices(database, years=10, group_name=SECTOR_GROUP)
    progress(f"섹터용 가격 {count}종목(11개 섹터 + SPY) / {rows:,}행 반영 완료")

    try:
        progress("차트용 나스닥100 현물 지수 갱신 중… (위험모델 NQ 입력 유지)")
        register_universe(database, ("^NDX",), group_name="DISPLAY_BENCHMARK")
        _, rows, _ = download_prices(database, years=2, group_name="DISPLAY_BENCHMARK")
        progress(f"차트용 NDX 종가 {rows:,}행 반영 완료")
    except (OSError, RuntimeError, ValueError, sqlite3.DatabaseError) as error:
        progress(f"NDX 차트 가격 갱신 보류 (선물값 대체 없음): {error}")

    as_of = latest_nq_date(database)
    if not as_of:
        progress("PDF exact 입력 갱신 보류: NQ 기준일이 없습니다.")
        return
    end_date = datetime.strptime(as_of[:10], "%Y-%m-%d").date()
    try:
        active, added, removed = sync_current_open_data_membership(
            database, current_korea_date()
        )
        progress(
            f"Breadth 공개 구성종목 {active}개 확인 "
            f"(신규 {added}, 제외 {removed})"
        )
    except (OSError, RuntimeError, ValueError, sqlite3.DatabaseError) as error:
        progress(f"Breadth 공개 구성종목 갱신 보류: {error}")
    try:
        counts, used_stored = refresh_turning_macro_inputs(
            database, end_date.isoformat()
        )
        if used_stored:
            progress("전환 매크로 연결 오류: 허용 시차 내 직전 유효값을 유지합니다.")
        else:
            progress(
                f"전환 매크로 8계열 최신화 완료 / 신규·수정 {sum(counts.values()):,}행"
            )
    except (OSError, RuntimeError, ValueError) as error:
        progress(f"전환 매크로 갱신 보류(별도 입력): {error}")

    with closing(sqlite3.connect(database)) as connection:
        has_membership = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sp500_membership'"
        ).fetchone()
    if not has_membership:
        progress("Breadth aggregate unavailable: S&P 500 구성종목이 필요합니다.")
        return
    try:
        price_start = end_date - timedelta(days=round(2 * 365.25))
        members = register_members_overlapping(database, price_start, end_date)
        with closing(sqlite3.connect(database)) as connection:
            priced_members = connection.execute(
                """
                SELECT COUNT(DISTINCT p.ticker)
                FROM universe u
                JOIN daily_prices p ON p.ticker=u.ticker
                WHERE u.group_name=? AND u.active=1
                """,
                (SP500_BREADTH_GROUP,),
            ).fetchone()[0]
        if priced_members * 10 < members * 9:
            progress(
                "Breadth 초기 가격이 필요합니다: "
                "market_data.py download --group-name SP500_BREADTH --years 2 --backfill"
            )
            return
        count, rows, _ = download_prices(
            database,
            years=2,
            group_name=SP500_BREADTH_GROUP,
        )
        breadth_rows = compute_breadth_metrics(database)
        valid_breadth = load_breadth_history(database, as_of, 1)
        if not valid_breadth or valid_breadth[0]["date"] != as_of:
            latest = valid_breadth[0]["date"] if valid_breadth else "없음"
            raise ValueError(f"Breadth 기준일 미달: 요청 {as_of}, 유효 {latest}")
        latest_source_row = None
        try:
            with closing(sqlite3.connect(database)) as connection:
                latest_source_row = connection.execute(
                    "SELECT source FROM breadth_metrics ORDER BY date DESC LIMIT 1"
                ).fetchone()
        except sqlite3.DatabaseError:
            pass
        latest_source = str(latest_source_row[0]) if latest_source_row else None
        source_label = (
            "공개 실시간"
            if latest_source == LIVE_BREADTH_SOURCE
            else "exact"
            if latest_source == EXACT_BREADTH_SOURCE
            else "대기"
        )
        progress(
            f"Breadth {source_label} {members}종목 / 가격 {count}종목·{rows:,}행 / "
            f"지표 {breadth_rows:,}일 반영 완료"
        )
    except (OSError, RuntimeError, ValueError, sqlite3.DatabaseError) as error:
        progress(f"Breadth aggregate unavailable: 구성종목·가격을 확인하세요. ({error})")


def sync_portfolio_risk_aggregate(
    source: Path,
    target: Path,
    progress: Callable[[str], None] = print,
) -> bool:
    """Persist and sync only the portfolio-level aggregate, never position rows."""
    try:
        target_as_of = latest_nq_date(target)
        if not target_as_of:
            raise ValueError("target market date unavailable")
        metric = calculate_portfolio_risk_from_database(
            source,
            target_as_of,
            prices_database=source,
            benchmark_database=target,
            benchmark="NQ=F",
            holdings_source=EVENT_HOLDINGS_SOURCE,
        )
        store_portfolio_risk_metric(source, metric)
        if source.resolve() != target.resolve():
            store_portfolio_risk_metric(target, metric)
    except (OSError, ValueError, sqlite3.DatabaseError):
        progress("포트폴리오 상관 aggregate unavailable: exact 입력을 확인하세요.")
        return False
    progress(f"포트폴리오 상관 aggregate 준비 완료: {metric['date']}")
    return True


def _has_portfolio_metric(database: Path, as_of: str) -> bool:
    with closing(sqlite3.connect(database)) as connection:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='portfolio_risk_metrics'"
        ).fetchone()
        if not exists:
            return False
        return bool(
            connection.execute(
                "SELECT 1 FROM portfolio_risk_metrics WHERE substr(date, 1, 10)=?",
                (as_of,),
            ).fetchone()
        )


def _remove_portfolio_metric(database: Path, as_of: str) -> None:
    with closing(sqlite3.connect(database)) as connection:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='portfolio_risk_metrics'"
        ).fetchone()
        if exists:
            connection.execute(
                "DELETE FROM portfolio_risk_metrics WHERE substr(date, 1, 10)=?",
                (as_of,),
            )
            connection.commit()


def _resolve_holdings_workbook(path: Path) -> Path | None:
    candidates = (path,) if path.suffix else (path.with_suffix(".xlsx"), path.with_suffix(".xlsm"), path)
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _latest_holdings_signature(database: Path) -> tuple[tuple[object, ...], ...]:
    with closing(sqlite3.connect(database)) as connection:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='daily_holdings'"
        ).fetchone()
        if not exists:
            return ()
        return tuple(
            connection.execute(
                """
                SELECT date, isin, ticker, quantity, market_value_base
                FROM daily_holdings
                WHERE date=(SELECT MAX(date) FROM daily_holdings)
                ORDER BY isin
                """
            ).fetchall()
        )


def _missing_portfolio_close_tickers(
    database: Path, tickers: tuple[str, ...], as_of: str
) -> tuple[str, ...]:
    """Return current holdings without a usable close on the decision date."""
    if not tickers:
        return ()
    placeholders = ",".join("?" for _ in tickers)
    with closing(sqlite3.connect(database)) as connection:
        rows = connection.execute(
            f"""
            SELECT ticker FROM daily_prices
            WHERE date=? AND ticker IN ({placeholders})
              AND COALESCE(adj_close, close) > 0
            """,
            (as_of, *tickers),
        ).fetchall()
    available = {str(row[0]) for row in rows}
    return tuple(ticker for ticker in tickers if ticker not in available)


def refresh_portfolio_inputs(
    source: Path,
    target_as_of: str,
    workbook: Path = DEFAULT_HOLDINGS_WORKBOOK,
    progress: Callable[[str], None] = print,
) -> bool:
    """Refresh event holdings only when changed, then update private prices daily."""
    changed = False
    resolved = _resolve_holdings_workbook(workbook)
    if resolved is None:
        progress("Cloudium 보유내역 연결 안 됨: 마지막 확인 포지션을 사용합니다.")
    else:
        stat = resolved.stat()
        fingerprint = f"{stat.st_size}:{stat.st_mtime_ns}"
        with closing(sqlite3.connect(source)) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            previous = connection.execute(
                "SELECT value FROM metadata WHERE key='event_holdings_workbook_fingerprint'"
            ).fetchone()
        if not previous or str(previous[0]) != fingerprint:
            try:
                holdings_before = _latest_holdings_signature(source)
                holdings, actions = import_trading_log(source, resolved, preserve_history=True)
                holdings_after = _latest_holdings_signature(source)
                with closing(sqlite3.connect(source)) as connection:
                    connection.execute(
                        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
                        ("event_holdings_workbook_fingerprint", fingerprint),
                    )
                    connection.commit()
                changed = not previous or holdings_before != holdings_after
                if changed:
                    progress(
                        f"Cloudium 보유내역 변경 반영: {holdings:,}행 / 행동 {actions:,}건"
                    )
                else:
                    progress("Cloudium 보유내역 변경 없음: 마지막 확인 포지션을 사용합니다.")
            except (OSError, ValueError, KeyError, sqlite3.DatabaseError):
                progress("Cloudium 보유내역 읽기 실패: 마지막 확인 포지션을 사용합니다.")
        else:
            progress("Cloudium 보유내역 변경 없음: 마지막 확인 포지션을 사용합니다.")

    try:
        tickers = latest_long_tickers(source, target_as_of)
        if len(tickers) < 2:
            raise ValueError("at least two current holdings are required")
        ordered_tickers = tuple(sorted(tickers))
        try:
            count, rows, _ = download_prices(
                source,
                years=2,
                tickers=ordered_tickers,
            )
        except RuntimeError as error:
            progress(f"포트폴리오 종가 일괄 수신 불완전: 개별 재시도합니다. ({error})")
            count, rows = len(ordered_tickers), 0
        missing = _missing_portfolio_close_tickers(source, ordered_tickers, target_as_of)
        if missing:
            for ticker in missing:
                try:
                    _, ticker_rows, _ = download_prices(
                        source,
                        years=2,
                        tickers=(ticker,),
                        require_date=date.fromisoformat(target_as_of),
                    )
                    rows += ticker_rows
                except RuntimeError as error:
                    progress(f"{ticker} {target_as_of} 종가 수신 실패: {error}")
            missing = _missing_portfolio_close_tickers(source, ordered_tickers, target_as_of)
        if missing:
            raise RuntimeError(
                f"포트폴리오 최신 종가 미수신({target_as_of}): {', '.join(missing)}"
            )
        else:
            progress(f"포트폴리오 비공개 가격 {count}종목 / {rows:,}행 반영 완료")
    except (OSError, RuntimeError, ValueError, sqlite3.DatabaseError) as error:
        progress(f"포트폴리오 비공개 가격 갱신 실패: {error}")
    return changed


def list_nq_trading_dates(database: Path, start: str, end: str) -> list[str]:
    ensure_sanitized_database(database)
    try:
        start_date = datetime.strptime(start, "%Y-%m-%d").date()
        end_date = datetime.strptime(end, "%Y-%m-%d").date()
    except ValueError as error:
        raise ValueError("백필 날짜는 YYYY-MM-DD 형식이어야 합니다.") from error
    if start_date > end_date:
        raise ValueError("백필 시작일은 종료일보다 늦을 수 없습니다.")
    with closing(sqlite3.connect(database)) as connection:
        rows = connection.execute(
            """
            SELECT DISTINCT substr(date, 1, 10) AS trading_date
            FROM daily_prices
            WHERE ticker='NQ=F' AND substr(date, 1, 10) BETWEEN ? AND ?
            ORDER BY trading_date
            """,
            (start_date.isoformat(), end_date.isoformat()),
        ).fetchall()
    return [str(row[0]) for row in rows]


def _positive_price(adj_close: object, close: object) -> float | None:
    for value in (adj_close, close):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        price = float(value)
        if math.isfinite(price) and price > 0:
            return price
    return None


def _price_rows(
    database: Path, tickers: tuple[str, ...], as_of: str
) -> list[tuple[object, ...]]:
    placeholders = ", ".join("?" for _ in tickers)
    try:
        with closing(sqlite3.connect(database)) as connection:
            return connection.execute(
                f"""
                SELECT ticker, substr(date, 1, 10), adj_close, close
                FROM daily_prices
                WHERE ticker IN ({placeholders}) AND substr(date, 1, 10)<=?
                ORDER BY substr(date, 1, 10) DESC
                """,
                (*tickers, as_of),
            ).fetchall()
    except sqlite3.DatabaseError:
        return []


def _latest_complete_prices(
    database: Path, tickers: tuple[str, ...], as_of: str, count: int
) -> list[tuple[str, dict[str, float]]]:
    by_date: dict[str, dict[str, float]] = {}
    for ticker, price_date, adj_close, close in _price_rows(database, tickers, as_of):
        price = _positive_price(adj_close, close)
        if price is not None:
            by_date.setdefault(str(price_date), {})[str(ticker)] = price
    complete = [
        (price_date, values)
        for price_date, values in sorted(by_date.items(), reverse=True)
        if all(ticker in values for ticker in tickers)
    ]
    return complete[:count]


def _prior_p_bear_history(
    database: Path, regime_date: str | None, model_version: str | None
) -> list[float]:
    if not regime_date or not model_version:
        return []
    try:
        with closing(sqlite3.connect(database)) as connection:
            rows = connection.execute(
                """
                SELECT p_bear
                FROM regime_model_probabilities
                WHERE model_version=? AND date<?
                  AND typeof(p_bear) IN ('integer', 'real')
                  AND p_bear BETWEEN 0.0 AND 1.0
                ORDER BY date DESC
                LIMIT 20
                """,
                (model_version, regime_date),
            ).fetchall()
    except sqlite3.DatabaseError:
        return []
    return [
        probability
        for row in rows
        if (probability := _probability(row[0])) is not None
    ]


def _probability(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    probability = float(value)
    return probability if math.isfinite(probability) and 0.0 <= probability <= 1.0 else None


def build_reentry_advisory(
    database: Path, as_of: str, regime: dict[str, Any] | None
) -> dict[str, Any]:
    """Build the four-signal re-entry advisory from point-in-time local data."""
    conditions: list[dict[str, str]] = []

    vix_rows = _latest_complete_prices(database, ("^VIX", "^VIX3M"), as_of, 1)
    if vix_rows and vix_rows[0][0] == as_of:
        price_date, prices = vix_rows[0]
        normalised = prices["^VIX"] <= prices["^VIX3M"]
        conditions.append(
            {
                "key": "vix_term_structure",
                "label": "VIX 기간구조 정상화",
                "status": "CONFIRMED" if normalised else "NOT_CONFIRMED",
                "detail": (
                    f"{price_date} ^VIX {prices['^VIX']:.2f} / "
                    f"^VIX3M {prices['^VIX3M']:.2f}"
                ),
                "data_quality": "EXACT",
                "as_of": as_of,
            }
        )
    else:
        conditions.append(
            {
                "key": "vix_term_structure",
                "label": "VIX 기간구조 정상화",
                "status": "UNAVAILABLE",
                "detail": "기준일과 같은 날짜의 양수 ^VIX·^VIX3M 가격이 모두 필요합니다.",
                "data_quality": "UNAVAILABLE",
                "as_of": as_of,
            }
        )

    oas_rows = load_hy_oas_history(database, as_of, 2)
    if len(oas_rows) == 2 and observation_is_recent(as_of, oas_rows[0][0]):
        current_date, current_oas = oas_rows[0]
        previous_date, previous_oas = oas_rows[1]
        conditions.append(
            {
                "key": "hy_spread_stabilized",
                "label": "HY OAS 상승 정지",
                "status": "CONFIRMED" if current_oas <= previous_oas else "NOT_CONFIRMED",
                "detail": (
                    f"HY OAS {previous_date} {previous_oas:.1f}bp → "
                    f"{current_date} {current_oas:.1f}bp"
                ),
                "data_quality": "EXACT",
                "as_of": current_date,
            }
        )
    else:
        conditions.append(
            {
                "key": "hy_spread_stabilized",
                "label": "HY OAS 상승 정지",
                "status": "UNAVAILABLE",
                "detail": "기준일까지 공개된 HY_OAS 최근 2개 exact 관측치가 필요합니다.",
                "data_quality": "UNAVAILABLE",
                "as_of": as_of,
            }
        )

    nq_ticker = "NQ=F"
    nq_rows = _latest_complete_prices(database, (nq_ticker,), as_of, 5)
    if len(nq_rows) == 5 and nq_rows[0][0] == as_of:
        latest_date, latest_prices = nq_rows[0]
        latest_price = latest_prices[nq_ticker]
        five_day_mean = sum(prices[nq_ticker] for _, prices in nq_rows) / 5.0
        conditions.append(
            {
                "key": "nq_above_5dma",
                "label": "NQ 5DMA 회복",
                "status": (
                    "CONFIRMED" if latest_price >= five_day_mean else "NOT_CONFIRMED"
                ),
                "detail": (
                    f"{nq_ticker}: "
                    f"{latest_date} 종가 {latest_price:.2f} / 5DMA {five_day_mean:.2f}"
                ),
                "data_quality": "EXACT",
                "as_of": as_of,
            }
        )
    else:
        conditions.append(
            {
                "key": "nq_above_5dma",
                "label": "NQ 5DMA 회복",
                "status": "UNAVAILABLE",
                "detail": "기준일까지 NQ=F exact 최근 양수 종가 5개가 필요합니다.",
                "data_quality": "UNAVAILABLE",
                "as_of": as_of,
            }
        )

    breadth_rows = load_breadth_history(database, as_of, 2)
    nq_low_rows = _latest_complete_prices(database, ("NQ=F",), as_of, 21)
    breadth_added = False
    if (
        len(breadth_rows) == 2
        and breadth_rows[0]["date"] == as_of
        and breadth_rows[0].get("source") == breadth_rows[1].get("source")
        and len(nq_low_rows) == 21
        and nq_low_rows[0][0] == as_of
    ):
        current, previous = breadth_rows
        current_lows = current.get("new_52w_lows")
        prior_lows = previous.get("new_52w_lows")
        current_advancers = current.get("advancers")
        prior_advancers = previous.get("advancers")
        if all(
            isinstance(value, float)
            for value in (current_lows, prior_lows, current_advancers, prior_advancers)
        ):
            current_nq = nq_low_rows[0][1]["NQ=F"]
            prior_20d_low = min(prices["NQ=F"] for _, prices in nq_low_rows[1:])
            fresh_nq_low = current_nq <= prior_20d_low
            confirmed = (
                fresh_nq_low
                and current_lows < prior_lows
                and current_advancers > prior_advancers
            )
            conditions.append(
                {
                    "key": "breadth_divergence",
                    "label": "Breadth divergence",
                    "status": "CONFIRMED" if confirmed else "NOT_CONFIRMED",
                    "detail": (
                        f"NQ=F {current_nq:.2f} / 직전 20D 저점 {prior_20d_low:.2f}, "
                        f"52주 신저가 {prior_lows:.0f}→{current_lows:.0f}, "
                        f"상승종목수 {prior_advancers:.0f}→{current_advancers:.0f}"
                    ),
                    "data_quality": (
                        "EXACT"
                        if current.get("source") == EXACT_BREADTH_SOURCE
                        else "PUBLIC_LIVE"
                    ),
                    "as_of": as_of,
                }
            )
            breadth_added = True
    if not breadth_added:
        conditions.append(
            {
                "key": "breadth_divergence",
                "label": "Breadth divergence",
                "status": "UNAVAILABLE",
                "detail": (
                    "NQ=F exact 21개 가격과 안전 aggregate의 52주 신저가·"
                    "상승종목수 최근 2개 관측치가 필요합니다."
                ),
                "data_quality": "UNAVAILABLE",
                "as_of": as_of,
            }
        )

    regime_date = str(regime.get("date")) if regime and regime.get("date") else None
    model_version = (
        str(regime.get("model_version"))
        if regime and regime.get("model_version")
        else None
    )
    current_p_bear = _probability(regime.get("p_bear")) if regime else None
    prior_history = _prior_p_bear_history(database, regime_date, model_version)
    previous_p_bear = prior_history[0] if prior_history else None
    recent_peak_p_bear = max(prior_history) if prior_history else None
    result = evaluate_reentry(
        current_p_bear,
        previous_p_bear,
        conditions,
        recent_peak_p_bear=recent_peak_p_bear,
    )
    metadata = {item["key"]: item for item in conditions}
    for item in result.get("conditions", []):
        source = metadata.get(item.get("key"))
        if source:
            item["data_quality"] = source["data_quality"]
            item["as_of"] = source["as_of"]
    return result


def _fresh_channel(channel: dict[str, Any] | None, as_of: str) -> dict[str, Any] | None:
    return channel if channel is not None and str(channel.get("date")) == as_of else None


def list_policy_decision_dates(
    database: Path,
    *,
    end: str,
    start_exclusive: str | None = None,
    model_version: str = HMM_MODEL_VERSION,
) -> list[str]:
    """Return the exact NQ/HMM decision calendar for one frozen model version."""
    query = """
        SELECT DISTINCT substr(r.date, 1, 10) AS decision_date
        FROM regime_model_probabilities r
        JOIN daily_prices p
          ON p.ticker='NQ=F' AND substr(p.date, 1, 10)=substr(r.date, 1, 10)
        WHERE r.model_version=? AND substr(r.date, 1, 10)<=?
    """
    parameters: list[object] = [model_version, end]
    if start_exclusive is not None:
        query += " AND substr(r.date, 1, 10)>?"
        parameters.append(start_exclusive)
    query += " ORDER BY decision_date"
    with closing(sqlite3.connect(database)) as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [str(row[0]) for row in rows]


def build_policy_decision(
    database: Path,
    as_of: str,
    *,
    previous_state: dict[str, Any] | None = None,
    model_version: str = HMM_MODEL_VERSION,
    slow_risk_score: int | None = None,
    policy_version: str = POLICY_VERSION,
) -> dict[str, Any]:
    """Build one production policy decision; live, backfill, and OOS share it."""
    try:
        regime = get_regime_status(database, as_of, model_version=model_version)
    except ValueError as error:
        raise ValueError(f"{as_of} NQ HMM 학습 결과가 없습니다.") from error
    fast_overlay = build_fast_signal_snapshot(database, as_of)
    reentry = build_reentry_advisory(database, as_of, regime)
    policy = evaluate_risk_policy(
        _fresh_channel(regime, as_of),
        fast_overlay,
        reentry,
        as_of=as_of,
        previous_state=previous_state,
        slow_risk_score=slow_risk_score,
        policy_version=policy_version,
    )
    return {"as_of": as_of, "regime": regime, "risk_policy": policy}


def build_live_risk_score_map(database: Path) -> dict[str, int]:
    """Return the immutable v2 Router Score input used by the live policy."""
    scores = build_router_scores_from_database(database)
    if "risk_score" not in scores:
        raise ValueError("Router Score v2 결과에 risk_score가 없습니다.")
    result: dict[str, int] = {}
    for timestamp, value in scores["risk_score"].dropna().items():
        date_value = timestamp.date() if callable(getattr(timestamp, "date", None)) else None
        if date_value is None:
            try:
                date_value = datetime.strptime(str(timestamp)[:10], "%Y-%m-%d").date()
            except ValueError as error:
                raise ValueError("Router Score v2 날짜 형식이 올바르지 않습니다.") from error
        score = int(value)
        if not 0 <= score <= 100:
            raise ValueError(f"{date_value.isoformat()} Router Score가 0~100 범위를 벗어났습니다.")
        result[date_value.isoformat()] = score
    if not result:
        raise ValueError("완전한 Router Score v2 관측치가 없습니다.")
    return result


def replay_policy_history(
    database: Path,
    *,
    end: str,
    previous_state: dict[str, Any] | None = None,
    start_exclusive: str | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    model_version: str = HMM_MODEL_VERSION,
    score_by_date: Mapping[str, int] | None = None,
    policy_version: str = POLICY_VERSION,
) -> list[dict[str, Any]]:
    """Fold the frozen policy over the operating calendar in ascending order."""
    if start_exclusive is None:
        start_exclusive = (
            str(previous_state.get("last_as_of"))
            if isinstance(previous_state, dict) and previous_state.get("last_as_of")
            else None
        )
    if (score_by_date is None) != (policy_version == POLICY_VERSION):
        raise ValueError("rule-score replay requires an isolated score series and policy version")
    if score_by_date is not None and policy_version not in {
        RULE_SCORE_POLICY_VERSION,
        RULE_STRESS_SCORE_POLICY_VERSION,
        RULE_STRESS_SCORE_V3_POLICY_VERSION,
        RULE_STRESS_SCORE_V3_LIVE_POLICY_VERSION,
    }:
        raise ValueError("unsupported score override policy version")
    dates = list_policy_decision_dates(
        database,
        end=end,
        start_exclusive=start_exclusive,
        model_version=model_version,
    )
    records: list[dict[str, Any]] = []
    state = dict(previous_state) if previous_state is not None else None
    total = len(dates)
    for index, decision_date in enumerate(dates, start=1):
        score = score_by_date.get(decision_date) if score_by_date is not None else None
        if score_by_date is not None and score is None:
            raise ValueError(f"{decision_date} has no rule-based risk score")
        decision_options: dict[str, Any] = {
            "previous_state": state,
            "model_version": model_version,
        }
        if score_by_date is not None:
            decision_options.update(
                slow_risk_score=score,
                policy_version=policy_version,
            )
        record = build_policy_decision(database, decision_date, **decision_options)
        policy = record["risk_policy"]
        if policy.get("decision_status") == "OK" and isinstance(
            policy.get("policy_state"), dict
        ):
            state = dict(policy["policy_state"])
        records.append(record)
        if progress is not None and (index == 1 or index % 100 == 0 or index == total):
            progress(index, total, decision_date)
    return records


def _previous_policy_state(
    previous_snapshot: dict[str, Any] | None,
    as_of: str,
) -> dict[str, Any] | None:
    if not isinstance(previous_snapshot, dict):
        return None
    if previous_snapshot.get("schema_version") != 3:
        return None
    history = previous_snapshot.get("history")
    if (
        not isinstance(history, dict)
        or history.get("series_version") != POLICY_SERIES_VERSION
    ):
        return None
    try:
        previous_date = datetime.strptime(str(previous_snapshot.get("as_of")), "%Y-%m-%d").date()
        current_date = datetime.strptime(as_of, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None
    if previous_date >= current_date:
        return None
    regime = previous_snapshot.get("regime")
    if not isinstance(regime, dict) or regime.get("model_version") != HMM_MODEL_VERSION:
        return None
    policy = previous_snapshot.get("risk_policy")
    if (
        not isinstance(policy, dict)
        or policy.get("model_version") != ACTIVE_POLICY_VERSION
        or policy.get("decision_status") not in {"OK", "NO_DECISION"}
    ):
        return None
    state = policy.get("policy_state")
    if not isinstance(state, dict):
        return None
    if state.get("policy_version") != ACTIVE_POLICY_VERSION or state.get(
        "regime_model_version"
    ) != HMM_MODEL_VERSION:
        return None
    try:
        state_date = datetime.strptime(str(state.get("last_as_of")), "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None
    if policy.get("decision_status") == "OK":
        if state_date != previous_date:
            return None
    elif state_date >= previous_date:
        return None
    return dict(state)


def load_previous_snapshot(path: Path, *, strict: bool = False) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        if strict:
            raise ValueError(f"기존 스냅샷을 안전하게 읽을 수 없습니다: {path}") from error
        return None
    if not isinstance(value, dict):
        if strict:
            raise ValueError(f"기존 스냅샷 최상위 값이 객체가 아닙니다: {path}")
        return None
    return value


def _add_risk_score_trend(
    policy: dict[str, Any],
    previous_snapshot: dict[str, Any] | None,
    as_of: str,
    *,
    previous_score_fallback: float | None = None,
) -> None:
    current_score = policy.get("risk_score")
    current_policy_version = policy.get("model_version")
    previous_score: object = previous_score_fallback
    if isinstance(previous_snapshot, dict):
        try:
            previous_date = datetime.strptime(
                str(previous_snapshot.get("as_of")), "%Y-%m-%d"
            ).date()
            current_date = datetime.strptime(as_of, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            previous_date = current_date = None
        previous_policy = previous_snapshot.get("risk_policy")
        previous_regime = previous_snapshot.get("regime")
        previous_schema = previous_snapshot.get("schema_version")
        previous_history = previous_snapshot.get("history")
        history_matches = previous_schema == 2 or (
            previous_schema == 3
            and isinstance(previous_history, dict)
            and previous_history.get("series_version") == POLICY_SERIES_VERSION
        )
        if (
            previous_date is not None
            and current_date is not None
            and previous_date < current_date
            and history_matches
            and isinstance(previous_regime, dict)
            and previous_regime.get("model_version") == HMM_MODEL_VERSION
            and isinstance(previous_policy, dict)
            and previous_policy.get("decision_status") == "OK"
            and isinstance(current_policy_version, str)
            and previous_policy.get("model_version") == current_policy_version
        ):
            previous_score = previous_policy.get("risk_score")

    if (
        isinstance(current_score, bool)
        or not isinstance(current_score, (int, float))
        or isinstance(previous_score, bool)
        or not isinstance(previous_score, (int, float))
        or not math.isfinite(float(current_score))
        or not math.isfinite(float(previous_score))
    ):
        policy["risk_score_delta"] = None
        policy["risk_trend"] = None
        return
    delta = float(current_score) - float(previous_score)
    policy["risk_score_delta"] = int(delta) if delta.is_integer() else delta
    policy["risk_trend"] = "UP" if delta > 0 else "DOWN" if delta < 0 else "FLAT"


def _previous_risk_score_from_database(
    database: Path,
    as_of: str,
    *,
    score_by_date: Mapping[str, int] | None = None,
) -> float | None:
    """Return the prior live Router Score, never the legacy HMM-weighted score."""
    scores = dict(score_by_date) if score_by_date is not None else build_live_risk_score_map(database)
    prior_dates = [day for day in scores if day < as_of]
    if not prior_dates:
        return None
    return float(scores[max(prior_dates)])


def build_snapshot(
    database: Path,
    target_date: str | None = None,
    previous_snapshot: dict[str, Any] | None = None,
    *,
    build_mode: str = "LIVE",
) -> dict[str, Any]:
    ensure_sanitized_database(database)
    if build_mode not in {"LIVE", "BACKFILL"}:
        raise ValueError("build_mode must be LIVE or BACKFILL")
    as_of = latest_nq_date(database, target_date)
    if not as_of:
        raise ValueError("NQ 가격 데이터가 없습니다.")

    score_by_date = build_live_risk_score_map(database)
    previous_state = _previous_policy_state(previous_snapshot, as_of)
    previous_snapshot_as_of = (
        str(previous_snapshot.get("as_of"))
        if previous_state is not None and isinstance(previous_snapshot, dict)
        else None
    )
    policy_history = replay_policy_history(
        database,
        end=as_of,
        previous_state=previous_state,
        start_exclusive=previous_snapshot_as_of,
        score_by_date=score_by_date,
        policy_version=ACTIVE_POLICY_VERSION,
    )
    seed_mode = "STRICT_PRIOR" if previous_state is not None else "FULL_REPLAY"
    if policy_history:
        current_decision = policy_history[-1]
        parent_as_of = (
            policy_history[-2]["as_of"]
            if len(policy_history) >= 2
            else previous_snapshot_as_of
            if previous_snapshot_as_of is not None
            else None
        )
    elif previous_state is not None and previous_state.get("last_as_of") == as_of:
        current_decision = build_policy_decision(
            database,
            as_of,
            previous_state=previous_state,
            slow_risk_score=score_by_date.get(as_of),
            policy_version=ACTIVE_POLICY_VERSION,
        )
        parent_as_of = previous_state.get("last_as_of")
    else:
        raise ValueError(
            "NQ HMM 정책 이력이 없습니다. 최초 1회는 --skip-refresh 없이 "
            "전체 최신화를 실행하세요."
        )
    regime = current_decision["regime"]
    risk_policy = current_decision["risk_policy"]

    try:
        crash_guard = get_crash_guard_status(database, as_of)
    except (ValueError, sqlite3.DatabaseError):
        crash_guard = None
    try:
        onset = get_shadow_hazard_status(database, as_of)
    except (ValueError, sqlite3.DatabaseError):
        onset = None
    previous_score_fallback = (
        policy_history[-2]["risk_policy"].get("risk_score")
        if len(policy_history) >= 2
        else _previous_risk_score_from_database(
            database,
            as_of,
            score_by_date=score_by_date,
        )
    )
    _add_risk_score_trend(
        risk_policy,
        previous_snapshot,
        as_of,
        previous_score_fallback=previous_score_fallback,
    )

    prices = load_price_matrix(database)
    rrg = latest_snapshot(build_rrg_signals(prices), as_of, tail_days=5)
    forecast, metadata = fit_leading_forecast(load_frame(database, HMM_MODEL_VERSION), as_of=as_of)
    forecast_fields = {
        row["ticker"]: row
        for row in forecast.to_dict(orient="records")
    }
    rows = []
    for row in rrg.to_dict(orient="records"):
        ticker = row["ticker"]
        predicted = forecast_fields[ticker]
        rows.append(
            {
                "ticker": ticker,
                "sector": row["sector"],
                "quadrant": row["quadrant"],
                "quadrant_transition": row["quadrant_transition"],
                "tail_heading": row["tail_heading"],
                "rs_ratio": row["rs_ratio"],
                "rs_momentum": row["rs_momentum"],
                "delta_rs_ratio": row["delta_rs_ratio"],
                "delta_rs_momentum": row["delta_rs_momentum"],
                "shadow_score": row["shadow_score"],
                "forecast_rank": predicted["forecast_rank"],
                "forecast_score": predicted["forecast_score"],
                "predicted_excess_20d": predicted["predicted_excess_20d"],
                "signal": predicted["signal"],
            }
        )
    rows.sort(key=lambda item: int(item["forecast_rank"]))
    generated_at = datetime.now(timezone.utc).isoformat()
    return _clean(
        {
            "schema_version": 3,
            "generated_at": generated_at,
            "as_of": as_of,
            "history": {
                "series_version": POLICY_SERIES_VERSION,
                "build_mode": build_mode,
                "seed_mode": seed_mode,
                "parent_as_of": parent_as_of,
            },
            "regime": regime,
            "crash_guard": crash_guard,
            "drawdown_onset": onset,
            "risk_policy": risk_policy,
            "sector": {
                "rrg_as_of": str(rrg["date"].max().date()),
                "forecast_as_of": metadata["as_of"],
                "target": metadata["target"],
                "training_weeks": metadata["training_weeks"],
                "rows": rows,
            },
            "notices": [
                "운영 참고지표이며 자동 주문과 연결되지 않습니다.",
                "섹터 예측은 다음 20거래일의 동일가중 섹터 대비 상대수익 추정치입니다.",
            ],
        }
    )


def upload_snapshot(
    snapshot: dict[str, Any],
    site_url: str,
    token: str,
    sites_bypass_token: str | None = None,
) -> dict[str, Any]:
    endpoint = site_url.rstrip("/") + "/api/snapshot"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    if sites_bypass_token:
        headers["OAI-Sites-Authorization"] = f"Bearer {sites_bypass_token}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(snapshot, ensure_ascii=False, allow_nan=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            if error.code not in {429, 500, 502, 503, 504} or attempt == 3:
                raise RuntimeError(f"업로드 실패 HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == 3:
                raise RuntimeError(f"업로드 연결 실패: {error}") from error
        wait_seconds = 2**attempt
        print(f"업로드 일시 오류: {wait_seconds}초 후 재시도합니다…")
        time.sleep(wait_seconds)
    raise RuntimeError("업로드 재시도 횟수를 초과했습니다.")


def download_latest_snapshot(
    site_url: str,
    token: str,
    sites_bypass_token: str | None = None,
) -> dict[str, Any] | None:
    query = urllib.parse.urlencode({"series": POLICY_SERIES_VERSION})
    endpoint = site_url.rstrip("/") + f"/api/snapshot?{query}"
    headers = {"Authorization": f"Bearer {token}"}
    if sites_bypass_token:
        headers["OAI-Sites-Authorization"] = f"Bearer {sites_bypass_token}"
    request = urllib.request.Request(endpoint, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            snapshot = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"기존 이력 조회 실패 HTTP {error.code}: {detail}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"기존 이력 조회 연결 실패: {error}") from error
    history = snapshot.get("history") if isinstance(snapshot, dict) else None
    if (
        not isinstance(snapshot, dict)
        or snapshot.get("schema_version") != 3
        or not isinstance(history, dict)
        or history.get("series_version") != POLICY_SERIES_VERSION
    ):
        raise ValueError("서버의 최신 스냅샷이 현재 운영 정책 시리즈와 일치하지 않습니다.")
    return snapshot


def _is_current_policy_snapshot(snapshot: dict[str, Any] | None, as_of: str) -> bool:
    history = snapshot.get("history") if isinstance(snapshot, dict) else None
    return bool(
        isinstance(snapshot, dict)
        and snapshot.get("schema_version") == 3
        and snapshot.get("as_of") == as_of
        and isinstance(history, dict)
        and history.get("series_version") == POLICY_SERIES_VERSION
    )


def publish_missing_policy_history(
    database: Path,
    *,
    latest_as_of: str,
    previous_snapshot: dict[str, Any] | None,
    config: Mapping[str, str],
) -> dict[str, Any] | None:
    """Fill a failed-upload gap without bypassing immutable policy parents."""
    history = previous_snapshot.get("history") if isinstance(previous_snapshot, dict) else None
    remote_as_of = previous_snapshot.get("as_of") if isinstance(previous_snapshot, dict) else None
    if (
        not isinstance(history, dict)
        or history.get("series_version") != POLICY_SERIES_VERSION
        or not isinstance(remote_as_of, str)
        or remote_as_of >= latest_as_of
    ):
        return None
    dates = list_policy_decision_dates(
        database, end=latest_as_of, start_exclusive=remote_as_of
    )
    if not dates:
        return None
    snapshot = previous_snapshot
    print(f"서버 이력 공백 {len(dates)}일을 순차 복구합니다: {dates[0]} ~ {dates[-1]}")
    for trading_date in dates:
        snapshot = build_snapshot(
            database, trading_date, previous_snapshot=snapshot, build_mode="BACKFILL"
        )
        upload_snapshot(
            snapshot,
            config["site_url"],
            config["upload_token"],
            config.get("sites_bypass_token"),
        )
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Regime/Sector HTTPS dashboard updater")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--source-database", type=Path, default=DEFAULT_SOURCE_DATABASE)
    parser.add_argument(
        "--holdings-workbook", type=Path, default=DEFAULT_HOLDINGS_WORKBOOK
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--date")
    parser.add_argument("--backfill-start", help="과거 스냅샷 게시 시작일(명시 실행 전용)")
    parser.add_argument("--backfill-end", help="과거 스냅샷 게시 종료일; 생략하면 DB 최신일")
    parser.add_argument("--skip-refresh", action="store_true", help="현재 DB로 스냅샷만 다시 게시")
    parser.add_argument("--no-upload", action="store_true", help="검증용: 파일만 만들고 HTTPS 전송 생략")
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="같은 기준일의 기존 스냅샷도 현재 DB 데이터로 다시 생성",
    )
    parser.add_argument(
        "--sanitize-only",
        action="store_true",
        help="게시 DB의 과거 내부 유니버스 흔적만 제거하고 종료",
    )
    parser.add_argument("--output", type=Path, default=APP_DIR / "outputs" / "site_snapshot.json")
    args = parser.parse_args()

    if args.date and args.backfill_start:
        parser.error("--date와 --backfill-start는 함께 사용할 수 없습니다.")
    if args.backfill_end and not args.backfill_start:
        parser.error("--backfill-end에는 --backfill-start가 필요합니다.")
    if args.backfill_start and not args.skip_refresh:
        parser.error("과거 백필은 기존 DB만 사용하도록 --skip-refresh를 함께 지정하세요.")

    if not args.database.exists():
        print("민감정보 없는 게시 전용 DB를 처음 생성합니다…")
        create_publish_database(args.source_database, args.database)
    sanitize_publish_database(args.database)
    if args.sanitize_only:
        print(f"게시 DB 안전 정리 완료: {args.database}")
        return
    if not args.no_upload and not args.config.exists():
        raise FileNotFoundError(f"업로드 설정 파일이 없습니다: {args.config}")
    config = (
        json.loads(args.config.read_text(encoding="utf-8-sig"))
        if not args.no_upload
        else None
    )
    portfolio_inputs_changed = False
    portfolio_metric_added = False
    if not args.skip_refresh:
        refresh_models(
            args.database,
            required_price_date=date.fromisoformat(args.date) if args.date else None,
        )
        target_as_of = latest_nq_date(args.database)
        if target_as_of:
            had_portfolio_metric = _has_portfolio_metric(args.database, target_as_of)
            portfolio_inputs_changed = refresh_portfolio_inputs(
                args.source_database,
                target_as_of,
                args.holdings_workbook,
            )
            portfolio_sync_ok = sync_portfolio_risk_aggregate(
                args.source_database, args.database
            )
            portfolio_metric_added = portfolio_sync_ok and not had_portfolio_metric
            if portfolio_inputs_changed and not portfolio_sync_ok:
                _remove_portfolio_metric(args.database, target_as_of)

    if args.backfill_start:
        backfill_end = args.backfill_end or latest_nq_date(args.database)
        if not backfill_end:
            raise ValueError("NQ 가격 데이터가 없습니다.")
        try:
            start_date = datetime.strptime(args.backfill_start, "%Y-%m-%d").date()
            end_date = datetime.strptime(backfill_end, "%Y-%m-%d").date()
        except ValueError as error:
            raise ValueError("백필 날짜는 YYYY-MM-DD 형식이어야 합니다.") from error
        if start_date > end_date:
            raise ValueError("백필 시작일은 종료일보다 늦을 수 없습니다.")
        dates = list_policy_decision_dates(
            args.database,
            end=end_date.isoformat(),
            start_exclusive=(start_date - timedelta(days=1)).isoformat(),
        )
        if not dates:
            raise ValueError("선택 범위에 NQ HMM 정책 거래일이 없습니다.")
        print(f"과거 스냅샷 {len(dates):,}거래일 게시 준비: {dates[0]} ~ {dates[-1]}")
        last_snapshot: dict[str, Any] | None = None
        if config is not None:
            last_snapshot = download_latest_snapshot(
                config["site_url"],
                config["upload_token"],
                config.get("sites_bypass_token"),
            )
            if last_snapshot is not None:
                remote_as_of = str(last_snapshot["as_of"])
                dates = [date for date in dates if date > remote_as_of]
                if not dates:
                    print(f"서버에 {remote_as_of}까지 이미 게시되어 있습니다.")
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(
                        json.dumps(last_snapshot, ensure_ascii=False, indent=2, allow_nan=False),
                        encoding="utf-8",
                    )
                    return
                print(f"서버 이력 {remote_as_of} 다음 거래일부터 재개합니다.")
        for index, trading_date in enumerate(dates, start=1):
            last_snapshot = build_snapshot(
                args.database,
                trading_date,
                previous_snapshot=last_snapshot,
                build_mode="BACKFILL",
            )
            if config is not None:
                upload_snapshot(
                    last_snapshot,
                    config["site_url"],
                    config["upload_token"],
                    config.get("sites_bypass_token"),
                )
            print(f"[{index:,}/{len(dates):,}] {trading_date} 게시 완료")
        if last_snapshot is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(last_snapshot, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
        return

    latest_as_of = latest_nq_date(args.database, args.date)
    if not latest_as_of:
        raise ValueError("NQ 가격 데이터가 없습니다.")
    previous_snapshot = (
        download_latest_snapshot(
            config["site_url"],
            config["upload_token"],
            config.get("sites_bypass_token"),
        )
        if config is not None
        else load_previous_snapshot(args.output, strict=True)
    )
    if config is not None:
        recovered_snapshot = publish_missing_policy_history(
            args.database,
            latest_as_of=latest_as_of,
            previous_snapshot=previous_snapshot,
            config=config,
        )
        if recovered_snapshot is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(recovered_snapshot, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            return
    published_date_locked = bool(
        config is not None
        and _is_current_policy_snapshot(previous_snapshot, latest_as_of)
    )
    if published_date_locked:
        print(
            f"서버 {latest_as_of} 기록 유지: 새 입력은 다음 거래일부터 반영됩니다."
        )
        snapshot = previous_snapshot
    elif (
        not args.force_rebuild
        and not portfolio_inputs_changed
        and not portfolio_metric_added
        and _is_current_policy_snapshot(previous_snapshot, latest_as_of)
    ):
        print(f"기존 운영 스냅샷 재사용: {latest_as_of}")
        snapshot = previous_snapshot
    else:
        print("Regime·Sector 공개용 스냅샷 생성 중…")
        snapshot = build_snapshot(
            args.database,
            args.date,
            previous_snapshot=previous_snapshot,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"스냅샷 저장: {args.output}")
    if config is not None and not published_date_locked:
        result = upload_snapshot(
            snapshot,
            config["site_url"],
            config["upload_token"],
            config.get("sites_bypass_token"),
        )
        print(f"HTTPS 게시 완료: 기준일 {result['as_of']}")
    elif published_date_locked:
        print(f"HTTPS 게시 유지: 기준일 {latest_as_of}")


if __name__ == "__main__":
    main()
