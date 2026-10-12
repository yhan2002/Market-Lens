"""Point-in-time fast risk signals built only from allowlisted market aggregates."""

from __future__ import annotations

import math
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path
from typing import Any

from breadth_metrics import (
    EXACT_BREADTH_SOURCE,
    LIVE_BREADTH_SOURCE,
    MIN_LIVE_BREADTH_COVERAGE,
)
from portfolio_risk import portfolio_holdings_date_is_valid
from roro_price_observations import load_observation_series


FAST_REQUIRED = 3
FAST_TOTAL = 8


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _read_only(database: Path) -> sqlite3.Connection:
    return sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    if table not in {
        "daily_prices",
        "macro_observations",
        "breadth_metrics",
        "portfolio_risk_metrics",
        "daily_holdings",
    }:
        return set()
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    if not exists:
        return set()
    return {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')}


def _price_history(
    connection: sqlite3.Connection, ticker: str, as_of: str, count: int
) -> list[tuple[str, float]]:
    columns = _table_columns(connection, "daily_prices")
    if not {"ticker", "date"}.issubset(columns) or not ({"adj_close", "close"} & columns):
        return []
    adj_close = '"adj_close"' if "adj_close" in columns else "NULL"
    close = '"close"' if "close" in columns else "NULL"
    rows = connection.execute(
        f"""
        SELECT substr(date, 1, 10), {adj_close}, {close}
        FROM daily_prices
        WHERE ticker=? AND substr(date, 1, 10)<=?
        ORDER BY substr(date, 1, 10) DESC
        """,
        (ticker, as_of),
    ).fetchall()
    result: list[tuple[str, float]] = []
    seen: set[str] = set()
    for raw_date, raw_adjusted, raw_close in rows:
        price_date = str(raw_date)
        if price_date in seen:
            continue
        seen.add(price_date)
        price = next(
            (
                candidate
                for raw in (raw_adjusted, raw_close)
                if (candidate := _finite(raw)) is not None and candidate > 0.0
            ),
            None,
        )
        if price is not None:
            result.append((price_date, price))
            if len(result) == count:
                break
    return result


def _oas_bps(value: object) -> float | None:
    number = _finite(value)
    if number is None or not 0.0 <= number <= 100.0:
        return None
    # macro_observations has one explicit contract: percentage points (3.50 = 350bp).
    return number * 100.0


def load_hy_oas_history(
    database: Path, as_of: str, count: int = 2
) -> list[tuple[str, float]]:
    """Return distinct HY OAS observations that were available by ``as_of``."""
    if count < 1 or not database.exists():
        return []
    try:
        with closing(_read_only(database)) as connection:
            columns = _table_columns(connection, "macro_observations")
            required = {"date", "series", "value", "available_at", "source"}
            if not required.issubset(columns):
                return []
            rows = connection.execute(
                """
                SELECT substr(date, 1, 10), value
                FROM macro_observations
                WHERE upper(series)='HY_OAS'
                  AND substr(date, 1, 10)<=?
                  AND substr(available_at, 1, 10)<=?
                  AND trim(source)<>''
                  AND upper(source) NOT LIKE '%PROXY%'
                ORDER BY substr(date, 1, 10) DESC, available_at DESC
                """,
                (as_of, as_of),
            ).fetchall()
    except sqlite3.DatabaseError:
        return []

    result: list[tuple[str, float]] = []
    seen: set[str] = set()
    for raw_date, raw_value in rows:
        observation_date = str(raw_date)
        if observation_date in seen:
            continue
        seen.add(observation_date)
        value = _oas_bps(raw_value)
        if value is not None:
            result.append((observation_date, value))
            if len(result) == count:
                break
    return result


def _breadth_percent_points(value: object) -> float | None:
    number = _finite(value)
    if number is None or abs(number) > 1.0:
        return None
    return number * 100.0


def _iso_day(value: object) -> str | None:
    text = str(value)
    try:
        parsed = date.fromisoformat(text)
    except ValueError:
        return None
    return text if parsed.isoformat() == text else None


def observation_is_recent(as_of: str, observation_date: str, max_age_days: int = 5) -> bool:
    try:
        age = date.fromisoformat(as_of) - date.fromisoformat(observation_date)
    except ValueError:
        return False
    return 0 <= age.days <= max_age_days


def load_breadth_history(
    database: Path, as_of: str, count: int = 2
) -> list[dict[str, float | str | None]]:
    if count < 1 or not database.exists():
        return []
    try:
        with closing(_read_only(database)) as connection:
            columns = _table_columns(connection, "breadth_metrics")
            required = {
                "date",
                "breadth_200dma",
                "breadth_change_1d",
                "new_52w_lows",
                "advancers",
                "decliners",
                "coverage",
                "source",
                "membership_as_of",
                "available_at",
            }
            if not required.issubset(columns):
                return []
            rows = connection.execute(
                """
                SELECT substr(date, 1, 10), breadth_200dma, breadth_change_1d,
                       new_52w_lows, advancers, decliners, coverage, source,
                       substr(membership_as_of, 1, 10), substr(available_at, 1, 10)
                FROM breadth_metrics
                WHERE substr(date, 1, 10)<=?
                  AND substr(membership_as_of, 1, 10)<=substr(date, 1, 10)
                  AND substr(available_at, 1, 10)<=substr(date, 1, 10)
                ORDER BY substr(date, 1, 10) DESC
                """,
                (as_of,),
            ).fetchall()
    except sqlite3.DatabaseError:
        return []

    result: list[dict[str, float | str | None]] = []
    seen: set[str] = set()
    for (
        raw_date,
        raw_level,
        raw_change,
        raw_lows,
        raw_advancers,
        raw_decliners,
        raw_coverage,
        raw_source,
        raw_membership_as_of,
        raw_available_at,
    ) in rows:
        observation_date = _iso_day(raw_date)
        membership_as_of = _iso_day(raw_membership_as_of)
        available_at = _iso_day(raw_available_at)
        if observation_date is None:
            continue
        if observation_date in seen:
            continue
        seen.add(observation_date)
        coverage = _finite(raw_coverage)
        source = str(raw_source)
        coverage_ok = coverage is not None and (
            (
                source == EXACT_BREADTH_SOURCE
                and math.isclose(coverage, 1.0, abs_tol=1e-12)
            )
            or (
                source == LIVE_BREADTH_SOURCE
                and MIN_LIVE_BREADTH_COVERAGE <= coverage <= 1.0
            )
        )
        level = _finite(raw_level)
        change = _breadth_percent_points(raw_change)
        lows = _finite(raw_lows)
        advancers = _finite(raw_advancers)
        decliners = _finite(raw_decliners)
        if (
            not coverage_ok
            or membership_as_of is None
            or available_at is None
            or membership_as_of > observation_date
            or available_at > observation_date
            or level is None
            or not 0.0 <= level <= 1.0
            or change is None
            or lows is None
            or advancers is None
            or decliners is None
            or min(lows, advancers, decliners) < 0.0
        ):
            continue
        result.append(
            {
                "date": observation_date,
                "breadth_200dma": level,
                "breadth_change_1d_pp": change,
                "new_52w_lows": lows,
                "advancers": advancers,
                "decliners": decliners,
                "coverage": coverage,
                "source": source,
                "membership_as_of": membership_as_of,
                "available_at": available_at,
            }
        )
        if len(result) == count:
            break
    return result


def _fallback_portfolio_correlation(
    connection: sqlite3.Connection, as_of: str
) -> tuple[str, float, str] | None:
    holding_columns = _table_columns(connection, "daily_holdings")
    price_columns = _table_columns(connection, "daily_prices")
    if not {"date", "ticker", "market_value_base"}.issubset(holding_columns):
        return None
    if not {"date", "ticker"}.issubset(price_columns) or not (
        {"adj_close", "close"} & price_columns
    ):
        return None
    holding_date_row = connection.execute(
        "SELECT MAX(substr(date, 1, 10)) FROM daily_holdings WHERE substr(date, 1, 10)<=?",
        (as_of,),
    ).fetchone()
    if not holding_date_row or not holding_date_row[0]:
        return None
    holding_date = str(holding_date_row[0])
    if not observation_is_recent(as_of, holding_date, max_age_days=7):
        return None
    holding_rows = connection.execute(
        """
        SELECT ticker, SUM(market_value_base)
        FROM daily_holdings
        WHERE substr(date, 1, 10)=? AND ticker IS NOT NULL AND market_value_base>0
        GROUP BY ticker
        """,
        (holding_date,),
    ).fetchall()
    weights = {
        str(ticker): value
        for ticker, raw_weight in holding_rows
        if ticker and (value := _finite(raw_weight)) is not None and value > 0.0
    }
    if len(weights) < 2:
        return None

    adjusted = '"adj_close"' if "adj_close" in price_columns else "NULL"
    close = '"close"' if "close" in price_columns else "NULL"
    placeholders = ", ".join("?" for _ in weights)
    price_rows = connection.execute(
        f"""
        SELECT substr(date, 1, 10), ticker, {adjusted}, {close}
        FROM daily_prices
        WHERE ticker IN ({placeholders}) AND substr(date, 1, 10)<=?
        ORDER BY substr(date, 1, 10)
        """,
        (*weights, as_of),
    ).fetchall()
    records: list[tuple[str, str, float]] = []
    for raw_date, raw_ticker, raw_adjusted, raw_close in price_rows:
        price = next(
            (
                candidate
                for raw in (raw_adjusted, raw_close)
                if (candidate := _finite(raw)) is not None and candidate > 0.0
            ),
            None,
        )
        if price is not None:
            records.append((str(raw_date), str(raw_ticker), price))
    if not records:
        return None
    try:
        import pandas as pd

        from portfolio_risk import average_portfolio_correlation

        prices = pd.DataFrame(records, columns=["date", "ticker", "price"]).pivot_table(
            index="date", columns="ticker", values="price", aggfunc="last"
        )
        available = [
            ticker
            for ticker in weights
            if ticker in prices.columns
            and prices[ticker].notna().sum() >= 21
            and str(prices[ticker].dropna().index[-1]) == as_of
        ]
        if len(available) < 2:
            return None
        if sum(weights[ticker] for ticker in available) < 0.90 * sum(weights.values()):
            return None
        complete = prices[available].dropna()
        if complete.empty or str(complete.index[-1]) != as_of:
            return None
        correlation = average_portfolio_correlation(
            prices[available],
            {ticker: weights[ticker] for ticker in available},
            window=20,
            min_observations=15,
        )
    except (ImportError, ValueError):
        return None
    return as_of, correlation, "EXACT"


def _portfolio_correlation(database: Path, as_of: str) -> tuple[str, float, str] | None:
    if not database.exists():
        return None
    try:
        with closing(_read_only(database)) as connection:
            columns = _table_columns(connection, "portfolio_risk_metrics")
            required = {
                "date",
                "holdings_date",
                "average_correlation",
                "beta",
                "coverage",
                "source",
            }
            if required.issubset(columns):
                row = connection.execute(
                    """
                    SELECT substr(date, 1, 10), substr(holdings_date, 1, 10),
                           average_correlation, coverage, source
                    FROM portfolio_risk_metrics
                    WHERE substr(date, 1, 10)<=?
                    ORDER BY substr(date, 1, 10) DESC
                    LIMIT 1
                    """,
                    (as_of,),
                ).fetchone()
                if row and str(row[0]) == as_of:
                    correlation = _finite(row[2])
                    coverage = _finite(row[3])
                    if (
                        correlation is not None
                        and -1.0 <= correlation <= 1.0
                        and coverage is not None
                        and 0.90 <= coverage <= 1.0
                        and portfolio_holdings_date_is_valid(
                            as_of, str(row[1]), str(row[4])
                        )
                    ):
                        return str(row[0]), correlation, "EXACT"
            return _fallback_portfolio_correlation(connection, as_of)
    except sqlite3.DatabaseError:
        return None


def _condition(
    key: str,
    label: str,
    as_of: str,
    triggered: bool | None,
    detail: str,
    quality: str = "EXACT",
) -> dict[str, str]:
    return {
        "key": key,
        "label": label,
        "status": "UNAVAILABLE" if triggered is None else "TRIGGERED" if triggered else "CLEAR",
        "detail": detail,
        "data_quality": "UNAVAILABLE" if triggered is None else quality,
        "as_of": as_of,
    }


def _return_condition(
    connection: sqlite3.Connection,
    *,
    ticker: str,
    periods: int,
    threshold: float,
    trigger_below: bool,
    key: str,
    label: str,
    as_of: str,
) -> dict[str, str]:
    selected = ticker
    rows = _price_history(connection, ticker, as_of, periods + 1)
    if len(rows) != periods + 1 or rows[0][0] != as_of:
        return _condition(
            key,
            label,
            as_of,
            None,
            f"기준일을 포함한 {periods + 1}개 양수 가격 관측치가 필요합니다.",
        )
    change = rows[0][1] / rows[-1][1] - 1.0
    tolerance = 1e-12
    triggered = (
        change <= threshold + tolerance
        if trigger_below
        else change >= threshold - tolerance
    )
    return _condition(
        key,
        label,
        as_of,
        triggered,
        f"{selected} {periods}일 변화 {change:+.2%}",
    )


def _macro_return_condition(
    connection: sqlite3.Connection,
    *,
    series: str,
    periods: int,
    threshold: float,
    trigger_below: bool,
    key: str,
    label: str,
    as_of: str,
) -> dict[str, str]:
    """Evaluate a revised FRED series only from vintages available by ``as_of``."""
    if not {"date", "series", "value", "available_at"}.issubset(
        _table_columns(connection, "macro_observations")
    ):
        return _condition(key, label, as_of, None, "FRED USDJPY 관측치가 없습니다.")
    rows = connection.execute(
        """
        SELECT date, value, available_at
        FROM macro_observations
        WHERE series=? AND date<=? AND substr(available_at, 1, 10)<=?
        ORDER BY date DESC, available_at DESC
        """,
        (series, as_of, as_of),
    ).fetchall()
    selected: list[tuple[str, float]] = []
    seen: set[str] = set()
    for raw_date, raw_value, _ in rows:
        value = _finite(raw_value)
        day = str(raw_date)[:10]
        if value is not None and day not in seen:
            selected.append((day, value))
            seen.add(day)
        if len(selected) == periods + 1:
            break
    if len(selected) != periods + 1 or selected[0][0] != as_of:
        return _condition(
            key, label, as_of, None,
            f"기준일을 포함한 {periods + 1}개 FRED USDJPY 관측치가 필요합니다.",
        )
    change = selected[0][1] / selected[-1][1] - 1.0
    triggered = change <= threshold + 1e-12 if trigger_below else change >= threshold - 1e-12
    return _condition(
        key, label, as_of, triggered,
        f"FRED DEXJPUS {periods}일 변화 {change:+.2%}",
    )


def _session_jpy_return_condition(
    connection: sqlite3.Connection, observation_database: Path, as_of: str
) -> dict[str, str]:
    """Use Yahoo's raw timestamp mapping, never its UTC-labelled daily rows."""
    rows = connection.execute(
        "SELECT date FROM daily_prices WHERE ticker='NQ=F' AND date<=? ORDER BY date",
        (as_of,),
    ).fetchall()
    sessions = [str(row[0])[:10] for row in rows]
    try:
        values = load_observation_series(
            observation_database, ticker="JPY=X", nq_sessions=sessions
        ).dropna()
    except (OSError, ValueError, sqlite3.DatabaseError):
        values = None
    if values is None or len(values) < 4 or values.index[-1].date().isoformat() != as_of:
        return _condition(
            "jpy_3d_drop", "USDJPY 3일 -4%", as_of, None,
            "기준일을 포함한 4개 Yahoo 세션 정렬 USDJPY 관측치가 필요합니다.",
        )
    change = float(values.iloc[-1] / values.iloc[-4] - 1.0)
    return _condition(
        "jpy_3d_drop", "USDJPY 3일 -4%", as_of, change <= -0.04 + 1e-12,
        f"Yahoo JPY=X 세션정렬 3일 변화 {change:+.2%}",
    )


def build_fast_signal_snapshot(
    database: Path, as_of: str, *, jpy_observation_database: Path | None = None
) -> dict[str, Any]:
    """Build the eight-condition fast overlay without reading holdings or future data."""
    conditions: list[dict[str, str]] = []
    if database.exists():
        try:
            with closing(_read_only(database)) as connection:
                conditions.append(
                    _return_condition(
                        connection,
                        ticker="^VIX",
                        periods=1,
                        threshold=0.20,
                        trigger_below=False,
                        key="vix_1d_spike",
                        label="VIX 1일 +20%",
                        as_of=as_of,
                    )
                )

                vix = _price_history(connection, "^VIX", as_of, 1)
                vix3m = _price_history(connection, "^VIX3M", as_of, 1)
                if vix and vix3m and vix[0][0] == as_of and vix3m[0][0] == as_of:
                    ratio = vix[0][1] / vix3m[0][1]
                    conditions.append(
                        _condition(
                            "vix_term_inversion",
                            "VIX / VIX3M > 1.00",
                            as_of,
                            ratio > 1.00,
                            f"^VIX/^VIX3M {ratio:.3f}",
                        )
                    )
                else:
                    conditions.append(
                        _condition(
                            "vix_term_inversion",
                            "VIX / VIX3M > 1.00",
                            as_of,
                            None,
                            "기준일의 ^VIX와 ^VIX3M 가격이 모두 필요합니다.",
                        )
                    )

                conditions.append(
                    _return_condition(
                        connection,
                        ticker="NQ=F",
                        periods=1,
                        threshold=-0.025,
                        trigger_below=True,
                        key="nq_1d_drop",
                        label="NQ 1일 -2.5%",
                        as_of=as_of,
                    )
                )
                conditions.append(
                    _return_condition(
                        connection,
                        ticker="^MOVE",
                        periods=1,
                        threshold=0.10,
                        trigger_below=False,
                        key="move_1d_spike",
                        label="MOVE 1일 +10%",
                        as_of=as_of,
                    )
                )
                conditions.append(
                    _session_jpy_return_condition(
                        connection, jpy_observation_database, as_of
                    ) if jpy_observation_database is not None else _macro_return_condition(
                        connection,
                        series="USDJPY_FRED",
                        periods=3,
                        threshold=-0.04,
                        trigger_below=True,
                        key="jpy_3d_drop",
                        label="USDJPY 3일 -4%",
                        as_of=as_of,
                    )
                )
        except sqlite3.DatabaseError:
            conditions = []

    if not conditions:
        conditions.extend(
            [
                _condition("vix_1d_spike", "VIX 1일 +20%", as_of, None, "가격 데이터가 없습니다."),
                _condition("vix_term_inversion", "VIX / VIX3M > 1.00", as_of, None, "가격 데이터가 없습니다."),
                _condition("nq_1d_drop", "NQ 1일 -2.5%", as_of, None, "가격 데이터가 없습니다."),
                _condition("move_1d_spike", "MOVE 1일 +10%", as_of, None, "가격 데이터가 없습니다."),
                _condition("jpy_3d_drop", "USDJPY 3일 -4%", as_of, None, "가격 데이터가 없습니다."),
            ]
        )

    oas = load_hy_oas_history(database, as_of, 2)
    if len(oas) == 2 and observation_is_recent(as_of, oas[0][0]):
        widening = oas[0][1] - oas[1][1]
        hy_condition = _condition(
            "hy_oas_widening",
            "HY OAS +20bp",
            oas[0][0],
            widening >= 20.0 - 1e-9,
            f"HY OAS {oas[1][1]:.1f}bp → {oas[0][1]:.1f}bp ({widening:+.1f}bp)",
        )
    else:
        hy_condition = _condition(
            "hy_oas_widening",
            "HY OAS +20bp",
            as_of,
            None,
            "기준일까지 공개된 HY_OAS 최근 2개 관측치가 필요합니다.",
        )

    breadth = load_breadth_history(database, as_of, 1)
    breadth_change = breadth[0].get("breadth_change_1d_pp") if breadth else None
    if breadth and breadth[0]["date"] == as_of and isinstance(breadth_change, float):
        breadth_quality = (
            "EXACT"
            if breadth[0].get("source") == EXACT_BREADTH_SOURCE
            else "PUBLIC_LIVE"
        )
        breadth_condition = _condition(
            "breadth_1d_collapse",
            "Breadth 하루 -10%p",
            as_of,
            breadth_change <= -10.0 + 1e-9,
            f"200DMA breadth 1일 변화 {breadth_change:+.1f}%p",
            breadth_quality,
        )
    else:
        breadth_condition = _condition(
            "breadth_1d_collapse",
            "Breadth 하루 -10%p",
            as_of,
            None,
            "기준일의 안전 aggregate breadth_change_1d가 필요합니다.",
        )

    correlation = _portfolio_correlation(database, as_of)
    if correlation and correlation[0] == as_of:
        correlation_condition = _condition(
            "portfolio_correlation",
            "Portfolio correlation > 0.70",
            as_of,
            correlation[1] > 0.70,
            f"20일 aggregate 평균 상관계수 {correlation[1]:.3f}",
            correlation[2],
        )
    else:
        correlation_condition = _condition(
            "portfolio_correlation",
            "Portfolio correlation > 0.70",
            as_of,
            None,
            "기준일의 안전 aggregate average_correlation이 필요합니다.",
        )

    # Preserve the PDF order: price signals, exact credit, FX, breadth, correlation.
    conditions.insert(4, hy_condition)
    conditions.extend((breadth_condition, correlation_condition))
    triggered = [item for item in conditions if item["status"] == "TRIGGERED"]
    complete = all(item["status"] != "UNAVAILABLE" for item in conditions)
    status = "TRIGGERED" if len(triggered) >= FAST_REQUIRED else "CLEAR" if complete else "INCOMPLETE"
    if triggered:
        primary_driver = ", ".join(item["label"] for item in triggered[:3])
    elif complete:
        primary_driver = "Fast 조건 3개 미만"
    else:
        primary_driver = "관측 불충분"
    advisory = {
        "TRIGGERED": "급변 위험 중첩: 포트폴리오 베타 축소를 검토합니다.",
        "CLEAR": "Fast 8 급변 조건은 3개 미만입니다.",
        "INCOMPLETE": "관측되지 않은 조건이 있어 Fast overlay 결정을 보류합니다.",
    }[status]
    return {
        "as_of": as_of,
        "status": status,
        "count": len(triggered),
        "trigger_count": len(triggered),
        "required": FAST_REQUIRED,
        "total": FAST_TOTAL,
        "complete": complete,
        "primary_driver": primary_driver,
        "advisory": advisory,
        "conditions": conditions,
    }
