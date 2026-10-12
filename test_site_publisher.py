import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from breadth_metrics import EXACT_BREADTH_SOURCE
from risk_policy import (
    POLICY_VERSION,
    RULE_STRESS_SCORE_V3_LIVE_POLICY_VERSION,
    evaluate_reentry,
    evaluate_risk_policy,
)

from site_publisher import (
    ACTIVE_POLICY_VERSION,
    FORBIDDEN_TABLES,
    HMM_MODEL_VERSION,
    LEGACY_POLICY_SERIES_VERSION,
    POLICY_SERIES_VERSION,
    PUBLISH_TABLES,
    _add_risk_score_trend,
    _is_current_policy_snapshot,
    _previous_policy_state,
    _previous_risk_score_from_database,
    build_policy_decision,
    build_reentry_advisory,
    build_snapshot,
    create_publish_database,
    download_latest_snapshot,
    ensure_sanitized_database,
    list_nq_trading_dates,
    list_policy_decision_dates,
    publish_missing_policy_history,
    replay_policy_history,
    load_previous_snapshot,
    refresh_models,
    refresh_portfolio_inputs,
    sanitize_publish_database,
    sync_portfolio_risk_aggregate,
)


def fast_overlay(*, triggered: int = 0, unavailable: int = 0) -> dict:
    keys = (
        "vix_1d_spike",
        "vix_term_inversion",
        "nq_1d_drop",
        "move_1d_spike",
        "hy_oas_widening",
        "jpy_3d_drop",
        "breadth_1d_collapse",
        "portfolio_correlation",
    )
    conditions = []
    for index, key in enumerate(keys):
        status = (
            "TRIGGERED"
            if index < triggered
            else "UNAVAILABLE"
            if index < triggered + unavailable
            else "CLEAR"
        )
        conditions.append(
            {
                "key": key,
                "label": f"Fast {index}",
                "status": status,
                "detail": "test",
                "data_quality": "UNAVAILABLE" if status == "UNAVAILABLE" else "EXACT",
                "as_of": "2026-08-07",
            }
        )
    complete = unavailable == 0
    status = "TRIGGERED" if triggered >= 3 else "CLEAR" if complete else "INCOMPLETE"
    return {
        "as_of": "2026-08-07",
        "status": status,
        "count": triggered,
        "trigger_count": triggered,
        "required": 3,
        "total": 8,
        "complete": complete,
        "primary_driver": "test",
        "advisory": "test",
        "conditions": conditions,
    }


def reentry_advisory(*, history: bool = True) -> dict:
    keys = (
        "vix_term_structure",
        "hy_spread_stabilized",
        "nq_above_5dma",
        "breadth_divergence",
    )
    conditions = [
        {
            "key": key,
            "label": f"Re-entry {index}",
            "status": "CONFIRMED",
            "detail": "test",
            "data_quality": "EXACT",
        }
        for index, key in enumerate(keys)
    ]
    return evaluate_reentry(
        0.00 if history else None,
        0.10 if history else None,
        conditions,
        recent_peak_p_bear=0.82 if history else None,
    )


class SanitizedDatabaseTests(unittest.TestCase):
    def test_cloudium_workbook_imports_only_when_changed_but_prices_refresh_daily(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "private.db"
            workbook = Path(directory) / "ss 가공용 v2.xlsx"
            workbook.write_bytes(b"test")
            progress: list[str] = []
            with (
                patch("site_publisher.import_trading_log", return_value=(2, 0)) as importer,
                patch("site_publisher.latest_long_tickers", return_value=("A", "B")),
                patch("site_publisher.download_prices", return_value=(3, 30, "2026-01-01")) as download,
                patch("site_publisher._missing_portfolio_close_tickers", return_value=()),
            ):
                first = refresh_portfolio_inputs(
                    source, "2026-08-11", workbook, progress.append
                )
                second = refresh_portfolio_inputs(
                    source, "2026-08-11", workbook, progress.append
                )
                workbook.write_bytes(b"refreshed cache")
                third = refresh_portfolio_inputs(
                    source, "2026-08-11", workbook, progress.append
                )

            self.assertTrue(first)
            self.assertFalse(second)
            self.assertFalse(third)
            self.assertEqual(importer.call_count, 2)
            importer.assert_called_with(source, workbook, preserve_history=True)
            self.assertEqual(download.call_count, 3)
            self.assertNotIn("require_date", download.call_args.kwargs)
            self.assertNotIn("A", " ".join(progress))
            self.assertNotIn("B", " ".join(progress))

    def test_portfolio_prices_retry_symbols_missing_from_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "private.db"
            progress: list[str] = []
            with (
                patch("site_publisher.latest_long_tickers", return_value=("A", "B")),
                patch("site_publisher.download_prices", return_value=(2, 20, "2026-08-10")) as download,
                patch(
                    "site_publisher._missing_portfolio_close_tickers",
                    side_effect=[("B",), ()],
                ),
            ):
                refresh_portfolio_inputs(source, "2026-08-17", Path(directory) / "missing.xlsx", progress.append)

            self.assertEqual(download.call_count, 2)
            self.assertNotIn("require_date", download.call_args_list[0].kwargs)
            self.assertEqual(download.call_args_list[1].kwargs["tickers"], ("B",))
            self.assertEqual(download.call_args_list[1].kwargs["require_date"], date(2026, 8, 17))
            self.assertTrue(any("반영 완료" in message for message in progress))

    def test_portfolio_prices_retry_every_symbol_after_batch_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "private.db"
            progress: list[str] = []
            download_results = [RuntimeError("batch incomplete"), (1, 3, "2026-08-15"), (1, 4, "2026-08-15")]
            with (
                patch("site_publisher.latest_long_tickers", return_value=("A", "B")),
                patch("site_publisher.download_prices", side_effect=download_results) as download,
                patch(
                    "site_publisher._missing_portfolio_close_tickers",
                    side_effect=[("A", "B"), ()],
                ),
            ):
                refresh_portfolio_inputs(
                    source, "2026-08-17", Path(directory) / "missing.xlsx", progress.append
                )

            self.assertEqual([call.kwargs["tickers"] for call in download.call_args_list], [("A", "B"), ("A",), ("B",)])
            self.assertTrue(any("개별 재시도" in message for message in progress))
            self.assertTrue(any("반영 완료" in message for message in progress))

    def test_rejects_wrong_database_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "regime_router.db"
            sqlite3.connect(database).close()
            with self.assertRaisesRegex(ValueError, "regime_router_publish.db"):
                ensure_sanitized_database(database)

    def test_rejects_sensitive_tables(self) -> None:
        for table in sorted(FORBIDDEN_TABLES):
            with self.subTest(table=table), tempfile.TemporaryDirectory() as directory:
                database = Path(directory) / "regime_router_publish.db"
                with closing(sqlite3.connect(database)) as connection:
                    connection.execute(f"CREATE TABLE {table}(id INTEGER)")
                    connection.commit()
                with self.assertRaisesRegex(ValueError, "민감 테이블"):
                    ensure_sanitized_database(database)

    def test_accepts_sanitized_shadow_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "regime_router_publish.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("CREATE TABLE daily_prices(ticker TEXT)")
                connection.commit()
            ensure_sanitized_database(database)

    def test_allows_public_reviewed_move_closes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "regime_router_publish.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    "CREATE TABLE reviewed_move_closes(date TEXT PRIMARY KEY, close REAL, source TEXT, imported_at TEXT)"
                )
                connection.execute(
                    "INSERT INTO reviewed_move_closes VALUES ('2026-08-10', 75.46, 'INVESTING_REVIEWED_CSV', '2026-08-14T00:00:00Z')"
                )
                connection.commit()
            ensure_sanitized_database(database)

    def test_create_publish_database_copies_only_allowlisted_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.db"
            target = Path(directory) / "regime_router_publish.db"
            with closing(sqlite3.connect(source)) as connection:
                connection.execute("CREATE TABLE daily_prices(ticker TEXT)")
                connection.execute("INSERT INTO daily_prices VALUES ('SPY')")
                connection.execute(
                    "CREATE TABLE macro_observations(date TEXT, series TEXT, value REAL, available_at TEXT, source TEXT)"
                )
                connection.execute(
                    "INSERT INTO macro_observations VALUES ('2026-08-07','HY_OAS',3.5,'2026-08-07','FRED')"
                )
                connection.execute("CREATE TABLE daily_holdings(secret TEXT)")
                connection.execute("INSERT INTO daily_holdings VALUES ('private')")
                connection.commit()
            create_publish_database(source, target)
            with closing(sqlite3.connect(target)) as connection:
                tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                price = connection.execute("SELECT ticker FROM daily_prices").fetchone()[0]
                oas = connection.execute("SELECT value FROM macro_observations").fetchone()[0]
            self.assertEqual(price, "SPY")
            self.assertEqual(oas, 3.5)
            self.assertNotIn("daily_holdings", tables)
            self.assertTrue(
                {"macro_observations", "breadth_metrics", "portfolio_risk_metrics"}
                <= PUBLISH_TABLES
            )

    def test_publish_database_filters_internal_universe_prices_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.db"
            target = Path(directory) / "regime_router_publish.db"
            with closing(sqlite3.connect(source)) as connection:
                connection.executescript(
                    """
                    CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT);
                    CREATE TABLE universe(
                        ticker TEXT, group_name TEXT, source TEXT, active INTEGER
                    );
                    CREATE TABLE daily_prices(ticker TEXT, date TEXT);
                    INSERT INTO metadata VALUES
                        ('universe_config', 'C:/private/portfolio.py'),
                        ('last_price_download_at', '2026-08-07');
                    INSERT INTO universe VALUES
                        ('SECRET', 'TRADING_LOG', 'private', 1),
                        ('^VIX', 'REGIME_CROSS_ASSET', 'built-in', 1),
                        ('SPY', 'REGIME_BENCHMARK', 'built-in', 1);
                    INSERT INTO daily_prices VALUES
                        ('SECRET', '2026-08-07'),
                        ('^VIX', '2026-08-07'),
                        ('SPY', '2026-08-07');
                    """
                )
                connection.commit()

            create_publish_database(source, target)
            with closing(sqlite3.connect(target)) as connection:
                groups = connection.execute(
                    "SELECT ticker, group_name FROM universe ORDER BY ticker"
                ).fetchall()
                prices = connection.execute(
                    "SELECT ticker FROM daily_prices ORDER BY ticker"
                ).fetchall()
                metadata = connection.execute("SELECT * FROM metadata").fetchall()

            self.assertEqual(groups, [("SPY", "REGIME_BENCHMARK"), ("^VIX", "REGIME_CROSS_ASSET")])
            self.assertEqual(prices, [("SPY",), ("^VIX",)])
            self.assertEqual(metadata, [])

    def test_sanitize_removes_legacy_internal_universe_traces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "regime_router_publish.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.executescript(
                    """
                    CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT);
                    CREATE TABLE universe(
                        ticker TEXT, group_name TEXT, source TEXT, active INTEGER
                    );
                    CREATE TABLE daily_prices(ticker TEXT, date TEXT);
                    INSERT INTO metadata VALUES ('universe_config', 'C:/private/config.py');
                    INSERT INTO universe VALUES
                        ('SECRET', 'CSQ_MP_CP', 'private', 1),
                        ('SPY', 'REGIME_BENCHMARK', 'built-in', 1);
                    INSERT INTO daily_prices VALUES
                        ('SECRET', '2026-08-07'),
                        ('SPY', '2026-08-07');
                    """
                )
                connection.commit()

            sanitize_publish_database(database)
            with closing(sqlite3.connect(database)) as connection:
                groups = connection.execute("SELECT ticker FROM universe").fetchall()
                prices = connection.execute("SELECT ticker FROM daily_prices").fetchall()
                metadata = connection.execute("SELECT * FROM metadata").fetchall()

            self.assertEqual(groups, [("SPY",)])
            self.assertEqual(prices, [("SPY",)])
            self.assertEqual(metadata, [])

    def test_lists_only_nq_trading_dates_in_requested_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "regime_router_publish.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("CREATE TABLE daily_prices(date TEXT, ticker TEXT)")
                connection.executemany(
                    "INSERT INTO daily_prices VALUES (?, ?)",
                    [
                        ("2026-08-03", "NQ=F"),
                        ("2026-08-04", "SPY"),
                        ("2026-08-05", "NQ=F"),
                        ("2026-08-07", "NQ=F"),
                    ],
                )
                connection.commit()
            self.assertEqual(
                list_nq_trading_dates(database, "2026-08-04", "2026-08-06"),
                ["2026-08-05"],
            )

    def test_rejects_reversed_backfill_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "regime_router_publish.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("CREATE TABLE daily_prices(date TEXT, ticker TEXT)")
                connection.commit()
            with self.assertRaisesRegex(ValueError, "시작일"):
                list_nq_trading_dates(database, "2026-08-07", "2026-08-01")

    def test_policy_calendar_excludes_nq_days_without_the_frozen_hmm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "regime_router_publish.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("CREATE TABLE daily_prices(date TEXT, ticker TEXT)")
                connection.execute(
                    "CREATE TABLE regime_model_probabilities(date TEXT, model_version TEXT)"
                )
                connection.executemany(
                    "INSERT INTO daily_prices VALUES (?, 'NQ=F')",
                    [("2026-08-04",), ("2026-08-05",), ("2026-08-06",)],
                )
                connection.executemany(
                    "INSERT INTO regime_model_probabilities VALUES (?, ?)",
                    [
                        ("2026-08-04", HMM_MODEL_VERSION),
                        ("2026-08-06", HMM_MODEL_VERSION),
                        ("2026-08-05", "other-model"),
                    ],
                )
                connection.commit()
            self.assertEqual(
                list_policy_decision_dates(database, end="2026-08-06"),
                ["2026-08-04", "2026-08-06"],
            )
            self.assertEqual(
                list_policy_decision_dates(
                    database,
                    end="2026-08-06",
                    model_version="other-model",
                ),
                ["2026-08-05"],
            )

    def test_syncs_only_portfolio_aggregate_from_private_source(self) -> None:
        dates = pd.bdate_range("2026-04-01", periods=90)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "private.db"
            target = Path(directory) / "regime_router_publish.db"
            holding_date = dates[-1].date().isoformat()
            with closing(sqlite3.connect(source)) as connection:
                connection.execute(
                    "CREATE TABLE daily_holdings(date TEXT, ticker TEXT, quantity REAL, market_value_base REAL)"
                )
                connection.executemany(
                    "INSERT INTO daily_holdings VALUES (?, ?, ?, ?)",
                    [
                        (holding_date, "SECRET_A", 1.0, 60.0),
                        (holding_date, "SECRET_B", 1.0, 40.0),
                    ],
                )
                connection.execute(
                    "CREATE TABLE daily_prices(ticker TEXT, date TEXT, close REAL, adj_close REAL)"
                )
                rows = []
                for index, current in enumerate(dates):
                    current_date = current.date().isoformat()
                    rows.extend(
                        [
                            ("NQ=F", current_date, 100 * 1.005**index, 100 * 1.005**index),
                            ("SECRET_A", current_date, 100 * 1.01**index, 100 * 1.01**index),
                            ("SECRET_B", current_date, 200 * 1.01**index, 200 * 1.01**index),
                        ]
                    )
                connection.executemany("INSERT INTO daily_prices VALUES (?, ?, ?, ?)", rows)
                connection.commit()
            with closing(sqlite3.connect(target)) as connection:
                connection.execute(
                    "CREATE TABLE daily_prices(ticker TEXT, date TEXT, close REAL, adj_close REAL)"
                )
                for index, current in enumerate(dates):
                    current_date = current.date().isoformat()
                    connection.execute(
                        "INSERT INTO daily_prices VALUES (?, ?, ?, ?)",
                        ("NQ=F", current_date, 100 * 1.005**index, 100 * 1.005**index),
                    )
                connection.commit()

            progress: list[str] = []
            self.assertTrue(sync_portfolio_risk_aggregate(source, target, progress.append))

            with closing(sqlite3.connect(target)) as connection:
                columns = [
                    row[1]
                    for row in connection.execute("PRAGMA table_info(portfolio_risk_metrics)")
                ]
                metric = connection.execute(
                    "SELECT date, average_correlation, coverage FROM portfolio_risk_metrics"
                ).fetchone()
            self.assertNotIn("ticker", columns)
            with closing(sqlite3.connect(target)) as connection:
                public_tickers = {
                    row[0] for row in connection.execute("SELECT DISTINCT ticker FROM daily_prices")
                }
            self.assertEqual(public_tickers, {"NQ=F"})
            self.assertEqual(metric[0], holding_date)
            self.assertGreater(metric[1], 0.99)
            self.assertEqual(metric[2], 1.0)
            self.assertNotIn("SECRET_A", str(progress))
            self.assertNotIn("SECRET_B", str(progress))

    def test_failed_portfolio_aggregate_does_not_replace_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "private.db"
            target = Path(directory) / "regime_router_publish.db"
            sqlite3.connect(source).close()
            with closing(sqlite3.connect(target)) as connection:
                connection.executescript(
                    """
                    CREATE TABLE daily_prices(ticker TEXT, date TEXT, close REAL, adj_close REAL);
                    INSERT INTO daily_prices VALUES ('SPY','2026-08-07',100,100);
                    CREATE TABLE portfolio_risk_metrics(
                        date TEXT PRIMARY KEY, holdings_date TEXT, average_correlation REAL,
                        beta REAL, coverage REAL, source TEXT
                    );
                    INSERT INTO portfolio_risk_metrics VALUES(
                        '2026-08-06','2026-08-06',0.55,NULL,1.0,'EXISTING'
                    );
                    """
                )
                connection.commit()

            progress: list[str] = []
            self.assertFalse(sync_portfolio_risk_aggregate(source, target, progress.append))
            with closing(sqlite3.connect(target)) as connection:
                rows = connection.execute(
                    "SELECT date, average_correlation, source FROM portfolio_risk_metrics"
                ).fetchall()
            self.assertEqual(rows, [("2026-08-06", 0.55, "EXISTING")])
            self.assertEqual(len(progress), 1)

    def test_stale_optional_oil_does_not_abort_unrelated_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "regime_router_publish.db"
            sqlite3.connect(database).close()
            messages = []
            with (
                patch("site_publisher.update_everything"),
                patch("site_publisher.register_sector_universe"),
                patch("site_publisher.download_prices", return_value=(12, 12, None)),
                patch("site_publisher.latest_nq_date", return_value="2026-09-09"),
                patch("site_publisher.sync_current_open_data_membership", return_value=(503, 0, 0)),
                patch("site_publisher.refresh_turning_macro_inputs", side_effect=ValueError("WTI_SPOT stale")),
            ):
                refresh_models(database, messages.append)
            self.assertTrue(any("WTI_SPOT stale" in message for message in messages))
            self.assertTrue(any("Breadth aggregate unavailable" in message for message in messages))

    def test_refresh_prepares_exact_macro_and_breadth_inputs_with_mocked_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "regime_router_publish.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    """
                    CREATE TABLE sp500_membership(
                        ticker TEXT, effective_from TEXT, effective_to TEXT,
                        membership_as_of TEXT, available_at TEXT, source TEXT
                    )
                    """
                )
                connection.executescript(
                    """
                    INSERT INTO sp500_membership VALUES (
                        'A', '2026-01-01', NULL, '2026-01-01', '2026-01-01',
                        'LICENSED_PIT:TEST'
                    );
                    CREATE TABLE universe(ticker TEXT, group_name TEXT, active INTEGER);
                    CREATE TABLE daily_prices(ticker TEXT, date TEXT, close REAL, adj_close REAL);
                    INSERT INTO universe VALUES ('A', 'SP500_BREADTH', 1);
                    INSERT INTO daily_prices VALUES ('A', '2026-08-07', 100, 100);
                    """
                )
                connection.commit()
            progress: list[str] = []
            with (
                patch("site_publisher.update_everything") as update,
                patch("site_publisher.register_sector_universe"),
                patch("site_publisher.register_universe") as chart_register,
                patch(
                    "site_publisher.download_prices",
                    side_effect=[(11, 22, None), (1, 5, None), (500, 1000, None)],
                ) as download,
                patch("site_publisher.latest_nq_date", return_value="2026-08-07"),
                patch(
                    "site_publisher.sync_current_open_data_membership",
                    return_value=(503, 0, 0),
                ) as public_members,
                patch(
                    "site_publisher.current_korea_date",
                    return_value=pd.Timestamp("2026-08-12").date(),
                ),
                patch(
                    "site_publisher.refresh_turning_macro_inputs",
                    return_value=({"HY_OAS": 1, "CASH_3M": 1}, False),
                ) as macro,
                patch("site_publisher.register_members_overlapping", return_value=1) as members,
                patch("site_publisher.compute_breadth_metrics", return_value=200) as breadth,
                patch("site_publisher.load_breadth_history", return_value=[{"date": "2026-08-07"}]),
            ):
                refresh_models(database, progress.append)

        macro.assert_called_once_with(database, "2026-08-07")
        public_members.assert_called_once()
        self.assertEqual(public_members.call_args.args[0], database)
        self.assertEqual(
            public_members.call_args.args[1], pd.Timestamp("2026-08-12").date()
        )
        update.assert_called_once_with(
            database,
            progress.append,
            include_equity_universe=False,
            required_price_date=None,
        )
        members.assert_called_once_with(
            database,
            pd.Timestamp("2024-08-07").date(),
            pd.Timestamp("2026-08-07").date(),
        )
        chart_register.assert_called_once_with(database, ("^NDX",), group_name="DISPLAY_BENCHMARK")
        self.assertEqual(download.call_count, 3)
        self.assertEqual(download.call_args_list[1].kwargs["group_name"], "DISPLAY_BENCHMARK")
        self.assertEqual(download.call_args_list[2].kwargs["group_name"], "SP500_BREADTH")
        self.assertEqual(download.call_args_list[2].kwargs["years"], 2)
        breadth.assert_called_once_with(database)
        self.assertTrue(any("전환 매크로 8계열" in message for message in progress))
        self.assertTrue(any("Breadth " in message and "반영 완료" in message for message in progress))


class ReentryAdvisoryDatabaseTests(unittest.TestCase):
    @staticmethod
    def _create_database(path: Path) -> None:
        with closing(sqlite3.connect(path)) as connection:
            connection.execute(
                """
                CREATE TABLE daily_prices(
                    ticker TEXT, date TEXT, close REAL, adj_close REAL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE regime_model_probabilities(
                    model_version TEXT, date TEXT, p_bear REAL
                )
                """
            )
            connection.execute(
                "CREATE TABLE macro_observations(date TEXT, series TEXT, value REAL, available_at TEXT, source TEXT)"
            )
            connection.execute(
                """
                CREATE TABLE breadth_metrics(
                    date TEXT, breadth_200dma REAL, breadth_change_1d REAL,
                    new_52w_lows REAL, advancers REAL, decliners REAL,
                    coverage REAL, source TEXT,
                    membership_as_of TEXT, available_at TEXT
                )
                """
            )
            connection.commit()

    def test_builds_exact_point_in_time_conditions_and_probability_trend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "local.db"
            self._create_database(database)
            with closing(sqlite3.connect(database)) as connection:
                connection.executemany(
                    "INSERT INTO daily_prices VALUES (?, ?, ?, ?)",
                    [
                        *(
                            ("NQ=F", current.date().isoformat(), 100.0, 100.0)
                            for current in pd.date_range("2026-07-18", periods=20)
                        ),
                        ("NQ=F", "2026-08-07", 90.0, 90.0),
                        ("HYG", "2026-08-06", 90.0, 90.0),
                        ("IEF", "2026-08-06", 100.0, 100.0),
                        ("HYG", "2026-08-07", 92.0, 92.0),
                        ("IEF", "2026-08-07", 100.0, 100.0),
                        ("^VIX", "2026-08-07", 18.0, 18.0),
                        ("^VIX3M", "2026-08-07", 20.0, 20.0),
                        ("QQQ", "2026-08-08", 1.0, 1.0),
                    ],
                )
                connection.executemany(
                    "INSERT INTO regime_model_probabilities VALUES (?, ?, ?)",
                    [
                        ("test-hmm", "2026-08-05", 0.82),
                        ("test-hmm", "2026-08-06", 0.70),
                        ("test-hmm", "2026-08-08", 0.10),
                    ],
                )
                connection.executemany(
                    "INSERT INTO macro_observations VALUES (?, ?, ?, ?, ?)",
                    [
                        ("2026-08-06", "HY_OAS", 4.0, "2026-08-06", "FRED"),
                        ("2026-08-07", "HY_OAS", 3.9, "2026-08-07", "FRED"),
                    ],
                )
                connection.executemany(
                    "INSERT INTO breadth_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            "2026-08-06", 0.40, -0.01, 100, 1000, 1500, 1.0,
                            EXACT_BREADTH_SOURCE, "2026-08-06", "2026-08-06",
                        ),
                        (
                            "2026-08-07", 0.42, 0.02, 80, 1200, 1300, 1.0,
                            EXACT_BREADTH_SOURCE, "2026-08-07", "2026-08-07",
                        ),
                    ],
                )
                connection.commit()

            result = build_reentry_advisory(
                database,
                "2026-08-07",
                {
                    "date": "2026-08-07",
                    "model_version": "test-hmm",
                    "p_bear": 0.60,
                },
            )
            conditions = {item["key"]: item for item in result["conditions"]}

            self.assertEqual(result["previous_p_bear"], 0.70)
            self.assertEqual(result["recent_peak_p_bear"], 0.82)
            self.assertTrue(result["p_bear_falling"])
            self.assertEqual(result["status"], "ELIGIBLE_ONE_STEP")
            self.assertEqual(conditions["hy_spread_stabilized"]["status"], "CONFIRMED")
            self.assertEqual(conditions["hy_spread_stabilized"]["data_quality"], "EXACT")
            self.assertEqual(conditions["nq_above_5dma"]["status"], "NOT_CONFIRMED")
            self.assertEqual(conditions["nq_above_5dma"]["data_quality"], "EXACT")
            self.assertEqual(conditions["vix_term_structure"]["status"], "CONFIRMED")
            self.assertEqual(conditions["breadth_divergence"]["status"], "CONFIRMED")
            self.assertIn("20D", conditions["breadth_divergence"]["detail"])
            self.assertNotIn("proxy", str(result).lower())

    def test_reentry_rejects_proxy_future_vintage_and_legacy_breadth(self) -> None:
        cases = (
            ("proxy", "PROXY", "2026-08-07", "2026-08-07", False),
            (
                "future-membership",
                EXACT_BREADTH_SOURCE,
                "2026-08-08",
                "2026-08-07",
                False,
            ),
            (
                "future-availability",
                EXACT_BREADTH_SOURCE,
                "2026-08-07",
                "2026-08-08",
                False,
            ),
            ("legacy", EXACT_BREADTH_SOURCE, "2026-08-07", "2026-08-07", True),
        )
        for name, source, membership_as_of, available_at, legacy in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                database = Path(directory) / "local.db"
                self._create_database(database)
                with closing(sqlite3.connect(database)) as connection:
                    connection.executemany(
                        "INSERT INTO daily_prices VALUES (?, ?, ?, ?)",
                        [
                            *(
                                ("NQ=F", current.date().isoformat(), 100.0, 100.0)
                                for current in pd.date_range("2026-07-18", periods=20)
                            ),
                            ("NQ=F", "2026-08-07", 90.0, 90.0),
                        ],
                    )
                    if legacy:
                        connection.execute("DROP TABLE breadth_metrics")
                        connection.execute(
                            """
                            CREATE TABLE breadth_metrics(
                                date TEXT, breadth_200dma REAL, breadth_change_1d REAL,
                                new_52w_lows REAL, advancers REAL, decliners REAL,
                                coverage REAL, source TEXT
                            )
                            """
                        )
                        connection.executemany(
                            "INSERT INTO breadth_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            [
                                ("2026-08-06", 0.40, -0.01, 100, 1000, 1500, 1.0, source),
                                ("2026-08-07", 0.42, 0.02, 80, 1200, 1300, 1.0, source),
                            ],
                        )
                    else:
                        connection.executemany(
                            "INSERT INTO breadth_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            [
                                (
                                    "2026-08-06", 0.40, -0.01, 100, 1000, 1500, 1.0,
                                    source, "2026-08-06", "2026-08-06",
                                ),
                                (
                                    "2026-08-07", 0.42, 0.02, 80, 1200, 1300, 1.0,
                                    source, membership_as_of, available_at,
                                ),
                            ],
                        )
                    connection.commit()

                conditions = {
                    item["key"]: item
                    for item in build_reentry_advisory(database, "2026-08-07", None)[
                        "conditions"
                    ]
                }
                self.assertEqual(conditions["breadth_divergence"]["status"], "UNAVAILABLE")

    def test_no_prior_probability_history_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "local.db"
            self._create_database(database)

            result = build_reentry_advisory(
                database,
                "2026-08-07",
                {
                    "date": "2026-08-07",
                    "model_version": "test-hmm",
                    "p_bear": 0.60,
                },
            )

            self.assertEqual(result["status"], "NO_HISTORY")
            self.assertIsNone(result["previous_p_bear"])
            self.assertIsNone(result["recent_peak_p_bear"])
            self.assertIsNone(result["p_bear_falling"])

    def test_low_recent_peak_waits_for_meaningful_risk_off_peak(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "local.db"
            self._create_database(database)
            with closing(sqlite3.connect(database)) as connection:
                connection.executemany(
                    "INSERT INTO daily_prices VALUES (?, ?, ?, ?)",
                    [
                        *(("NQ=F", f"2026-08-0{day}", 90.0 + day, 90.0 + day) for day in range(3, 8)),
                        ("HYG", "2026-08-06", 90.0, 90.0),
                        ("IEF", "2026-08-06", 100.0, 100.0),
                        ("HYG", "2026-08-07", 92.0, 92.0),
                        ("IEF", "2026-08-07", 100.0, 100.0),
                        ("^VIX", "2026-08-07", 18.0, 18.0),
                        ("^VIX3M", "2026-08-07", 20.0, 20.0),
                    ],
                )
                connection.executemany(
                    "INSERT INTO regime_model_probabilities VALUES (?, ?, ?)",
                    [
                        ("test-hmm", "2026-08-05", 0.25),
                        ("test-hmm", "2026-08-06", 0.30),
                    ],
                )
                connection.executemany(
                    "INSERT INTO macro_observations VALUES (?, ?, ?, ?, ?)",
                    [
                        ("2026-08-06", "HY_OAS", 4.0, "2026-08-06", "FRED"),
                        ("2026-08-07", "HY_OAS", 3.9, "2026-08-07", "FRED"),
                    ],
                )
                connection.executemany(
                    "INSERT INTO breadth_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            "2026-08-06", 0.40, -0.01, 100, 1000, 1500, 1.0,
                            EXACT_BREADTH_SOURCE, "2026-08-06", "2026-08-06",
                        ),
                        (
                            "2026-08-07", 0.42, 0.02, 80, 1200, 1300, 1.0,
                            EXACT_BREADTH_SOURCE, "2026-08-07", "2026-08-07",
                        ),
                    ],
                )
                connection.commit()

            result = build_reentry_advisory(
                database,
                "2026-08-07",
                {
                    "date": "2026-08-07",
                    "model_version": "test-hmm",
                    "p_bear": 0.20,
                },
            )

            self.assertEqual(result["recent_peak_p_bear"], 0.30)
            conditions = {item["key"]: item for item in result["conditions"]}
            self.assertEqual(result["confirmed"], 3)
            self.assertEqual(conditions["breadth_divergence"]["status"], "UNAVAILABLE")
            self.assertEqual(result["status"], "WAIT_RISK_OFF_PEAK")

    def test_stale_exact_and_current_proxy_prices_do_not_confirm_reentry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "local.db"
            self._create_database(database)
            with closing(sqlite3.connect(database)) as connection:
                connection.executemany(
                    "INSERT INTO daily_prices VALUES (?, ?, ?, ?)",
                    [
                        ("^VIX", "2026-08-06", 18.0, 18.0),
                        ("^VIX3M", "2026-08-06", 20.0, 20.0),
                        ("HYG", "2026-08-05", 90.0, 90.0),
                        ("IEF", "2026-08-05", 100.0, 100.0),
                        ("HYG", "2026-08-06", 91.0, 91.0),
                        ("IEF", "2026-08-06", 100.0, 100.0),
                        *(("QQQ", f"2026-08-0{day}", 90.0 + day, 90.0 + day) for day in range(2, 7)),
                    ],
                )
                connection.commit()

            result = build_reentry_advisory(
                database,
                "2026-08-07",
                {
                    "date": "2026-08-07",
                    "model_version": "test-hmm",
                    "p_bear": 0.60,
                },
            )
            conditions = {item["key"]: item for item in result["conditions"]}

            self.assertEqual(conditions["vix_term_structure"]["status"], "UNAVAILABLE")
            self.assertEqual(conditions["hy_spread_stabilized"]["status"], "UNAVAILABLE")
            self.assertEqual(conditions["nq_above_5dma"]["status"], "UNAVAILABLE")


class SnapshotRiskPolicyTests(unittest.TestCase):
    def test_recognizes_only_the_current_live_snapshot(self) -> None:
        self.assertEqual(
            ACTIVE_POLICY_VERSION, RULE_STRESS_SCORE_V3_LIVE_POLICY_VERSION
        )
        snapshot = {
            "schema_version": 3,
            "as_of": "2026-08-10",
            "history": {"series_version": POLICY_SERIES_VERSION},
        }
        self.assertTrue(_is_current_policy_snapshot(snapshot, "2026-08-10"))
        self.assertFalse(_is_current_policy_snapshot(snapshot, "2026-08-11"))
        self.assertFalse(
            _is_current_policy_snapshot(
                {**snapshot, "history": {"series_version": "old-series"}},
                "2026-08-10",
            )
        )

    def test_downloads_latest_snapshot_for_the_explicit_live_series(self) -> None:
        payload = {
            "schema_version": 3,
            "as_of": "2026-08-10",
            "history": {"series_version": POLICY_SERIES_VERSION},
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(payload).encode("utf-8")

        with patch(
            "site_publisher.urllib.request.urlopen",
            return_value=FakeResponse(),
        ) as urlopen:
            self.assertEqual(
                download_latest_snapshot("https://example.test", "token"),
                payload,
            )
        self.assertIn("series=", urlopen.call_args.args[0].full_url)

    def test_adds_optional_policy_without_hiding_stale_raw_channel(self) -> None:
        rrg = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-08-07"),
                    "ticker": "XLK",
                    "sector": "Technology",
                    "quadrant": "LEADING",
                    "quadrant_transition": "LEADING→LEADING",
                    "tail_heading": "NORTH_EAST",
                    "rs_ratio": 101.0,
                    "rs_momentum": 102.0,
                    "delta_rs_ratio": 0.2,
                    "delta_rs_momentum": 0.3,
                    "shadow_score": 1.0,
                }
            ]
        )
        forecast = pd.DataFrame(
            [
                {
                    "ticker": "XLK",
                    "forecast_rank": 1,
                    "forecast_score": 0.5,
                    "predicted_excess_20d": 0.01,
                    "signal": "OVERWEIGHT",
                }
            ]
        )
        stale_regime = {
            "date": "2026-08-06",
            "model_version": "test-hmm",
            "state": "BULL",
            "p_bull": 0.8,
            "p_sideways": 0.1,
            "p_bear": 0.1,
        }
        crash_guard = {"date": "2026-08-07", "risk_band": "NORMAL_SHADOW"}
        onset = {"date": "2026-08-07", "risk_band": "DORMANT"}
        reentry = reentry_advisory(history=False)
        stale_policy = evaluate_risk_policy(
            None,
            fast_overlay(),
            reentry,
            as_of="2026-08-07",
            slow_risk_score=35,
            policy_version=ACTIVE_POLICY_VERSION,
        )

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "regime_router_publish.db"
            sqlite3.connect(database).close()
            with (
                patch("site_publisher.latest_nq_date", return_value="2026-08-07"),
                patch("site_publisher.get_regime_status", return_value=stale_regime),
                patch("site_publisher.get_crash_guard_status", return_value=crash_guard),
                patch("site_publisher.get_shadow_hazard_status", return_value=onset),
                patch(
                    "site_publisher.build_live_risk_score_map",
                    return_value={"2026-08-06": 40, "2026-08-07": 35},
                ),
                patch(
                    "site_publisher.replay_policy_history",
                    return_value=[
                        {
                            "as_of": "2026-08-07",
                            "regime": stale_regime,
                            "risk_policy": stale_policy,
                        }
                    ],
                ) as replay,
                patch("site_publisher.load_price_matrix", return_value=object()),
                patch("site_publisher.build_rrg_signals", return_value=object()),
                patch("site_publisher.latest_snapshot", return_value=rrg),
                patch("site_publisher.load_frame", return_value=object()),
                patch(
                    "site_publisher.fit_leading_forecast",
                    return_value=(
                        forecast,
                        {
                            "as_of": "2026-08-07",
                            "target": "relative_20d",
                            "training_weeks": 52,
                        },
                    ),
                ),
            ):
                snapshot = build_snapshot(
                    database,
                    previous_snapshot={
                        "schema_version": 2,
                        "as_of": "2026-08-06",
                        "regime": {"model_version": HMM_MODEL_VERSION},
                        "risk_policy": {
                            "decision_status": "OK",
                            "recommended_exposure": 0.5,
                        },
                    },
                )

        self.assertEqual(snapshot["schema_version"], 3)
        self.assertEqual(snapshot["history"]["series_version"], POLICY_SERIES_VERSION)
        self.assertNotEqual(POLICY_SERIES_VERSION, LEGACY_POLICY_SERIES_VERSION)
        self.assertTrue(POLICY_SERIES_VERSION.endswith("__r5"))
        self.assertTrue(LEGACY_POLICY_SERIES_VERSION.endswith("__r4"))
        self.assertEqual(snapshot["history"]["seed_mode"], "FULL_REPLAY")
        self.assertEqual(
            replay.call_args.kwargs["policy_version"], ACTIVE_POLICY_VERSION
        )
        self.assertEqual(
            replay.call_args.kwargs["score_by_date"]["2026-08-07"], 35
        )
        self.assertEqual(snapshot["regime"], stale_regime)
        self.assertEqual(snapshot["crash_guard"], crash_guard)
        self.assertEqual(snapshot["drawdown_onset"], onset)
        self.assertNotIn("fast_overlay", snapshot)
        self.assertIn("risk_policy", snapshot)
        self.assertEqual(snapshot["risk_policy"]["decision_status"], "NO_DECISION")
        self.assertIn("HMM_REGIME", snapshot["risk_policy"]["missing_channels"])
        self.assertEqual(snapshot["risk_policy"]["reentry"]["status"], "NO_HISTORY")
        self.assertEqual(snapshot["risk_policy"]["fast_overlay"]["total"], 8)
        self.assertEqual(
            set(snapshot["risk_policy"]["fast_overlay"]),
            {"status", "complete", "trigger_count", "required", "total", "conditions"},
        )
        self.assertTrue(
            all(
                "as_of" not in condition
                for condition in snapshot["risk_policy"]["fast_overlay"]["conditions"]
            )
        )
        self.assertIsNone(snapshot["risk_policy"]["risk_score_delta"])
        self.assertIsNone(snapshot["risk_policy"]["risk_trend"])
        self.assertIn("운영 참고지표", snapshot["notices"][0])
        self.assertNotIn("shadow", snapshot["notices"][0].lower())

    def test_previous_state_requires_exact_series_policy_hmm_and_date(self) -> None:
        decision = evaluate_risk_policy(
            {
                "date": "2026-08-06",
                "model_version": HMM_MODEL_VERSION,
                "state": "BULL",
                "p_bull": 0.85,
                "p_sideways": 0.10,
                "p_bear": 0.05,
            },
            fast_overlay(),
            None,
            as_of="2026-08-06",
            slow_risk_score=20,
            policy_version=ACTIVE_POLICY_VERSION,
        )
        snapshot = {
            "schema_version": 3,
            "as_of": "2026-08-06",
            "history": {"series_version": POLICY_SERIES_VERSION},
            "regime": {"model_version": HMM_MODEL_VERSION},
            "risk_policy": decision,
        }
        self.assertEqual(
            _previous_policy_state(snapshot, "2026-08-07"),
            decision["policy_state"],
        )
        invalid = (
            {**snapshot, "schema_version": 2},
            {**snapshot, "as_of": "2026-08-07"},
            {**snapshot, "as_of": "2026-08-08"},
            {**snapshot, "history": {"series_version": "old-series"}},
            {**snapshot, "regime": {"model_version": "legacy-spy-model"}},
            {
                **snapshot,
                "risk_policy": {**decision, "model_version": "risk-policy-old"},
            },
        )
        for candidate in invalid:
            with self.subTest(candidate=candidate):
                self.assertIsNone(_previous_policy_state(candidate, "2026-08-07"))

    def test_no_decision_snapshot_preserves_the_last_valid_state(self) -> None:
        prior = evaluate_risk_policy(
            {
                "date": "2026-08-06",
                "model_version": HMM_MODEL_VERSION,
                "state": "BEAR",
                "p_bull": 0.10,
                "p_sideways": 0.25,
                "p_bear": 0.65,
            },
            fast_overlay(),
            None,
            as_of="2026-08-06",
            slow_risk_score=80,
            policy_version=ACTIVE_POLICY_VERSION,
        )
        unavailable = evaluate_risk_policy(
            {
                "date": "2026-08-07",
                "model_version": HMM_MODEL_VERSION,
                "state": "BULL",
                "p_bull": 0.85,
                "p_sideways": 0.10,
                "p_bear": 0.05,
            },
            None,
            None,
            as_of="2026-08-07",
            previous_state=prior["policy_state"],
            slow_risk_score=20,
            policy_version=ACTIVE_POLICY_VERSION,
        )
        snapshot = {
            "schema_version": 3,
            "as_of": "2026-08-07",
            "history": {"series_version": POLICY_SERIES_VERSION},
            "regime": {"model_version": HMM_MODEL_VERSION},
            "risk_policy": unavailable,
        }
        self.assertEqual(unavailable["decision_status"], "NO_DECISION")
        self.assertEqual(
            _previous_policy_state(snapshot, "2026-08-10"),
            prior["policy_state"],
        )

    def test_full_and_split_replay_produce_identical_exposure_path(self) -> None:
        dates = [
            "2026-01-02",
            "2026-01-05",
            "2026-01-06",
            "2026-01-07",
            "2026-01-08",
            "2026-01-09",
            "2026-01-12",
        ]
        probabilities = {
            dates[0]: (100, 0, 0),
            dates[1]: (0, 0, 100),
            **{day: (100, 0, 0) for day in dates[2:]},
        }

        def selected_dates(
            database, *, end, start_exclusive=None, model_version=HMM_MODEL_VERSION
        ):
            del database
            return [
                day
                for day in dates
                if day <= end and (start_exclusive is None or day > start_exclusive)
            ]

        def one_decision(
            database,
            as_of,
            *,
            previous_state=None,
            model_version=HMM_MODEL_VERSION,
        ):
            del database
            bull, sideways, bear = probabilities[as_of]
            values = {"BULL": bull, "SIDEWAYS": sideways, "BEAR": bear}
            regime = {
                "date": as_of,
                "model_version": model_version,
                "state": max(values, key=values.get),
                "p_bull": bull / 100,
                "p_sideways": sideways / 100,
                "p_bear": bear / 100,
            }
            policy = evaluate_risk_policy(
                regime,
                fast_overlay(),
                None,
                as_of=as_of,
                previous_state=previous_state,
            )
            return {"as_of": as_of, "regime": regime, "risk_policy": policy}

        with (
            patch(
                "site_publisher.list_policy_decision_dates",
                side_effect=selected_dates,
            ),
            patch("site_publisher.build_policy_decision", side_effect=one_decision),
        ):
            full = replay_policy_history(Path("ignored.db"), end=dates[-1])
            first = replay_policy_history(Path("ignored.db"), end=dates[2])
            second = replay_policy_history(
                Path("ignored.db"),
                end=dates[-1],
                previous_state=first[-1]["risk_policy"]["policy_state"],
                start_exclusive=dates[2],
            )

        full_exposure = [row["risk_policy"]["recommended_exposure"] for row in full]
        split_exposure = [
            row["risk_policy"]["recommended_exposure"] for row in first + second
        ]
        self.assertEqual(full_exposure, split_exposure)
        self.assertEqual(full_exposure, [0.95, 0.40, 0.40, 0.40, 0.60, 0.80, 0.95])

    def test_live_replay_fails_closed_when_a_score_is_missing(self) -> None:
        with patch(
            "site_publisher.list_policy_decision_dates",
            return_value=["2026-08-06"],
        ):
            with self.assertRaisesRegex(ValueError, "no rule-based risk score"):
                replay_policy_history(
                    Path("ignored.db"),
                    end="2026-08-06",
                    score_by_date={},
                    policy_version=ACTIVE_POLICY_VERSION,
                )

    def test_previous_snapshot_file_is_optional_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.json"
            self.assertIsNone(load_previous_snapshot(path))
            path.write_text("not-json", encoding="utf-8")
            self.assertIsNone(load_previous_snapshot(path))
            with self.assertRaisesRegex(ValueError, "안전하게 읽을 수 없습니다"):
                load_previous_snapshot(path, strict=True)
            path.write_text('{"as_of": "2026-08-06"}', encoding="utf-8")
            self.assertEqual(
                load_previous_snapshot(path),
                {"as_of": "2026-08-06"},
            )

    def test_same_day_rerun_holds_upward_step_until_next_trading_day(self) -> None:
        previous_state = {
            "state_version": 1,
            "policy_version": POLICY_VERSION,
            "regime_model_version": HMM_MODEL_VERSION,
            "last_as_of": "2026-08-06",
            "exposure": 0.50,
            "raw_target": 0.95,
            "recovery_clear_streak": 2,
            "episode_peak_p_bear": 0.82,
            "episode_started_as_of": "2026-08-01",
        }
        eligible_reentry = reentry_advisory()
        same_day = evaluate_risk_policy(
            {
                "date": "2026-08-06",
                "model_version": HMM_MODEL_VERSION,
                "p_bull": 1.0,
                "p_sideways": 0.0,
                "p_bear": 0.0,
            },
            fast_overlay(),
            eligible_reentry,
            as_of="2026-08-06",
            previous_state=previous_state,
        )
        next_day = evaluate_risk_policy(
            {
                "date": "2026-08-07",
                "model_version": HMM_MODEL_VERSION,
                "p_bull": 1.0,
                "p_sideways": 0.0,
                "p_bear": 0.0,
            },
            fast_overlay(),
            eligible_reentry,
            as_of="2026-08-07",
            previous_state=same_day["policy_state"],
        )

        self.assertEqual(eligible_reentry["status"], "ELIGIBLE_ONE_STEP")
        self.assertEqual(same_day["raw_recommended_exposure"], 0.95)
        self.assertEqual(same_day["recommended_exposure"], 0.5)
        self.assertEqual(next_day["recommended_exposure"], 0.6)

    def test_same_day_guard_does_not_delay_immediate_risk_reduction(self) -> None:
        previous_state = {
            "state_version": 1,
            "policy_version": POLICY_VERSION,
            "regime_model_version": HMM_MODEL_VERSION,
            "last_as_of": "2026-08-07",
            "exposure": 0.80,
            "raw_target": 0.95,
            "recovery_clear_streak": 3,
            "episode_peak_p_bear": 0.60,
            "episode_started_as_of": "2026-08-01",
        }
        result = evaluate_risk_policy(
            {
                "date": "2026-08-07",
                "model_version": HMM_MODEL_VERSION,
                "p_bull": 0.0,
                "p_sideways": 0.0,
                "p_bear": 1.0,
            },
            fast_overlay(),
            reentry_advisory(),
            as_of="2026-08-07",
            previous_state=previous_state,
        )

        self.assertEqual(result["raw_recommended_exposure"], 0.40)
        self.assertEqual(result["recommended_exposure"], 0.40)

    def test_risk_score_delta_uses_only_strictly_prior_normal_snapshot(self) -> None:
        previous = {
            "schema_version": 2,
            "as_of": "2026-08-06",
            "regime": {"model_version": HMM_MODEL_VERSION},
            "risk_policy": {
                "model_version": POLICY_VERSION,
                "decision_status": "OK",
                "risk_score": 40,
            },
        }
        rising = {
            "model_version": POLICY_VERSION,
            "decision_status": "OK",
            "risk_score": 55,
        }
        _add_risk_score_trend(rising, previous, "2026-08-07")
        self.assertEqual(rising["risk_score_delta"], 15)
        self.assertEqual(rising["risk_trend"], "UP")

        flat = {
            "model_version": POLICY_VERSION,
            "decision_status": "OK",
            "risk_score": 40,
        }
        _add_risk_score_trend(flat, previous, "2026-08-07")
        self.assertEqual(flat["risk_score_delta"], 0)
        self.assertEqual(flat["risk_trend"], "FLAT")

        same_day = {
            "model_version": POLICY_VERSION,
            "decision_status": "OK",
            "risk_score": 55,
        }
        _add_risk_score_trend(same_day, previous, "2026-08-06")
        self.assertIsNone(same_day["risk_score_delta"])
        self.assertIsNone(same_day["risk_trend"])

        legacy = {
            "schema_version": 1,
            "as_of": "2026-08-06",
            "risk_policy": {"decision_status": "OK", "market_health_score": 80},
        }
        no_cross_version_guess = {"decision_status": "OK", "risk_score": 20}
        _add_risk_score_trend(no_cross_version_guess, legacy, "2026-08-07")
        self.assertIsNone(no_cross_version_guess["risk_score_delta"])
        self.assertIsNone(no_cross_version_guess["risk_trend"])

        wrong_policy = {
            "schema_version": 2,
            "as_of": "2026-08-06",
            "regime": {"model_version": HMM_MODEL_VERSION},
            "risk_policy": {
                "model_version": "risk-policy-old",
                "decision_status": "OK",
                "risk_score": 40,
            },
        }
        no_policy_mix = {
            "model_version": POLICY_VERSION,
            "decision_status": "OK",
            "risk_score": 55,
        }
        _add_risk_score_trend(no_policy_mix, wrong_policy, "2026-08-07")
        self.assertIsNone(no_policy_mix["risk_score_delta"])
        self.assertIsNone(no_policy_mix["risk_trend"])

    def test_previous_score_fallback_is_used_for_same_day_snapshot(self) -> None:
        current = {"decision_status": "OK", "risk_score": 20}
        same_day = {
            "schema_version": 2,
            "as_of": "2026-08-07",
            "risk_policy": {"decision_status": "OK", "risk_score": 20},
        }
        _add_risk_score_trend(
            current,
            same_day,
            "2026-08-07",
            previous_score_fallback=35,
        )
        self.assertEqual(current["risk_score_delta"], -15)
        self.assertEqual(current["risk_trend"], "DOWN")

        wrong_model = {
            "schema_version": 2,
            "as_of": "2026-08-06",
            "regime": {"model_version": "legacy-spy-model"},
            "risk_policy": {"decision_status": "OK", "risk_score": 90},
        }
        current = {"decision_status": "OK", "risk_score": 20}
        _add_risk_score_trend(
            current,
            wrong_model,
            "2026-08-07",
            previous_score_fallback=35,
        )
        self.assertEqual(current["risk_score_delta"], -15)

    def test_previous_score_uses_router_score_not_hmm_posterior(self) -> None:
        with patch(
            "site_publisher.build_live_risk_score_map",
            return_value={"2026-08-05": 18, "2026-08-06": 37, "2026-08-07": 90},
        ):
            score = _previous_risk_score_from_database(
                Path("ignored.db"), "2026-08-07"
            )
        self.assertEqual(score, 37)

    def test_missing_remote_days_are_uploaded_in_parent_order(self) -> None:
        remote = {
            "schema_version": 3,
            "as_of": "2026-08-13",
            "history": {"series_version": POLICY_SERIES_VERSION},
        }
        first = {"as_of": "2026-08-14", "history": {"series_version": POLICY_SERIES_VERSION}}
        second = {"as_of": "2026-08-17", "history": {"series_version": POLICY_SERIES_VERSION}}
        with (
            patch("site_publisher.list_policy_decision_dates", return_value=["2026-08-14", "2026-08-17"]),
            patch("site_publisher.build_snapshot", side_effect=[first, second]) as build,
            patch("site_publisher.upload_snapshot") as upload,
        ):
            result = publish_missing_policy_history(
                Path("ignored.db"),
                latest_as_of="2026-08-17",
                previous_snapshot=remote,
                config={"site_url": "https://example.test", "upload_token": "token"},
            )
        self.assertEqual(result, second)
        self.assertEqual(build.call_args_list[0].kwargs["previous_snapshot"], remote)
        self.assertEqual(build.call_args_list[1].kwargs["previous_snapshot"], first)
        self.assertEqual(upload.call_count, 2)


if __name__ == "__main__":
    unittest.main()
