import sqlite3
import os
import sys
import tempfile
import unittest
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from market_data import (
    CROSS_ASSET_TICKERS,
    EQUITY_UNIVERSE_GROUP,
    RESEARCH_UNIVERSE_GROUP,
    _download_cboe_index_rows,
    _configure_yahoo_ca_bundle,
    _configure_yahoo_cache,
    _download_cboe_vix3m_rows,
    _download_direct_chart_rows,
    _yahoo_end_date_kst,
    _download_start_dates,
    database_status,
    download_prices,
    initialize_database,
    import_move_close_csv,
    probe_latest_completed_market_date,
    read_universe_group,
    register_universe,
)


class MarketDataTest(unittest.TestCase):
    def test_move_csv_stores_reviewed_overrides_and_rejects_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "market.db"
            csv_path = root / "move.csv"
            register_universe(database, ("^MOVE",), group_name="test")
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("INSERT INTO daily_prices VALUES ('^MOVE','2026-07-20',NULL,NULL,NULL,72.66,72.66,NULL,'old')")
                connection.commit()
            csv_path.write_text('날짜,종가\n2026- 07- 20,72.66\n2026- 07- 21,74.67\n', encoding="utf-8-sig")
            inserted, verified = import_move_close_csv(database, csv_path)
            self.assertEqual((inserted, verified), (2, 0))
            with closing(sqlite3.connect(database)) as connection:
                self.assertEqual(connection.execute("SELECT close FROM daily_prices WHERE ticker='^MOVE' AND date='2026-07-20'").fetchone()[0], 72.66)
                self.assertEqual(connection.execute("SELECT close FROM reviewed_move_closes WHERE date='2026-07-21'").fetchone()[0], 74.67)
                self.assertIsNone(connection.execute("SELECT 1 FROM daily_prices WHERE ticker='^MOVE' AND date='2026-07-21'").fetchone())
            csv_path.write_text('날짜,종가\n2026-07-20,99\n', encoding="utf-8-sig")
            with self.assertRaisesRegex(ValueError, "reviewed value"):
                import_move_close_csv(database, csv_path)
    def test_register_does_not_recreate_legacy_private_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "router.db"
            register_universe(database, ("SPY",), group_name="REGIME_BENCHMARK")

            with closing(sqlite3.connect(database)) as connection:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }

            self.assertTrue({"metadata", "universe", "daily_prices"} <= tables)
            self.assertTrue({"portfolio_nav", "drawdown_events"}.isdisjoint(tables))

    def test_yahoo_end_date_waits_for_completed_us_session(self) -> None:
        self.assertEqual(
            _yahoo_end_date_kst(datetime(2026, 8, 6, 20, 59, tzinfo=timezone.utc)),
            date(2026, 8, 6),
        )
        self.assertEqual(
            _yahoo_end_date_kst(datetime(2026, 8, 6, 21, 0, tzinfo=timezone.utc)),
            date(2026, 8, 7),
        )
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            _yahoo_end_date_kst(datetime(2026, 8, 6, 21, 0))

    def test_download_is_sequential_and_pauses_between_small_batches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "router.db"
            tickers = tuple(f"T{index:02d}" for index in range(45))
            register_universe(database, tickers, group_name="TEST")
            calls: list[tuple[int, bool, str]] = []

            def fake_download(*, tickers, threads, end, **_kwargs):
                calls.append((len(tickers), threads, end))
                columns = pd.MultiIndex.from_product(
                    [tickers, ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
                )
                return pd.DataFrame(
                    [[100.0] * len(columns)],
                    index=[pd.Timestamp("2026-08-06")],
                    columns=columns,
                )

            with (
                patch.dict("sys.modules", {"yfinance": SimpleNamespace(download=fake_download)}),
                patch("market_data._download_direct_chart_batch", return_value=[]),
                patch("market_data.sleep") as pause,
            ):
                ticker_count, row_count, _ = download_prices(database, group_name="TEST")

            self.assertEqual((ticker_count, row_count), (45, 45))
            expected_end = _yahoo_end_date_kst().isoformat()
            self.assertEqual(
                calls,
                [
                    (10, False, expected_end),
                    (10, False, expected_end),
                    (10, False, expected_end),
                    (10, False, expected_end),
                    (5, False, expected_end),
                ],
            )
            self.assertEqual(pause.call_count, 4)

    def test_download_stops_after_first_fully_blocked_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "router.db"
            register_universe(
                database, tuple(f"T{index:02d}" for index in range(25)), group_name="TEST"
            )
            download = unittest.mock.Mock(return_value=pd.DataFrame())

            with (
                patch.dict(
                    "sys.modules", {"yfinance": SimpleNamespace(download=download)}
                ),
                patch("market_data._download_direct_chart_batch", return_value=[]),
            ):
                with self.assertRaisesRegex(RuntimeError, "Yahoo Finance returned no rows"):
                    download_prices(database, group_name="TEST")

            download.assert_called_once()
            self.assertFalse(download.call_args.kwargs["threads"])

    def test_complete_direct_chart_batches_skip_yfinance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "router.db"
            register_universe(
                database, tuple(f"T{index:02d}" for index in range(15)), group_name="TEST"
            )
            download = unittest.mock.Mock()

            def direct_rows(tickers, _start, _end, _fetched_at):
                return [
                    (ticker, "2026-08-06", 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, "test")
                    for ticker in tickers
                ]

            with (
                patch.dict(
                    "sys.modules", {"yfinance": SimpleNamespace(download=download)}
                ),
                patch("market_data._download_direct_chart_batch", side_effect=direct_rows) as direct,
                patch("market_data.sleep"),
            ):
                ticker_count, row_count, _ = download_prices(database, group_name="TEST")

            self.assertEqual((ticker_count, row_count), (15, 15))
            download.assert_not_called()
            self.assertEqual(direct.call_count, 2)

    def test_required_close_uses_yfinance_when_direct_chart_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "router.db"
            register_universe(database, ("AMD",), group_name="TEST")

            options: dict[str, object] = {}

            def fake_download(*, tickers, **kwargs):
                options.update(kwargs)
                columns = pd.MultiIndex.from_product(
                    [tickers, ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
                )
                return pd.DataFrame(
                    [[100.0] * len(columns)],
                    index=[pd.Timestamp("2026-08-17")],
                    columns=columns,
                )

            with (
                patch.dict("sys.modules", {"yfinance": SimpleNamespace(download=fake_download)}),
                patch(
                    "market_data._download_direct_chart_batch",
                    return_value=[("AMD", "2026-08-14", 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, "test")],
                ),
            ):
                download_prices(
                    database,
                    tickers=("AMD",),
                    require_date=date(2026, 8, 17),
                )

            with closing(sqlite3.connect(database)) as connection:
                close = connection.execute(
                    "SELECT close FROM daily_prices WHERE ticker='AMD' AND date='2026-08-17'"
                ).fetchone()
            self.assertEqual(close, (100.0,))
            self.assertEqual(options["start"], "2026-08-17")
            self.assertEqual(options["end"], "2026-08-18")

    def test_yahoo_uses_ascii_runtime_certificate_bundle(self) -> None:
        import certifi

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "Lib" / "site-packages" / "pip" / "_vendor" / "certifi" / "cacert.pem"
            bundle.parent.mkdir(parents=True)
            bundle.write_text("certificate", encoding="ascii")
            previous_where = certifi.where
            try:
                with patch("market_data.sys.base_prefix", str(root)):
                    _configure_yahoo_ca_bundle()
                self.assertEqual(os.environ["CURL_CA_BUNDLE"], str(bundle))
                self.assertEqual(certifi.where(), str(bundle))
            finally:
                certifi.where = previous_where

    def test_yahoo_uses_local_writable_cache(self) -> None:
        calls: list[str] = []
        _configure_yahoo_cache(
            SimpleNamespace(set_tz_cache_location=lambda path: calls.append(path))
        )
        self.assertEqual(calls, [str(Path(__file__).resolve().parent / ".yfinance-cache")])

    def test_direct_chart_parser_builds_rows(self) -> None:
        payload = {
            "chart": {
                "result": [
                    {
                        "timestamp": [1786032000],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [100.0],
                                    "high": [102.0],
                                    "low": [99.0],
                                    "close": [101.0],
                                    "volume": [1234],
                                }
                            ],
                            "adjclose": [{"adjclose": [100.5]}],
                        },
                    }
                ]
            }
        }
        response = unittest.mock.MagicMock()
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        with (
            patch("market_data.urlopen", return_value=response),
            patch("market_data.json.load", return_value=payload),
        ):
            rows = _download_direct_chart_rows(
                "AAPL", date(2026, 8, 1), date(2026, 8, 7), "test"
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "AAPL")
        self.assertEqual(rows[0][6], 100.5)

    def test_market_probe_excludes_today_until_new_york_17(self) -> None:
        rows = [
            ("ticker", "2026-08-12", None, None, None, 1, 1, None, "test"),
            ("ticker", "2026-08-13", None, None, None, 1, 1, None, "test"),
        ]
        with patch("market_data._download_direct_chart_rows", return_value=rows):
            before = probe_latest_completed_market_date(
                now=datetime(2026, 8, 13, 20, 59, tzinfo=timezone.utc)
            )
            at_cutoff = probe_latest_completed_market_date(
                now=datetime(2026, 8, 13, 21, 0, tzinfo=timezone.utc)
            )

        self.assertEqual(before, date(2026, 8, 12))
        self.assertEqual(at_cutoff, date(2026, 8, 13))

    def test_market_probe_rejects_ticker_date_mismatch_and_missing_rows(self) -> None:
        def mismatched(ticker, *_args):
            observed = "2026-08-12" if ticker == "SPY" else "2026-08-11"
            return [(ticker, observed, None, None, None, 1, 1, None, "test")]

        instant = datetime(2026, 8, 13, 20, 0, tzinfo=timezone.utc)
        with patch("market_data._download_direct_chart_rows", side_effect=mismatched):
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                probe_latest_completed_market_date(now=instant)
        with patch("market_data._download_direct_chart_rows", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "probe failed"):
                probe_latest_completed_market_date(now=instant)

    def test_market_probe_rejects_unverified_database_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "market.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    "CREATE TABLE daily_prices ("
                    "ticker TEXT NOT NULL, date TEXT NOT NULL, close REAL, adj_close REAL, "
                    "fetched_at TEXT NOT NULL, PRIMARY KEY (ticker,date))"
                )
                connection.executemany(
                    "INSERT INTO daily_prices "
                    "(ticker,date,close,adj_close,fetched_at) VALUES (?,?,?,?,?)",
                    [
                        ("SPY", "2026-08-17", 772.67, 772.67, "manual"),
                        ("NQ=F", "2026-08-17", 30096.0, 30096.0, "test"),
                    ],
                )
                connection.commit()

            def mismatched(ticker, *_args):
                if ticker == "SPY":
                    return []
                observed = "2026-08-17"
                return [(ticker, observed, None, None, None, 1, 1, None, "test")]

            with patch("market_data._download_direct_chart_rows", side_effect=mismatched), self.assertRaisesRegex(RuntimeError, "최신 거래일 확인 불가"):
                probe_latest_completed_market_date(
                    now=datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc),
                    fallback_database=database,
                )


    def test_cboe_vix3m_parser_filters_requested_dates(self) -> None:
        payload = (
            "DATE,OPEN,HIGH,LOW,CLOSE\n"
            "08/07/2026,18.50,19.00,18.40,18.72\n"
            "08/10/2026,18.80,19.10,18.70,18.98\n"
            "08/11/2026,18.90,19.05,18.75,18.91\n"
        ).encode()
        response = unittest.mock.MagicMock()
        response.read.return_value = payload
        response.__enter__.return_value = response
        response.__exit__.return_value = False

        with patch("market_data.urlopen", return_value=response):
            rows = _download_cboe_vix3m_rows(
                date(2026, 8, 10), date(2026, 8, 12), "test"
            )

        self.assertEqual([row[1] for row in rows], ["2026-08-10", "2026-08-11"])
        self.assertEqual(rows[-1][0], "^VIX3M")
        self.assertEqual(rows[-1][5:7], (18.91, 18.91))
        self.assertIsNone(rows[-1][7])

    def test_cboe_vix_parser_uses_official_vix_history(self) -> None:
        payload = b"DATE,OPEN,HIGH,LOW,CLOSE\n08/17/2026,16.0,17.0,15.0,16.5\n"
        response = unittest.mock.MagicMock()
        response.read.return_value = payload
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        with patch("market_data.urlopen", return_value=response):
            rows = _download_cboe_index_rows("^VIX", date(2026, 8, 17), date(2026, 8, 18), "test")
        self.assertEqual(rows[0][0], "^VIX")
        self.assertEqual(rows[0][1], "2026-08-17")

    def test_literal_universe_and_database_init(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.py"
            config.write_text(
                "universe_groups = {\n"
                "    'CSQ_MP': ['META', 'NVDA', 'META'],\n"
                "    'CSQ_CP': ['AMD', 'META'],\n"
                "}\n"
                "raise RuntimeError('must not execute')\n",
                encoding="utf-8-sig",
            )
            database = root / "router.db"

            self.assertEqual(read_universe_group(config, "CSQ_MP"), ["META", "NVDA"])
            self.assertEqual(
                read_universe_group(config, EQUITY_UNIVERSE_GROUP),
                ["AMD", "META", "NVDA"],
            )
            self.assertEqual(initialize_database(database, config), (3, 2))
            self.assertEqual(dict(database_status(database))["active_tickers"], 5)
            with closing(sqlite3.connect(database)) as connection:
                groups = dict(connection.execute("SELECT ticker, group_name FROM universe"))
            self.assertEqual(groups["META"], EQUITY_UNIVERSE_GROUP)
            self.assertEqual(groups["SPY"], "REGIME_BENCHMARK")

    def test_research_group_combines_mp_cp_and_univ(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.py"
            config.write_text(
                "universe_groups = {\n"
                "    'CSQ_MP': ['META', 'NVDA'],\n"
                "    'CSQ_CP': ['AMD', 'META'],\n"
                "    'CSQ_UNIV': ['AMD', 'AAPL'],\n"
                "}\n",
                encoding="utf-8",
            )
            database = root / "router.db"

            self.assertEqual(
                read_universe_group(config, RESEARCH_UNIVERSE_GROUP),
                ["AAPL", "AMD", "META", "NVDA"],
            )
            initialize_database(database, config, benchmarks=())
            self.assertEqual(
                initialize_database(
                    database,
                    config,
                    group_name=RESEARCH_UNIVERSE_GROUP,
                    benchmarks=(),
                ),
                (4, 0),
            )
            with closing(sqlite3.connect(database)) as connection:
                operating = {
                    ticker
                    for (ticker,) in connection.execute(
                        "SELECT ticker FROM universe WHERE group_name=? AND active=1",
                        (EQUITY_UNIVERSE_GROUP,),
                    )
                }
            self.assertEqual(operating, {"AMD", "META", "NVDA"})

    def test_default_composite_group_has_deduplicated_scoped_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mp = [f"MP{index}" for index in range(18)]
            cp = mp[:13] + [f"CP{index}" for index in range(67)]
            config = root / "config.py"
            config.write_text(
                f"universe_groups = {{'CSQ_MP': {mp!r}, 'CSQ_CP': {cp!r}}}\n",
                encoding="utf-8",
            )
            database = root / "router.db"

            self.assertEqual(initialize_database(database, config, benchmarks=()), (85, 0))
            starts = _download_start_dates(
                database,
                years=2,
                today=date(2026, 8, 5),
                group_name=EQUITY_UNIVERSE_GROUP,
            )
            self.assertEqual(len(starts), 85)
            self.assertEqual(set(starts), set(mp) | set(cp))

    def test_literal_reader_never_executes_config_expression(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "executed.txt"
            config = root / "config.py"
            config.write_text(
                "universe_groups = __import__('pathlib').Path("
                f"{str(marker)!r}).write_text('executed')\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                read_universe_group(config, EQUITY_UNIVERSE_GROUP)
            self.assertFalse(marker.exists())

    def test_explicit_single_group_remains_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.py"
            config.write_text(
                "universe_groups = {'CSQ_MP': ['META'], 'CSQ_CP': ['AMD']}\n",
                encoding="utf-8",
            )
            database = root / "router.db"

            self.assertEqual(
                initialize_database(
                    database, config, group_name="CSQ_MP", benchmarks=()
                ),
                (1, 0),
            )
            self.assertEqual(
                set(
                    _download_start_dates(
                        database,
                        years=2,
                        today=date(2026, 8, 5),
                        group_name="CSQ_MP",
                    )
                ),
                {"META"},
            )

    def test_download_start_is_calculated_per_ticker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.py"
            config.write_text(
                "universe_groups = {'CSQ_MP': ['NEW', 'OLD']}\n", encoding="utf-8"
            )
            database = root / "router.db"
            initialize_database(database, config, group_name="CSQ_MP", benchmarks=())
            today = date(2026, 8, 5)
            old_latest = today - timedelta(days=1)
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    """
                    INSERT INTO daily_prices(ticker, date, close, adj_close, fetched_at)
                    VALUES ('OLD', ?, 100, 100, 'test')
                    """,
                    (old_latest.isoformat(),),
                )
                connection.commit()

            starts = _download_start_dates(database, years=10, today=today)
            self.assertEqual(starts["OLD"], old_latest - timedelta(days=7))
            self.assertEqual(
                starts["NEW"], today - timedelta(days=round(10 * 365.25))
            )
            backfill = _download_start_dates(
                database, years=10, today=today, tickers=("OLD",), backfill=True
            )
            self.assertEqual(
                backfill["OLD"], today - timedelta(days=round(10 * 365.25))
            )

    def test_cross_asset_registration_and_scoped_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "router.db"
            self.assertEqual(
                register_universe(database, CROSS_ASSET_TICKERS), len(CROSS_ASSET_TICKERS)
            )
            starts = _download_start_dates(
                database,
                years=20,
                today=date(2026, 8, 5),
                group_name="REGIME_CROSS_ASSET",
            )
            self.assertEqual(set(starts), set(CROSS_ASSET_TICKERS))
            self.assertEqual(
                set(
                    _download_start_dates(
                        database,
                        years=20,
                        today=date(2026, 8, 5),
                        tickers=("HYG",),
                    )
                ),
                {"HYG"},
            )

            register_universe(database, ("HYG",))
            with closing(sqlite3.connect(database)) as connection:
                active = {
                    ticker
                    for (ticker,) in connection.execute(
                        "SELECT ticker FROM universe WHERE group_name='REGIME_CROSS_ASSET' AND active=1"
                    )
                }
            self.assertEqual(active, {"HYG"})


if __name__ == "__main__":
    unittest.main()
