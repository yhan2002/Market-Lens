import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from roro_price_observations import (
    CONTRACT_VERSION,
    SOURCE,
    TABLE,
    ImmutablePriceObservationConflict,
    YahooChartError,
    download_and_persist,
    download_yahoo_observations,
    load_observation_series,
    map_observations_to_nq_sessions,
    persist_price_observations,
    refresh_from_publish_database,
)


class JsonResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def chart_response(
    *,
    ticker: str,
    timezone: str = "America/New_York",
    timestamps: list[int] | None = None,
    closes: list[float | None] | None = None,
    adjusted: list[float | None] | None = None,
    error: object = None,
) -> JsonResponse:
    timestamps = timestamps or [1767637800]
    closes = closes or [410.0 if ticker == "GLD" else 156.0]
    adjusted = adjusted or list(closes)
    payload = {
        "chart": {
            "error": error,
            "result": [
                {
                    "meta": {"exchangeTimezoneName": timezone},
                    "timestamp": timestamps,
                    "indicators": {
                        "quote": [{"close": closes}],
                        "adjclose": [{"adjclose": adjusted}],
                    },
                }
            ],
        }
    }
    return JsonResponse(json.dumps(payload).encode("utf-8"))


def observation(
    *,
    ticker: str = "JPY=X",
    raw_timestamp_utc: str = "2026-01-05T22:00:00Z",
    source_timezone: str = "America/New_York",
    observation_session_date: str = "2026-01-05",
    available_at: str = "2026-01-05T22:05:00Z",
    close: float = 156.0,
    adj_close: float | None = None,
    fetched_at: str = "2026-01-06T01:00:00Z",
) -> dict[str, object]:
    return {
        "contract_version": CONTRACT_VERSION,
        "source": SOURCE,
        "ticker": ticker,
        "raw_timestamp_utc": raw_timestamp_utc,
        "source_timezone": source_timezone,
        "observation_session_date": observation_session_date,
        "available_at": available_at,
        "close": close,
        "adj_close": adj_close,
        "fetched_at": fetched_at,
    }


class RoroPriceObservationsTest(unittest.TestCase):
    @patch("roro_price_observations.urlopen")
    def test_download_preserves_frozen_yahoo_session_fields(self, mocked_urlopen) -> None:
        mocked_urlopen.return_value = chart_response(ticker="GLD")

        rows = download_yahoo_observations(
            ticker="GLD",
            start="2026-01-05",
            end="2026-01-05",
            as_of="2026-01-06T00:00:00Z",
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ticker"], "GLD")
        self.assertEqual(rows[0]["source_timezone"], "America/New_York")
        self.assertEqual(rows[0]["observation_session_date"], "2026-01-05")
        self.assertEqual(rows[0]["raw_timestamp_utc"], "2026-01-05T18:30:00Z")
        self.assertEqual(rows[0]["available_at"], rows[0]["raw_timestamp_utc"])
        self.assertEqual(rows[0]["fetched_at"], "2026-01-06T00:00:00Z")
        self.assertEqual(rows[0]["close"], 410.0)
        request = mocked_urlopen.call_args.args[0]
        self.assertIn("GLD", request.full_url)
        self.assertEqual(mocked_urlopen.call_args.kwargs["timeout"], 20)

    @patch("roro_price_observations.urlopen")
    def test_two_ticker_download_is_atomic_and_exact_rerun_is_idempotent(
        self, mocked_urlopen
    ) -> None:
        mocked_urlopen.side_effect = [
            chart_response(ticker="GLD"),
            chart_response(ticker="JPY=X"),
            chart_response(ticker="GLD"),
            chart_response(ticker="JPY=X"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "observations.db"
            first = download_and_persist(
                database,
                start="2026-01-05",
                end="2026-01-05",
                as_of="2026-01-06T00:00:00Z",
            )
            second = download_and_persist(
                database,
                start="2026-01-05",
                end="2026-01-05",
                as_of="2026-01-06T00:00:00Z",
            )

        self.assertEqual(first["inserted"], 2)
        self.assertEqual(second["idempotent"], 2)

    @patch("roro_price_observations.urlopen")
    def test_changed_as_of_conflicts_instead_of_rewriting_history(
        self, mocked_urlopen
    ) -> None:
        mocked_urlopen.side_effect = [
            chart_response(ticker="GLD"),
            chart_response(ticker="JPY=X"),
            chart_response(ticker="GLD"),
            chart_response(ticker="JPY=X"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "observations.db"
            download_and_persist(
                database,
                start="2026-01-05",
                end="2026-01-05",
                as_of="2026-01-06T00:00:00Z",
            )
            with self.assertRaises(ImmutablePriceObservationConflict):
                download_and_persist(
                    database,
                    start="2026-01-05",
                    end="2026-01-05",
                    as_of="2026-01-07T00:00:00Z",
                )

    @patch("roro_price_observations.urlopen")
    def test_second_ticker_failure_creates_no_partial_database(
        self, mocked_urlopen
    ) -> None:
        mocked_urlopen.side_effect = [
            chart_response(ticker="GLD"),
            chart_response(ticker="JPY=X", error={"code": "Unavailable"}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "observations.db"
            with self.assertRaises(YahooChartError):
                download_and_persist(
                    database,
                    start="2026-01-05",
                    end="2026-01-05",
                    as_of="2026-01-06T00:00:00Z",
                )
            self.assertFalse(database.exists())

    @patch("roro_price_observations.urlopen")
    def test_chart_error_and_network_error_fail_closed(self, mocked_urlopen) -> None:
        mocked_urlopen.return_value = chart_response(
            ticker="GLD", error={"code": "Not Found"}
        )
        with self.assertRaisesRegex(YahooChartError, "chart error"):
            download_yahoo_observations(
                ticker="GLD",
                start="2026-01-05",
                end="2026-01-05",
                as_of="2026-01-06T00:00:00Z",
            )
        mocked_urlopen.side_effect = OSError("offline")
        with self.assertRaisesRegex(YahooChartError, "request failed"):
            download_yahoo_observations(
                ticker="GLD",
                start="2026-01-05",
                end="2026-01-05",
                as_of="2026-01-06T00:00:00Z",
            )

    @patch("roro_price_observations.urlopen")
    def test_timezone_nonpositive_and_malformed_arrays_fail_closed(
        self, mocked_urlopen
    ) -> None:
        cases = (
            chart_response(ticker="GLD", timezone="NOT/A_ZONE"),
            chart_response(ticker="GLD", closes=[0.0]),
            chart_response(ticker="GLD", closes=[410.0, 411.0]),
        )
        for response in cases:
            with self.subTest(response=response):
                mocked_urlopen.return_value = response
                with self.assertRaises(YahooChartError):
                    download_yahoo_observations(
                        ticker="GLD",
                        start="2026-01-05",
                        end="2026-01-05",
                        as_of="2026-01-06T00:00:00Z",
                    )

    @patch("roro_price_observations.urlopen")
    def test_null_close_rows_are_skipped_without_fill(self, mocked_urlopen) -> None:
        mocked_urlopen.return_value = chart_response(
            ticker="GLD",
            timestamps=[1767637800, 1767724200],
            closes=[None, 411.0],
            adjusted=[None, 411.0],
        )
        rows = download_yahoo_observations(
            ticker="GLD",
            start="2026-01-05",
            end="2026-01-06",
            as_of="2026-01-07T00:00:00Z",
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["observation_session_date"], "2026-01-06")
        self.assertEqual(rows[0]["close"], 411.0)

        mocked_urlopen.return_value = chart_response(ticker="GLD", closes=[None])
        with self.assertRaisesRegex(YahooChartError, "no in-range rows"):
            download_yahoo_observations(
                ticker="GLD",
                start="2026-01-05",
                end="2026-01-05",
                as_of="2026-01-06T00:00:00Z",
            )

    def test_download_range_and_as_of_are_explicit_and_ordered(self) -> None:
        with self.assertRaisesRegex(ValueError, "start"):
            download_yahoo_observations(
                ticker="GLD",
                start="2026-01-06",
                end="2026-01-05",
                as_of="2026-01-06T00:00:00Z",
            )
        with self.assertRaisesRegex(ValueError, "end"):
            download_yahoo_observations(
                ticker="GLD",
                start="2026-01-05",
                end="2026-01-07",
                as_of="2026-01-06T00:00:00Z",
            )
        with self.assertRaisesRegex(ValueError, "aware UTC"):
            download_yahoo_observations(
                ticker="GLD",
                start="2026-01-05",
                end="2026-01-05",
                as_of="2026-01-06",
            )

    def test_append_is_idempotent_and_changed_same_key_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "observations.db"
            record = observation()
            first = persist_price_observations(database, [record])
            second = persist_price_observations(database, [record])
            changed = {**record, "close": 157.0}
            with self.assertRaises(ImmutablePriceObservationConflict):
                persist_price_observations(database, [changed])

            with closing(sqlite3.connect(database)) as connection:
                stored = connection.execute(
                    f"SELECT contract_version,source,ticker,raw_timestamp_utc,"
                    f"source_timezone,observation_session_date,available_at,close,"
                    f"adj_close,fetched_at FROM {TABLE}"
                ).fetchone()
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(f"UPDATE {TABLE} SET close=999")
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(f"DELETE FROM {TABLE}")

        self.assertEqual(first["inserted"], 1)
        self.assertEqual(second["idempotent"], 1)
        self.assertEqual(stored[3], record["raw_timestamp_utc"])
        self.assertEqual(stored[4], record["source_timezone"])
        self.assertEqual(stored[7], record["close"])

    def test_missing_timezone_available_at_and_naive_timestamps_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "observations.db"
            missing_timezone = observation()
            del missing_timezone["source_timezone"]
            with self.assertRaisesRegex(ValueError, "source_timezone"):
                persist_price_observations(database, [missing_timezone])
            with self.assertRaisesRegex(ValueError, "IANA"):
                persist_price_observations(
                    database, [{**observation(), "source_timezone": "NOT/A_ZONE"}]
                )
            missing_available = observation()
            del missing_available["available_at"]
            with self.assertRaisesRegex(ValueError, "available_at"):
                persist_price_observations(database, [missing_available])
            with self.assertRaisesRegex(ValueError, "aware UTC"):
                persist_price_observations(
                    database,
                    [{**observation(), "raw_timestamp_utc": "2026-01-04T22:00:00"}],
                )
            self.assertFalse(database.exists())

    def test_inconsistent_session_and_timestamp_order_fail_closed(self) -> None:
        invalid_records = (
            (
                {**observation(), "observation_session_date": "2026-01-06"},
                "observation_session_date",
            ),
            (
                {**observation(), "available_at": "2026-01-05T21:59:59Z"},
                "available_at.*raw_timestamp_utc",
            ),
            (
                {**observation(), "fetched_at": "2026-01-05T22:04:59Z"},
                "fetched_at.*available_at",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "observations.db"
            for record, message in invalid_records:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        persist_price_observations(database, [record])
            self.assertFalse(database.exists())

    def test_contract_source_and_ticker_are_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "observations.db"
            for field, value in (
                ("contract_version", "future-contract"),
                ("source", "OTHER_SOURCE"),
                ("ticker", "GC=F"),
            ):
                with self.subTest(field=field):
                    with self.assertRaises(ValueError):
                        persist_price_observations(
                            database, [{**observation(), field: value}]
                        )
            self.assertFalse(database.exists())

    def test_values_apply_at_the_first_nq_close_plus_one_hour_cutoff(self) -> None:
        sessions = pd.DatetimeIndex(
            ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]
        )
        sunday_fx_for_monday = observation(
            raw_timestamp_utc="2026-01-04T22:00:00Z",
            source_timezone="Asia/Tokyo",
            observation_session_date="2026-01-05",
            available_at="2026-01-04T22:05:00Z",
        )
        same_day_gld = observation(
            ticker="GLD",
            raw_timestamp_utc="2026-01-05T21:00:00Z",
            observation_session_date="2026-01-05",
            available_at="2026-01-05T21:05:00Z",
            close=410.0,
        )

        fx = map_observations_to_nq_sessions([sunday_fx_for_monday], sessions)
        gold = map_observations_to_nq_sessions([same_day_gld], sessions)

        self.assertTrue(pd.isna(fx.loc["2026-01-02"]))
        self.assertEqual(fx.loc["2026-01-05"], 156.0)
        self.assertTrue(pd.isna(fx.loc["2026-01-06"]))
        self.assertTrue(pd.isna(fx.loc["2026-01-07"]))
        self.assertEqual(gold.loc["2026-01-05"], 410.0)
        self.assertTrue(pd.isna(gold.loc["2026-01-07"]))

    def test_holiday_collision_uses_latest_available_observation_without_fill(self) -> None:
        sessions = pd.DatetimeIndex(["2026-01-02", "2026-01-05", "2026-01-06"])
        saturday = observation(
            raw_timestamp_utc="2026-01-03T20:00:00Z",
            observation_session_date="2026-01-03",
            available_at="2026-01-03T20:05:00Z",
            close=155.0,
        )
        sunday = observation(
            raw_timestamp_utc="2026-01-04T20:00:00Z",
            observation_session_date="2026-01-04",
            available_at="2026-01-04T20:05:00Z",
            close=156.0,
        )

        mapped = map_observations_to_nq_sessions([sunday, saturday], sessions)

        self.assertTrue(pd.isna(mapped.loc["2026-01-02"]))
        self.assertEqual(mapped.loc["2026-01-05"], 156.0)
        self.assertTrue(pd.isna(mapped.loc["2026-01-06"]))

    def test_available_at_date_can_delay_the_effective_session(self) -> None:
        sessions = pd.DatetimeIndex(
            ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
        )
        late = observation(
            observation_session_date="2026-01-05",
            available_at="2026-01-07T00:01:00Z",
            fetched_at="2026-01-07T01:00:00Z",
        )

        mapped = map_observations_to_nq_sessions([late], sessions)

        self.assertTrue(mapped.loc[:"2026-01-06"].isna().all())
        self.assertEqual(mapped.loc["2026-01-07"], 156.0)

    def test_query_only_loader_reads_close_from_the_separate_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "observations.db"
            persist_price_observations(database, [observation()])
            before = database.read_bytes()
            values = load_observation_series(
                database,
                ticker="JPY=X",
                nq_sessions=pd.DatetimeIndex(
                    ["2026-01-05", "2026-01-06", "2026-01-07"]
                ),
            )
            after = database.read_bytes()

        self.assertEqual(before, after)
        self.assertEqual(values.loc["2026-01-06"], 156.0)
        self.assertTrue(pd.isna(values.loc["2026-01-07"]))

    @patch("roro_price_observations.download_yahoo_observations", return_value=[])
    def test_refresh_fetches_only_sessions_after_a_complete_seed(self, mocked_download) -> None:
        with tempfile.TemporaryDirectory() as directory:
            observation_database = Path(directory) / "observations.db"
            publish_database = Path(directory) / "publish.db"
            persist_price_observations(
                observation_database,
                [
                    observation(),
                    observation(ticker="GLD", close=410.0),
                ],
            )
            with closing(sqlite3.connect(publish_database)) as connection:
                connection.execute("CREATE TABLE daily_prices(ticker TEXT, date TEXT)")
                connection.execute("INSERT INTO daily_prices VALUES ('NQ=F','2026-01-06')")
                connection.commit()
            result = refresh_from_publish_database(
                observation_database,
                publish_database,
                as_of="2026-01-07T00:00:00Z",
            )

        self.assertEqual(result["latest_nq_date"], "2026-01-06")
        self.assertEqual(result["last_session_date"], "2026-01-05")
        self.assertEqual(mocked_download.call_count, 2)
        for call in mocked_download.call_args_list:
            self.assertEqual(call.kwargs["start"], "2026-01-06")
            self.assertEqual(call.kwargs["end"], "2026-01-06")

    @patch("roro_price_observations.download_yahoo_observations", return_value=[])
    def test_refresh_skips_when_current_and_fetches_only_lagging_ticker(self, mocked_download) -> None:
        with tempfile.TemporaryDirectory() as directory:
            observation_database = Path(directory) / "observations.db"
            publish_database = Path(directory) / "publish.db"
            persist_price_observations(
                observation_database,
                [
                    observation(),
                    observation(ticker="GLD", close=410.0),
                ],
            )
            with closing(sqlite3.connect(publish_database)) as connection:
                connection.execute("CREATE TABLE daily_prices(ticker TEXT, date TEXT)")
                connection.execute("INSERT INTO daily_prices VALUES ('NQ=F','2026-01-05')")
                connection.commit()
            result = refresh_from_publish_database(observation_database, publish_database)
            self.assertEqual(result["inserted"], 0)
            mocked_download.assert_not_called()
            with closing(sqlite3.connect(observation_database)) as connection:
                connection.execute(f"DROP TRIGGER {TABLE}_no_update")
                connection.execute(
                    f"UPDATE {TABLE} SET observation_session_date='2026-01-04' WHERE ticker='GLD'"
                )
                connection.commit()
            refresh_from_publish_database(observation_database, publish_database)
            mocked_download.assert_called_once()
            self.assertEqual(mocked_download.call_args.kwargs["ticker"], "GLD")
            self.assertEqual(mocked_download.call_args.kwargs["start"], "2026-01-05")
            self.assertEqual(mocked_download.call_args.kwargs["end"], "2026-01-05")


if __name__ == "__main__":
    unittest.main()
