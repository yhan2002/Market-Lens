import sqlite3
import tempfile
import unittest
import zipfile
from contextlib import closing
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, parse_qsl, urlparse
from unittest.mock import patch

from macro_data import (
    FRED_SERIES,
    TURNING_MACRO_BACKFILL_SERIES,
    TURNING_MACRO_LIVE_SERIES,
    backfill_cash_rate_alfred,
    backfill_hy_oas_alfred,
    backfill_hy_oas_vintages,
    backfill_turning_macro_alfred,
    download_alfred_vintages,
    fetch_cash_rate,
    fetch_hy_oas,
    import_hy_oas_csv,
    import_hy_oas_revised_history_csv,
    import_usdjpy_revised_history_csv,
    fetch_usdjpy_revised_history,
    latest_as_of,
    parse_fred_csv,
    parse_fred_vintage_json,
    parse_alfred_vintage_zip,
    refresh_turning_macro_inputs,
    series_as_of,
    upsert_observations,
)


class _Response:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.closed = False

    def read(self) -> bytes:
        return self.body

    def close(self) -> None:
        self.closed = True


class _TimeoutResponse(_Response):
    def read(self) -> bytes:
        raise TimeoutError("slow ALFRED response")


class MacroDataTests(unittest.TestCase):
    def test_turning_macro_refresh_is_atomic_and_idempotent(self) -> None:
        requested = []

        def opener(request, *, timeout):
            series_id = parse_qs(urlparse(request.full_url).query)["id"][0]
            requested.append(series_id)
            return _Response(
                f"DATE,{series_id}\n2026-08-29,1.25\n".encode("utf-8")
            )

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "risk.db"
            first, fallback = refresh_turning_macro_inputs(
                database,
                "2026-09-02",
                opener=opener,
                available_at="2026-09-02T09:00:00Z",
            )
            repeated, repeated_fallback = refresh_turning_macro_inputs(
                database,
                "2026-09-02",
                opener=opener,
                available_at="2026-09-03T09:00:00Z",
            )
            with closing(sqlite3.connect(database)) as connection:
                stored = connection.execute(
                    "SELECT COUNT(DISTINCT series) FROM macro_observations"
                ).fetchone()[0]

        self.assertEqual(set(requested), {FRED_SERIES[s] for s in TURNING_MACRO_LIVE_SERIES})
        self.assertEqual(first, {series: 1 for series in TURNING_MACRO_LIVE_SERIES})
        self.assertEqual(repeated, {series: 0 for series in TURNING_MACRO_LIVE_SERIES})
        self.assertFalse(fallback or repeated_fallback)
        self.assertEqual(stored, len(TURNING_MACRO_LIVE_SERIES))

    def test_turning_macro_parse_failure_writes_nothing(self) -> None:
        def opener(request, *, timeout):
            series_id = parse_qs(urlparse(request.full_url).query)["id"][0]
            if series_id == FRED_SERIES["UST_10Y"]:
                return _Response(b"DATE,WRONG\n2026-08-29,1.25\n")
            return _Response(
                f"DATE,{series_id}\n2026-08-29,1.25\n".encode("utf-8")
            )

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "risk.db"
            with self.assertRaisesRegex(ValueError, "coverage is stale"):
                refresh_turning_macro_inputs(
                    database, "2026-09-02", opener=opener
                )
            self.assertFalse(database.exists())

    def test_turning_macro_backfill_reuses_annual_alfred_path(self) -> None:
        with patch("macro_data._backfill_alfred_series", return_value=2) as backfill:
            counts = backfill_turning_macro_alfred(
                Path("risk.db"), "2008-06-01", "2026-09-02"
            )
        self.assertEqual(counts, {series: 2 for series in TURNING_MACRO_BACKFILL_SERIES})
        self.assertEqual(backfill.call_count, len(TURNING_MACRO_BACKFILL_SERIES))

    def test_import_usdjpy_revised_history_uses_fred_series(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "risk.db"
            source = Path(directory) / "dexjpus.csv"
            source.write_text(
                "observation_date,DEXJPUS\n2026-08-03,160.0\n2026-08-04,.\n2026-08-05,158.0\n",
                encoding="utf-8",
            )
            count = import_usdjpy_revised_history_csv(database, source)
            rows = series_as_of(database, "USDJPY_FRED", "2026-08-05")
        self.assertEqual(count, 2)
        self.assertEqual([(row[0], row[1], row[3]) for row in rows], [
            ("2026-08-05", 158.0, "FRED_REVISED_HISTORY:DEXJPUS"),
            ("2026-08-03", 160.0, "FRED_REVISED_HISTORY:DEXJPUS"),
        ])

    def test_fetch_usdjpy_uses_bounded_fred_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "risk.db"
            requested = []
            def opener(request, *, timeout):
                requested.append(request.full_url)
                return _Response(b"observation_date,DEXJPUS\n2026-08-10,158.0\n")
            count = fetch_usdjpy_revised_history(
                database, "2026-08-10", "2026-08-10", opener=opener
            )
        self.assertEqual(count, 1)
        self.assertIn("id=DEXJPUS", requested[0])
    @staticmethod
    def _alfred_zip(csv_text: str) -> bytes:
        payload = BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr("README.txt", "official ALFRED test fixture")
            archive.writestr("vintages_starting_2026-07-01.csv", csv_text)
        return payload.getvalue()

    def test_alfred_backfill_restores_point_in_time_history_without_api_key(self) -> None:
        page = b"""
        <select id="form_selected_vintage_dates">
          <option value="2026-06-30">2026-06-30</option>
          <option value="2026-07-01">2026-07-01</option>
          <option value="2026-07-02">2026-07-02</option>
        </select>
        """
        archive = self._alfred_zip(
            "observation_date,BAMLH0A0HYM2_20260630,"
            "BAMLH0A0HYM2_20260701,BAMLH0A0HYM2_20260702\n"
            "2026-06-30,2.75,,\n"
            "2026-07-01,,2.74,\n"
            "2026-07-01,,,2.76\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "risk.db"
            calls = []

            def opener(request, *, timeout):
                calls.append((request, timeout))
                return _Response(page if request.data is None else archive)

            first = backfill_hy_oas_alfred(
                database, "2026-06-30", "2026-07-02", opener=opener
            )
            repeated = backfill_hy_oas_alfred(
                database, "2026-06-30", "2026-07-02", opener=opener
            )
            july_first = series_as_of(database, "HY_OAS", "2026-07-01")
            july_second = series_as_of(database, "HY_OAS", "2026-07-02")

        self.assertEqual((first, repeated), (3, 0))
        self.assertEqual(july_first[0][0:2], ("2026-07-01", 2.74))
        self.assertEqual(july_second[0][0:2], ("2026-07-01", 2.76))
        posted = parse_qsl(calls[1][0].data.decode("utf-8"))
        self.assertIn(("form[file_type]", "3"), posted)
        self.assertEqual(
            [value for key, value in posted if key == "form[selected_vintage_dates][]"],
            ["2026-06-30", "2026-07-01", "2026-07-02"],
        )

    def test_alfred_download_retries_a_read_timeout(self) -> None:
        page = b"""
        <select id="form_selected_vintage_dates">
          <option value="2026-01-02">2026-01-02</option>
        </select>
        """
        archive = self._alfred_zip(
            "observation_date,DGS3MO_20260102\n2026-01-01,3.65\n"
        )
        timed_out = _TimeoutResponse(b"")
        responses = iter((timed_out, _Response(page), _Response(archive)))

        with patch("macro_data.sleep") as sleeper:
            payload = download_alfred_vintages(
                "DGS3MO", "2026-01-01", "2026-01-02", opener=lambda *_args, **_kwargs: next(responses)
            )

        self.assertEqual(payload, archive)
        self.assertTrue(timed_out.closed)
        sleeper.assert_called_once_with(1.0)

    def test_alfred_download_falls_back_after_post_timeout(self) -> None:
        page = b"""
        <select id="form_selected_vintage_dates">
          <option value="2026-01-02">2026-01-02</option>
        </select>
        """
        fallback = b"observation_date,DGS3MO_20260102\n2026-01-01,3.65\n"
        calls = []

        def opener(request, *, timeout):
            calls.append(request)
            if request.data is not None:
                raise TimeoutError("slow ALFRED POST")
            if "alfredgraph.csv" in request.full_url:
                return _Response(fallback)
            return _Response(page)

        with patch("macro_data.sleep") as sleeper:
            payload = download_alfred_vintages(
                "DGS3MO", "2026-01-01", "2026-01-02", opener=opener
            )

        self.assertEqual(payload, fallback)
        self.assertEqual(sum(request.data is not None for request in calls), 1)
        self.assertEqual(sum("alfredgraph.csv" in request.full_url for request in calls), 1)
        sleeper.assert_not_called()

    def test_alfred_download_splits_a_timed_out_batch(self) -> None:
        days = [f"2026-01-0{day}" for day in range(1, 5)]
        page = (
            '<select id="form_selected_vintage_dates">'
            + "".join(f'<option value="{day}">{day}</option>' for day in days)
            + "</select>"
        ).encode()
        posts = []

        def opener(request, *, timeout):
            if request.data is None:
                return _Response(page)
            selected = parse_qs(request.data.decode("utf-8"))[
                "form[selected_vintage_dates][]"
            ]
            posts.append(selected)
            if len(selected) > 2:
                return _TimeoutResponse(b"")
            header = "observation_date," + ",".join(
                f"DGS3MO_{day.replace('-', '')}" for day in selected
            )
            rows = [
                f"{day}," + ",".join(
                    "3.65" if day == selected_day else ""
                    for selected_day in selected
                )
                for day in selected
            ]
            return _Response(self._alfred_zip(header + "\n" + "\n".join(rows) + "\n"))

        with patch("macro_data.ALFRED_VINTAGE_BATCH_SIZE", 4):
            payload = download_alfred_vintages(
                "DGS3MO", days[0], days[-1], opener=opener
            )

        self.assertEqual(posts, [days, days[:2], days[2:]])
        self.assertEqual(len(parse_alfred_vintage_zip(payload, "DGS3MO")), 4)

    def test_alfred_download_does_not_retry_permanent_http_error(self) -> None:
        page = b"""
        <select id="form_selected_vintage_dates">
          <option value="2026-01-02">2026-01-02</option>
        </select>
        """
        calls = []

        def opener(request, *, timeout):
            calls.append((request, timeout))
            if request.data is None:
                return _Response(page)
            raise HTTPError(request.full_url, 404, "Not Found", None, None)

        with patch("macro_data.sleep") as sleeper:
            with self.assertRaises(HTTPError):
                download_alfred_vintages(
                    "DGS3MO", "2026-01-01", "2026-01-02", opener=opener
                )

        self.assertEqual(len(calls), 2)
        sleeper.assert_not_called()

    def test_alfred_download_batches_vintages_without_narrowing_observations(self) -> None:
        page = b"""
        <select id="form_selected_vintage_dates">
          <option value="2026-01-01">2026-01-01</option>
          <option value="2026-01-02">2026-01-02</option>
          <option value="2026-01-03">2026-01-03</option>
        </select>
        """
        posts = []

        def opener(request, *, timeout):
            if request.data is None:
                return _Response(page)
            form = parse_qs(request.data.decode("utf-8"))
            selected = form["form[selected_vintage_dates][]"]
            posts.append((selected, form["form[obs_start_date]"][0]))
            header = "observation_date," + ",".join(
                f"DGS3MO_{day.replace('-', '')}" for day in selected
            )
            rows = [
                f"{day}," + ",".join("3.65" if day == selected_day else "" for selected_day in selected)
                for day in selected
            ]
            return _Response(self._alfred_zip(header + "\n" + "\n".join(rows) + "\n"))

        with patch("macro_data.ALFRED_VINTAGE_BATCH_SIZE", 2):
            payload = download_alfred_vintages(
                "DGS3MO",
                "2026-01-01",
                "2026-01-03",
                opener=opener,
                observation_start="2025-12-01",
            )

        self.assertEqual([selected for selected, _ in posts], [
            ["2026-01-01", "2026-01-02"],
            ["2026-01-03"],
        ])
        self.assertEqual([start for _, start in posts], ["2025-12-01", "2025-12-01"])
        self.assertEqual(len(parse_alfred_vintage_zip(payload, "DGS3MO")), 3)

    def test_cash_rate_uses_dgs3mo_for_live_and_alfred_history(self) -> None:
        page = b"""
        <select id="form_selected_vintage_dates">
          <option value="2026-01-02">2026-01-02</option>
        </select>
        """
        archive = self._alfred_zip(
            "observation_date,DGS3MO_20260102\n2026-01-01,3.65\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "risk.db"
            calls = []

            def alfred_opener(request, *, timeout):
                calls.append(request)
                return _Response(page if request.data is None else archive)

            historical = backfill_cash_rate_alfred(
                database, "2026-01-01", "2026-01-02", opener=alfred_opener
            )

            def fred_opener(request, *, timeout):
                calls.append(request)
                return _Response(b"DATE,DGS3MO\n2026-01-05,3.70\n")

            live = fetch_cash_rate(
                database,
                "2026-01-05",
                "2026-01-05",
                opener=fred_opener,
                available_at="2026-01-06T00:00:00Z",
            )
            with closing(sqlite3.connect(database)) as connection:
                rows = connection.execute(
                    "SELECT series,date,value,available_at,source "
                    "FROM macro_observations ORDER BY date"
                ).fetchall()

        self.assertEqual((historical, live), (1, 1))
        self.assertEqual(rows[0], (
            "CASH_3M", "2026-01-01", 3.65,
            "2026-01-02", "ALFRED:DGS3MO:OUTPUT3"
        ))
        self.assertEqual(rows[1], (
            "CASH_3M", "2026-01-05", 3.70,
            "2026-01-06T00:00:00+00:00", "FRED:DGS3MO"
        ))

    def test_alfred_backfill_splits_long_requests_by_calendar_year(self) -> None:
        page = b"""
        <select id="form_selected_vintage_dates">
          <option value="2025-12-31">2025-12-31</option>
          <option value="2026-01-02">2026-01-02</option>
        </select>
        """
        archives = {
            "2025-12-31": self._alfred_zip(
                "observation_date,DGS3MO_20251231\n2025-12-30,3.60\n"
            ),
            "2026-01-02": self._alfred_zip(
                "observation_date,DGS3MO_20260102\n"
                "2025-12-30,3.61\n2026-01-01,3.65\n"
            ),
        }
        posts = []

        def opener(request, *, timeout):
            if request.data is None:
                return _Response(page)
            form = parse_qs(request.data.decode("utf-8"))
            selected = form["form[selected_vintage_dates][]"]
            posts.append((selected, form["form[obs_start_date]"][0]))
            return _Response(archives[selected[0]])

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "risk.db"
            count = backfill_cash_rate_alfred(
                database,
                "2025-12-30",
                "2026-01-02",
                opener=opener,
            )
            latest = series_as_of(database, "CASH_3M", "2026-01-02")

        self.assertEqual(count, 3)
        self.assertEqual(posts, [
            (["2025-12-31"], "2025-12-30"),
            (["2026-01-02"], "2025-12-30"),
        ])
        self.assertEqual([(row[0], row[1]) for row in latest], [
            ("2026-01-01", 3.65),
            ("2025-12-30", 3.61),
        ])

    def test_alfred_parser_fails_closed_for_html_or_future_availability(self) -> None:
        with self.assertRaisesRegex(ValueError, "ZIP archive"):
            parse_alfred_vintage_zip(b"<html>error</html>", "BAMLH0A0HYM2")
        archive = self._alfred_zip(
            "observation_date,BAMLH0A0HYM2_20260701\n"
            "2026-07-02,2.74\n"
        )
        with self.assertRaisesRegex(ValueError, "must not precede"):
            parse_alfred_vintage_zip(archive, "BAMLH0A0HYM2")

    def test_fred_vintage_backfill_restores_point_in_time_history(self) -> None:
        body = (
            b'{"count":3,"observations":['
            b'{"realtime_start":"2026-07-01","date":"2026-06-30","value":"2.75"},'
            b'{"realtime_start":"2026-07-02","date":"2026-07-01","value":"2.74"},'
            b'{"realtime_start":"2026-07-05","date":"2026-07-01","value":"2.76"}'
            b']}'
        )
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "risk.db"
            calls = []

            def opener(request, *, timeout):
                calls.append((request, timeout))
                return _Response(body)

            first = backfill_hy_oas_vintages(
                database,
                "2026-06-30",
                "2026-07-01",
                api_key="test-key",
                realtime_end="2026-07-05",
                opener=opener,
            )
            repeated = backfill_hy_oas_vintages(
                database,
                "2026-06-30",
                "2026-07-01",
                api_key="test-key",
                realtime_end="2026-07-05",
                opener=opener,
            )
            july_first = series_as_of(database, "HY_OAS", "2026-07-01")
            july_second = series_as_of(database, "HY_OAS", "2026-07-02")
            revised = series_as_of(database, "HY_OAS", "2026-07-05")

        self.assertEqual((first, repeated), (3, 0))
        self.assertEqual(july_first[0][0:2], ("2026-06-30", 2.75))
        self.assertEqual(july_second[0][0:2], ("2026-07-01", 2.74))
        self.assertEqual(revised[0][0:2], ("2026-07-01", 2.76))
        query = parse_qs(urlparse(calls[0][0].full_url).query)
        self.assertEqual(query["output_type"], ["3"])
        self.assertEqual(query["observation_start"], ["2026-06-30"])
        self.assertEqual(query["observation_end"], ["2026-07-01"])
        self.assertEqual(query["realtime_end"], ["2026-07-05"])

    def test_fred_vintage_input_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "FRED_API_KEY"):
            backfill_hy_oas_vintages(
                Path("unused.db"),
                "2026-07-01",
                "2026-07-01",
                api_key="",
                opener=lambda *_args, **_kwargs: self.fail("network should not run"),
            )
        with self.assertRaisesRegex(ValueError, "truncated"):
            parse_fred_vintage_json(
                '{"count":2,"observations":['
                '{"realtime_start":"2026-07-02","date":"2026-07-01","value":"2.74"}]}'
                ,
                "BAMLH0A0HYM2",
            )

    def test_licensed_history_requires_explicit_availability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "risk.db"
            csv_path = root / "hy_oas.csv"
            csv_path.write_text(
                "date,value,available_at\n"
                "2008-01-02,5.25,2008-01-03\n"
                "2008-01-03,5.40,2008-01-04T08:00:00Z\n",
                encoding="utf-8",
            )

            count = import_hy_oas_csv(database, csv_path, source="licensed-test")
            before = latest_as_of(database, "HY_OAS", "2008-01-02")
            after = series_as_of(database, "HY_OAS", "2008-01-04")

            self.assertEqual(count, 2)
            self.assertIsNone(before)
            self.assertEqual([row[1] for row in after], [5.4, 5.25])

    def test_revised_history_uses_observation_date_for_score_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "risk.db"
            csv_path = root / "fred.csv"
            csv_path.write_text(
                "observation_date,BAMLH0A0HYM2\n"
                "2023-08-14,3.79\n"
                "2023-08-15,.\n"
                "2023-08-16,3.85\n",
                encoding="utf-8",
            )

            count = import_hy_oas_revised_history_csv(database, csv_path)
            rows = series_as_of(database, "HY_OAS", "2023-08-16")

            self.assertEqual(count, 2)
            self.assertEqual([row[1] for row in rows], [3.85, 3.79])
            self.assertEqual({row[3] for row in rows}, {"FRED_REVISED_HISTORY:BAMLH0A0HYM2"})

    def test_parser_omits_blank_and_dot_but_rejects_non_finite_values(self) -> None:
        body = (
            b"observation_date,BAMLH0A0HYM2\n"
            b"2026-08-03,3.50\n"
            b"2026-08-04,.\n"
            b"2026-08-05,\n"
        )
        self.assertEqual(
            parse_fred_csv(body, "BAMLH0A0HYM2"),
            [("2026-08-03", 3.5)],
        )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            parse_fred_csv(
                "DATE,BAMLH0A0HYM2\n2026-08-03,nan\n", "BAMLH0A0HYM2"
            )

    def test_fetch_uses_bounded_official_csv_and_stores_native_units(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "risk.db"
            calls = []
            response = _Response(
                b"DATE,BAMLH0A0HYM2\n2026-08-03,3.50\n2026-08-04,.\n2026-08-05,3.72\n"
            )

            def opener(request, *, timeout):
                calls.append((request, timeout))
                return response

            count = fetch_hy_oas(
                database,
                "2026-08-03",
                "2026-08-05",
                opener=opener,
                available_at="2026-08-06T01:02:03Z",
            )
            with closing(sqlite3.connect(database)) as connection:
                rows = connection.execute(
                    "SELECT date, series, value, available_at, source "
                    "FROM macro_observations ORDER BY date"
                ).fetchall()

        self.assertEqual(count, 2)
        self.assertTrue(response.closed)
        self.assertEqual(len(calls), 1)
        parsed_url = urlparse(calls[0][0].full_url)
        self.assertEqual(parsed_url.netloc, "fred.stlouisfed.org")
        self.assertEqual(
            parse_qs(parsed_url.query),
            {
                "id": ["BAMLH0A0HYM2"],
                "cosd": ["2026-08-03"],
                "coed": ["2026-08-05"],
            },
        )
        self.assertEqual(
            rows,
            [
                (
                    "2026-08-03",
                    "HY_OAS",
                    3.5,
                    "2026-08-06T01:02:03+00:00",
                    "FRED:BAMLH0A0HYM2",
                ),
                (
                    "2026-08-05",
                    "HY_OAS",
                    3.72,
                    "2026-08-06T01:02:03+00:00",
                    "FRED:BAMLH0A0HYM2",
                ),
            ],
        )

    def test_as_of_selects_latest_available_vintage_without_lookahead(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "risk.db"
            upsert_observations(
                database,
                "HY_OAS",
                [("2026-08-01", 3.0), ("2026-08-02", 3.2)],
                available_at="2026-08-02",
                source="first",
            )
            upsert_observations(
                database,
                "HY_OAS",
                [("2026-08-01", 3.1), ("2026-08-03", 9.9)],
                available_at="2026-08-04",
                source="revision",
            )

            before_revision = series_as_of(database, "hy_oas", "2026-08-03")
            after_revision = series_as_of(database, "HY_OAS", "2026-08-04")
            latest_before_revision = latest_as_of(database, "HY_OAS", "2026-08-03")

        self.assertEqual(
            before_revision,
            [
                ("2026-08-02", 3.2, "2026-08-02", "first"),
                ("2026-08-01", 3.0, "2026-08-02", "first"),
            ],
        )
        self.assertEqual(after_revision[0][0:2], ("2026-08-03", 9.9))
        self.assertEqual(after_revision[-1][0:2], ("2026-08-01", 3.1))
        self.assertIsNotNone(latest_before_revision)
        self.assertEqual(latest_before_revision[0], "2026-08-02")

    def test_repeated_fetch_stores_only_new_or_revised_vintages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "risk.db"
            bodies = [
                b"DATE,BAMLH0A0HYM2\n2026-08-03,3.50\n2026-08-04,3.60\n",
                b"DATE,BAMLH0A0HYM2\n2026-08-03,3.50\n2026-08-04,3.60\n",
                b"DATE,BAMLH0A0HYM2\n2026-08-03,3.50\n2026-08-04,3.65\n",
            ]

            def opener(_request, *, timeout):
                self.assertEqual(timeout, 30.0)
                return _Response(bodies.pop(0))

            first = fetch_hy_oas(
                database,
                "2026-08-03",
                "2026-08-04",
                opener=opener,
                available_at="2026-08-05",
            )
            unchanged = fetch_hy_oas(
                database,
                "2026-08-03",
                "2026-08-04",
                opener=opener,
                available_at="2026-08-06",
            )
            revised = fetch_hy_oas(
                database,
                "2026-08-03",
                "2026-08-04",
                opener=opener,
                available_at="2026-08-07",
            )
            with closing(sqlite3.connect(database)) as connection:
                rows = connection.execute(
                    "SELECT date, value, available_at FROM macro_observations "
                    "ORDER BY date, available_at"
                ).fetchall()

        self.assertEqual((first, unchanged, revised), (2, 0, 1))
        self.assertEqual(
            rows,
            [
                ("2026-08-03", 3.5, "2026-08-05"),
                ("2026-08-04", 3.6, "2026-08-05"),
                ("2026-08-04", 3.65, "2026-08-07"),
            ],
        )

    def test_earlier_same_value_vintage_is_retained_for_point_in_time_backfill(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "risk.db"
            upsert_observations(
                database,
                "HY_OAS",
                [("2026-08-01", 3.5)],
                available_at="2026-08-05",
                source="licensed",
            )
            inserted = upsert_observations(
                database,
                "HY_OAS",
                [("2026-08-01", 3.5)],
                available_at="2026-08-02",
                source="licensed",
            )
            as_of = series_as_of(database, "HY_OAS", "2026-08-02")

        self.assertEqual(inserted, 1)
        self.assertEqual(as_of[0][0:2], ("2026-08-01", 3.5))


if __name__ == "__main__":
    unittest.main()
