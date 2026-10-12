import io
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing, nullcontext
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import refresh_pipeline as pipeline


class RefreshPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        breadth = patch("refresh_pipeline.load_breadth_history", return_value=[])
        breadth.start()
        self.addCleanup(breadth.stop)
        status = patch("refresh_pipeline._holdings_status", return_value=("2026-08-14", None))
        status.start()
        self.addCleanup(status.stop)

    def _paths(self, root: Path):
        python = root / "python.exe"
        config = root / "site_upload_config.json"
        python.touch()
        config.write_text("{}", encoding="utf-8")
        return python, config

    def test_one_pipeline_runs_all_surfaces_and_keeps_optional_warning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python, _ = self._paths(root)
            results = [value for value in (0, 0, 1, 0, 0, 0, 0, 0, 0)]
            with (
                patch("refresh_pipeline.BASE_DIR", root),
                patch("refresh_pipeline.PYTHON", python),
                patch("refresh_pipeline.refresh_lock", return_value=nullcontext()),
                patch("refresh_pipeline._export_holdings", return_value=None),
                patch("refresh_pipeline._run", side_effect=results) as run,
            ):
                result = pipeline.run_refresh_pipeline(
                    date(2026, 8, 17), log_path=root / "update.log"
                )
        self.assertEqual(result.requested_as_of, "2026-08-17")
        self.assertEqual(result.holdings_as_of, "2026-08-14")
        self.assertIn("JPY·GLD 입력", result.warnings[0])
        self.assertTrue(any("Fast 8 Breadth 미완료" in item for item in result.warnings))
        self.assertEqual(run.call_count, 9)
        commands = [call.args[1] for call in run.call_args_list]
        self.assertIn("site_publisher.py", commands[0])
        self.assertEqual(commands[0][commands[0].index("--holdings-workbook") + 1], str(pipeline.HOLDINGS_CACHE))
        self.assertEqual(commands[0][commands[0].index("--date") + 1], "2026-08-17")
        self.assertIn("reports/rotation_entry_publish.py", commands[1])
        self.assertIn("--record", commands[1])
        self.assertIn("--upload", commands[1])
        self.assertIn("roro_site_publisher.py", commands[4])
        self.assertIn("--require-as-of", commands[4])
        self.assertIn("--repair-missing-breadth", commands[4])
        self.assertIn("--required-as-of", commands[3])
        self.assertEqual(commands[3][commands[3].index("--required-as-of") + 1], "2026-08-17")
        self.assertIn("--publish-database", commands[3])
        self.assertIn("sector_shadow_store.py", commands[5])
        self.assertIn("sector_site_publisher.py", commands[6])
        self.assertIn("it_internals_shadow.py", commands[7])
        self.assertIn("--upload", commands[7])
        self.assertEqual(commands[7][commands[7].index("--sector-database") + 1], str(pipeline.PUBLISH_DATABASE))
        self.assertIn("sector_hybrid_publisher.py", commands[8])
        self.assertIn("--upload", commands[8])
        self.assertEqual(commands[8][commands[8].index("--as-of") + 1], "2026-08-17")

    def test_required_failure_is_not_reported_as_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python, _ = self._paths(root)
            with (
                patch("refresh_pipeline.BASE_DIR", root),
                patch("refresh_pipeline.PYTHON", python),
                patch("refresh_pipeline.refresh_lock", return_value=nullcontext()),
                patch("refresh_pipeline._export_holdings", return_value=None),
                patch(
                    "refresh_pipeline._run",
                    side_effect=[value for value in (9, 0, 0, 0, 0, 0, 0)],
                ) as run,
            ):
                with self.assertRaisesRegex(pipeline.RefreshFailed, "기본 대시보드"):
                    pipeline.run_refresh_pipeline(
                        date(2026, 8, 17), log_path=root / "update.log"
                    )

        self.assertIn("reports/rotation_entry_publish.py", run.call_args_list[1].args[1])
        self.assertIn("--record", run.call_args_list[1].args[1])
        self.assertIn("it_internals_shadow.py", run.call_args_list[-2].args[1])

    def test_denied_network_holdings_uses_cache_warning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = Mock()
            source.stat.side_effect = PermissionError(5, "access denied")
            log = io.StringIO()
            with (
                patch("refresh_pipeline.BASE_DIR", root),
                patch("refresh_pipeline.PRIVATE_DIR", root / "private"),
                patch("refresh_pipeline.HOLDINGS_SOURCE", source),
                patch("refresh_pipeline.time.sleep") as sleep,
                patch("refresh_pipeline._run") as run,
            ):
                warning = pipeline._export_holdings(log)
        self.assertIn("이전 확인 포지션", warning)
        self.assertIn("PermissionError", log.getvalue())
        self.assertIn("3/3", log.getvalue())
        self.assertEqual(sleep.call_count, 2)
        run.assert_not_called()

    def test_automatic_target_uses_nq_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python, _ = self._paths(root)
            with (
                patch("refresh_pipeline.BASE_DIR", root),
                patch("refresh_pipeline.PYTHON", python),
                patch("refresh_pipeline.refresh_lock", return_value=nullcontext()),
                patch("refresh_pipeline._export_holdings", return_value=None),
                patch("refresh_pipeline.probe_latest_completed_market_date", return_value=date(2026, 8, 17)) as probe,
                patch("refresh_pipeline._run", return_value=0),
            ):
                pipeline.run_refresh_pipeline(None, log_path=root / "update.log")
        probe.assert_called_once_with(
            tickers=("NQ=F",), fallback_database=pipeline.PUBLISH_DATABASE
        )

    def test_holdings_import_warning_reaches_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python, _ = self._paths(root)
            with (
                patch("refresh_pipeline.BASE_DIR", root),
                patch("refresh_pipeline.PYTHON", python),
                patch("refresh_pipeline.refresh_lock", return_value=nullcontext()),
                patch("refresh_pipeline._export_holdings", return_value=None),
                patch("refresh_pipeline._holdings_status", return_value=("2026-08-01", "DB 반영 미확인")),
                patch("refresh_pipeline._run", return_value=0),
            ):
                result = pipeline.run_refresh_pipeline(date(2026, 8, 17), log_path=root / "update.log")
        self.assertEqual(result.holdings_as_of, "2026-08-01")
        self.assertIn("DB 반영 미확인", result.warnings)


class HoldingsRefreshTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.private = self.root / "private"
        self.private.mkdir()
        self.source = self.root / "cloud.xlsx"
        self.source.write_bytes(b"source")
        self.inbox = self.private / "incoming" / "holdings.xlsx"
        self.cache = self.private / "cache.xlsx"
        self.cache.write_bytes(b"previous valid cache")
        self.database = self.root / "private.db"
        (self.root / "EXPORT_CLOUDIUM_HOLDINGS.vbs").touch()
        (self.root / "System32").mkdir()
        (self.root / "System32" / "cscript.exe").touch()
        paths = patch.multiple(
            pipeline, BASE_DIR=self.root, PRIVATE_DIR=self.private,
            HOLDINGS_SOURCE=self.source,
            HOLDINGS_CACHE=self.cache, HOLDINGS_DATABASE=self.database,
        )
        paths.start()
        self.addCleanup(paths.stop)
        environment = patch.dict(os.environ, {"SystemRoot": str(self.root)})
        environment.start()
        self.addCleanup(environment.stop)

    @staticmethod
    def _export(name, command, log):
        Path(command[-1]).write_bytes(b"validated new cache")
        return 0

    def test_mp_source_wins_over_old_local_copy_and_replaces_cache(self) -> None:
        self.inbox.parent.mkdir()
        self.inbox.write_bytes(b"local")
        log = io.StringIO()
        with (
            patch.object(pipeline, "_run", side_effect=self._export) as run,
            patch.object(pipeline, "read_holdings_workbook", return_value=([{"date": "2026-09-22"}], {})),
        ):
            self.assertIsNone(pipeline._export_holdings(log))
        self.assertEqual(run.call_args.args[1][-2], str(self.source))
        self.assertEqual(self.cache.read_bytes(), b"validated new cache")
        self.assertIn("입력=MP 원본", log.getvalue())
        self.assertIn("2026-09-22", log.getvalue())

    def test_mp_source_used_when_no_local_input(self) -> None:
        with (
            patch.object(pipeline, "_run", side_effect=self._export) as run,
            patch.object(pipeline, "read_holdings_workbook", return_value=([{"date": "2026-09-22"}], {})),
        ):
            self.assertIsNone(pipeline._export_holdings(io.StringIO()))
        self.assertEqual(run.call_args.args[1][-2], str(self.source))

    def test_missing_mp_does_not_fall_back_to_old_local_copy(self) -> None:
        self.source.unlink()
        self.inbox.parent.mkdir()
        self.inbox.write_bytes(b"old local copy")
        with (
            patch.object(pipeline.time, "sleep"),
            patch.object(pipeline, "_run") as run,
        ):
            warning = pipeline._export_holdings(io.StringIO())
        self.assertIn("MP 원본", warning)
        self.assertIn("이전 확인 포지션", warning)
        self.assertEqual(self.cache.read_bytes(), b"previous valid cache")
        run.assert_not_called()

    def test_mp_source_access_can_recover_on_retry(self) -> None:
        source = Mock()
        source.stat.side_effect = [FileNotFoundError(2, "not ready"), self.source.stat()]
        with (
            patch.object(pipeline, "HOLDINGS_SOURCE", source),
            patch.object(pipeline.time, "sleep") as sleep,
            patch.object(pipeline, "_run", side_effect=self._export),
            patch.object(pipeline, "read_holdings_workbook", return_value=([{"date": "2026-09-22"}], {})),
        ):
            self.assertIsNone(pipeline._export_holdings(io.StringIO()))
        sleep.assert_called_once_with(2)

    def test_missing_export_script_is_identified(self) -> None:
        (self.root / "EXPORT_CLOUDIUM_HOLDINGS.vbs").unlink()
        with patch.object(pipeline, "_run") as run:
            warning = pipeline._export_holdings(io.StringIO())
        self.assertIn("Excel 내보내기 스크립트 없음", warning)
        run.assert_not_called()

    def test_bridge_failure_preserves_cache(self) -> None:
        with patch.object(pipeline, "_run", return_value=1):
            warning = pipeline._export_holdings(io.StringIO())
        self.assertIn("Excel 내보내기 실패", warning)
        self.assertEqual(self.cache.read_bytes(), b"previous valid cache")

    def test_invalid_export_preserves_cache_and_does_not_log_sensitive_error(self) -> None:
        log = io.StringIO()
        with (
            patch.object(pipeline, "_run", side_effect=self._export),
            patch.object(pipeline, "read_holdings_workbook", side_effect=ValueError("SECRET_POSITION")),
        ):
            warning = pipeline._export_holdings(log)
        self.assertIn("검증", warning)
        self.assertNotIn("SECRET_POSITION", log.getvalue())
        self.assertEqual(self.cache.read_bytes(), b"previous valid cache")

    def test_source_date_cannot_roll_back_private_history(self) -> None:
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute("CREATE TABLE daily_holdings(date TEXT)")
            connection.execute("INSERT INTO daily_holdings VALUES ('2026-09-22')")
        with (
            patch.object(pipeline, "_run", side_effect=self._export),
            patch.object(pipeline, "read_holdings_workbook", return_value=([{"date": "2026-08-28"}], {})),
        ):
            warning = pipeline._export_holdings(io.StringIO())
        self.assertIn("과거", warning)
        self.assertEqual(self.cache.read_bytes(), b"previous valid cache")

    def test_old_mp_file_is_explicitly_flagged(self) -> None:
        os.utime(self.source, (1, 1))
        with (
            patch.object(pipeline, "_run", side_effect=self._export),
            patch.object(pipeline, "read_holdings_workbook", return_value=([{"date": "2026-09-22"}], {})),
        ):
            warning = pipeline._export_holdings(io.StringIO())
        self.assertIn("10일 초과", warning)
        self.assertIn("포지션 불변", warning)

    def test_db_status_distinguishes_position_date_and_missing_import(self) -> None:
        cache_stat = self.cache.stat()
        fingerprint = f"{cache_stat.st_size}:{cache_stat.st_mtime_ns}"
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute("CREATE TABLE daily_holdings(date TEXT)")
            connection.executemany("INSERT INTO daily_holdings VALUES (?)", [("2026-08-28",), ("2026-09-23",)])
            connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
            connection.execute("INSERT INTO metadata VALUES ('event_holdings_workbook_fingerprint', ?)", (fingerprint,))
        self.assertEqual(pipeline._holdings_status("2026-09-22"), ("2026-08-28", None))
        self.cache.write_bytes(b"new but not imported")
        latest, warning = pipeline._holdings_status("2026-09-22")
        self.assertEqual(latest, "2026-08-28")
        self.assertIn("DB 반영", warning)


if __name__ == "__main__":
    unittest.main()
