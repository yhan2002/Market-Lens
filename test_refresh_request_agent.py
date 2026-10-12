import json
import unittest
from contextlib import nullcontext
from datetime import date
from unittest.mock import patch

import refresh_request_agent as agent
from refresh_pipeline import RefreshResult


class FakeResponse:
    def __init__(self, body: dict | None = None, status: int = 200) -> None:
        self.status = status
        self._body = json.dumps(body).encode("utf-8") if body is not None else b""

    def __enter__(self): return self
    def __exit__(self, *_args): return False
    def getcode(self) -> int: return self.status
    def read(self) -> bytes: return self._body


class RefreshRequestAgentTests(unittest.TestCase):
    def test_post_refresh_uses_existing_auth_headers(self) -> None:
        config = {
            "site_url": "https://example.test",
            "upload_token": "secret-upload",
            "sites_bypass_token": "secret-bypass",
        }
        with patch(
            "refresh_request_agent.urllib.request.urlopen",
            return_value=FakeResponse({"request": {"id": 7}}),
        ) as urlopen:
            response = agent.post_refresh(config, {"action": "claim"})
        request = urlopen.call_args.args[0]
        self.assertEqual(response, {"request": {"id": 7}})
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-upload")
        self.assertEqual(request.get_header("Oai-sites-authorization"), "Bearer secret-bypass")

    def test_busy_runner_does_not_claim_request(self) -> None:
        with (
            patch("refresh_request_agent.is_refresh_busy", return_value=True),
            patch("refresh_request_agent.post_refresh") as post,
        ):
            self.assertEqual(agent.main(), 0)
        post.assert_not_called()

    def test_empty_queue_is_noop(self) -> None:
        with (
            patch("refresh_request_agent.is_refresh_busy", return_value=False),
            patch("refresh_request_agent.load_config", return_value={"site_url": "https://x", "upload_token": "x"}),
            patch("refresh_request_agent.post_refresh", return_value=None),
            patch("refresh_request_agent.run_refresh_pipeline") as run,
        ):
            self.assertEqual(agent.main(), 0)
        run.assert_not_called()

    def test_same_nq_date_refreshes_turning_macro_only(self) -> None:
        calls = []
        def post(_config, payload):
            calls.append(payload)
            return {"id": 1} if payload["action"] == "claim" else {"ok": True}
        with (
            patch("refresh_request_agent.is_refresh_busy", return_value=False),
            patch("refresh_request_agent.load_config", return_value={"site_url": "https://x", "upload_token": "x"}),
            patch("refresh_request_agent.post_refresh", side_effect=post),
            patch("refresh_request_agent.server_snapshot_as_of", return_value="2026-08-17"),
            patch("refresh_request_agent.server_roro_as_of", return_value="2026-08-17"),
            patch("refresh_request_agent.probe_latest_completed_market_date", return_value=date(2026, 8, 17)) as probe,
            patch("refresh_request_agent.PUBLISH_DATABASE") as database,
            patch("refresh_request_agent.refresh_lock", return_value=nullcontext()),
            patch(
                "refresh_request_agent.refresh_turning_macro_inputs",
                return_value=({}, False),
            ) as macro,
            patch("refresh_request_agent.run_refresh_pipeline") as run,
        ):
            database.is_file.return_value = True
            self.assertEqual(agent.main(), 0)
        probe.assert_called_once_with(tickers=("NQ=F",), fallback_database=database)
        run.assert_not_called()
        macro.assert_called_once_with(database, "2026-08-17")
        self.assertEqual(
            calls[-1]["detail"],
            "새 거래일 없음 · 2026-08-17 유지 · 전환 매크로 최신화 완료"
            " · 보유내역 재확인 안 함"
            " · 보유내역 반영은 다음 예약 갱신 또는 UPDATE_SITE.cmd 수동 갱신 필요",
        )
        self.assertEqual(calls[-1]["status"], "SUCCEEDED")

    def test_stale_outputs_use_one_pipeline_and_verify_servers(self) -> None:
        calls = []
        def post(_config, payload):
            calls.append(payload)
            return {"request": {"id": 2}} if payload["action"] == "claim" else {"ok": True}
        with (
            patch("refresh_request_agent.is_refresh_busy", return_value=False),
            patch("refresh_request_agent.load_config", return_value={"site_url": "https://x", "upload_token": "x"}),
            patch("refresh_request_agent.post_refresh", side_effect=post),
            patch("refresh_request_agent.server_snapshot_as_of", side_effect=["2026-08-14", "2026-08-17"]),
            patch("refresh_request_agent.server_roro_as_of", side_effect=["2026-08-14", "2026-08-17"]),
            patch("refresh_request_agent.probe_latest_completed_market_date", return_value=date(2026, 8, 17)),
            patch("refresh_request_agent.run_refresh_pipeline", return_value=RefreshResult("2026-08-17", ())) as run,
        ):
            self.assertEqual(agent.main(), 0)
        run.assert_called_once_with(date(2026, 8, 17))
        self.assertEqual(calls[-1]["status"], "SUCCEEDED")
        self.assertEqual(calls[-1]["as_of"], "2026-08-17")
        self.assertEqual(calls[-1]["detail"], "갱신 및 게시 완료 · 2026-08-17")

    def test_warnings_report_partial_completion_and_holdings_date(self) -> None:
        for holdings_as_of in ("2026-08-10", None):
            with self.subTest(holdings_as_of=holdings_as_of):
                result = RefreshResult(
                    "2026-08-17", ("SS 보유내역 갱신 실패: 직전 내역 유지",),
                    holdings_as_of=holdings_as_of,
                )
                with (
                    patch("refresh_request_agent.is_refresh_busy", return_value=False),
                    patch("refresh_request_agent.load_config", return_value={"site_url": "https://x", "upload_token": "x"}),
                    patch("refresh_request_agent.post_refresh", side_effect=[{"id": 5}, {"ok": True}]) as post,
                    patch("refresh_request_agent.server_snapshot_as_of", side_effect=["2026-08-14", "2026-08-17"]),
                    patch("refresh_request_agent.server_roro_as_of", side_effect=["2026-08-14", "2026-08-17"]),
                    patch("refresh_request_agent.probe_latest_completed_market_date", return_value=date(2026, 8, 17)),
                    patch("refresh_request_agent.run_refresh_pipeline", return_value=result),
                ):
                    self.assertEqual(agent.main(), 0)
                finish = post.call_args.args[1]
                self.assertEqual(finish["status"], "SUCCEEDED")
                self.assertEqual(finish["as_of"], "2026-08-17")
                self.assertTrue(finish["detail"].startswith("부분 완료 · 2026-08-17"))
                self.assertNotIn("갱신 및 게시 완료", finish["detail"])
                self.assertIn(result.warnings[0], finish["detail"])
                if holdings_as_of:
                    self.assertIn(f"보유내역 기준일 {holdings_as_of}", finish["detail"])
                else:
                    self.assertNotIn("보유내역 기준일", finish["detail"])

    def test_stale_market_lens_is_reported_as_failure(self) -> None:
        calls = []
        def post(_config, payload):
            calls.append(payload)
            return {"id": 3} if payload["action"] == "claim" else {"ok": True}
        with (
            patch("refresh_request_agent.is_refresh_busy", return_value=False),
            patch("refresh_request_agent.load_config", return_value={"site_url": "https://x", "upload_token": "x"}),
            patch("refresh_request_agent.post_refresh", side_effect=post),
            patch("refresh_request_agent.server_snapshot_as_of", side_effect=["2026-08-14", "2026-08-17"]),
            patch("refresh_request_agent.server_roro_as_of", side_effect=["2026-08-14", "2026-08-14"]),
            patch("refresh_request_agent.probe_latest_completed_market_date", return_value=date(2026, 8, 17)),
            patch("refresh_request_agent.run_refresh_pipeline", return_value=RefreshResult("2026-08-17", ())),
            patch("refresh_request_agent._log_exception"),
        ):
            self.assertEqual(agent.main(), 1)
        self.assertEqual(calls[-1]["status"], "FAILED")
        self.assertIn("Market Lens 게시일", calls[-1]["detail"])

    def test_base_postcondition_failure_is_reported(self) -> None:
        calls = []
        def post(_config, payload):
            calls.append(payload)
            return {"id": 4} if payload["action"] == "claim" else {"ok": True}
        with (
            patch("refresh_request_agent.is_refresh_busy", return_value=False),
            patch("refresh_request_agent.load_config", return_value={"site_url": "https://x", "upload_token": "x"}),
            patch("refresh_request_agent.post_refresh", side_effect=post),
            patch("refresh_request_agent.server_snapshot_as_of", side_effect=["2026-08-14", "2026-08-14"]),
            patch("refresh_request_agent.server_roro_as_of", side_effect=["2026-08-14", "2026-08-14"]),
            patch("refresh_request_agent.probe_latest_completed_market_date", return_value=date(2026, 8, 17)),
            patch("refresh_request_agent.run_refresh_pipeline", return_value=RefreshResult("2026-08-17", ())),
            patch("refresh_request_agent._log_exception"),
        ):
            self.assertEqual(agent.main(), 1)
        self.assertEqual(calls[-1]["status"], "FAILED")
        self.assertIn("게시일", calls[-1]["detail"])


if __name__ == "__main__":
    unittest.main()
