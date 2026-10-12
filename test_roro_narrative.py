from __future__ import annotations

import json
import tempfile
import unittest
import copy
import io
from urllib.error import HTTPError
from pathlib import Path

from roro_narrative import generate_summary, load_api_key, regenerate_observation, fallback_summary


class RoroNarrativeTest(unittest.TestCase):
    def test_diagnostics_redact_secrets_and_reject_incomplete_output(self):
        payload = {"market_risk_score": 43, "policy": {"fast_status": "CLEAR", "fast_trigger_count": 0}}
        diagnostics = {}
        def fail(*args, **kwargs):
            raise HTTPError('https://example.invalid', 429, 'secret-test-key', {}, io.BytesIO(b'secret-test-key'))
        with self.assertLogs('roro_narrative', level='WARNING') as logs:
            self.assertIsNone(generate_summary(payload, 'secret-test-key', request=fail, diagnostics=diagnostics))
        self.assertEqual(diagnostics['http_status'], 429)
        self.assertNotIn('secret-test-key', str(logs.output))
        class Response:
            def read(self): return json.dumps({'choices': [{'finish_reason': 'length', 'message': {'content': 'unfinished'}}]}).encode()
            def __enter__(self): return self
            def __exit__(self, *args): return False
        self.assertIsNone(generate_summary(payload, 'test', request=lambda *a, **k: Response(), diagnostics=diagnostics))
        self.assertEqual(diagnostics['reason'], 'response_not_completed')
        self.assertIsNone(generate_summary(payload, '', request=lambda *a, **k: self.fail('no key must not call'), diagnostics=diagnostics))
        self.assertEqual(diagnostics['reason'], 'missing_api_key')
        self.assertIn('43점', fallback_summary(payload))

    def test_regeneration_preserves_observation(self):
        observation = {'date': '2026-09-08', 'series_version': 'test', 'provenance_hash': 'a' * 64,
                       'payload': {'market_risk_score': 43, 'policy': {'fast_status': 'CLEAR', 'fast_trigger_count': 0, 'primary_driver': 'RORO SCORE'}}}
        original = copy.deepcopy(observation)
        class Response:
            def read(self): return json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': '위험은 완화됐습니다.'}}]}).encode()
            def __enter__(self): return self
            def __exit__(self, *args): return False
        result = regenerate_observation(observation, 'test', request=lambda *a, **k: Response())
        self.assertEqual(result['summary'], '위험은 완화됐습니다.')
        self.assertEqual(result['provenance_hash'], original['provenance_hash'])
        self.assertEqual(observation, original)

    def test_loads_only_openrouter_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("OPENAI_API_KEY='old-key'\nOPENROUTER_API_KEY='test-key'\n", encoding="utf-8")
            self.assertEqual(load_api_key(path), "test-key")

    def test_one_line_output_and_safe_failure(self) -> None:
        payload = {
            "risk_regime": "YELLOW", "trend_phase": "TRANSITION", "market_risk_score": 39,
            "recommended_long": 64, "trajectory_5d": -8.0, "risk_drivers": [], "relief_drivers": ["market"],
            "macro_risk_score": 42, "market_trend_score": 37, "portfolio_internal_score": 35,
            "turn_evidence": ["VIX_CURVE_5D_DOWN", "NQ_ABOVE_MA5"],
            "policy": {"fast_status": "CLEAR", "fast_trigger_count": 0, "phase": "RECOVERY_CONFIRM"},
        }
        class Response:
            def read(self): return json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "위험 완화가 이어지지만 전환 확인 구간입니다."}}]}).encode()
            def __enter__(self): return self
            def __exit__(self, *args): return False
        self.assertEqual(generate_summary(payload, "test", request=lambda *_args, **_kwargs: Response()), "위험 완화가 이어지지만 전환 확인 구간입니다.")
        self.assertIsNone(generate_summary(payload, "test", request=lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError())))
        payload["macro_risk_score"] = float("nan")
        self.assertIsNone(generate_summary(payload, "test", request=lambda *_args, **_kwargs: self.fail("request should not run")))

    def test_prompt_uses_market_evidence_not_model_codes(self) -> None:
        payload = {
            "risk_regime": "YELLOW", "trend_phase": "BOTTOMING_CANDIDATE", "market_risk_score": 38,
            "recommended_long": 64, "trajectory_5d": -21.0, "risk_drivers": [], "relief_drivers": [],
            "macro_risk_score": 43, "market_trend_score": 37, "portfolio_internal_score": 35,
            "turn_evidence": ["VIX_CURVE_5D_DOWN", "NQ_ABOVE_MA5"],
            "policy": {"fast_status": "INCOMPLETE", "fast_trigger_count": 0, "phase": "RECOVERY_HOLD"},
        }
        captured: dict[str, object] = {}
        class Response:
            def read(self): return json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "시황 요약"}}]}, ensure_ascii=False).encode("utf-8")
            def __enter__(self): return self
            def __exit__(self, *args): return False
        def request(request, **_kwargs):
            captured.update(json.loads(request.data.decode("utf-8")))
            captured["url"] = request.full_url
            return Response()
        crash_guard = {
            "score": 0.67, "score_1d": 0.67, "score_5d": 0.56,
            "risk_band": "VULNERABLE_SHADOW",
            "vulnerable_cutoff": 0.60, "crash_watch_cutoff": 0.80,
            "components": {
                "vix_jump_1d": {"raw": 0.0752, "percentile": 0.795},
                "vix_acceleration_1d": {"raw": 0.051, "percentile": 0.854},
            },
        }
        generate_summary(payload, "test", crash_guard=crash_guard, request=request)
        facts = json.loads(captured["messages"][1]["content"])
        self.assertEqual(captured["model"], "google/gemini-2.5-flash-lite")
        self.assertEqual(captured["provider"], {"data_collection": "deny"})
        self.assertEqual(captured["url"], "https://openrouter.ai/api/v1/chat/completions")
        serialized = json.dumps(facts, ensure_ascii=False)
        self.assertEqual(len(facts["12개_세부팩터"]), 12)
        self.assertEqual(facts["Crash_Guard"]["점수"], 0.67)
        self.assertEqual(
            facts["Crash_Guard"]["세부요인"][0]["과거백분위"], 79.5
        )
        self.assertIn("변동성 곡선 안정", serialized)
        self.assertNotIn("BOTTOMING_CANDIDATE", serialized)
