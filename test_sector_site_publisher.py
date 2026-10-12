from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from roro_shadow import SERIES_VERSION as RORO_SERIES_VERSION
from sector_rotation import SECTOR_ETFS
from sector_roro_shadow import MODEL_VERSION
from sector_shadow_store import PAYLOAD_FIELDS, SERIES_VERSION
from sector_site_publisher import (
    APP_DIR,
    DEFAULT_SHADOW_DATABASE,
    ENVELOPE_FIELDS,
    PAYLOAD_WEB_FIELDS,
    ROW_WEB_FIELDS,
    WEB_SCHEMA_VERSION,
    _group_provenance,
    main,
    research_history_payload,
    upload_snapshots,
    web_snapshot,
    web_snapshots,
)


def _payload(sector: str, rank: int) -> dict[str, object]:
    values: dict[str, object] = {
        field: None for field in PAYLOAD_FIELDS
    }
    values.update(
        {
            "sector_name": SECTOR_ETFS[sector],
            "rrg_score": 0.9 - rank * 0.03,
            "predicted_excess_20d": 0.02 - rank * 0.002,
            "forecast_score": (12 - rank) / 11,
            "forecast_rank": rank,
            "signal": "MODEL_TOP3" if rank <= 3 else "NEGATIVE",
            "quadrant": "LEADING" if rank <= 3 else "LAGGING",
            "quadrant_transition": "IMPROVING->LEADING",
            "tail_heading": "NE_STRENGTHENING" if rank <= 3 else "SW_DETERIORATING",
            "delta_rs_ratio": 1.0 - rank / 10,
            "delta_rs_momentum": 0.5 - rank / 20,
            "sector_beta_60": 0.5 + rank / 10,
            "beta_rank": (rank - 0.5) / 11,
            "roro_fit": rank / 11,
            "alpha_positive": rank <= 9,
            "alpha_top3": rank <= 3,
            "rrg_supportive": rank <= 3,
            "roro_context": "FIT" if rank >= 6 else "RORO_MISMATCH",
            "alignment": "ALIGNED",
            "sector_status": "SELECT" if rank <= 3 else "WATCH",
            "market_risk_score": 35,
            "risk_regime": "YELLOW",
            "roro_trajectory_5d": -5,
            "roro_data_status": "CONDITIONAL_EVENT_PIT",
            "forecast_target": "next-20-trading-day sector return minus equal-sector return",
            "training_rows": 2000,
            "training_weeks": 156,
            "training_start": "2017-01-06",
            "training_end": "2025-12-05",
            "latest_training_label_end": "2025-12-31",
            "alpha": 10.0,
            "top_k": 3,
            "context_type": "RORO_EXACT_DATE_BETA_FIT",
            "status_contract": "SELECT/WATCH/AVOID; no forced Top3 fill",
        }
    )
    return values


def _observations(day: str) -> list[dict[str, object]]:
    rows = []
    for rank, sector in enumerate(SECTOR_ETFS, 1):
        rows.append(
            {
                "series_version": SERIES_VERSION,
                "date": day,
                "sector": sector,
                "input_cutoff": day,
                "model_version": MODEL_VERSION,
                "roro_series_version": RORO_SERIES_VERSION,
                "provenance_hash": hashlib.sha256(f"{day}:{sector}".encode()).hexdigest(),
                "created_at": "2026-08-13T00:00:00.000000Z",
                "payload": _payload(sector, rank),
            }
        )
    return rows


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


class SectorSitePublisherTest(unittest.TestCase):
    def test_research_history_keeps_alpha_when_roro_is_na(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "history.csv"
            rows = []
            for rank, (ticker, name) in enumerate(SECTOR_ETFS.items(), 1):
                rows.append(
                    {
                        "decision_date": "2025-01-03", "ticker": ticker, "sector": name,
                        "forecast_rank": rank, "predicted_excess_20d": 0.01,
                        "actual_excess_20d": "", "alpha_hit": "", "quadrant": "LEADING",
                        "quadrant_transition": "LEADING->LEADING", "tail_heading": "NE_STRENGTHENING",
                        "roro_fit": "", "roro_context": "N/A", "fit_change": "N/A",
                        "sector_status": "N/A",
                    }
                )
            pd.DataFrame(rows).to_csv(source, index=False)

            payload = research_history_payload(source)

        self.assertEqual(payload["start"], "2025-01-03")
        self.assertEqual(len(payload["rows"]), 11)
        self.assertEqual(payload["rows"][0]["roro_context"], "N/A")
        self.assertIsNone(payload["rows"][0]["roro_fit"])
        self.assertEqual(payload["rows"][0]["predicted_excess_20d"], 0.01)

    def test_default_database_follows_the_store_contract(self) -> None:
        from sector_shadow_store import DEFAULT_TARGET

        self.assertEqual(DEFAULT_SHADOW_DATABASE, APP_DIR / DEFAULT_TARGET)

    def test_envelope_has_363_rows_across_33_dates_and_latest_has_11(self) -> None:
        observations = [
            observation
            for day in pd.bdate_range("2026-01-07", periods=33).strftime("%Y-%m-%d")
            for observation in _observations(day)
        ]
        snapshots = web_snapshots(observations)
        latest = web_snapshots(observations, latest_only=True)

        self.assertEqual(len(snapshots), 33)
        self.assertEqual(sum(len(item["payload"]["rows"]) for item in snapshots), 363)
        self.assertEqual(len(latest), 1)
        self.assertEqual(len(latest[0]["payload"]["rows"]), 11)
        self.assertEqual(tuple(snapshots[0]), ENVELOPE_FIELDS)
        self.assertEqual(snapshots[0]["schema_version"], WEB_SCHEMA_VERSION)
        self.assertEqual(tuple(snapshots[0]["payload"]), PAYLOAD_WEB_FIELDS)
        self.assertTrue(
            all(tuple(row) == ROW_WEB_FIELDS for row in snapshots[0]["payload"]["rows"])
        )
        self.assertEqual(snapshots[0], web_snapshot(_observations("2026-01-07")))
        self.assertEqual(snapshots[0]["payload"]["relative_benchmark"], "SPY")
        self.assertEqual(snapshots[0]["payload"]["market_context"], "NQ=F")

        snapshot = snapshots[0]
        self.assertEqual(
            snapshot["provenance_hash"],
            _group_provenance(
                series_version=snapshot["series_version"],
                date=snapshot["as_of"],
                input_cutoff=snapshot["input_cutoff"],
                model_versions=snapshot["model_versions"],
                payload=snapshot["payload"],
            ),
        )

    def test_public_provenance_has_a_fixed_cross_language_encoding(self) -> None:
        actual = _group_provenance(
            series_version="sector-series",
            date="2026-01-07",
            input_cutoff="2026-01-07",
            model_versions={"sector_roro": "sector-model", "roro": "roro-series"},
            payload={
                "alpha": 10.0,
                "rows": [{"sector": "XLK", "score": -0.0}],
                "korean": "순환매",
            },
        )
        self.assertEqual(
            actual,
            "bd779f950a535012441102bf5e1df25650a4a1f9bb497cb190a05b08ec6b93c6",
        )

    def test_identity_fields_fail_closed_and_never_reach_envelope(self) -> None:
        rows = _observations("2026-01-07")
        rows[0]["payload"]["holding_name"] = "SECRET"
        with self.assertRaisesRegex(ValueError, "holding identity"):
            web_snapshot(rows)

        clean = web_snapshot(_observations("2026-01-07"))
        serialized = json.dumps(clean).lower()
        for forbidden in ("ticker", "isin", "quantity", "holding_name", "positions"):
            self.assertNotIn(f'"{forbidden}"', serialized)

    def test_mixed_series_date_or_common_metadata_fails_closed(self) -> None:
        rows = _observations("2026-01-07")
        rows[0]["series_version"] = "other"
        with self.assertRaisesRegex(ValueError, "supported"):
            web_snapshot(rows)

        rows = _observations("2026-01-07")
        rows[0]["payload"]["market_risk_score"] = 99
        with self.assertRaisesRegex(ValueError, "must agree"):
            web_snapshot(rows)

    @patch("sector_site_publisher.urllib.request.urlopen")
    def test_uploads_one_atomic_date_per_request(self, urlopen) -> None:
        urlopen.side_effect = [
            FakeResponse({"dates": 1, "rows": 11, "inserted": 11, "idempotent": 0}),
            FakeResponse({"dates": 1, "rows": 11, "inserted": 11, "idempotent": 0}),
        ]
        sample = web_snapshot(_observations("2026-01-07"))
        second = deepcopy(sample)
        second["as_of"] = "2026-01-08"
        second["input_cutoff"] = "2026-01-08"
        snapshots = [sample, second]
        result = upload_snapshots(
            snapshots, "https://example.test", "secret", "bypass"
        )
        self.assertEqual(result["dates"], 2)
        self.assertEqual(result["rows"], 22)
        self.assertEqual(urlopen.call_count, 2)
        for call in urlopen.call_args_list:
            request = call.args[0]
            self.assertEqual(request.full_url, "https://example.test/api/sector-roro")
            self.assertEqual(request.headers["Authorization"], "Bearer secret")
            self.assertEqual(len(json.loads(request.data)), 1)

    @patch("sector_site_publisher.urllib.request.urlopen")
    @patch("sector_site_publisher.build_and_publish")
    def test_no_upload_cli_writes_json_with_zero_network_calls(
        self, build, urlopen
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site_payload.json"
            snapshot = web_snapshot(_observations("2026-01-07"))
            build.return_value = (
                {
                    "series_version": SERIES_VERSION,
                    "dates": 1,
                    "rows": 11,
                    "as_of": "2026-01-07",
                    "source_database_mode": "mode=ro; query_only=ON",
                },
                [snapshot],
                None,
            )
            printed = io.StringIO()
            with redirect_stdout(printed):
                main(
                    [
                        "--shadow-database",
                        str(Path(directory) / "sector.db"),
                        "--as-of",
                        "2026-01-07",
                        "--latest-only",
                        "--no-upload",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), [snapshot])
            self.assertIsNone(build.call_args.kwargs["config"])
            urlopen.assert_not_called()
            self.assertIsNone(json.loads(printed.getvalue())["upload"])

    @patch("sector_site_publisher.build_and_publish")
    def test_cli_requires_explicit_external_upload_consent(self, build) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(
                '{"site_url":"https://example.test","upload_token":"secret"}',
                encoding="utf-8",
            )
            snapshot = web_snapshot(_observations("2026-01-07"))
            build.return_value = (
                {"dates": 1, "rows": 11, "as_of": "2026-01-07"},
                [snapshot],
                None,
            )
            printed = io.StringIO()
            with redirect_stdout(printed):
                main([
                    "--shadow-database", str(root / "sector.db"),
                    "--config", str(config),
                    "--consent", str(root / "missing-consent.json"),
                    "--output", str(root / "payload.json"),
                ])
            self.assertIsNone(build.call_args.kwargs["config"])
            self.assertIn("명시 동의", printed.getvalue())


if __name__ == "__main__":
    unittest.main()
