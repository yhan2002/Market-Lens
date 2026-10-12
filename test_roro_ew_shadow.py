from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from roro_engine import BLOCKS, INPUT_COLUMNS, MODEL_VERSION as RORO_MODEL_VERSION
from roro_ew_shadow import (
    BLOCK_PAYLOAD_COLUMNS,
    GLOBAL_PAYLOAD_COLUMNS,
    ImmutableSeriesConflict,
    SERIES_VERSION,
    TABLE,
    build_ew_shadow_series,
    latest_ew_shadow,
    main,
    persist_ew_shadow,
    read_ew_shadow_history,
    _payloads,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _panel() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        ["2026-01-02", "2026-01-05", "2026-01-06"], name="date"
    )
    panel = pd.DataFrame(1.0, index=index, columns=INPUT_COLUMNS)
    panel.loc[index[0], ["usdjpy", "gold", "breadth_200dma"]] = np.nan
    panel.attrs["database_mode"] = "mode=ro; query_only=ON"
    return panel


def _scores(last_credit: float = 60.0) -> pd.DataFrame:
    index = _panel().index
    frame = pd.DataFrame(index=index)
    for block in BLOCKS:
        partial = block in {
            "credit",
            "rates_volatility",
            "equity_volatility",
            "equity_trend",
        }
        values = [np.nan, 40.0 if partial else np.nan, last_credit if block == "credit" else 41.0]
        frame[f"{block}_score"] = values
        frame[f"{block}_regime"] = [
            "NO_SCORE",
            "YELLOW" if partial else "NO_SCORE",
            "YELLOW",
        ]
        frame[f"{block}_contribution"] = [np.nan, np.nan, -0.5]
        for lag in (1, 5, 20):
            frame[f"{block}_trajectory_{lag}d"] = [
                np.nan,
                -1.5 if partial else np.nan,
                2.0,
            ]
    frame["available_block_count"] = [0, 4, 7]
    frame["completeness"] = [0.0, 400 / 7, 100.0]
    frame["score_raw"] = [np.nan, np.nan, 42.0]
    frame["risk_score"] = pd.Series([pd.NA, pd.NA, 42], index=index, dtype="Int64")
    frame["disagreement"] = [np.nan, np.nan, 3.0]
    frame["agreement"] = [np.nan, np.nan, 94.0]
    frame["data_status"] = ["DATA_BLOCKED", "NO_SCORE", "FULL"]
    frame["roro_regime"] = ["NO_SCORE", "NO_SCORE", "YELLOW"]
    for lag in (1, 5, 20):
        frame[f"trajectory_{lag}d"] = [np.nan, np.nan, 2.0]
    frame["risk_drivers"] = [(), (), ("credit",)]
    frame["relief_drivers"] = [(), (), ("equity_trend",)]
    frame["ticker"] = ["SECRET_A", "SECRET_B", "SECRET_C"]
    frame["holding_name"] = ["Private A", "Private B", "Private C"]
    return frame


class RoroEwShadowTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / "source.db"
        self.target = self.root / "shadow" / "roro_ew.db"
        with closing(sqlite3.connect(self.source)) as connection:
            connection.execute("CREATE TABLE sentinel(value TEXT)")
            connection.execute("INSERT INTO sentinel VALUES('SOURCE')")
            connection.commit()

    def _build(self, scores: pd.DataFrame | None = None, **kwargs: object) -> dict:
        panel = _panel()
        with (
            patch("roro_ew_shadow.load_roro_panel", return_value=panel),
            patch(
                "roro_ew_shadow.build_roro_scores",
                return_value=scores if scores is not None else _scores(),
            ),
        ):
            return build_ew_shadow_series(
                self.source, self.target, as_of="2026-01-06", **kwargs
            )

    def test_keeps_no_score_diagnostics_and_filters_identity(self) -> None:
        before = _digest(self.source)
        result = self._build()
        history = read_ew_shadow_history(self.target)

        self.assertEqual(_digest(self.source), before)
        self.assertEqual(result["inserted"], 3)
        self.assertEqual(history[0]["date"], "2026-01-02")
        blocked = history[0]["payload"]
        self.assertEqual(blocked["data_status"], "DATA_BLOCKED")
        self.assertIsNone(blocked["risk_score"])
        for column in (*GLOBAL_PAYLOAD_COLUMNS, *BLOCK_PAYLOAD_COLUMNS):
            self.assertIn(column, blocked)
        self.assertFalse(blocked["block_availability"]["credit"])
        self.assertEqual(blocked["available_blocks"], [])
        self.assertFalse(blocked["input_availability"]["gold"])
        partial = history[1]["payload"]
        self.assertEqual(partial["data_status"], "NO_SCORE")
        self.assertIsNone(partial["risk_score"])
        self.assertEqual(partial["credit_score"], 40.0)
        self.assertEqual(partial["credit_trajectory_20d"], -1.5)
        self.assertIsNone(partial["credit_contribution"])
        self.assertEqual(len(partial["available_blocks"]), 4)
        encoded = json.dumps([row["payload"] for row in history])
        self.assertNotIn("SECRET", encoded)
        self.assertNotIn("holding_name", encoded)

    def test_identity_free_schema_and_read_is_non_mutating(self) -> None:
        self._build()
        before = _digest(self.target)
        with closing(sqlite3.connect(self.target)) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            columns = {
                row[1] for row in connection.execute(f"PRAGMA table_info({TABLE})")
            }
        latest = latest_ew_shadow(self.target)
        self.assertEqual(_digest(self.target), before)
        self.assertEqual(tables, {TABLE})
        self.assertFalse(columns & {"ticker", "security", "holding_name", "quantity"})
        self.assertEqual(latest["roro_model_version"], RORO_MODEL_VERSION)
        self.assertEqual(len(latest["provenance_hash"]), 64)

    def test_idempotence_conflict_and_atomic_rollback(self) -> None:
        first = self._build()
        created_at = latest_ew_shadow(self.target)["created_at"]
        second = self._build()
        self.assertEqual(first["inserted"], 3)
        self.assertEqual(second["idempotent"], 3)
        self.assertEqual(latest_ew_shadow(self.target)["created_at"], created_at)

        base = _scores().loc[[pd.Timestamp("2026-01-06")]]
        rollback_target = self.root / "rollback.db"
        persist_ew_shadow(
            rollback_target, _payloads(base, _panel()), as_of="2026-01-06"
        )
        changed = _scores(last_credit=61.0).loc[
            [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")]
        ]
        with self.assertRaisesRegex(ImmutableSeriesConflict, "2026-01-06"):
            persist_ew_shadow(
                rollback_target,
                _payloads(changed, _panel()),
                as_of="2026-01-06",
            )
        self.assertEqual(
            [row["date"] for row in read_ew_shadow_history(rollback_target)],
            ["2026-01-06"],
        )

    def test_source_alias_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "separate"):
            build_ew_shadow_series(self.source, self.source, as_of="2026-01-06")

    def test_target_is_dedicated_and_stored_rows_are_tamper_evident(self) -> None:
        operating = self.root / "other_operating.db"
        with closing(sqlite3.connect(operating)) as connection:
            connection.execute("CREATE TABLE sentinel(value TEXT)")
            connection.commit()
        with self.assertRaisesRegex(ValueError, "only its aggregate table"):
            persist_ew_shadow(
                operating, _payloads(_scores(), _panel()), as_of="2026-01-06"
            )

        self._build()
        with closing(sqlite3.connect(self.target)) as connection:
            triggers = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='trigger'"
                )
            }
            self.assertEqual(
                triggers, {f"{TABLE}_no_update", f"{TABLE}_no_delete"}
            )
            with self.assertRaises(sqlite3.DatabaseError):
                connection.execute(
                    f"UPDATE {TABLE} SET input_cutoff='2099-01-01'"
                )
            connection.rollback()
            connection.execute(f"DROP TRIGGER {TABLE}_no_update")
            payload = json.loads(
                connection.execute(
                    f"SELECT payload_json FROM {TABLE} ORDER BY date LIMIT 1"
                ).fetchone()[0]
            )
            payload["data_status"] = "TAMPERED"
            connection.execute(
                f"UPDATE {TABLE} SET payload_json=? WHERE date='2026-01-02'",
                (json.dumps(payload, sort_keys=True, separators=(",", ":")),),
            )
            connection.commit()
        with self.assertRaisesRegex(ValueError, "provenance hash mismatch"):
            read_ew_shadow_history(self.target)

    def test_latest_only_and_cli(self) -> None:
        result = self._build(latest_only=True)
        self.assertEqual(result["rows_seen"], 1)
        self.assertEqual(
            [row["date"] for row in read_ew_shadow_history(self.target)],
            ["2026-01-06"],
        )

        expected = {"series_version": SERIES_VERSION, "inserted": 1}
        output = io.StringIO()
        with (
            patch("roro_ew_shadow.build_ew_shadow_series", return_value=expected) as build,
            redirect_stdout(output),
        ):
            main(
                [
                    "build",
                    str(self.source),
                    str(self.target),
                    "--as-of",
                    "2026-01-06",
                    "--latest-only",
                ]
            )
        build.assert_called_once_with(
            str(self.source),
            str(self.target),
            as_of="2026-01-06",
            latest_only=True,
        )
        self.assertEqual(json.loads(output.getvalue()), expected)


if __name__ == "__main__":
    unittest.main()
