import unittest
import json
import tempfile
import sqlite3
from contextlib import closing
from pathlib import Path
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd

from it_internals_shadow import (
    SECTOR_ETFS,
    STATES,
    build_it_daily_frame,
    build_it_daily_history,
    build_history,
    refresh_prices,
    classify_it_state,
    web_payload,
    load_it_price_matrix,
)


class ItInternalsShadowTest(unittest.TestCase):
    def test_requested_close_is_required_before_writing(self) -> None:
        target = date(2026, 9, 9)
        with (
            patch("it_internals_shadow.register_universe", return_value=3),
            patch("it_internals_shadow.download_prices", return_value=(3, 3, "2026-09-02")) as download,
        ):
            refresh_prices(Path("unused.db"), require_date=target)
        self.assertEqual(download.call_args.kwargs["require_date"], target)
        prices = pd.DataFrame(100.0, index=pd.bdate_range(end="2026-09-08", periods=100), columns=["SOXX", "IGV", "XLK"])
        with tempfile.TemporaryDirectory() as directory, patch("it_internals_shadow.load_it_price_matrix", return_value=prices):
            output = Path(directory) / "history.csv"
            with self.assertRaisesRegex(ValueError, "stale"):
                build_history(Path("unused.db"), output, Path(directory) / "metadata.json", as_of=target.isoformat())
            self.assertFalse(output.exists())

    def test_states_daily_history_and_forward_actuals(self) -> None:
        index = pd.RangeIndex(4)
        states = classify_it_state(
            pd.Series([0.1, -0.1, 0.1, -0.1], index=index),
            pd.Series([0.1, -0.1, -0.1, 0.1], index=index),
        )
        self.assertEqual(
            states.tolist(),
            [
                "BROAD_STRENGTH",
                "BROAD_WEAKNESS",
                "SEMICONDUCTOR_LEAD",
                "SOFTWARE_LEAD",
            ],
        )
        self.assertEqual(set(states), set(STATES))

        dates = pd.bdate_range("2025-01-06", periods=150)
        step = np.arange(len(dates), dtype=float)
        prices = pd.DataFrame(
            {
                "SOXX": 100 * np.exp(0.0012 * step),
                "IGV": 100 * np.exp(0.0002 * step),
                "XLK": 100 * np.exp(0.0006 * step),
            },
            index=dates,
        )
        daily = build_it_daily_frame(prices)
        cutoff = dates[-30]
        history = build_it_daily_history(daily, as_of=cutoff)

        self.assertEqual(len(history), dates.get_loc(cutoff) - 60 + 1)
        self.assertEqual(history["decision_date"].max(), cutoff)
        self.assertEqual(history.iloc[0]["state_transition"], "N/A")
        self.assertEqual(history.iloc[1]["state_transition"], "—")
        self.assertEqual(set(history["state_20d"]), {"SEMICONDUCTOR_LEAD"})
        self.assertGreater(history["soxx_minus_igv_20d"].min(), 0)
        labelled = history.loc[history["actual_available"]]
        self.assertTrue((labelled["pair_forward_actual_20d"] > 0).all())
        self.assertEqual(set(labelled["pair_actual_winner"]), {"SEMICONDUCTOR"})
        self.assertFalse(bool(history.iloc[-1]["actual_available"]))
        self.assertTrue(pd.isna(history.iloc[-1]["pair_forward_actual_20d"]))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "history.csv"
            metadata_path = root / "metadata.json"
            history.to_csv(csv_path, index=False, date_format="%Y-%m-%d")
            metadata_path.write_text(
                json.dumps({"generated_at_utc": "2026-09-02T00:00:00+00:00"}),
                encoding="utf-8",
            )
            payload = web_payload(csv_path, metadata_path)
            database = root / "prices.db"
            with closing(sqlite3.connect(database)) as connection:
                pd.DataFrame([
                    {"ticker": ticker, "date": day.date().isoformat(), "adj_close": 100 + i * scale}
                    for i, day in enumerate(dates)
                    for ticker, scale in [(ticker, 0.1 * (j + 1)) for j, ticker in enumerate([*SECTOR_ETFS, "SPY"])]
                ]).to_sql("daily_prices", connection, index=False)
            enriched = web_payload(csv_path, metadata_path, sector_database=database)
            self.assertEqual(enriched["daily_pairs"]["comm_energy"][-1][0], cutoff.date().isoformat())
            self.assertEqual(len(enriched["daily_pairs"]["hw_sw"]), len(history))
            self.assertEqual(enriched["sector_prices"]["tickers"], list(SECTOR_ETFS))
            self.assertEqual(len(enriched["sector_prices"]["rows"][-1]), 13)
            self.assertEqual(enriched["sector_prices"]["rows"][-1][0], cutoff.date().isoformat())
            self.assertLess(len(json.dumps(enriched, separators=(",", ":"))), 1_250_000)
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("DELETE FROM daily_prices WHERE ticker='XLE' AND date=?", (cutoff.date().isoformat(),))
                connection.commit()
            with self.assertRaisesRegex(ValueError, "stale"):
                web_payload(csv_path, metadata_path, sector_database=database)
            missing = root / "missing.db"
            with self.assertRaises(sqlite3.OperationalError):
                load_it_price_matrix(missing)
            self.assertFalse(missing.exists())
        self.assertEqual(payload["benchmark"], "XLK")
        self.assertEqual(payload["cadence"], "DAILY")
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["proxies"], {"semiconductor": "SOXX", "software": "IGV"})
        self.assertEqual(payload["rows"][-1][-1], None)


if __name__ == "__main__":
    unittest.main()
