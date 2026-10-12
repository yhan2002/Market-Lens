from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from research_drawdown_onset import drawdown_onset_labels
from turning_macro_shadow_experiment import (
    EXPECTED_MACRO_SERIES,
    PIT_STATUS,
    SOURCE,
    build_event_frame,
    build_macro_feature_sets,
    ensure_output_boundary,
    load_macro_history,
    merge_macro_next_session,
)


class TurningMacroShadowExperimentTest(unittest.TestCase):
    def test_macro_is_available_strictly_next_session_and_future_mutation_is_causal(self) -> None:
        macro = pd.DataFrame(
            {"feature": [1.0, 2.0]},
            index=pd.to_datetime(["2026-01-05", "2026-01-06"]),
        )
        sessions = pd.DatetimeIndex(pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]))
        original = merge_macro_next_session(
            sessions, macro, ["feature"], asof_column="macro_date"
        )
        self.assertTrue(pd.isna(original.loc[pd.Timestamp("2026-01-05"), "feature"]))
        self.assertEqual(original.loc[pd.Timestamp("2026-01-06"), "feature"], 1.0)
        self.assertEqual(original.loc[pd.Timestamp("2026-01-07"), "feature"], 2.0)

        changed = macro.copy()
        changed.loc[pd.Timestamp("2026-01-06"), "feature"] = 999.0
        mutated = merge_macro_next_session(
            sessions, changed, ["feature"], asof_column="macro_date"
        )
        assert_frame_equal(original.iloc[:2], mutated.iloc[:2])
        self.assertEqual(mutated.loc[pd.Timestamp("2026-01-07"), "feature"], 999.0)

    def test_rotation_in_labels_are_symmetric_reciprocal_onsets(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=330)
        relative = pd.Series(
            np.r_[np.linspace(100, 130, 280), np.linspace(130, 118, 50)],
            index=dates,
        )
        rotation_out = build_event_frame(relative, "ROTATION_OUT")
        rotation_in = build_event_frame(relative, "ROTATION_IN")
        expected_out = drawdown_onset_labels(relative)
        expected_in = drawdown_onset_labels(1.0 / relative)
        np.testing.assert_array_equal(rotation_out["onset"], expected_out["onset"])
        np.testing.assert_array_equal(rotation_in["onset"], expected_in["onset"])

    def test_negative_wti_is_preserved_without_log_transform(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "macro_history.csv"
            dates = pd.bdate_range("2020-03-02", periods=30)
            rows = []
            for series_index, series in enumerate(sorted(EXPECTED_MACRO_SERIES)):
                for position, date in enumerate(dates):
                    value = 10.0 + series_index + position
                    if series == "WTI_SPOT" and position == 20:
                        value = -37.63
                    rows.append(
                        {
                            "series": series,
                            "fred_series_id": "TEST",
                            "security": "TEST Index",
                            "bloomberg_field": "PX_LAST",
                            "date": date.strftime("%Y-%m-%d"),
                            "value": value,
                            "unit": "%" if "UST" in series or "BREAKEVEN" in series else "USD/bbl",
                            "currency": "USD",
                            "exact_or_proxy": "PROXY",
                            "pit_status": PIT_STATUS,
                            "retrieved_at_utc": "2026-09-02T00:00:00+00:00",
                            "source": SOURCE,
                        }
                    )
            pd.DataFrame(rows).to_csv(path, index=False)
            history = load_macro_history(path)
            features, _, wti_columns = build_macro_feature_sets(history)
            event_date = dates[20]
            wide = history.pivot(index="date", columns="series", values="value")
            self.assertAlmostEqual(
                features.loc[event_date, "curve_2s10s_level"],
                wide.loc[event_date, "UST_10Y"] - wide.loc[event_date, "UST_2Y"],
            )
            self.assertEqual(features.loc[event_date, "wti_spot_level"], -37.63)
            self.assertTrue(np.isfinite(features.loc[event_date, "wti_spot_change_1"]))
            self.assertIn("wti_spot_change_20", wti_columns)

    def test_output_must_be_outside_both_input_packages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gics = root / "gics_input"
            macro = root / "macro_input"
            gics.mkdir()
            macro.mkdir()
            with self.assertRaises(ValueError):
                ensure_output_boundary(gics / "results", gics, macro)
            with self.assertRaises(ValueError):
                ensure_output_boundary(macro, gics, macro)
            ensure_output_boundary(root / "results", gics, macro)


if __name__ == "__main__":
    unittest.main()
