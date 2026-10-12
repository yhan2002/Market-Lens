from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from research_drawdown_onset import causal_trailing_percentile
from sector_turning_pooled_shadow import (
    _direction_frame,
    _early_event_metrics,
    _labels_with_calendar_gaps,
    _wilson_interval,
    directional_change_labels,
    effect_code,
    effect_deviations,
    paired_date_block_auc_delta_ci,
    paired_date_block_pre_pivot_recall_delta_ci,
    append_prospective_signals,
    build_prospective_outcomes,
    code_bundle_sha256,
    ensure_frozen_manifest,
    ensure_prospective_input_sessions,
    gics_input_fingerprints,
    prospective_session_fingerprints,
    purged_training_mask,
    shadow_signal_level,
    summarize_prospective_outcomes,
    validate_expected_latest_date,
    walk_forward_direction,
)

RUN_MANIFEST_SHA256 = "A" * 64
INPUT_SESSION_SHA256 = "C" * 64


def synthetic_turns() -> pd.Series:
    returns = (
        [0.001] * 75
        + [-0.006] * 8
        + [-0.002] * 24
        + [0.004] * 3
        + [-0.003] * 12
        + [0.011] * 4
        + [0.002] * 15
        + [-0.012] * 4
        + [-0.002] * 15
    )
    dates = pd.bdate_range("2020-01-01", periods=len(returns) + 1)
    return pd.Series(100 * np.exp(np.r_[0.0, np.cumsum(returns)]), index=dates)


class DirectionalChangeTests(unittest.TestCase):
    def test_score_calibration_is_strictly_causal(self) -> None:
        history = pd.Series([1.0, 2.0, 3.0, 4.0])
        calibrated = causal_trailing_percentile(history, window=3, min_history=2)
        self.assertTrue(calibrated.iloc[:2].isna().all())
        self.assertEqual(calibrated.iloc[2], 1.0)

        extended = causal_trailing_percentile(
            pd.concat([history, pd.Series([-999.0, 999.0])], ignore_index=True),
            window=3,
            min_history=2,
        )
        pd.testing.assert_series_equal(calibrated, extended.iloc[: len(history)])

    def test_code_bundle_fingerprint_covers_runtime_sources(self) -> None:
        fingerprint = code_bundle_sha256()
        self.assertEqual(len(fingerprint), 64)
        self.assertEqual(fingerprint, fingerprint.upper())

    def test_gics_fingerprint_ignores_future_append_but_detects_history_revision(self) -> None:
        master = pd.DataFrame(
            {
                "industry_group_code": ["1010", "1510"],
                "industry_group_name": ["Energy", "Materials"],
                "parent_sector_code": ["10", "15"],
                "group_security": ["A Index", "B Index"],
                "group_bbgid": ["A", "B"],
                "parent_security": ["PA Index", "PB Index"],
                "parent_bbgid": ["PA", "PB"],
            }
        )
        groups = pd.DataFrame(
            {
                "industry_group_code": ["1010", "1510", "1010", "1510"],
                "date": pd.to_datetime(
                    ["2026-08-31", "2026-08-31", "2026-09-01", "2026-09-01"]
                ),
                "total_return_level": [100.0, 200.0, 101.0, 202.0],
            }
        )
        benchmark = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-08-31", "2026-09-01"]),
                "total_return_level": [300.0, 303.0],
            }
        )
        original = gics_input_fingerprints(master, groups, benchmark, ["1010", "1510"])
        future_changed = groups.copy()
        future_changed.loc[future_changed["date"].eq("2026-09-01"), "total_return_level"] *= 2
        self.assertEqual(
            original,
            gics_input_fingerprints(master, future_changed, benchmark, ["1510", "1010"]),
        )
        history_changed = groups.copy()
        history_changed.loc[0, "total_return_level"] += 0.01
        self.assertNotEqual(
            original[1],
            gics_input_fingerprints(master, history_changed, benchmark, ["1010", "1510"])[1],
        )

    def test_frozen_manifest_is_immutable_per_model_version(self) -> None:
        manifest = {
            "model_version": "v1",
            "code_bundle_sha256": "A" * 64,
            "model_spec": {"horizon": 10},
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "frozen_manifest.json"
            first = ensure_frozen_manifest(path, manifest)
            self.assertEqual(first, ensure_frozen_manifest(path, manifest))
            with self.assertRaisesRegex(RuntimeError, "without a model version bump"):
                ensure_frozen_manifest(path, {**manifest, "code_bundle_sha256": "B" * 64})
            second = ensure_frozen_manifest(path, {**manifest, "model_version": "v2"})
            self.assertNotEqual(first, second)

    def test_prospective_input_ledger_detects_revisions(self) -> None:
        prices = pd.DataFrame(
            {
                "SPX": [100.0, 101.0],
                "1010": [50.0, 51.0],
                "1510": [25.0, np.nan],
            },
            index=pd.to_datetime(["2026-09-01", "2026-09-02"]),
        )
        sessions = prospective_session_fingerprints(prices, ["1010", "1510"])
        self.assertEqual(sessions["present_group_count"].tolist(), [2, 1])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sessions.csv"
            self.assertEqual(ensure_prospective_input_sessions(path, sessions), 2)
            self.assertEqual(ensure_prospective_input_sessions(path, sessions), 0)
            revised = sessions.copy()
            revised.loc[0, "input_session_sha256"] = "D" * 64
            with self.assertRaisesRegex(RuntimeError, "input was revised"):
                ensure_prospective_input_sessions(path, revised)
            extension = pd.DataFrame(
                {"SPX": [102.0], "1010": [52.0], "1510": [26.0]},
                index=pd.to_datetime(["2026-09-03"]),
            )
            extended = prospective_session_fingerprints(
                pd.concat([prices, extension]), ["1010", "1510"]
            )
            self.assertEqual(ensure_prospective_input_sessions(path, extended), 1)

    def test_labels_are_symmetric_and_cooldown_is_enforced(self) -> None:
        price = synthetic_turns()
        labels = directional_change_labels(price)
        inverse = directional_change_labels(1.0 / price)
        self.assertEqual(labels["top_end"].tolist(), inverse["bottom_end"].tolist())
        self.assertEqual(labels["bottom_end"].tolist(), inverse["top_end"].tolist())
        mapping = {"UNKNOWN": "UNKNOWN", "UP_ACTIVE": "DOWN_ACTIVE", "DOWN_ACTIVE": "UP_ACTIVE"}
        self.assertEqual(labels["state"].map(mapping).tolist(), inverse["state"].tolist())
        for event_position in np.flatnonzero(labels["event"].ne("").to_numpy()):
            cooldown = labels.iloc[event_position + 1 : event_position + 6]
            self.assertEqual(len(cooldown), 5)
            self.assertFalse(cooldown[["top_at_risk", "bottom_at_risk"]].to_numpy().any())
        active = labels["state"].ne("UNKNOWN") & labels["cooldown_remaining"].eq(0)
        self.assertTrue(labels.loc[active & labels["state"].eq("UP_ACTIVE"), "top_at_risk"].all())
        self.assertTrue(labels.loc[active & labels["state"].eq("DOWN_ACTIVE"), "bottom_at_risk"].all())
        self.assertTrue(labels.loc[labels["event"].ne(""), "pivot_date"].notna().all())
        self.assertTrue(labels.loc[labels["event"].ne(""), "confirmation_date"].notna().all())
        self.assertTrue(set(labels["state"]).issubset({"UNKNOWN", "UP_ACTIVE", "DOWN_ACTIVE"}))

    def test_bottom_target_has_both_classes(self) -> None:
        labels = directional_change_labels(synthetic_turns())
        target = labels.loc[labels["bottom_at_risk"], "bottom_target"].dropna()
        self.assertEqual(set(target), {0.0, 1.0})
        self.assertGreater(int(labels["bottom_end"].sum()), 0)

    def test_unknown_requires_a_full_barrier_move(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=75)
        price = pd.Series(100 * np.exp(np.arange(75) * 0.001), index=dates)
        labels = directional_change_labels(price)
        self.assertTrue(labels["state"].eq("UNKNOWN").all())

    def test_historical_labels_do_not_change_when_future_prices_change(self) -> None:
        price = synthetic_turns()
        cutoff = 120
        changed = price.copy()
        changed.iloc[cutoff:] = changed.iloc[cutoff] * np.exp(
            np.linspace(0.0, 1.0, len(changed) - cutoff)
        )
        original = directional_change_labels(price)
        revised = directional_change_labels(changed)
        columns = ["state", "pivot_date", "barrier", "event", "top_at_risk", "bottom_at_risk"]
        pd.testing.assert_frame_equal(original.iloc[:cutoff][columns], revised.iloc[:cutoff][columns])
        pd.testing.assert_frame_equal(
            original.iloc[: cutoff - 10][["top_target", "bottom_target"]],
            revised.iloc[: cutoff - 10][["top_target", "bottom_target"]],
        )

    def test_effect_coding_and_deviations_sum_to_zero(self) -> None:
        sectors = pd.Series(["A", "B", "C", "A", "C"])
        encoded, levels = effect_code(sectors)
        self.assertEqual(encoded.shape[1], 3)
        self.assertAlmostEqual(float(encoded.loc[2].sum()), 0.0)
        deviations = effect_deviations([0.4, -0.1, 0.2], levels)
        self.assertAlmostEqual(sum(deviations.values()), 0.0)

    def test_centered_effect_fit_is_invariant_to_level_order(self) -> None:
        sectors = pd.Series(list("ABC") * 20)
        target = np.tile([0, 1, 0, 1, 1, 0], 10)
        first, _ = effect_code(sectors, ["A", "B", "C"])
        second, _ = effect_code(sectors, ["C", "A", "B"])
        model_a = LogisticRegression(C=0.1, random_state=42).fit(first, target)
        model_b = LogisticRegression(C=0.1, random_state=42).fit(second, target)
        np.testing.assert_allclose(
            model_a.predict_proba(first)[:, 1],
            model_b.predict_proba(second)[:, 1],
            atol=1e-10,
        )

    def test_gap_keeps_canonical_session_positions(self) -> None:
        price = synthetic_turns()
        price.iloc[100] = np.nan
        labels = _labels_with_calendar_gaps(price, horizon=10)
        self.assertEqual(labels.iloc[100]["state"], "UNKNOWN")
        self.assertFalse(bool(labels.iloc[100]["top_at_risk"]))
        known_ends = labels["top_label_end_position"].dropna()
        for date, position in known_ends.items():
            self.assertEqual(price.index[int(position)], labels.loc[date, "top_label_end_date"])

    def test_purge_excludes_unfinished_labels(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=5)
        frame = pd.DataFrame(
            {
                "date": dates,
                "at_risk": True,
                "target": [0.0, 1.0, 0.0, 1.0, np.nan],
                "label_end_date": [dates[1], dates[2], dates[3], dates[4], pd.NaT],
            }
        )
        mask = purged_training_mask(frame, dates[3])
        self.assertEqual(mask.tolist(), [True, True, False, False, False])

    def test_pre_pivot_target_excludes_pivot_and_confirmation_rows(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=5)
        frame = pd.DataFrame(
            {
                "date": dates,
                "_position": np.arange(5),
                "sector": "A",
                "bottom_at_risk": True,
                "bottom_target": [1.0, 1.0, 1.0, 1.0, np.nan],
                "bottom_label_end_position": [4.0, 4.0, 4.0, 4.0, np.nan],
                "bottom_label_end_date": [dates[4], dates[4], dates[4], dates[4], pd.NaT],
                "event": ["", "", "", "", "BOTTOM_END"],
                "pivot_position": [np.nan, np.nan, np.nan, np.nan, 2.0],
                "pivot_date": [pd.NaT, pd.NaT, pd.NaT, pd.NaT, dates[2]],
                "confirmation_position": [np.nan, np.nan, np.nan, np.nan, 4.0],
                "confirmation_date": [pd.NaT, pd.NaT, pd.NaT, pd.NaT, dates[4]],
            }
        )
        result = _direction_frame(frame, "ROTATION_IN", target_mode="PRE_PIVOT")
        self.assertEqual(result["target"].iloc[:4].tolist(), [1.0, 1.0, 0.0, 0.0])
        self.assertTrue(pd.isna(result["target"].iloc[4]))
        self.assertTrue((result.loc[:3, "label_end_date"] == dates[4]).all())

    def test_paired_ci_resamples_calendar_dates(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=60)
        index = pd.MultiIndex.from_product([dates, ["A", "B", "C"]], names=["date", "sector"])
        target = pd.Series(np.tile([0, 1, 0], 60), index=index)
        result = paired_date_block_auc_delta_ci(
            target, target.astype(float), 1.0 - target.astype(float), samples=100, block_dates=20
        )
        self.assertAlmostEqual(result["delta_auc"], 1.0)
        self.assertGreater(result["ci_low"], 0.0)

    def test_short_sample_cannot_create_zero_width_confidence_interval(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=40)
        index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "sector"])
        target = pd.Series(np.tile([0, 1], 40), index=index)
        result = paired_date_block_auc_delta_ci(
            target, target.astype(float), 1.0 - target.astype(float), samples=20
        )
        self.assertTrue(np.isnan(result["ci_low"]))

    def test_auc_ci_requires_positive_dates_not_only_positive_rows(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=80)
        index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "sector"])
        target = pd.Series(0, index=index)
        target.loc[(dates[0], "A")] = 1
        result = paired_date_block_auc_delta_ci(
            target, target.astype(float), 1.0 - target.astype(float), samples=100
        )
        self.assertEqual(result["positive_dates"], 1)
        self.assertTrue(np.isnan(result["ci_low"]))

    def test_primary_ci_is_paired_pre_pivot_capture_uplift(self) -> None:
        calendar = pd.bdate_range("2020-01-01", periods=200)
        records = []
        for number, confirmation in enumerate(range(12, 172, 4)):
            records.append(
                {
                    "date": calendar[confirmation - 3],
                    "sector": f"S{number % 4}",
                    "_position": confirmation - 3,
                    "target": 1.0,
                    "target_event_pivot_position": confirmation - 2,
                    "target_event_confirmation_position": confirmation,
                    "target_event_confirmation_date": calendar[confirmation],
                }
            )
        candidate = pd.DataFrame(records).assign(score=0.9)
        baseline = pd.DataFrame(records).assign(score=0.1)
        result = paired_date_block_pre_pivot_recall_delta_ci(
            candidate, baseline, samples=100, calendar_dates=calendar
        )
        self.assertAlmostEqual(result["delta_pre_pivot_recall"], 1.0)
        self.assertGreater(result["early_capture_ci_low"], 0.0)

    def test_one_continuous_alert_episode_maps_to_only_one_event(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=30)
        positions = np.arange(30)
        first = (positions >= 2) & (positions <= 11)
        second = (positions >= 13) & (positions <= 21)
        scores = np.full(30, 0.1)
        scores[(positions >= 5) & (positions <= 11)] = 0.9
        scores[(positions >= 15) & (positions <= 21)] = 0.9
        confirmation_dates = pd.Series(pd.NaT, index=np.arange(30), dtype="datetime64[ns]")
        confirmation_dates.loc[first] = dates[12]
        confirmation_dates.loc[second] = dates[22]
        rows = pd.DataFrame(
            {
                "date": dates,
                "sector": "A",
                "_position": positions,
                "score": scores,
                "target": (first | second).astype(float),
                "target_event_pivot_position": np.where(first, 10, np.where(second, 20, np.nan)),
                "target_event_confirmation_position": np.where(
                    first, 12, np.where(second, 22, np.nan)
                ),
                "target_event_confirmation_date": confirmation_dates,
            }
        )
        metrics = _early_event_metrics(rows, pd.DataFrame(), "ROTATION_OUT", horizon=10)
        self.assertEqual(metrics["eligible_confirmations"], 2)
        self.assertEqual(metrics["event_hits"], 1)
        self.assertEqual(metrics["matched_alert_clusters"], 1)

    def test_alert_episode_straddling_target_window_is_not_split_false(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=20)
        positions = np.arange(20)
        linked = (positions >= 2) & (positions <= 11)
        confirmation_dates = pd.Series(pd.NaT, index=np.arange(20), dtype="datetime64[ns]")
        confirmation_dates.loc[linked] = dates[12]
        rows = pd.DataFrame(
            {
                "date": dates,
                "sector": "A",
                "_position": positions,
                "score": np.where((positions >= 1) & (positions <= 9), 0.9, 0.1),
                "target": linked.astype(float),
                "target_event_pivot_position": np.where(linked, 10, np.nan),
                "target_event_confirmation_position": np.where(linked, 12, np.nan),
                "target_event_confirmation_date": confirmation_dates,
            }
        )
        metrics = _early_event_metrics(rows, pd.DataFrame(), "ROTATION_OUT", horizon=10)
        self.assertEqual(metrics["alert_clusters"], 1)
        self.assertEqual(metrics["matched_alert_clusters"], 1)
        self.assertEqual(metrics["false_alert_clusters"], 0)

    def test_event_denominator_does_not_depend_on_early_target(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=5)
        rows = pd.DataFrame(
            {
                "date": dates[:3],
                "sector": "A",
                "_position": [0, 1, 2],
                "score": [0.1, 0.1, 0.1],
                "target": [0.0, 0.0, 0.0],
                "target_event_pivot_position": [2.0, 2.0, np.nan],
                "target_event_confirmation_position": [4.0, 4.0, np.nan],
                "target_event_confirmation_date": [dates[4], dates[4], pd.NaT],
            }
        )
        metrics = _early_event_metrics(rows, pd.DataFrame(), "ROTATION_IN", horizon=10)
        self.assertEqual(metrics["eligible_confirmations"], 1)
        self.assertEqual(metrics["pre_pivot_hits"], 0)

    def test_unmatured_latest_rows_are_scored_but_not_trained(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=180)
        parts = []
        for sector, offset in (("A", 0.0), ("B", 0.2)):
            part = pd.DataFrame(
                {
                    "date": dates,
                    "_position": np.arange(len(dates)),
                    "sector": sector,
                    "sector_name": sector,
                    "state": "DOWN_ACTIVE",
                    "pivot_position": np.nan,
                    "pivot_date": pd.NaT,
                    "confirmation_position": np.nan,
                    "confirmation_date": pd.NaT,
                    "barrier": 0.05,
                    "cooldown_remaining": 0,
                    "event": "",
                    "at_risk": True,
                    "target": np.where(np.arange(len(dates)) < 170, np.arange(len(dates)) % 2, np.nan),
                    "label_end_position": np.where(
                        np.arange(len(dates)) < 170, np.arange(len(dates)) + 10, np.nan
                    ),
                    "label_end_date": [
                        dates[position + 10] if position < 170 else pd.NaT
                        for position in range(len(dates))
                    ],
                    "target_event_pivot_position": np.nan,
                    "target_event_pivot_date": pd.NaT,
                    "target_event_confirmation_position": np.nan,
                    "target_event_confirmation_date": pd.NaT,
                    "macro_asof_date": dates - pd.offsets.BDay(1),
                }
            )
            for number, feature in enumerate(
                (
                    "relative_return_5",
                    "relative_return_20",
                    "relative_volatility_20",
                    "relative_ma200_gap",
                    "relative_drawdown_60",
                    "relative_acceleration_5",
                    "curve_2s10s_change_20",
                    "real_rate_10y_change_20",
                    "breakeven_10y_change_20",
                    "brent_spot_change_20",
                )
            ):
                part[feature] = np.sin(np.arange(len(dates)) / (number + 3)) + offset
            parts.append(part)
        frame = pd.concat(parts, ignore_index=True)
        with patch("sector_turning_pooled_shadow._direction_frame", return_value=frame):
            predictions, _, walk = walk_forward_direction(
                pd.DataFrame(),
                "ROTATION_IN",
                min_train_dates=60,
                refit_every=30,
                target_mode="PRE_PIVOT",
            )
        latest = predictions.loc[predictions["date"].eq(dates[-1])]
        self.assertEqual(len(latest), 6)
        self.assertTrue(latest["target"].isna().all())
        self.assertEqual(walk["scorable_rows"], 360)
        self.assertEqual(walk["eligible_rows"], 340)

    def test_price_only_scoring_does_not_require_macro_columns(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=180)
        parts = []
        for sector, offset in (("A", 0.0), ("B", 0.2)):
            part = pd.DataFrame(
                {
                    "date": dates,
                    "_position": np.arange(len(dates)),
                    "sector": sector,
                    "sector_name": sector,
                    "state": "DOWN_ACTIVE",
                    "pivot_position": np.nan,
                    "pivot_date": pd.NaT,
                    "confirmation_position": np.nan,
                    "confirmation_date": pd.NaT,
                    "barrier": 0.05,
                    "cooldown_remaining": 0,
                    "event": "",
                    "at_risk": True,
                    "target": np.where(np.arange(len(dates)) < 170, np.arange(len(dates)) % 2, np.nan),
                    "label_end_position": np.where(
                        np.arange(len(dates)) < 170, np.arange(len(dates)) + 10, np.nan
                    ),
                    "label_end_date": [
                        dates[position + 10] if position < 170 else pd.NaT
                        for position in range(len(dates))
                    ],
                    "target_event_pivot_position": np.nan,
                    "target_event_pivot_date": pd.NaT,
                    "target_event_confirmation_position": np.nan,
                    "target_event_confirmation_date": pd.NaT,
                    "macro_asof_date": pd.NaT,
                }
            )
            for number, feature in enumerate(
                (
                    "relative_return_5",
                    "relative_return_20",
                    "relative_volatility_20",
                    "relative_ma200_gap",
                    "relative_drawdown_60",
                    "relative_acceleration_5",
                )
            ):
                part[feature] = np.sin(np.arange(len(dates)) / (number + 3)) + offset
            parts.append(part)
        frame = pd.concat(parts, ignore_index=True)
        with patch("sector_turning_pooled_shadow._direction_frame", return_value=frame):
            predictions, _, walk = walk_forward_direction(
                pd.DataFrame(),
                "ROTATION_IN",
                min_train_dates=60,
                refit_every=30,
                target_mode="PRE_PIVOT",
                models=("POOLED_PRICE",),
            )
        latest = predictions.loc[predictions["date"].eq(dates[-1])]
        self.assertEqual(len(latest), 2)
        self.assertEqual(set(latest["model"]), {"POOLED_PRICE"})
        self.assertEqual(walk["scorable_rows"], 360)

    def test_prospective_log_is_idempotent_and_rejects_revisions(self) -> None:
        latest = pd.DataFrame(
            {
                "direction": ["ROTATION_IN"],
                "date": ["2026-09-03"],
                "sector": ["4510"],
                "raw_score": [0.8],
                "score": [0.95],
            }
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "prospective.csv"
            self.assertEqual(
                append_prospective_signals(
                    path,
                    latest,
                    run_manifest_sha256=RUN_MANIFEST_SHA256,
                    input_session_sha256=INPUT_SESSION_SHA256,
                ),
                1,
            )
            self.assertEqual(
                append_prospective_signals(
                    path,
                    latest,
                    run_manifest_sha256=RUN_MANIFEST_SHA256,
                    input_session_sha256=INPUT_SESSION_SHA256,
                ),
                0,
            )
            changed = latest.assign(score=0.94)
            with self.assertRaisesRegex(RuntimeError, "immutable prospective signal conflict"):
                append_prospective_signals(
                    path,
                    changed,
                    run_manifest_sha256=RUN_MANIFEST_SHA256,
                    input_session_sha256=INPUT_SESSION_SHA256,
                )
            with self.assertRaisesRegex(RuntimeError, "immutable prospective signal conflict"):
                append_prospective_signals(
                    path,
                    latest,
                    run_manifest_sha256="B" * 64,
                    input_session_sha256=INPUT_SESSION_SHA256,
                )
            with self.assertRaisesRegex(RuntimeError, "immutable prospective signal conflict"):
                append_prospective_signals(
                    path,
                    latest,
                    run_manifest_sha256=RUN_MANIFEST_SHA256,
                    input_session_sha256="D" * 64,
                )
            self.assertEqual(len(pd.read_csv(path)), 1)

    def test_prospective_log_rejects_historical_backfill(self) -> None:
        latest = pd.DataFrame(
            {
                "direction": ["ROTATION_IN"],
                "date": ["2026-08-31"],
                "sector": ["4510"],
                "raw_score": [0.8],
                "score": [0.95],
            }
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "prospective.csv"
            with self.assertRaisesRegex(ValueError, "cannot backfill"):
                append_prospective_signals(
                    path,
                    latest,
                    run_manifest_sha256=RUN_MANIFEST_SHA256,
                    input_session_sha256=INPUT_SESSION_SHA256,
                )
            self.assertFalse(path.exists())

    def test_prospective_outcomes_mature_without_changing_signal_log(self) -> None:
        latest = pd.DataFrame(
            {
                "direction": ["ROTATION_IN"],
                "date": ["2026-09-03"],
                "sector": ["4510"],
                "raw_score": [0.8],
                "score": [0.95],
            }
        )
        label = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-09-03")],
                "sector": ["4510"],
                "_position": [100],
                "target": [1.0],
                "label_end_date": [pd.Timestamp("2026-09-17")],
                "target_event_pivot_position": [104.0],
                "target_event_pivot_date": [pd.Timestamp("2026-09-09")],
                "target_event_confirmation_position": [108.0],
                "target_event_confirmation_date": [pd.Timestamp("2026-09-15")],
            }
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "prospective.csv"
            append_prospective_signals(
                path,
                latest,
                run_manifest_sha256=RUN_MANIFEST_SHA256,
                input_session_sha256=INPUT_SESSION_SHA256,
            )
            original = path.read_bytes()
            with patch("sector_turning_pooled_shadow._direction_frame", return_value=label):
                outcomes = build_prospective_outcomes(path, pd.DataFrame())
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(outcomes.loc[0, "outcome_status"], "MATURED")
            self.assertEqual(outcomes.loc[0, "pre_pivot_lead_sessions"], 4.0)
            summary = summarize_prospective_outcomes(outcomes)
            self.assertEqual(summary["independent_confirmed_events"], 1)
            self.assertEqual(
                summary["directions"]["ROTATION_IN"]["PRIORITY"]["pre_pivot_hits"], 1
            )
            self.assertFalse(summary["automatic_promotion"])

    def test_wilson_interval_requires_observations(self) -> None:
        self.assertEqual(_wilson_interval(0, 0), (None, None))
        low, high = _wilson_interval(15, 20)
        self.assertGreater(low, 0.05)
        self.assertLess(high, 1.0)

    def test_prospective_latest_date_must_match_completed_market_date(self) -> None:
        self.assertEqual(
            validate_expected_latest_date(pd.Timestamp("2026-09-03 12:00"), "2026-09-03"),
            pd.Timestamp("2026-09-03"),
        )
        with self.assertRaisesRegex(RuntimeError, "expected completed market date"):
            validate_expected_latest_date(pd.Timestamp("2026-09-02"), "2026-09-03")
        with self.assertRaisesRegex(ValueError, "is required"):
            validate_expected_latest_date(pd.Timestamp("2026-09-03"), None)

    def test_shadow_signal_levels_use_frozen_two_tier_cutoffs(self) -> None:
        levels = shadow_signal_level(pd.Series([0.1, 0.9, 0.9499, 0.95, 1.0]))
        self.assertEqual(
            levels.tolist(), ["NONE", "WATCH", "WATCH", "PRIORITY", "PRIORITY"]
        )


if __name__ == "__main__":
    unittest.main()
