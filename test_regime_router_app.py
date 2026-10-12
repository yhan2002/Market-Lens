import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import date
from pathlib import Path
from unittest.mock import call, patch

from regime_router_app import (
    QUANTBOT_CONFIG,
    _current_model_date,
    build_status_view,
    friendly_error,
    latest_nq_date,
    normalize_date,
    update_everything,
)


REGIME = {
    "date": "2026-08-04",
    "state": "BULL",
    "p_bull": 0.8915,
    "p_sideways": 0.071,
    "p_bear": 0.0375,
}


class RegimeRouterAppTest(unittest.TestCase):
    def test_windows_launcher_uses_crlf(self) -> None:
        launcher = Path(__file__).with_name("START_REGIME_ROUTER.cmd").read_bytes()
        self.assertNotIn(b"\n", launcher.replace(b"\r\n", b""))

    def test_normalize_date(self) -> None:
        self.assertEqual(normalize_date(" 2026-08-05 "), "2026-08-05")
        with self.assertRaisesRegex(ValueError, "YYYY-MM-DD"):
            normalize_date("2026/08/05")

    def test_missing_quantbot_config_has_specific_error(self) -> None:
        error = FileNotFoundError(2, "not found", str(QUANTBOT_CONFIG))
        self.assertIn("QuantBot config.py", friendly_error(error))
        self.assertIn(str(QUANTBOT_CONFIG), friendly_error(error))

    def test_alert_copy_explains_action_and_score(self) -> None:
        hazard = {
            "date": "2026-08-04",
            "risk_band": "ALERT_SHADOW",
            "score": 0.81,
        }
        headline, details, tone = build_status_view(
            REGIME,
            hazard,
            requested_date="2026-08-05",
            price_date="2026-08-04",
        )
        self.assertIn("위험 축소 검토", headline)
        self.assertIn("자동 매도 신호는 아닙니다", details)
        self.assertIn("하락 확률이 아니라", details)
        self.assertEqual(tone, "alert")

    def test_current_view_warns_when_model_is_stale(self) -> None:
        hazard = {
            "date": "2026-08-03",
            "risk_band": "NORMAL_SHADOW",
            "score": 0.4,
        }
        _, details, _ = build_status_view(
            REGIME,
            hazard,
            requested_date="2026-08-06",
            price_date="2026-08-05",
        )
        self.assertIn("HMM, 조기경보 결과가 가격 데이터보다 오래됐습니다", details)

    def test_historical_view_does_not_raise_false_stale_warning(self) -> None:
        historical = {**REGIME, "date": "2022-01-05"}
        hazard = {
            "date": "2022-01-05",
            "risk_band": "ALERT_SHADOW",
            "score": 0.729,
        }
        _, details, _ = build_status_view(
            historical,
            hazard,
            requested_date="2022-01-05",
            price_date="2026-08-04",
        )
        self.assertNotIn("결과가 가격 데이터보다 오래됐습니다", details)

    def test_historical_weekend_uses_previous_trading_day(self) -> None:
        historical = {**REGIME, "date": "2022-01-07"}
        hazard = {
            "date": "2022-01-07",
            "risk_band": "NORMAL_SHADOW",
            "score": 0.4,
        }
        _, details, _ = build_status_view(
            historical,
            hazard,
            requested_date="2022-01-08",
            price_date="2022-01-07",
        )
        self.assertNotIn("결과가 가격 데이터보다 오래됐습니다", details)

    def test_missing_hazard_has_clear_next_action(self) -> None:
        headline, details, tone = build_status_view(
            REGIME,
            None,
            requested_date="2026-08-04",
            price_date="2026-08-04",
        )
        self.assertIn("조기경보 미계산", headline)
        self.assertIn("전체 최신화", details)
        self.assertEqual(tone, "neutral")

    def test_guard_watch_overrides_dormant_onset(self) -> None:
        hazard = {
            "date": "2026-08-04",
            "risk_band": "DORMANT",
            "score": None,
        }
        guard = {
            "date": "2026-08-04",
            "risk_band": "CRASH_WATCH_SHADOW",
            "score": 0.91,
            "score_1d": 0.91,
            "score_5d": 0.74,
        }
        headline, details, tone = build_status_view(
            REGIME,
            hazard,
            requested_date="2026-08-05",
            price_date="2026-08-04",
            crash_guard=guard,
        )
        self.assertIn("급락 주의", headline)
        self.assertIn("Crash Guard(상시 Shadow): 급락 주의", details)
        self.assertIn("1일 급변 0.910", details)
        self.assertIn("급락 확률이 아니라", details)
        self.assertEqual(tone, "alert")

    def test_dormant_onset_says_guard_keeps_watching(self) -> None:
        hazard = {
            "date": "2026-08-04",
            "risk_band": "DORMANT",
            "score": None,
        }
        guard = {
            "date": "2026-08-04",
            "risk_band": "NORMAL_SHADOW",
            "score": 0.33,
            "score_1d": 0.21,
            "score_5d": 0.33,
        }
        headline, details, tone = build_status_view(
            REGIME,
            hazard,
            requested_date="2026-08-05",
            price_date="2026-08-04",
            crash_guard=guard,
        )
        self.assertIn("Onset 적용 밖", headline)
        self.assertIn("Crash Guard는 계속 감시합니다", details)
        self.assertIn("상시 감시 정상", details)
        self.assertEqual(tone, "caution")

    def test_two_alerts_are_shown_as_combined_warning(self) -> None:
        hazard = {
            "date": "2026-08-04",
            "risk_band": "ALERT_SHADOW",
            "score": 0.81,
        }
        guard = {
            "date": "2026-08-04",
            "risk_band": "CRASH_WATCH_SHADOW",
            "score": 0.93,
            "score_1d": 0.93,
            "score_5d": 0.8,
        }
        headline, _, tone = build_status_view(
            REGIME,
            hazard,
            requested_date="2026-08-05",
            price_date="2026-08-04",
            crash_guard=guard,
        )
        self.assertIn("복합 경보", headline)
        self.assertEqual(tone, "alert")

    def test_guard_staleness_is_checked_independently(self) -> None:
        hazard = {
            "date": "2026-08-04",
            "risk_band": "NORMAL_SHADOW",
            "score": 0.4,
        }
        guard = {
            "date": "2026-08-03",
            "risk_band": "NORMAL_SHADOW",
            "score": 0.3,
            "score_1d": 0.2,
            "score_5d": 0.3,
        }
        _, details, _ = build_status_view(
            REGIME,
            hazard,
            requested_date="2026-08-04",
            price_date="2026-08-04",
            crash_guard=guard,
        )
        self.assertIn("Crash Guard 결과가 가격 데이터보다 오래됐습니다", details)

    def test_latest_nq_date(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "test.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("CREATE TABLE daily_prices(ticker TEXT, date TEXT)")
                connection.executemany(
                    "INSERT INTO daily_prices VALUES (?, ?)",
                    (("NQ=F", "2026-08-01"), ("SPY", "2026-08-05"), ("NQ=F", "2026-08-04")),
                )
                connection.commit()
            self.assertEqual(latest_nq_date(database), "2026-08-04")
            self.assertEqual(latest_nq_date(database, "2026-08-02"), "2026-08-01")

    def test_current_model_date_requires_all_four_dates_to_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "test.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.executescript(
                    """
                    CREATE TABLE daily_prices(ticker TEXT, date TEXT);
                    CREATE TABLE regime_model_probabilities(model_version TEXT, date TEXT);
                    CREATE TABLE drawdown_hazard_scores(model_version TEXT, date TEXT);
                    INSERT INTO daily_prices VALUES ('SPY', '2026-08-12'), ('NQ=F', '2026-08-12');
                    INSERT INTO regime_model_probabilities VALUES ('hmm-live', '2026-08-12');
                    INSERT INTO drawdown_hazard_scores VALUES ('drawdown-live', '2026-08-12');
                    """
                )
            self.assertEqual(
                _current_model_date(
                    database,
                    hmm_model_version="hmm-live",
                    drawdown_model_version="drawdown-live",
                ),
                "2026-08-12",
            )
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("INSERT INTO daily_prices VALUES ('NQ=F', '2026-08-13')")
                connection.commit()
            self.assertIsNone(
                _current_model_date(
                    database,
                    hmm_model_version="hmm-live",
                    drawdown_model_version="drawdown-live",
                )
            )

    @patch("research_drawdown_onset.write_shadow_scores")
    @patch("regime_router.fit_candidate_from_database")
    @patch("market_data.download_prices")
    @patch("market_data.register_universe")
    def test_full_update_skips_current_models(
        self,
        register_universe,
        download_prices,
        fit_candidate,
        write_shadow,
    ) -> None:
        download_prices.side_effect = ((10, 0, "2026-08-12"), (3, 0, "2026-08-12"))
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "regime_router.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.executescript(
                    """
                    CREATE TABLE daily_prices(ticker TEXT, date TEXT);
                    CREATE TABLE regime_model_probabilities(model_version TEXT, date TEXT);
                    CREATE TABLE drawdown_hazard_scores(model_version TEXT, date TEXT);
                    INSERT INTO daily_prices VALUES ('SPY', '2026-08-12'), ('NQ=F', '2026-08-12');
                    INSERT INTO regime_model_probabilities VALUES
                        ('hmm3-sticky-nq-independent-a2d-logvol-v1-diag-mc25-k25-m252-w1260-r20-s42-n3', '2026-08-12');
                    INSERT INTO drawdown_hazard_scores VALUES
                        ('onset-fixed-core-nq-v2-shadow', '2026-08-12');
                    """
                )
            progress: list[str] = []
            summary = update_everything(
                database,
                progress.append,
                include_equity_universe=False,
            )

        fit_candidate.assert_not_called()
        write_shadow.assert_not_called()
        self.assertEqual(len(progress), 6)
        self.assertIn("재계산 생략", progress[4])
        self.assertIn("재계산 생략", progress[5])
        self.assertIn("HMM·조기경보 2026-08-12 최신", summary)

    @patch("research_drawdown_onset.write_shadow_scores", return_value=4_778)
    @patch("regime_router.fit_candidate_from_database", return_value=[object()] * 4_000)
    @patch("market_data.download_prices")
    @patch("market_data.register_universe")
    @patch("market_data.initialize_database", return_value=(85, 2))
    def test_full_update_sequence_without_network(
        self,
        initialize_database,
        register_universe,
        download_prices,
        fit_candidate,
        write_shadow,
    ) -> None:
        download_prices.side_effect = (
            (85, 850, "2016-08-05"),
            (10, 100, "2026-07-01"),
            (3, 30, "2026-07-01"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "regime_router.db"
            database.touch()
            progress: list[str] = []
            required = date(2026, 8, 17)
            summary = update_everything(
                database, progress.append, required_price_date=required
            )

        initialize_database.assert_called_once_with(database, QUANTBOT_CONFIG)
        from market_data import CROSS_ASSET_TICKERS

        register_universe.assert_called_once_with(database, CROSS_ASSET_TICKERS)
        self.assertEqual(
            download_prices.call_args_list,
            [
                call(database, years=10, group_name="CSQ_MP_CP"),
                call(
                    database,
                    years=20,
                    group_name="REGIME_CROSS_ASSET",
                    require_date=required,
                ),
                call(
                    database,
                    years=20,
                    tickers=("NQ=F",),
                    require_date=required,
                ),
            ],
        )
        fit_candidate.assert_called_once_with(
            database,
            model_version=(
                "hmm3-sticky-nq-independent-a2d-logvol-v1-diag-mc25-"
                "k25-m252-w1260-r20-s42-n3"
            ),
            benchmark="NQ=F",
            feature_set="independent-a2c",
            covariance_type="diag",
            covariance_floor=0.25,
            sticky_strength=25.0,
            min_train=252,
            train_window=1260,
            refit_every=20,
            random_state=42,
            n_init=3,
        )
        write_shadow.assert_called_once_with(database)
        self.assertEqual(len(progress), 6)
        self.assertIn("MP+CP 합집합 85종목", summary)
        self.assertIn("가격 98종목 / 980행", summary)


if __name__ == "__main__":
    unittest.main()
