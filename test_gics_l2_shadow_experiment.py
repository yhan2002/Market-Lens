import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from gics_l2_shadow_experiment import (
    BASE_FEATURE_COLUMNS,
    build_cohort_prices,
    build_feature_frame,
    cohort_codes,
    evaluate_rrg,
    load_package,
    weekly_complete_rows,
)


CODES = [
    "1010", "1510", "2010", "2020", "2030", "2510", "2520", "2530", "2550",
    "3010", "3020", "3030", "3510", "3520", "4010", "4020", "4030", "4510",
    "4520", "4530", "5010", "5020", "5510", "6010", "6020",
]


class GicsL2ShadowExperimentTest(unittest.TestCase):
    def test_validation_no_fill_centered_target_and_short_rrg_evaluation(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=280)
        benchmark = pd.DataFrame(
            {
                "date": dates.strftime("%Y-%m-%d"),
                "benchmark_name": "S&P 500",
                "security": "SPX Index",
                "bbgid": "BBG_TEST_SPX",
                "total_return_level": 100 * np.exp(0.0003 * np.arange(len(dates))),
                "currency": "USD",
                "bloomberg_field": "TOT_RETURN_INDEX_GROSS_DVDS",
                "source": "Bloomberg Desktop API",
            }
        )
        master = pd.DataFrame(
            {
                "industry_group_code": CODES,
                "industry_group_name": [f"Group {code}" for code in CODES],
                "parent_sector_code": [code[:2] for code in CODES],
                "parent_sector_name": [f"Sector {code[:2]}" for code in CODES],
                "group_security": [f"TEST_{code} Index" for code in CODES],
                "group_bbgid": [f"BBG_TEST_{code}" for code in CODES],
                "provider": "Standard & Poor's",
                "universe": "S&P 500",
                "currency": "USD",
                "return_type": "GROSS_TOTAL_RETURN",
                "pit_status": "PIT_NOT_VERIFIED",
            }
        )
        rows = []
        step = np.arange(len(dates))
        for index, code in enumerate(CODES):
            level = 80 * np.exp((0.00015 + index * 0.00001) * step + 0.003 * np.sin(step / 9 + index))
            rows.extend(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "industry_group_code": code,
                    "industry_group_name": f"Group {code}",
                    "parent_sector_code": code[:2],
                    "security": f"TEST_{code} Index",
                    "bbgid": f"BBG_TEST_{code}",
                    "total_return_level": value,
                    "currency": "USD",
                    "bloomberg_field": "TOT_RETURN_INDEX_GROSS_DVDS",
                    "source": "Bloomberg Desktop API",
                }
                for date, value in zip(dates, level, strict=True)
            )
        groups = pd.DataFrame(rows)
        missing_date = dates[180]
        groups = groups.loc[
            ~((groups["industry_group_code"] == "1010") & (groups["date"] == missing_date.strftime("%Y-%m-%d")))
        ]
        groups.loc[len(groups)] = {
            "date": "2023-12-30",
            "industry_group_code": "1010",
            "industry_group_name": "Group 1010",
            "parent_sector_code": "10",
            "security": "TEST_1010 Index",
            "bbgid": "BBG_TEST_1010",
            "total_return_level": 77.0,
            "currency": "USD",
            "bloomberg_field": "TOT_RETURN_INDEX_GROSS_DVDS",
            "source": "Bloomberg Desktop API",
        }

        with tempfile.TemporaryDirectory() as temp:
            package = Path(temp)
            master.to_csv(package / "industry_group_master.csv", index=False)
            groups.to_csv(package / "industry_group_total_return.csv", index=False)
            benchmark.to_csv(package / "benchmark_total_return.csv", index=False)
            loaded_master, loaded_groups, loaded_benchmark = load_package(package)
            codes = cohort_codes(loaded_master, "full25")
            prices = build_cohort_prices(loaded_groups, loaded_benchmark, codes)
            self.assertEqual(prices.index.tolist(), list(dates))
            self.assertTrue(pd.isna(prices.at[missing_date, "1010"]))
            self.assertNotIn(pd.Timestamp("2023-12-30"), prices.index)

            frame = build_feature_frame(prices, codes)
            centred = frame.groupby("date").filter(
                lambda group: len(group) == 25 and group["target_excess_20d"].notna().all()
            ).groupby("date")["target_excess_20d"].mean()
            self.assertLess(float(centred.abs().max()), 1e-12)
            weekly = weekly_complete_rows(frame, codes)
            feature_columns = ["shadow_score", *BASE_FEATURE_COLUMNS]
            feature_complete = frame.groupby("date").filter(
                lambda group: len(group) == len(codes)
                and group["ticker"].nunique() == len(codes)
                and group[feature_columns].notna().all().all()
            )
            feature_week_ends = feature_complete.groupby(
                feature_complete["date"].dt.to_period("W-FRI")
            )["date"].max()
            for signal_date in weekly["date"].drop_duplicates():
                self.assertEqual(
                    signal_date,
                    feature_week_ends.loc[signal_date.to_period("W-FRI")],
                )
            summary, observations = evaluate_rrg(weekly, top_k=5, bootstrap_samples=100)
            self.assertEqual(summary["status"], "OK")
            self.assertFalse(observations.empty)
            self.assertNotIn(missing_date, set(weekly["date"]))

            duplicate = pd.concat([groups, groups.iloc[[0]]], ignore_index=True)
            duplicate.to_csv(package / "industry_group_total_return.csv", index=False)
            with self.assertRaisesRegex(ValueError, "duplicate code/date"):
                load_package(package)

    def test_rrg_full_tie_is_neutral_and_does_not_warn(self) -> None:
        tied = pd.DataFrame(
            {
                "date": pd.Timestamp("2026-08-28"),
                "ticker": ["4510", "4520", "4530"],
                "shadow_score": [2 / 3, 2 / 3, 2 / 3],
                "target_excess_20d": [-0.03, 0.01, 0.02],
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            summary, observations = evaluate_rrg(
                tied, top_k=1, bootstrap_samples=100
            )
        row = observations.iloc[0]
        self.assertEqual(row["selected"], "4510,4520,4530")
        self.assertEqual(
            row["selection_weights"],
            "4510:0.333333,4520:0.333333,4530:0.333333",
        )
        self.assertEqual(row["top_k_excess"], 0.0)
        self.assertTrue(pd.isna(row["hit"]))
        self.assertTrue(pd.isna(row["rank_ic"]))
        self.assertEqual(summary["active_observations"], 0)
        self.assertEqual(summary["no_edge_observations"], 1)
        self.assertEqual(summary["boundary_tie_observations"], 1)


if __name__ == "__main__":
    unittest.main()
