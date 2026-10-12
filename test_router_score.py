import unittest
from contextlib import closing

import numpy as np
import pandas as pd

from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3

from router_score import build_router_scores
from router_score_v2 import build_router_scores_v2
from regime_router import LIVE_HMM_MODEL_VERSION
from validate_router_score import build_validation_frame


def sample_market(count: int, *, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0004, 0.012, count)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2005-01-03", periods=count),
            "close": 100 * np.exp(np.cumsum(returns)),
        }
    )


class RouterScoreTest(unittest.TestCase):
    def test_score_is_integer_and_bounded_after_minimum_history(self) -> None:
        result = build_router_scores(sample_market(700))
        available = result["risk_score"].dropna()
        self.assertGreater(len(available), 0)
        self.assertTrue(available.between(0, 100).all())
        self.assertTrue(all(float(value).is_integer() for value in available))

    def test_future_rows_do_not_change_existing_scores(self) -> None:
        market = sample_market(700)
        base = build_router_scores(market)
        future = sample_market(30, seed=7)
        future["date"] = pd.bdate_range(market["date"].iloc[-1] + pd.Timedelta(days=1), periods=30)
        future["close"] = market["close"].iloc[-1] * np.exp(
            np.cumsum(np.linspace(-0.08, 0.08, len(future)))
        )
        extended = build_router_scores(pd.concat([market, future], ignore_index=True))
        pd.testing.assert_series_equal(
            base["risk_score"],
            extended.loc[base.index, "risk_score"],
            check_names=False,
        )

    def test_components_use_three_equal_risk_blocks(self) -> None:
        result = build_router_scores(sample_market(700))
        row = result.dropna(subset=["risk_score"]).iloc[-1]
        expected = 100 * (
            row["trend_risk"] + row["volatility_risk"] + row["drawdown_risk"]
        ) / 3
        self.assertAlmostEqual(row["score_raw"], expected)

    def test_v2_scores_only_above_median_block_stress(self) -> None:
        result = build_router_scores_v2(sample_market(700))
        row = result.dropna(subset=["risk_score"]).iloc[-1]
        expected = 100 * sum(
            max(0.0, 2 * row[name] - 1)
            for name in ("trend_risk", "volatility_risk", "drawdown_risk")
        ) / 3
        self.assertAlmostEqual(row["score_raw"], expected)
        self.assertEqual(row["risk_score"], int(expected + 0.5))

    def test_v2_constant_history_fails_closed(self) -> None:
        market = pd.DataFrame(
            {
                "date": pd.bdate_range("2005-01-03", periods=700),
                "close": np.full(700, 100.0),
            }
        )
        self.assertTrue(build_router_scores_v2(market)["risk_score"].isna().all())

    def test_validation_excludes_dates_without_forward_outcome(self) -> None:
        market = sample_market(700)
        with TemporaryDirectory() as directory:
            database = Path(directory) / "router.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.executescript(
                    """
                    CREATE TABLE daily_prices(
                        ticker TEXT, date TEXT, close REAL, adj_close REAL
                    );
                    CREATE TABLE regime_model_probabilities(
                        model_version TEXT, date TEXT, p_bull REAL,
                        p_sideways REAL, p_bear REAL
                    );
                    """
                )
                connection.executemany(
                    "INSERT INTO daily_prices VALUES (?, ?, ?, ?)",
                    [
                        ("NQ=F", row.date.date().isoformat(), row.close, row.close)
                        for row in market.itertuples(index=False)
                    ],
                )
                connection.executemany(
                    "INSERT INTO regime_model_probabilities VALUES (?, ?, ?, ?, ?)",
                    [
                        (LIVE_HMM_MODEL_VERSION, row.date.date().isoformat(), 0.4, 0.4, 0.2)
                        for row in market.itertuples(index=False)
                    ],
                )
                connection.commit()
            labelled, full = build_validation_frame(database, horizon=20)
            self.assertEqual(full["date"].max(), market["date"].max())
            self.assertLess(labelled["date"].max(), full["date"].max())


if __name__ == "__main__":
    unittest.main()
