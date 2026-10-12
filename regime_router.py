from __future__ import annotations

import argparse
import sqlite3
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from breadth_metrics import EXACT_BREADTH_SOURCE
from portfolio_risk import (
    APPROVED_PORTFOLIO_SOURCES,
    portfolio_holdings_date_is_valid,
)


PROBABILITY_COLUMNS = ["p_bull", "p_sideways", "p_bear"]
BENCHMARK_HMM_FEATURE_COLUMNS = (
    "momentum_20",
    "momentum_60",
    "volatility_20",
    "drawdown_60",
    "ma200_distance",
    "ma200_slope_20",
)
INDEPENDENT_HMM_FEATURE_COLUMNS = (
    "trend_acceleration_20_60",
    "slow_trend_ma200_slope_20",
    "volatility_shock_20_120",
)
INDEPENDENT_A2B_HMM_FEATURE_COLUMNS = (
    "trend_acceleration_20_60",
    "slow_trend_ma200_slope_20",
    "absolute_volatility_20",
)
INDEPENDENT_A2C_HMM_FEATURE_COLUMNS = (
    "trend_acceleration_20_60",
    "slow_trend_ma200_slope_20",
    "log_absolute_volatility_20",
)
INDEPENDENT_A3A_HMM_FEATURE_COLUMNS = (
    *INDEPENDENT_A2C_HMM_FEATURE_COLUMNS,
    "vix_curve",
)
HMM_FEATURE_SETS = (
    "core",
    "independent-a2",
    "independent-a2b",
    "independent-a2c",
    "independent-a3a",
)
HMM_COVARIANCE_TYPES = ("diag", "tied", "full")
LIVE_HMM_MODEL_VERSION = "hmm3-sticky-nq-core-v1-k25-m252-w1260-r20-s42-n3"
INDEPENDENT_HMM_MODEL_VERSION = (
    "hmm3-sticky-nq-independent-a2-v1-diag-mc10-k25-m252-w1260-r20-s42-n3"
)
INDEPENDENT_A2B_HMM_MODEL_VERSION = (
    "hmm3-sticky-nq-independent-a2b-v1-diag-mc10-k25-m252-w1260-r20-s42-n3"
)
INDEPENDENT_A2C_HMM_MODEL_VERSION = (
    "hmm3-sticky-nq-independent-a2c-logvol-v1-diag-mc10-k25-m252-w1260-r20-s42-n3"
)
INDEPENDENT_A3A_HMM_MODEL_VERSION = (
    "hmm3-sticky-nq-crossasset-a3a-vixcurve-v1-diag-mc10-k25-m252-w1260-r20-s42-n3"
)

PDF_HMM_MODEL_VERSION = (
    "hmm3-sticky-pdf8-block4-v1-diag-mc10-i500-k25-m756-w1260-r20-s42-44"
)
PDF_HMM_FEATURE_COLUMNS = (
    "nq_trend20",
    "nq_rv20",
    "vix_curve",
    "move_level",
    "hy_oas_level",
    "yen_carry5",
    "breadth_risk",
    "correlation20",
)
PDF_HMM_BLOCK_COLUMNS = ("trend", "cross_asset_stress", "carry", "internals")
PDF_HMM_FROZEN_SETTINGS = {
    "sticky_strength": 25.0,
    "min_train": 756,
    "train_window": 1260,
    "refit_every": 20,
    "random_state": 42,
    "n_init": 3,
    "covariance_type": "diag",
    "max_iter": 500,
    "tolerance": 1e-4,
    "min_covar": 0.10,
}


class _FlooredDiagGaussianHMM(GaussianHMM):
    """Apply an actual variance floor after every Gaussian HMM M-step."""

    covariance_floor: float

    def _do_mstep(self, stats: dict[str, np.ndarray]) -> None:
        super()._do_mstep(stats)
        if self.covariance_type != "diag":
            raise ValueError("covariance_floor is supported only for diag covariance")
        self._covars_ = np.maximum(self._covars_, self.covariance_floor)


class _FlooredFullGaussianHMM(GaussianHMM):
    """Apply an eigenvalue floor to every full state covariance after M-step."""

    covariance_floor: float

    def _do_mstep(self, stats: dict[str, np.ndarray]) -> None:
        super()._do_mstep(stats)
        if self.covariance_type != "full":
            raise ValueError("full covariance floor requires full covariance")
        for state, covariance in enumerate(self._covars_):
            values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
            self._covars_[state] = (vectors * np.maximum(values, self.covariance_floor)) @ vectors.T


def build_features(market: pd.DataFrame) -> pd.DataFrame:
    """Build causal market features from date/close and optional breadth."""
    required = {"date", "close"}
    missing = required - set(market.columns)
    if missing:
        raise ValueError(f"market CSV missing columns: {sorted(missing)}")

    frame = market.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    close = pd.to_numeric(frame["close"], errors="coerce")
    daily_return = close.pct_change()

    features = pd.DataFrame(
        {
            "return_5": close.pct_change(5).to_numpy(),
            "momentum_20": close.pct_change(20).to_numpy(),
            "volatility_20": (daily_return.rolling(20).std() * np.sqrt(252)).to_numpy(),
            "drawdown_60": (close / close.rolling(60).max() - 1).to_numpy(),
        },
        index=frame["date"],
    )
    if "breadth" in frame:
        features["breadth"] = pd.to_numeric(frame["breadth"], errors="coerce").to_numpy()
    return features.replace([np.inf, -np.inf], np.nan).dropna()


def build_features_v2(market: pd.DataFrame) -> pd.DataFrame:
    """Build causal benchmark-only trend and risk features for sticky HMM v2."""
    required = {"date", "close"}
    missing = required - set(market.columns)
    if missing:
        raise ValueError(f"market data missing columns: {sorted(missing)}")
    frame = market.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    close = pd.to_numeric(frame["close"], errors="coerce")
    daily_return = close.pct_change()
    ma200 = close.rolling(200).mean()
    features = pd.DataFrame(
        {
            "momentum_20": close.pct_change(20).to_numpy(),
            "momentum_60": close.pct_change(60).to_numpy(),
            "volatility_20": (daily_return.rolling(20).std() * np.sqrt(252)).to_numpy(),
            "drawdown_60": (close / close.rolling(60).max() - 1).to_numpy(),
            "ma200_distance": (close / ma200 - 1).to_numpy(),
            "ma200_slope_20": (ma200 / ma200.shift(20) - 1).to_numpy(),
        },
        index=frame["date"],
    )
    return features.replace([np.inf, -np.inf], np.nan).dropna()


def build_independent_hmm_features(market: pd.DataFrame) -> pd.DataFrame:
    """Build three causal features with distinct slow-market roles."""
    required = {"date", "close"}
    missing = required - set(market.columns)
    if missing:
        raise ValueError(f"market data missing columns: {sorted(missing)}")
    frame = market.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    close = pd.to_numeric(frame["close"], errors="coerce")
    daily_return = close.pct_change()
    ma200 = close.rolling(200).mean()
    volatility_20 = daily_return.rolling(20).std() * np.sqrt(252)
    volatility_120 = daily_return.rolling(120).std() * np.sqrt(252)
    return_20 = close.pct_change(20)
    return_60_as_20 = (close / close.shift(60)).pow(1 / 3) - 1
    features = pd.DataFrame(
        {
            "trend_acceleration_20_60": (return_20 - return_60_as_20).to_numpy(),
            "slow_trend_ma200_slope_20": (ma200 / ma200.shift(20) - 1).to_numpy(),
            "volatility_shock_20_120": np.log(volatility_20 / volatility_120).to_numpy(),
        },
        index=frame["date"],
    )
    return features.replace([np.inf, -np.inf], np.nan).dropna()


def build_independent_a2b_hmm_features(market: pd.DataFrame) -> pd.DataFrame:
    """Replace only A2's volatility ratio with absolute 20-day volatility."""
    required = {"date", "close"}
    missing = required - set(market.columns)
    if missing:
        raise ValueError(f"market data missing columns: {sorted(missing)}")
    frame = market.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    close = pd.to_numeric(frame["close"], errors="coerce")
    daily_return = close.pct_change()
    ma200 = close.rolling(200).mean()
    return_20 = close.pct_change(20)
    return_60_as_20 = (close / close.shift(60)).pow(1 / 3) - 1
    features = pd.DataFrame(
        {
            "trend_acceleration_20_60": (return_20 - return_60_as_20).to_numpy(),
            "slow_trend_ma200_slope_20": (ma200 / ma200.shift(20) - 1).to_numpy(),
            "absolute_volatility_20": (
                daily_return.rolling(20).std() * np.sqrt(252)
            ).to_numpy(),
        },
        index=frame["date"],
    )
    return features.replace([np.inf, -np.inf], np.nan).dropna()


def build_independent_a2c_hmm_features(market: pd.DataFrame) -> pd.DataFrame:
    """Log-transform only A2-B's positive absolute-volatility input."""
    features = build_independent_a2b_hmm_features(market).rename(
        columns={"absolute_volatility_20": "log_absolute_volatility_20"}
    )
    features["log_absolute_volatility_20"] = np.log(
        features["log_absolute_volatility_20"]
    )
    return features.loc[:, list(INDEPENDENT_A2C_HMM_FEATURE_COLUMNS)]


def build_independent_a3a_hmm_features(market: pd.DataFrame) -> pd.DataFrame:
    """Add only the same-day VIX/VIX3M curve to the frozen A2-C inputs."""
    required = {"date", "close", "vix", "vix3m"}
    missing = required - set(market.columns)
    if missing:
        raise ValueError(f"A3-A market data missing columns: {sorted(missing)}")
    frame = market.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    frame = frame.drop_duplicates("date", keep="last")
    vix = pd.to_numeric(frame["vix"], errors="coerce")
    vix3m = pd.to_numeric(frame["vix3m"], errors="coerce")
    valid_curve = (vix > 0) & (vix3m > 0)
    curve = pd.Series(
        np.log(vix.where(valid_curve) / vix3m.where(valid_curve)).to_numpy(),
        index=frame["date"],
        name="vix_curve",
    )
    features = build_independent_a2c_hmm_features(
        frame.loc[:, ["date", "close"]]
    ).join(curve, how="inner")
    return features.loc[:, list(INDEPENDENT_A3A_HMM_FEATURE_COLUMNS)].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()


def build_features_v3(market: pd.DataFrame) -> pd.DataFrame:
    """Build the PDF eight-variable HMM candidate feature set causally.

    This is deliberately a pure research transform. It does not replace or
    persist the currently deployed benchmark-only HMM.
    """
    source_columns = {
        "date",
        "nq_close",
        "vix",
        "vix3m",
        "move",
        "hy_oas",
        "usdjpy",
        "breadth_200dma",
        "portfolio_correlation",
    }
    missing = source_columns - set(market.columns)
    if missing:
        raise ValueError(f"PDF HMM market data missing columns: {sorted(missing)}")

    frame = market.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    frame = frame.drop_duplicates("date", keep="last")
    numeric_columns = source_columns - {"date"}
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    nq_close = frame["nq_close"]
    nq_daily_return = np.log(nq_close / nq_close.shift(1))
    features = pd.DataFrame(
        {
            "nq_trend20": (-np.log(nq_close / nq_close.shift(20))).to_numpy(),
            "nq_rv20": (
                nq_daily_return.rolling(20).std() * np.sqrt(252)
            ).to_numpy(),
            "vix_curve": np.log(frame["vix"] / frame["vix3m"]).to_numpy(),
            "move_level": np.log(frame["move"]).to_numpy(),
            # macro_observations stores BAMLH0A0HYM2 in percentage points.
            "hy_oas_level": np.log(frame["hy_oas"] * 100.0).to_numpy(),
            "yen_carry5": (
                -np.log(frame["usdjpy"] / frame["usdjpy"].shift(5))
            ).to_numpy(),
            "breadth_risk": (1.0 - frame["breadth_200dma"]).to_numpy(),
            "correlation20": frame["portfolio_correlation"].to_numpy(),
        },
        index=frame["date"],
    )
    return features[list(PDF_HMM_FEATURE_COLUMNS)].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()


def build_pdf_market_from_database(
    database: Path, as_of: date | str | None = None
) -> pd.DataFrame:
    """Assemble exact point-in-time inputs for the PDF eight-variable candidate."""
    cutoff = (
        date.today()
        if as_of is None
        else date.fromisoformat(as_of)
        if isinstance(as_of, str)
        else as_of
    )
    required_tables = {
        "daily_prices",
        "macro_observations",
        "breadth_metrics",
        "portfolio_risk_metrics",
    }
    uri = database.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        missing = required_tables - tables
        if missing:
            raise ValueError(f"PDF HMM database missing tables: {sorted(missing)}")
        breadth_columns = {
            str(row[1])
            for row in connection.execute('PRAGMA table_info("breadth_metrics")')
        }
        required_breadth_columns = {
            "date",
            "breadth_200dma",
            "coverage",
            "source",
            "membership_as_of",
            "available_at",
        }
        missing_breadth_columns = required_breadth_columns - breadth_columns
        if missing_breadth_columns:
            raise ValueError(
                "PDF HMM breadth provenance columns missing: "
                f"{sorted(missing_breadth_columns)}"
            )
        price_rows = connection.execute(
            """
            SELECT substr(date, 1, 10), ticker, COALESCE(adj_close, close)
            FROM daily_prices
            WHERE ticker IN ('NQ=F', '^VIX', '^VIX3M', '^MOVE', 'JPY=X')
              AND substr(date, 1, 10)<=?
            ORDER BY substr(date, 1, 10), ticker
            """,
            (cutoff.isoformat(),),
        ).fetchall()
        hy_rows = connection.execute(
            """
            SELECT substr(date, 1, 10), value, available_at, source
            FROM macro_observations
            WHERE upper(series)='HY_OAS'
              AND substr(date, 1, 10)<=?
              AND substr(available_at, 1, 10)<=?
              AND upper(source) NOT LIKE '%PROXY%'
            ORDER BY substr(available_at, 1, 10), substr(date, 1, 10), available_at
            """,
            (cutoff.isoformat(), cutoff.isoformat()),
        ).fetchall()
        breadth_rows = connection.execute(
            """
            SELECT substr(date, 1, 10), breadth_200dma, coverage, source,
                   substr(membership_as_of, 1, 10), substr(available_at, 1, 10)
            FROM breadth_metrics WHERE substr(date, 1, 10)<=? ORDER BY date
            """,
            (cutoff.isoformat(),),
        ).fetchall()
        correlation_rows = connection.execute(
            """
            SELECT substr(date, 1, 10), substr(holdings_date, 1, 10),
                   average_correlation, coverage, source
            FROM portfolio_risk_metrics WHERE substr(date, 1, 10)<=? ORDER BY date
            """,
            (cutoff.isoformat(),),
        ).fetchall()

    prices = pd.DataFrame(price_rows, columns=["date", "ticker", "value"])
    if prices.empty:
        raise ValueError("PDF HMM exact market prices are unavailable")
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    price_matrix = prices.pivot(index="date", columns="ticker", values="value")
    price_matrix = price_matrix.rename(
        columns={
            "NQ=F": "nq_close",
            "^VIX": "vix",
            "^VIX3M": "vix3m",
            "^MOVE": "move",
            "JPY=X": "usdjpy",
        }
    )
    missing_prices = {
        "nq_close", "vix", "vix3m", "move", "usdjpy"
    } - set(price_matrix.columns)
    if missing_prices:
        raise ValueError(f"PDF HMM exact price series missing: {sorted(missing_prices)}")
    price_matrix = price_matrix.dropna(
        subset=["nq_close", "vix", "vix3m", "move", "usdjpy"]
    )

    hy = pd.DataFrame(
        hy_rows,
        columns=["observation_date", "hy_oas", "available_at", "source"],
    )
    if hy.empty:
        raise ValueError("point-in-time HY OAS is unavailable")
    hy["observation_date"] = pd.to_datetime(hy["observation_date"], errors="coerce")
    hy["available_date"] = pd.to_datetime(
        hy["available_at"].astype(str).str[:10], errors="coerce"
    )
    hy["hy_oas"] = pd.to_numeric(hy["hy_oas"], errors="coerce")
    hy = hy.dropna(subset=["observation_date", "available_date", "hy_oas"])
    hy["effective_date"] = hy[["observation_date", "available_date"]].max(axis=1)
    hy = hy[hy["effective_date"] <= pd.Timestamp(cutoff)]
    hy = hy.sort_values(["effective_date", "observation_date", "available_at"])
    events = hy.to_dict(orient="records")
    active: dict[pd.Timestamp, tuple[str, float]] = {}
    event_index = 0
    hy_values: list[float] = []
    for market_date in price_matrix.index:
        while (
            event_index < len(events)
            and events[event_index]["effective_date"] <= market_date
        ):
            event = events[event_index]
            observation_date = event["observation_date"]
            available_at = str(event["available_at"])
            previous = active.get(observation_date)
            if previous is None or available_at >= previous[0]:
                active[observation_date] = (available_at, float(event["hy_oas"]))
            event_index += 1
        latest_observation = max(active, default=None)
        hy_values.append(
            float("nan") if latest_observation is None else active[latest_observation][1]
        )
    price_matrix["hy_oas"] = hy_values

    breadth = pd.DataFrame(
        breadth_rows,
        columns=[
            "date",
            "breadth_200dma",
            "coverage",
            "source",
            "membership_as_of",
            "available_at",
        ],
    )
    correlation = pd.DataFrame(
        correlation_rows,
        columns=[
            "date",
            "holdings_date",
            "portfolio_correlation",
            "coverage",
            "source",
        ],
    )
    for frame, value_column, required_source in (
        (breadth, "breadth_200dma", EXACT_BREADTH_SOURCE),
        (correlation, "portfolio_correlation", APPROVED_PORTFOLIO_SOURCES),
    ):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
        frame["coverage"] = pd.to_numeric(frame["coverage"], errors="coerce")
        source = frame["source"].astype(str)
        minimum_coverage = 1.0 if value_column == "breadth_200dma" else 0.90
        valid = frame["coverage"].between(minimum_coverage, 1.0) & ~source.str.contains(
            "PROXY", case=False, na=False
        )
        if required_source is not None:
            valid &= source.isin(required_source) if isinstance(required_source, set) else source.eq(required_source)
        if value_column == "breadth_200dma":
            frame["membership_as_of"] = pd.to_datetime(
                frame["membership_as_of"], errors="coerce"
            )
            frame["available_at"] = pd.to_datetime(
                frame["available_at"], errors="coerce"
            )
            valid &= frame["membership_as_of"].le(frame["date"])
            valid &= frame["available_at"].le(frame["date"])
        else:
            frame["holdings_date"] = pd.to_datetime(
                frame["holdings_date"], errors="coerce"
            )
            valid &= pd.Series(
                [
                    portfolio_holdings_date_is_valid(metric_date, holdings_date, source_name)
                    if pd.notna(metric_date) and pd.notna(holdings_date)
                    else False
                    for metric_date, holdings_date, source_name in zip(
                        frame["date"], frame["holdings_date"], source
                    )
                ],
                index=frame.index,
            )
        frame.drop(frame.index[~valid], inplace=True)
        frame.dropna(subset=["date", value_column], inplace=True)
        if value_column == "breadth_200dma":
            frame.drop(frame.index[~frame[value_column].between(0.0, 1.0)], inplace=True)
        else:
            frame.drop(frame.index[~frame[value_column].between(-1.0, 1.0)], inplace=True)
        frame.drop_duplicates("date", keep="last", inplace=True)
        frame.set_index("date", inplace=True)
        frame.drop(
            columns=[
                column
                for column in (
                    "coverage",
                    "source",
                    "holdings_date",
                    "membership_as_of",
                    "available_at",
                )
                if column in frame
            ],
            inplace=True,
        )
    market = price_matrix.join(breadth, how="left").join(correlation, how="left")
    return market.reset_index().rename(columns={"index": "date"}).dropna()


def _state_labels(model: GaussianHMM, momentum_column: int) -> dict[int, str]:
    order = sorted(
        range(model.n_components),
        key=lambda state: (float(model.means_[state, momentum_column]), state),
    )
    return {order[0]: "bear", order[1]: "sideways", order[2]: "bull"}


def _composite_state_labels(model: GaussianHMM, columns: pd.Index) -> dict[int, str]:
    risk_weights = {
        "momentum_20": -1.0,
        "momentum_60": -0.75,
        "volatility_20": 0.75,
        "drawdown_60": -0.75,
        "ma200_distance": -1.0,
        "ma200_slope_20": -0.5,
        "nq_return_20": -1.0,
        "nq_volatility_20": 1.0,
        "vix_term_ratio": 1.0,
        "move_level": 0.75,
        "hy_oas": 1.0,
        "usdjpy_change_5": -0.75,
        "breadth_200dma": -1.0,
        "portfolio_correlation": 1.0,
        "trend_acceleration_20_60": -1.0,
        "slow_trend_ma200_slope_20": -1.0,
        "volatility_shock_20_120": 1.0,
        "absolute_volatility_20": 1.0,
        "log_absolute_volatility_20": 1.0,
        "vix_curve": 1.0,
    }
    active_weights = {
        name: weight for name, weight in risk_weights.items() if name in columns
    }
    if not active_weights:
        raise ValueError("no recognized risk features for composite state labels")
    scores = []
    for state in range(model.n_components):
        score = sum(
            risk_weights[name] * float(model.means_[state, columns.get_loc(name)])
            for name in active_weights
        )
        scores.append((score, state))
    order = [state for _, state in sorted(scores)]
    if model.n_components == 3:
        return {order[0]: "bull", order[1]: "sideways", order[2]: "bear"}
    if model.n_components == 4:
        return {
            order[0]: "bull",
            order[1]: "sideways",
            order[2]: "bear",
            order[3]: "bear",
        }
    raise ValueError("composite state labels support only three or four states")


def _four_state_diagnostics(
    model: GaussianHMM, scaled_train: np.ndarray, columns: pd.Index
) -> dict[str, object]:
    if model.n_components != 4:
        raise ValueError("four-state diagnostics require four components")
    labels = _composite_state_labels(model, columns)
    risk_weights = {
        "trend_acceleration_20_60": -1.0,
        "slow_trend_ma200_slope_20": -1.0,
        "log_absolute_volatility_20": 1.0,
    }
    active = {name: weight for name, weight in risk_weights.items() if name in columns}
    if len(active) != 3:
        raise ValueError("four-state candidate requires the frozen A2-C features")
    scores = np.array(
        [
            sum(
                weight * float(model.means_[state, columns.get_loc(name)])
                for name, weight in active.items()
            )
            / len(active)
            for state in range(4)
        ]
    )
    ordered_scores = np.sort(scores)
    latent_states = model.predict(scaled_train)
    occupancy = np.bincount(latent_states, minlength=4) / len(latent_states)
    minimum_gap = float(np.min(np.diff(ordered_scores)))
    admissible = bool(
        np.all((occupancy >= 0.05) & (occupancy <= 0.80)) and minimum_gap >= 0.25
    )
    return {
        "admissible": admissible,
        "occupancy": occupancy,
        "minimum_centroid_gap": minimum_gap,
        "labels": labels,
    }


def _initial_transition_matrix(n_components: int, stay_probability: float = 0.96) -> np.ndarray:
    off_diagonal = (1 - stay_probability) / (n_components - 1)
    matrix = np.full((n_components, n_components), off_diagonal)
    np.fill_diagonal(matrix, stay_probability)
    return matrix


def _aggregate_public_posterior(
    posterior: np.ndarray, labels: dict[int, str]
) -> dict[str, float]:
    public = {"bull": 0.0, "sideways": 0.0, "bear": 0.0}
    for state, probability in enumerate(posterior):
        public[labels[state]] += float(probability)
    return public


def _require_complete_pdf8_market(market: pd.DataFrame) -> None:
    numeric = {
        "nq_close",
        "vix",
        "vix3m",
        "move",
        "hy_oas",
        "usdjpy",
        "breadth_200dma",
        "portfolio_correlation",
    }
    missing = {"date", *numeric} - set(market.columns)
    if missing:
        raise ValueError(f"PDF HMM market data missing columns: {sorted(missing)}")
    values = market.loc[:, sorted(numeric)].apply(pd.to_numeric, errors="coerce")
    if pd.to_datetime(market["date"], errors="coerce").isna().any() or not np.isfinite(
        values.to_numpy(dtype=float)
    ).all():
        raise ValueError("PDF8 requires complete exact rows")
    positive = ["nq_close", "vix", "vix3m", "move", "hy_oas", "usdjpy"]
    if (values[positive] <= 0).any().any():
        raise ValueError("PDF8 price, volatility, MOVE, HY OAS, and USDJPY must be positive")
    if not values["breadth_200dma"].between(0.0, 1.0).all():
        raise ValueError("PDF8 breadth_200dma must be between zero and one")
    if not values["portfolio_correlation"].between(-1.0, 1.0).all():
        raise ValueError("PDF8 portfolio_correlation must be between minus one and one")


def _pdf8_blocks(scaled_raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trend": scaled_raw["nq_trend20"],
            "cross_asset_stress": scaled_raw[
                ["nq_rv20", "vix_curve", "move_level", "hy_oas_level"]
            ].mean(axis=1),
            "carry": scaled_raw["yen_carry5"],
            "internals": scaled_raw[["breadth_risk", "correlation20"]].mean(axis=1),
        },
        index=scaled_raw.index,
    )


def _fit_pdf8_preprocessor(
    train: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, StandardScaler]:
    lower = train.quantile(0.01)
    upper = train.quantile(0.99)
    winsorized = train.clip(lower=lower, upper=upper, axis=1)
    median = winsorized.median()
    scale = (winsorized - median).abs().median() * 1.4826
    if (scale <= np.finfo(float).eps).any():
        raise ValueError("PDF8 training feature has zero MAD")
    robust = ((winsorized - median) / scale).clip(-4.0, 4.0)
    blocks = _pdf8_blocks(robust)
    block_scaler = StandardScaler().fit(blocks)
    if (block_scaler.var_ <= np.finfo(float).eps).any():
        raise ValueError("PDF8 training block has zero variance")
    return lower, upper, median, scale, block_scaler


def _transform_pdf8_blocks(
    frame: pd.DataFrame,
    preprocessor: tuple[pd.Series, pd.Series, pd.Series, pd.Series, StandardScaler],
) -> pd.DataFrame:
    lower, upper, median, scale, block_scaler = preprocessor
    winsorized = frame.clip(lower=lower, upper=upper, axis=1)
    robust = ((winsorized - median) / scale).clip(-4.0, 4.0)
    blocks = _pdf8_blocks(robust)
    return pd.DataFrame(
        block_scaler.transform(blocks),
        index=frame.index,
        columns=PDF_HMM_BLOCK_COLUMNS,
    )


def _prepare_pdf8_blocks(
    train: pd.DataFrame, observed: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the frozen, train-only robust PDF8 block transform."""
    preprocessor = _fit_pdf8_preprocessor(train)
    return (
        _transform_pdf8_blocks(train, preprocessor),
        _transform_pdf8_blocks(observed, preprocessor),
    )


def _pdf8_state_labels(model: GaussianHMM) -> tuple[dict[int, str], list[float]]:
    centroids = np.asarray(model.means_, dtype=float).mean(axis=1)
    order = sorted(range(3), key=lambda state: (float(centroids[state]), state))
    labels = {order[0]: "bull", order[1]: "sideways", order[2]: "bear"}
    return labels, [float(centroids[state]) for state in order]


def diagnose_pdf8_candidate(
    probabilities: pd.DataFrame, *, risk_centroids: list[float] | None = None
) -> dict[str, object]:
    reasons: list[str] = []
    matrix = probabilities.reindex(columns=PROBABILITY_COLUMNS).to_numpy(dtype=float)
    finite = bool(matrix.size and np.isfinite(matrix).all())
    sums_valid = bool(finite and np.allclose(matrix.sum(axis=1), 1.0, atol=1e-8))
    bounds_valid = bool(finite and ((matrix >= 0.0) & (matrix <= 1.0)).all())
    if not finite:
        reasons.append("probabilities_not_finite")
    if not bounds_valid:
        reasons.append("probabilities_out_of_bounds")
    if not sums_valid:
        reasons.append("probabilities_do_not_sum_to_one")

    occupancy: dict[str, float] = {}
    saturation_share = float("nan")
    if finite and bounds_valid and sums_valid:
        hard_states = np.argmax(matrix, axis=1)
        occupancy = {
            name: float(np.mean(hard_states == state))
            for state, name in enumerate(PROBABILITY_COLUMNS)
        }
        if any(value < 0.05 or value > 0.80 for value in occupancy.values()):
            reasons.append("hard_state_occupancy_outside_5_80pct")
        recent = matrix[-252:]
        saturation_share = float(np.mean(np.max(recent, axis=1) >= 0.99))
        if saturation_share > 0.80:
            reasons.append("recent_posterior_saturation_above_80pct")

    centroid_gap: float | None = None
    if risk_centroids is not None:
        ordered = np.sort(np.asarray(risk_centroids, dtype=float))
        if len(ordered) == 3 and np.isfinite(ordered).all():
            centroid_gap = float(np.diff(ordered).min())
            if centroid_gap < 0.25:
                reasons.append("adjacent_risk_centroid_gap_below_0p25z")

    return {
        "valid": not reasons,
        "reasons": reasons,
        "hard_state_occupancy": occupancy,
        "minimum_adjacent_risk_centroid_gap": centroid_gap,
        "recent_252_saturation_share": saturation_share,
    }


def walk_forward_pdf8_candidate(features: pd.DataFrame) -> pd.DataFrame:
    """Run only the immutable research PDF8 block4 HMM specification."""
    settings = PDF_HMM_FROZEN_SETTINGS
    if list(features.columns) != list(PDF_HMM_FEATURE_COLUMNS):
        raise ValueError("PDF8 feature columns do not match the frozen specification")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("PDF8 requires complete exact feature rows")
    min_train = int(settings["min_train"])
    if len(features) <= min_train:
        raise ValueError(f"need more than {min_train} complete PDF8 feature rows")

    output = pd.DataFrame(np.nan, index=features.index, columns=PROBABILITY_COLUMNS)
    model: GaussianHMM | None = None
    preprocessor: tuple[pd.Series, pd.Series, pd.Series, pd.Series, StandardScaler] | None = None
    labels: dict[int, str] | None = None
    risk_centroids: list[float] | None = None
    context_start = 0
    selected_seeds: list[int] = []
    unstable_refits = 0
    refit_attempts = 0
    rejected_refits = 0
    rejected_refit_reasons: dict[str, int] = {}
    for position in range(min_train, len(features)):
        should_refit = (position - min_train) % int(settings["refit_every"]) == 0
        if should_refit:
            refit_attempts += 1
            proposed_context_start = max(0, position - int(settings["train_window"]))
            train = features.iloc[proposed_context_start:position]
            attempt_reasons: set[str] = set()
            valid_models: list[
                tuple[
                    float,
                    int,
                    GaussianHMM,
                    tuple[pd.Series, pd.Series, pd.Series, pd.Series, StandardScaler],
                    dict[int, str],
                    list[float],
                ]
            ] = []
            stable_candidate_seen = False
            try:
                proposed_preprocessor = _fit_pdf8_preprocessor(train)
                scaled_train = _transform_pdf8_blocks(train, proposed_preprocessor)
            except ValueError:
                attempt_reasons.add("preprocessor_invalid")
            else:
                transition_prior = np.ones((3, 3))
                np.fill_diagonal(
                    transition_prior, float(settings["sticky_strength"])
                )
                for seed in range(
                    int(settings["random_state"]),
                    int(settings["random_state"]) + int(settings["n_init"]),
                ):
                    candidate = GaussianHMM(
                        n_components=3,
                        covariance_type=str(settings["covariance_type"]),
                        n_iter=int(settings["max_iter"]),
                        tol=float(settings["tolerance"]),
                        min_covar=float(settings["min_covar"]),
                        random_state=seed,
                        transmat_prior=transition_prior,
                        init_params="mc",
                    )
                    candidate.startprob_ = np.full(3, 1 / 3)
                    candidate.transmat_ = np.full((3, 3), 0.02)
                    np.fill_diagonal(candidate.transmat_, 0.96)
                    try:
                        candidate.fit(scaled_train)
                        history = list(candidate.monitor_.history)
                        stable = bool(candidate.monitor_.converged) and (
                            len(history) < 2 or history[-1] >= history[-2] - 1e-3
                        )
                        stable_candidate_seen |= stable
                        train_loglik = float(candidate.score(scaled_train))
                        candidate_labels, candidate_centroids = _pdf8_state_labels(
                            candidate
                        )
                        raw_probabilities = candidate.predict_proba(scaled_train)
                    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                        attempt_reasons.add(f"fit_error_{type(error).__name__}")
                        continue
                    semantic_probabilities = pd.DataFrame(
                        np.nan,
                        index=scaled_train.index,
                        columns=PROBABILITY_COLUMNS,
                    )
                    for state, state_label in candidate_labels.items():
                        semantic_probabilities[f"p_{state_label}"] = raw_probabilities[
                            :, state
                        ]
                    diagnostics = diagnose_pdf8_candidate(
                        semantic_probabilities,
                        risk_centroids=candidate_centroids,
                    )
                    if not stable:
                        attempt_reasons.add("hmm_not_stable")
                    if not np.isfinite(train_loglik):
                        attempt_reasons.add("train_loglik_not_finite")
                    attempt_reasons.update(diagnostics["reasons"])
                    if stable and np.isfinite(train_loglik) and diagnostics["valid"]:
                        valid_models.append(
                            (
                                train_loglik,
                                seed,
                                candidate,
                                proposed_preprocessor,
                                candidate_labels,
                                candidate_centroids,
                            )
                        )
                unstable_refits += int(not stable_candidate_seen)

            if valid_models:
                (
                    _,
                    selected_seed,
                    model,
                    preprocessor,
                    labels,
                    risk_centroids,
                ) = max(valid_models, key=lambda item: item[0])
                context_start = proposed_context_start
                selected_seeds.append(selected_seed)
            else:
                rejected_refits += 1
                for reason in attempt_reasons or {"no_valid_hmm_fit"}:
                    rejected_refit_reasons[reason] = (
                        rejected_refit_reasons.get(reason, 0) + 1
                    )

        if model is None or preprocessor is None or labels is None:
            continue
        observed = _transform_pdf8_blocks(
            features.iloc[context_start : position + 1], preprocessor
        )
        posterior = model.predict_proba(observed)[-1]
        for state, probability in enumerate(posterior):
            output.loc[features.index[position], f"p_{labels[state]}"] = probability

    completed = output.dropna()
    diagnostics = diagnose_pdf8_candidate(
        completed, risk_centroids=risk_centroids
    )
    completed.attrs.update(
        {
            "model_version": PDF_HMM_MODEL_VERSION,
            "research_only": True,
            "promotable": False,
            "validation_status": "VALID_RESEARCH_ONLY"
            if diagnostics["valid"]
            else "INVALID_RESEARCH_CANDIDATE",
            "invalid_reasons": list(diagnostics["reasons"]),
            "diagnostics": diagnostics,
            "refits": len(selected_seeds),
            "refit_attempts": refit_attempts,
            "rejected_refits": rejected_refits,
            "rejected_refit_reasons": dict(sorted(rejected_refit_reasons.items())),
            "unstable_refits": unstable_refits,
            "selected_seed_counts": {
                seed: selected_seeds.count(seed) for seed in sorted(set(selected_seeds))
            },
            "feature_columns": list(PDF_HMM_FEATURE_COLUMNS),
            "block_columns": list(PDF_HMM_BLOCK_COLUMNS),
            "covariance_type": "diag",
            "frozen_settings": dict(settings),
        }
    )
    return completed


def fit_pdf_candidate(
    market: pd.DataFrame,
    *,
    model_version: str = PDF_HMM_MODEL_VERSION,
    sticky_strength: float = 25.0,
    min_train: int = 756,
    train_window: int = 1260,
    refit_every: int = 20,
    random_state: int = 42,
    n_init: int = 3,
    covariance_type: str = "diag",
    max_iter: int = 500,
    tolerance: float = 1e-4,
    min_covar: float = 0.10,
) -> pd.DataFrame:
    """Prepare the frozen PDF8 candidate without persistence or promotion."""
    requested = {
        "sticky_strength": sticky_strength,
        "min_train": min_train,
        "train_window": train_window,
        "refit_every": refit_every,
        "random_state": random_state,
        "n_init": n_init,
        "covariance_type": covariance_type,
        "max_iter": max_iter,
        "tolerance": tolerance,
        "min_covar": min_covar,
    }
    if model_version != PDF_HMM_MODEL_VERSION or requested != PDF_HMM_FROZEN_SETTINGS:
        raise ValueError("PDF8 model version and settings are frozen; alternate settings refused")
    _require_complete_pdf8_market(market)
    return walk_forward_pdf8_candidate(build_features_v3(market))


def fit_pdf_candidate_from_database(
    database: Path,
    *,
    as_of: date | str | None = None,
    **kwargs: object,
) -> pd.DataFrame:
    """Build and fit the PDF candidate without persisting or promoting it."""
    return fit_pdf_candidate(build_pdf_market_from_database(database, as_of), **kwargs)


def walk_forward_probabilities(
    features: pd.DataFrame,
    *,
    min_train: int = 252,
    train_window: int = 1260,
    refit_every: int = 20,
    random_state: int = 42,
    n_init: int = 1,
    sticky_strength: float | None = None,
    composite_labels: bool = False,
    covariance_type: str = "diag",
    covariance_floor: float | None = None,
) -> pd.DataFrame:
    """Estimate three regime probabilities without using observations after t."""
    if min_train < 60:
        raise ValueError("min_train must be at least 60 observations")
    if len(features) <= min_train:
        raise ValueError(f"need more than {min_train} complete feature rows")
    if n_init < 1:
        raise ValueError("n_init must be at least 1")
    n_components = 3
    if covariance_type not in HMM_COVARIANCE_TYPES:
        raise ValueError(
            f"covariance_type must be one of {list(HMM_COVARIANCE_TYPES)}"
        )
    if covariance_floor is not None:
        if covariance_type not in {"diag", "full"}:
            raise ValueError("covariance_floor is supported only for diag or full covariance")
        if not np.isfinite(covariance_floor) or covariance_floor <= 0:
            raise ValueError("covariance_floor must be a finite positive number")

    output = pd.DataFrame(np.nan, index=features.index, columns=PROBABILITY_COLUMNS)
    model: GaussianHMM | None = None
    scaler: StandardScaler | None = None
    labels: dict[int, str] | None = None
    context_start = 0
    selected_seeds: list[int] = []
    unstable_refits = 0

    for position in range(min_train, len(features)):
        should_refit = model is None or (position - min_train) % refit_every == 0
        if should_refit:
            context_start = max(0, position - train_window)
            train = features.iloc[context_start:position]
            scaler = StandardScaler().fit(train)
            scaled_train = scaler.transform(train)
            model_kwargs: dict[str, object] = {}
            if sticky_strength is not None:
                transition_prior = np.ones((n_components, n_components))
                np.fill_diagonal(transition_prior, sticky_strength)
                model_kwargs = {
                    "transmat_prior": transition_prior,
                    "init_params": "mc",
                }
            fitted_models: list[tuple[bool, float, int, GaussianHMM]] = []
            for seed in range(random_state, random_state + n_init):
                model_type = GaussianHMM
                if covariance_floor is not None:
                    model_type = (
                        _FlooredDiagGaussianHMM
                        if covariance_type == "diag"
                        else _FlooredFullGaussianHMM
                    )
                candidate = model_type(
                    n_components=n_components,
                    covariance_type=covariance_type,
                    n_iter=200,
                    min_covar=(covariance_floor if covariance_floor is not None else 1e-4),
                    random_state=seed,
                    **model_kwargs,
                )
                if covariance_floor is not None:
                    candidate.covariance_floor = covariance_floor
                if sticky_strength is not None:
                    candidate.startprob_ = np.full(n_components, 1 / n_components)
                    candidate.transmat_ = _initial_transition_matrix(n_components)
                candidate.fit(scaled_train)
                history = list(candidate.monitor_.history)
                stable = bool(candidate.monitor_.converged) and (
                    len(history) < 2 or history[-1] >= history[-2] - 1e-3
                )
                fitted_models.append(
                    (stable, float(candidate.score(scaled_train)), seed, candidate)
                )
            stable, _, selected_seed, model = max(
                fitted_models, key=lambda item: (item[0], item[1])
            )
            selected_seeds.append(selected_seed)
            unstable_refits += int(not stable)
            labels = (
                _composite_state_labels(model, train.columns)
                if composite_labels
                else _state_labels(model, train.columns.get_loc("momentum_20"))
            )

        assert model is not None and scaler is not None and labels is not None
        observed = scaler.transform(features.iloc[context_start : position + 1])
        posterior = model.predict_proba(observed)[-1]
        public = _aggregate_public_posterior(posterior, labels)
        for label, probability in public.items():
            output.loc[features.index[position], f"p_{label}"] = probability

    completed = output.dropna()
    if not np.allclose(completed.sum(axis=1), 1.0, atol=1e-8):
        raise RuntimeError("HMM probabilities do not sum to one")
    completed.attrs.update(
        {
            "refits": len(selected_seeds),
            "unstable_refits": unstable_refits,
            "selected_seed_counts": {
                seed: selected_seeds.count(seed) for seed in sorted(set(selected_seeds))
            },
            "feature_columns": list(features.columns),
            "covariance_type": covariance_type,
            "covariance_floor": covariance_floor,
            "n_components": n_components,
        }
    )
    return completed


def fit_regimes(market_csv: Path, output_csv: Path, **kwargs: int) -> pd.DataFrame:
    market = pd.read_csv(market_csv)
    probabilities = walk_forward_probabilities(build_features(market), **kwargs)
    probabilities.rename_axis("date").reset_index().to_csv(output_csv, index=False)
    return probabilities


def build_market_from_database(
    database: Path,
    *,
    benchmark: str = "NQ=F",
    universe_group: str = "CSQ_MP_CP",
    breadth_window: int = 200,
) -> pd.DataFrame:
    """Build benchmark price and point-in-time synthetic-union breadth from SQLite."""
    with closing(sqlite3.connect(database)) as connection:
        tickers = [
            row[0]
            for row in connection.execute(
                "SELECT ticker FROM universe WHERE group_name=? AND active=1 ORDER BY ticker",
                (universe_group,),
            )
        ]
        prices = pd.read_sql_query(
            """
            SELECT ticker, date, COALESCE(adj_close, close) AS price
            FROM daily_prices
            WHERE ticker=? OR ticker IN (
                SELECT ticker FROM universe WHERE group_name=? AND active=1
            )
            ORDER BY date, ticker
            """,
            connection,
            params=(benchmark, universe_group),
            parse_dates=["date"],
        )
    if not tickers:
        raise ValueError(f"active universe group not found: {universe_group}")
    if prices.empty:
        raise ValueError("daily_prices is empty; run market_data.py download first")

    price_matrix = prices.pivot(index="date", columns="ticker", values="price")
    if benchmark not in price_matrix:
        raise ValueError(f"benchmark has no prices: {benchmark}")
    mp_prices = price_matrix.reindex(columns=tickers)
    moving_average = mp_prices.rolling(breadth_window, min_periods=breadth_window).mean()
    eligible = mp_prices.notna() & moving_average.notna()
    breadth = (mp_prices > moving_average).sum(axis=1) / eligible.sum(axis=1).replace(0, np.nan)
    return (
        pd.DataFrame(
            {"close": price_matrix[benchmark], "breadth": breadth},
            index=price_matrix.index,
        )
        .dropna()
        .rename_axis("date")
        .reset_index()
    )


def load_benchmark_from_database(database: Path, benchmark: str = "NQ=F") -> pd.DataFrame:
    with closing(sqlite3.connect(database)) as connection:
        market = pd.read_sql_query(
            """
            SELECT date, COALESCE(adj_close, close) AS close
            FROM daily_prices
            WHERE ticker=?
            ORDER BY date
            """,
            connection,
            params=(benchmark,),
            parse_dates=["date"],
        )
    if market.empty:
        raise ValueError(f"benchmark has no prices: {benchmark}")
    return market


def load_vix_curve_market_from_database(
    database: Path,
    benchmark: str = "NQ=F",
    as_of: date | str | None = None,
) -> pd.DataFrame:
    """Load the exact-date NQ/VIX/VIX3M intersection without forward filling."""
    cutoff = None if as_of is None else str(as_of)
    query = """
        SELECT ticker, date, COALESCE(adj_close, close) AS price
        FROM daily_prices
        WHERE ticker IN (?, ?, ?)
    """
    params: tuple[object, ...] = (benchmark, "^VIX", "^VIX3M")
    if cutoff is not None:
        query += " AND date <= ?"
        params += (cutoff,)
    query += " ORDER BY date, ticker"
    uri = f"file:{database.resolve().as_posix()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        prices = pd.read_sql_query(
            query,
            connection,
            params=params,
            parse_dates=["date"],
        )
    if prices.empty:
        raise ValueError("A3-A has no NQ/VIX/VIX3M prices")
    matrix = prices.pivot(index="date", columns="ticker", values="price")
    required = [benchmark, "^VIX", "^VIX3M"]
    missing = set(required) - set(matrix.columns)
    if missing:
        raise ValueError(f"A3-A missing price series: {sorted(missing)}")
    market = matrix.loc[:, required].rename(
        columns={benchmark: "close", "^VIX": "vix", "^VIX3M": "vix3m"}
    )
    market = market.apply(pd.to_numeric, errors="coerce")
    market = market.where(market > 0).dropna()
    if market.empty:
        raise ValueError("A3-A has no exact positive NQ/VIX/VIX3M dates")
    return market.rename_axis("date").reset_index()


def fit_regimes_from_database(
    database: Path,
    *,
    benchmark: str = "NQ=F",
    universe_group: str = "CSQ_MP_CP",
    **kwargs: int,
) -> pd.DataFrame:
    market = build_market_from_database(
        database, benchmark=benchmark, universe_group=universe_group
    )
    probabilities = walk_forward_probabilities(build_features(market), **kwargs)
    model_version = "hmm3-v1:" + ":".join(
        str(value)
        for value in (
            benchmark,
            universe_group,
            kwargs.get("min_train", 252),
            kwargs.get("train_window", 1260),
            kwargs.get("refit_every", 20),
        )
    )
    records = [
        (
            date.strftime("%Y-%m-%d"),
            float(row.p_bull),
            float(row.p_sideways),
            float(row.p_bear),
            model_version,
        )
        for date, row in probabilities.iterrows()
    ]
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("DELETE FROM regime_probabilities")
        connection.executemany(
            """
            INSERT INTO regime_probabilities(
                date, p_bull, p_sideways, p_bear, model_version
            ) VALUES (?, ?, ?, ?, ?)
            """,
            records,
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES ('last_regime_fit_at', ?)",
            (datetime.now(timezone.utc).isoformat(),),
        )
        connection.commit()
    return probabilities


def fit_candidate_from_database(
    database: Path,
    *,
    model_version: str | None = None,
    benchmark: str = "NQ=F",
    sticky_strength: float = 25.0,
    min_train: int = 252,
    train_window: int = 1260,
    refit_every: int = 20,
    random_state: int = 42,
    n_init: int = 3,
    feature_columns: tuple[str, ...] | None = None,
    feature_set: str = "core",
    covariance_type: str = "diag",
    covariance_floor: float | None = None,
) -> pd.DataFrame:
    if feature_set not in HMM_FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {list(HMM_FEATURE_SETS)}")
    if feature_set in {
        "independent-a2",
        "independent-a2b",
        "independent-a2c",
        "independent-a3a",
    }:
        if feature_columns is not None:
            raise ValueError(f"feature_columns cannot override the {feature_set} feature set")
        selected_features = {
            "independent-a2": INDEPENDENT_HMM_FEATURE_COLUMNS,
            "independent-a2b": INDEPENDENT_A2B_HMM_FEATURE_COLUMNS,
            "independent-a2c": INDEPENDENT_A2C_HMM_FEATURE_COLUMNS,
            "independent-a3a": INDEPENDENT_A3A_HMM_FEATURE_COLUMNS,
        }[feature_set]
    else:
        selected_features = tuple(
            BENCHMARK_HMM_FEATURE_COLUMNS if feature_columns is None else feature_columns
        )
    if (
        not selected_features
        or len(set(selected_features)) != len(selected_features)
        or (
            feature_set == "core"
            and set(selected_features) - set(BENCHMARK_HMM_FEATURE_COLUMNS)
        )
    ):
        raise ValueError("feature_columns must be unique benchmark HMM features")
    if covariance_type not in HMM_COVARIANCE_TYPES:
        raise ValueError(
            f"covariance_type must be one of {list(HMM_COVARIANCE_TYPES)}"
        )
    non_default = (
        feature_set != "core"
        or selected_features != BENCHMARK_HMM_FEATURE_COLUMNS
        or covariance_type != "diag"
        or covariance_floor is not None
    )
    if non_default and model_version is None:
        raise ValueError("model_version is required for non-default HMM semantics")
    if non_default and model_version == LIVE_HMM_MODEL_VERSION:
        raise ValueError("non-default HMM semantics cannot overwrite the live model version")
    if model_version == INDEPENDENT_A3A_HMM_MODEL_VERSION:
        frozen_a3a = (
            benchmark == "NQ=F"
            and sticky_strength == 25.0
            and min_train == 252
            and train_window == 1260
            and refit_every == 20
            and random_state == 42
            and n_init == 3
            and covariance_type == "diag"
            and covariance_floor == 0.10
            and feature_set == "independent-a3a"
        )
        if not frozen_a3a:
            raise ValueError("canonical A3-A model version requires its frozen specification")

    strength = f"{sticky_strength:g}".replace(".", "p")
    model_version = model_version or (
        f"hmm3-sticky-v2-{benchmark.lower()}-k{strength}"
        f"-m{min_train}-w{train_window}-r{refit_every}-s{random_state}-n{n_init}"
    )
    market = (
        load_vix_curve_market_from_database(database, benchmark)
        if feature_set == "independent-a3a"
        else load_benchmark_from_database(database, benchmark)
    )
    if feature_set == "independent-a2":
        features = build_independent_hmm_features(market)
    elif feature_set == "independent-a2b":
        features = build_independent_a2b_hmm_features(market)
    elif feature_set == "independent-a2c":
        features = build_independent_a2c_hmm_features(market)
    elif feature_set == "independent-a3a":
        features = build_independent_a3a_hmm_features(market)
    else:
        features = build_features_v2(market).loc[:, list(selected_features)]
    probabilities = walk_forward_probabilities(
        features,
        sticky_strength=sticky_strength,
        composite_labels=True,
        min_train=min_train,
        train_window=train_window,
        refit_every=refit_every,
        random_state=random_state,
        n_init=n_init,
        covariance_type=covariance_type,
        covariance_floor=covariance_floor,
    )
    records = [
        (
            model_version,
            current_date.strftime("%Y-%m-%d"),
            float(row.p_bull),
            float(row.p_sideways),
            float(row.p_bear),
        )
        for current_date, row in probabilities.iterrows()
    ]
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS regime_model_probabilities (
                model_version TEXT NOT NULL,
                date TEXT NOT NULL,
                p_bull REAL NOT NULL,
                p_sideways REAL NOT NULL,
                p_bear REAL NOT NULL,
                PRIMARY KEY (model_version, date)
            )
            """
        )
        connection.execute(
            "DELETE FROM regime_model_probabilities WHERE model_version=?", (model_version,)
        )
        connection.executemany(
            "INSERT INTO regime_model_probabilities VALUES (?, ?, ?, ?, ?)", records
        )
        connection.commit()
    probabilities.attrs["model_version"] = model_version
    probabilities.attrs["feature_set"] = feature_set
    return probabilities


def get_regime_status(
    database: Path,
    target_date: str,
    *,
    model_version: str | None = None,
) -> dict[str, str | float]:
    """Return the latest point-in-time regime on or before target_date."""
    try:
        date.fromisoformat(target_date)
    except ValueError as error:
        raise ValueError("date must use YYYY-MM-DD format") from error

    with closing(sqlite3.connect(database)) as connection:
        if model_version is None:
            row = connection.execute(
                """
                SELECT date, p_bull, p_sideways, p_bear
                FROM regime_probabilities
                WHERE date <= ?
                ORDER BY date DESC
                LIMIT 1
                """,
                (target_date,),
            ).fetchone()
        else:
            row = connection.execute(
                """
                SELECT date, p_bull, p_sideways, p_bear
                FROM regime_model_probabilities
                WHERE model_version=? AND date <= ?
                ORDER BY date DESC
                LIMIT 1
                """,
                (model_version, target_date),
            ).fetchone()
    if row is None:
        raise ValueError(f"no regime probability on or before {target_date}")

    probabilities = {"BULL": float(row[1]), "SIDEWAYS": float(row[2]), "BEAR": float(row[3])}
    state = max(probabilities, key=probabilities.get)
    return {
        "date": row[0],
        "model_version": model_version or "hmm3-v1",
        "state": state,
        **{f"p_{key.lower()}": value for key, value in probabilities.items()},
    }


def _simulate_episode(
    dates: pd.DatetimeIndex,
    returns: pd.Series,
    target_exposure: pd.Series,
    transaction_cost_bps: float,
) -> tuple[pd.Series, float]:
    wealth = 1.0
    exposure = 1.0
    total_cost = 0.0
    path: list[float] = []

    for offset, date in enumerate(dates):
        if offset:
            wealth *= 1 + exposure * float(returns.loc[date])
        target = float(np.clip(target_exposure.loc[date], 0.0, 1.0))
        cost = abs(target - exposure) * transaction_cost_bps / 10_000
        wealth *= 1 - cost
        total_cost += cost
        exposure = target
        path.append(wealth)
    return pd.Series(path, index=dates), total_cost


def evaluate_actions(
    portfolio: pd.DataFrame,
    probabilities: pd.DataFrame,
    events: pd.DataFrame,
    *,
    transaction_cost_bps: float = 10.0,
    minimum_hmm_exposure: float = 0.0,
) -> pd.DataFrame:
    """Compare actual sell, hold, 50% de-risk, and dynamic HMM exposure."""
    for name, frame, required in (
        ("portfolio", portfolio, {"date", "nav"}),
        ("probabilities", probabilities, {"date", "p_bear"}),
        ("events", events, {"sell_date", "reentry_date", "end_date"}),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} CSV missing columns: {sorted(missing)}")

    prices = portfolio.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    prices["nav"] = pd.to_numeric(prices["nav"], errors="raise")
    prices["return"] = prices["nav"].pct_change().fillna(0.0)

    probs = probabilities.copy()
    probs["date"] = pd.to_datetime(probs["date"])
    probs = probs.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    prices["p_bear"] = pd.to_numeric(probs["p_bear"], errors="coerce").reindex(prices.index).ffill()

    rows: list[dict[str, object]] = []
    for number, event in events.reset_index(drop=True).iterrows():
        sell_date = pd.Timestamp(event["sell_date"])
        reentry_date = pd.Timestamp(event["reentry_date"])
        end_date = pd.Timestamp(event["end_date"])
        if not sell_date < reentry_date <= end_date:
            raise ValueError("each event must satisfy sell_date < reentry_date <= end_date")

        episode = prices.loc[sell_date:end_date].copy()
        if episode.empty or episode.index[0] != sell_date or episode.index[-1] != end_date:
            raise ValueError(f"event dates must exist in portfolio data: event {number + 1}")
        if episode["p_bear"].isna().any():
            raise ValueError(f"p_bear unavailable in event {number + 1}")

        before_reentry = episode.index < reentry_date
        exposures = {
            "ACTUAL_FULL_SELL": pd.Series(np.where(before_reentry, 0.0, 1.0), index=episode.index),
            "HOLD": pd.Series(1.0, index=episode.index),
            "DE_RISK_50": pd.Series(np.where(before_reentry, 0.5, 1.0), index=episode.index),
            "HMM_POLICY": (1.0 - episode["p_bear"]).clip(lower=minimum_hmm_exposure, upper=1.0),
        }
        results: dict[str, tuple[pd.Series, float]] = {
            action: _simulate_episode(
                episode.index, episode["return"], exposure, transaction_cost_bps
            )
            for action, exposure in exposures.items()
        }
        hold_path = results["HOLD"][0]
        event_name = event.get("event", f"event_{number + 1}")

        for action, (path, cost) in results.items():
            drawdown = path / path.cummax() - 1
            rows.append(
                {
                    "event": event_name,
                    "action": action,
                    "sell_date": sell_date.date(),
                    "reentry_date": reentry_date.date(),
                    "end_date": end_date.date(),
                    "p_bear_on_sell": float(episode["p_bear"].iloc[0]),
                    "ending_wealth": float(path.iloc[-1]),
                    "total_return": float(path.iloc[-1] - 1),
                    "max_drawdown": float(drawdown.min()),
                    "loss_avoided_vs_hold": float(max(0.0, path.min() - hold_path.min())),
                    "missed_rebound_vs_hold": float(max(0.0, hold_path.iloc[-1] - path.iloc[-1])),
                    "transaction_cost": cost,
                }
            )
    return pd.DataFrame(rows)


def evaluate_from_csv(
    portfolio_csv: Path,
    probabilities_csv: Path,
    events_csv: Path,
    output_csv: Path,
    **kwargs: float,
) -> pd.DataFrame:
    result = evaluate_actions(
        pd.read_csv(portfolio_csv),
        pd.read_csv(probabilities_csv),
        pd.read_csv(events_csv),
        **kwargs,
    )
    result.to_csv(output_csv, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-state market regime router")
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("fit", help="create point-in-time regime probabilities")
    fit.add_argument("market_csv", type=Path)
    fit.add_argument("output_csv", type=Path)
    fit.add_argument("--min-train", type=int, default=252)
    fit.add_argument("--train-window", type=int, default=1260)
    fit.add_argument("--refit-every", type=int, default=20)

    fit_db = commands.add_parser("fit-db", help="fit regimes from SQLite market data")
    fit_db.add_argument("database", type=Path)
    fit_db.add_argument("--benchmark", default="NQ=F")
    fit_db.add_argument("--universe-group", default="CSQ_MP_CP")
    fit_db.add_argument("--min-train", type=int, default=252)
    fit_db.add_argument("--train-window", type=int, default=1260)
    fit_db.add_argument("--refit-every", type=int, default=20)

    fit_db_v2 = commands.add_parser(
        "fit-db-v2", help="fit an isolated sticky-HMM candidate from SQLite market data"
    )
    fit_db_v2.add_argument("database", type=Path)
    fit_db_v2.add_argument("--model-version")
    fit_db_v2.add_argument("--benchmark", default="NQ=F")
    fit_db_v2.add_argument("--min-train", type=int, default=252)
    fit_db_v2.add_argument("--train-window", type=int, default=1260)
    fit_db_v2.add_argument("--refit-every", type=int, default=20)
    fit_db_v2.add_argument("--sticky-strength", type=float, default=25.0)
    fit_db_v2.add_argument("--random-state", type=int, default=42)
    fit_db_v2.add_argument("--n-init", type=int, default=3)
    fit_db_v2.add_argument(
        "--feature-set",
        choices=HMM_FEATURE_SETS,
        default="core",
    )
    fit_db_v2.add_argument(
        "--feature-columns",
        nargs="+",
        choices=BENCHMARK_HMM_FEATURE_COLUMNS,
    )
    fit_db_v2.add_argument(
        "--covariance-type",
        choices=HMM_COVARIANCE_TYPES,
        default="diag",
    )
    fit_db_v2.add_argument("--covariance-floor", type=float)

    status_date = commands.add_parser("status-date", help="show regime on or before a date")
    status_date.add_argument("database", type=Path)
    status_date.add_argument("date")
    status_date.add_argument("--model-version")

    evaluate = commands.add_parser("evaluate", help="evaluate drawdown actions")
    evaluate.add_argument("portfolio_csv", type=Path)
    evaluate.add_argument("probabilities_csv", type=Path)
    evaluate.add_argument("events_csv", type=Path)
    evaluate.add_argument("output_csv", type=Path)
    evaluate.add_argument("--cost-bps", type=float, default=10.0)
    evaluate.add_argument("--minimum-hmm-exposure", type=float, default=0.0)

    args = parser.parse_args()
    if args.command == "fit":
        result = fit_regimes(
            args.market_csv,
            args.output_csv,
            min_train=args.min_train,
            train_window=args.train_window,
            refit_every=args.refit_every,
        )
    elif args.command == "fit-db":
        result = fit_regimes_from_database(
            args.database,
            benchmark=args.benchmark,
            universe_group=args.universe_group,
            min_train=args.min_train,
            train_window=args.train_window,
            refit_every=args.refit_every,
        )
    elif args.command == "fit-db-v2":
        result = fit_candidate_from_database(
            args.database,
            model_version=args.model_version,
            benchmark=args.benchmark,
            sticky_strength=args.sticky_strength,
            min_train=args.min_train,
            train_window=args.train_window,
            refit_every=args.refit_every,
            random_state=args.random_state,
            n_init=args.n_init,
            feature_columns=(
                None if args.feature_columns is None else tuple(args.feature_columns)
            ),
            feature_set=args.feature_set,
            covariance_type=args.covariance_type,
            covariance_floor=args.covariance_floor,
        )
    elif args.command == "status-date":
        status = get_regime_status(
            args.database, args.date, model_version=args.model_version
        )
        print(
            f"{status['date']} | {status['model_version']} | {status['state']} | "
            f"BULL {status['p_bull']:.2%} / SIDEWAYS {status['p_sideways']:.2%} / BEAR {status['p_bear']:.2%}"
        )
        from research_drawdown_onset import get_shadow_hazard_status

        try:
            hazard = get_shadow_hazard_status(args.database, args.date)
        except (ValueError, sqlite3.OperationalError):
            hazard = None
        if hazard is not None:
            score = "N/A" if hazard["score"] is None else f"{hazard['score']:.3f}"
            print(
                f"ONSET SHADOW {hazard['date']} | {hazard['risk_band']} | score {score} | "
                "next-session review only; not an order"
            )
        return
    else:
        result = evaluate_from_csv(
            args.portfolio_csv,
            args.probabilities_csv,
            args.events_csv,
            args.output_csv,
            transaction_cost_bps=args.cost_bps,
            minimum_hmm_exposure=args.minimum_hmm_exposure,
        )
    if isinstance(result.index, pd.DatetimeIndex):
        if result.attrs.get("model_version"):
            print(f"model_version={result.attrs['model_version']}")
        if result.attrs.get("refits") is not None:
            print(
                f"refits={result.attrs['refits']} | "
                f"unstable_refits={result.attrs['unstable_refits']} | "
                f"selected_seeds={result.attrs['selected_seed_counts']}"
            )
        if result.attrs.get("feature_columns"):
            print(
                f"features={result.attrs['feature_columns']} | "
                f"covariance={result.attrs['covariance_type']} | "
                f"covariance_floor={result.attrs['covariance_floor']} | "
                f"states={result.attrs['n_components']}"
            )
        result = result.rename_axis("date").reset_index()
    print(result.tail().to_string(index=False))


if __name__ == "__main__":
    main()
