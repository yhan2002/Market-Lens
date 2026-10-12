"""Reproduce the authorized ETF-ratio study; cached data by default, no trading."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/analysis/nasdaq_ratio_20260909"
TICKERS = ["IWN", "SPY", "XLY", "XLP", "HYG", "LQD", "QQQ"]
DAYS = [1, 5, 20]


def hac(y, x, lag):
    """OLS with Bartlett Newey-West covariance, no small-sample correction."""
    x = np.column_stack([np.ones(len(y)), np.asarray(x)])
    y = np.asarray(y)
    bread = np.linalg.inv(x.T @ x)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    scores = x * (y - x @ beta)[:, None]
    meat = scores.T @ scores
    for k in range(1, min(lag, len(y) - 1) + 1):
        cross = scores[k:].T @ scores[:-k]
        meat += (1 - k / (lag + 1)) * (cross + cross.T)
    se = np.sqrt(np.maximum(np.diag(bread @ meat @ bread), 0))
    return beta, se, 2 * norm.sf(np.abs(beta / se))


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    result = np.empty(len(p))
    result[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return result


def future(q, h, gap=0):
    # gap=0: close(t) to close(t+h); gap=1: one full session before entry.
    return q.shift(-(h + gap)) - q.shift(-gap)


def oos(frame, controls):
    """Annual expanding fits, labels must finish before the evaluation year."""
    pieces = []
    for year in range(2015, 2027):
        start = pd.Timestamp(year, 1, 1)
        train = frame.loc[frame.end < start]
        test = frame.loc[frame.index.year == year]
        if len(train) < 750 or test.empty:
            continue
        result = test[["y", "signal"]].copy()
        result["mean"] = train.y.mean()
        for name, cols in [("signal_only", ["signal"]), ("baseline", controls),
                           ("augmented", controls + ["signal"])]:
            a = np.column_stack([np.ones(len(train)), train[cols]])
            b = np.linalg.lstsq(a, train.y, rcond=None)[0]
            result[name] = np.column_stack([np.ones(len(test)), test[cols]]) @ b
        pieces.append(result)
    return pd.concat(pieces)


def self_test():
    q = pd.Series([0., 1., 3., 6., 10.])
    assert future(q, 2).iloc[0] == 3
    assert future(q, 2, 1).iloc[0] == 5
    assert np.allclose(holm([.01, .04, .03]), [.03, .06, .06])
    rng = np.random.default_rng(47)
    x = rng.normal(size=(200, 2))
    y = x @ [0.3, -0.7] + rng.normal(size=200)
    b, se, _ = hac(y, x, 0)
    a = np.column_stack([np.ones(200), x])
    inv = np.linalg.inv(a.T @ a)
    u = y - a @ b
    expected = inv @ (a.T @ np.diag(u ** 2) @ a) @ inv
    assert np.allclose(se ** 2, np.diag(expected))
    _, se2, _ = hac(y, x, 2)
    weights = np.maximum(0, 1 - np.abs(np.arange(200)[:, None] - np.arange(200)) / 3)
    expected2 = inv @ (a.T @ (np.outer(u, u) * weights) @ a) @ inv
    assert np.allclose(se2 ** 2, np.diag(expected2))
    dates = pd.bdate_range("2010-01-01", "2016-12-31")
    f = pd.DataFrame({"y": rng.normal(size=len(dates)), "signal": rng.normal(size=len(dates)),
                      "control": rng.normal(size=len(dates)), "end": dates + pd.offsets.BDay(20)}, index=dates)
    before = oos(f, ["control"])
    changed = f.copy()
    changed.loc[changed.end >= "2015-01-01", "y"] += 100
    after = oos(changed, ["control"])
    assert np.allclose(before.loc["2015", "augmented"], after.loc["2015", "augmented"])
    print("PASS: forward labels, Holm correction, HAC covariance, OOS label embargo")


def run(download=False):
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "yahoo_daily.pkl"
    if download:
        import yfinance as yf
        yf.set_tz_cache_location(str(OUT / "yf_cache"))
        data = yf.download(TICKERS, start="2007-01-01", end="2026-09-09",
                           auto_adjust=False, actions=False, progress=False, threads=False, timeout=20)
        if data.empty or any(data["Close"][s].count() < 4000 for s in TICKERS):
            raise ValueError("Incomplete download; existing cache preserved")
        data.to_pickle(cache)
    data = pd.read_pickle(cache)  # Only this script's locally generated cache.
    if data.index.has_duplicates or not data.index.is_monotonic_increasing:
        raise ValueError("Dates must be unique and increasing")
    prices = data.loc[:, pd.IndexSlice[["Close", "Adj Close"], TICKERS]]
    common = prices.dropna()
    assert len(common) > 4000 and np.isfinite(common.to_numpy()).all() and (common > 0).all().all()
    assert (common.index < pd.Timestamp("2026-09-09")).all()
    common.to_csv(OUT / "prices.csv")
    quality = {}
    for field in ["Close", "Adj Close"]:
        p = common[field]
        quality[field] = {"max_abs_daily_log_return": np.log(p).diff().abs().max().to_dict()}
    metadata = {"source": "Yahoo Finance via yfinance", "end_exclusive": "2026-09-09",
                "start": str(common.index.min().date()), "end": str(common.index.max().date()),
                "common_sessions": len(common), "raw_rows": len(data),
                "counts": data["Close"].count().to_dict(), "quality": quality,
                "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
                "oos": "2015 onward; yearly expanding OLS; target end strictly before test year",
                "controls": "QQQ past 1/5/20-session log returns and 20-session daily volatility",
                "inference": "Bartlett HAC maxlags=max(20,lookback+horizon); asymptotic two-sided normal p",
                "multiplicity": "Holm across all 144 univariate/controlled tests, both price modes, all 4 ratios",
                "limitations": "Retrospective historical OOS, not prospective; no costs/trading performance; no turning-point test"}
    rows, subperiods, delayed, predictions = [], [], [], []
    for field in ["Adj Close", "Close"]:
        p = np.log(common[field])
        q = p.QQQ
        ratios = pd.DataFrame({"IWN_SPY": p.IWN - p.SPY, "XLY_XLP": p.XLY - p.XLP,
                               "HYG_LQD": p.HYG - p.LQD})
        ratios["composite"] = ratios.sum(axis=1)
        controls = pd.DataFrame({f"qqq_{k}": q.diff(k) for k in DAYS})
        controls["vol20"] = q.diff().rolling(20).std()
        for name in ratios:
            for lookback in DAYS:
                signal = ratios[name].diff(lookback)
                for h in DAYS:
                    f = controls.assign(signal=signal, y=future(q, h),
                                        end=pd.Series(q.index, index=q.index).shift(-h)).dropna()
                    x = f.signal.to_numpy() / f.signal.std()
                    beta, se, pv = hac(f.y, x, max(20, lookback + h))
                    bc, sc, pc = hac(f.y, np.column_stack([f[controls.columns], x]), max(20, lookback + h))
                    pred = oos(f, list(controls.columns))
                    mse = {m: float(((pred.y - pred[m]) ** 2).mean())
                           for m in ["mean", "signal_only", "baseline", "augmented"]}
                    row = {"price": field, "factor": name, "lookback": lookback, "horizon": h,
                           "n": len(f), "pearson": f.signal.corr(f.y),
                           "spearman": f.signal.corr(f.y, method="spearman"),
                           "concurrent_corr": signal.corr(q.diff(lookback)),
                           "beta_1sd": beta[-1], "ci_low": beta[-1] - 1.96 * se[-1],
                           "ci_high": beta[-1] + 1.96 * se[-1], "p_hac": pv[-1],
                           "controlled_beta_1sd": bc[-1], "p_controlled": pc[-1],
                           "oos_n": len(pred), "oos_r2_vs_mean": 1 - mse["signal_only"] / mse["mean"],
                           "oos_increment_vs_qqq": 1 - mse["augmented"] / mse["baseline"],
                           "oos_direction_accuracy": ((pred.signal_only > 0) == (pred.y > 0)).mean(),
                           "oos_always_up_accuracy": (pred.y > 0).mean()}
                    rows.append(row)
                    if name == "composite":
                        for start, stop in [("2007", "2014"), ("2015", "2019"), ("2020", "2026")]:
                            sub = f.loc[start:stop]
                            sb, ss, sp = hac(sub.y, sub.signal / sub.signal.std(), max(20, lookback + h))
                            subperiods.append({"price": field, "lookback": lookback, "horizon": h,
                                               "period": start + "-" + stop, "n": len(sub),
                                               "correlation": sub.signal.corr(sub.y), "p_hac_unadjusted": sp[-1]})
                        delay = pd.concat([signal.rename("x"), future(q, h, 1).rename("y")], axis=1).dropna()
                        db, ds, dp = hac(delay.y, delay.x / delay.x.std(), max(20, lookback + h + 1))
                        delayed.append({"price": field, "lookback": lookback, "horizon": h,
                                        "n": len(delay), "correlation": delay.x.corr(delay.y), "p_hac": dp[-1]})
                        if field == "Adj Close":
                            predictions.append(pred.assign(lookback=lookback, horizon=h))
    results = pd.DataFrame(rows)
    corrected = holm(np.r_[results.p_hac, results.p_controlled])
    results["p_holm"] = corrected[:len(results)]
    results["p_controlled_holm"] = corrected[len(results):]
    results.to_csv(OUT / "statistics.csv", index=False)
    pd.DataFrame(subperiods).to_csv(OUT / "subperiods.csv", index=False)
    d = pd.DataFrame(delayed)
    d["p_holm"] = holm(d.p_hac)
    d.to_csv(OUT / "one_session_delay.csv", index=False)
    pd.concat(predictions).to_csv(OUT / "oos_predictions.csv")
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(results.loc[results.factor == "composite"].to_string(index=False))
    print("SIGNIFICANT AFTER HOLM:")
    print(results.loc[(results.p_holm < .05) | (results.p_controlled_holm < .05)].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Refresh the 7-ETF Yahoo cache")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run(args.download)
