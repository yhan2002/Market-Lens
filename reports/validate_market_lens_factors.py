"""Read-only Market Lens factor attribution; no downloads or production updates."""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from validate_nasdaq_ratio import future, hac, holm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/analysis/market_lens_factors_20260909"
VERSION = "roro-3axis-v1-w756-m252-p126-u20__roro-phase-v1__p-c85-isqc__shadow-r19-holdings-history-recalc"
FACTORS = ["credit_score", "rates_volatility_score", "equity_volatility_score",
           "fx_carry_score", "safe_haven_score", "equity_trend_score", "breadth_risk_percentile"]
AXES = ["macro_risk_score", "market_trend_score", "portfolio_internal_score"]


def shapley_r2(x, y):
    """Exact LMG/Shapley allocation of in-sample OLS R2, including shared variance."""
    x, y = np.asarray(x), np.asarray(y)
    n, p = x.shape
    total = ((y - y.mean()) ** 2).sum()
    values = {0: 0.0}
    for mask in range(1, 1 << p):
        cols = [j for j in range(p) if mask & (1 << j)]
        a = np.column_stack([np.ones(n), x[:, cols]])
        residual = y - a @ np.linalg.lstsq(a, y, rcond=None)[0]
        values[mask] = 1 - residual @ residual / total
    shares = np.zeros(p)
    for j in range(p):
        for mask in range(1 << p):
            if not mask & (1 << j):
                k = mask.bit_count()
                shares[j] += (values[mask | (1 << j)] - values[mask]) / (p * math.comb(p - 1, k))
    assert np.isclose(shares.sum(), values[(1 << p) - 1])
    return shares, values[(1 << p) - 1]


def load():
    rows = []
    for relative in ["outputs/roro_backfill_202509/roro_shadow_r19.db", "outputs/roro_shadow/roro_shadow.db"]:
        with sqlite3.connect((ROOT / relative).resolve().as_uri() + "?mode=ro", uri=True) as c:
            for date, payload in c.execute(
                "SELECT date,payload_json FROM roro_shadow_observations WHERE series_version=? ORDER BY date", (VERSION,)
            ):
                j = json.loads(payload)
                rows.append({"date": pd.Timestamp(date), **{k: j.get(k) for k in FACTORS + AXES}})
    scores = pd.DataFrame(rows).set_index("date").sort_index()
    if scores.index.has_duplicates:
        raise ValueError("Overlapping snapshot dates must be reviewed; refusing automatic overwrite")
    with sqlite3.connect((ROOT / "regime_router_publish.db").resolve().as_uri() + "?mode=ro", uri=True) as c:
        q = pd.read_sql_query("SELECT date, COALESCE(adj_close,close) AS nq FROM daily_prices WHERE ticker='NQ=F' AND date<='2026-09-08' ORDER BY date", c, parse_dates=["date"]).set_index("date").nq
    assert q.index.is_unique
    expected = q.loc[scores.index.min():scores.index.max()].index
    assert scores.index.equals(expected), "Missing score sessions would distort differences"
    assert ((scores.ge(0) & scores.le(100)) | scores.isna()).all().all()
    return scores, np.log(q)


def holdout(frame, columns):
    """Fixed April 2026 holdout; monthly expanding refits with label purging."""
    pieces = []
    baseline = ["nq1", "nq5", "nq20", "vol20"]
    for start in pd.date_range("2026-04-01", "2026-09-01", freq="MS"):
        train = frame.loc[frame.end < start]
        test = frame.loc[(frame.index >= start) & (frame.index < start + pd.offsets.MonthBegin(1))]
        if len(train) < 80 or test.empty:
            continue
        result = test[["y"]].copy()
        result["mean"] = train.y.mean()
        for label, cols in [("factors", columns), ("baseline", baseline), ("augmented", baseline + columns)]:
            mu, sd = train[cols].mean(), train[cols].std()
            if (sd <= 1e-12).any():
                raise ValueError("Constant training predictor")
            a = np.column_stack([np.ones(len(train)), (train[cols] - mu) / sd])
            b = np.linalg.lstsq(a, train.y, rcond=None)[0]
            result[label] = np.column_stack([np.ones(len(test)), (test[cols] - mu) / sd]) @ b
        pieces.append(result)
    return pd.concat(pieces)


def self_test():
    x = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    s, r = shapley_r2(x, x @ [1., 2.])
    assert np.allclose(s, [.2, .8]) and np.isclose(r, 1)
    s, r = shapley_r2(np.column_stack([x[:, 0], x[:, 0]]), x[:, 0])
    assert np.allclose(s, [.5, .5]) and np.isclose(r, 1)
    rng = np.random.default_rng(12)
    dates = pd.bdate_range("2025-09-01", "2026-09-01")
    cols = ["signal", "nq1", "nq5", "nq20", "vol20"]
    f = pd.DataFrame(rng.normal(size=(len(dates), len(cols))), index=dates, columns=cols)
    f["y"] = rng.normal(size=len(dates))
    f["end"] = dates + pd.offsets.BDay(20)
    before = holdout(f, ["signal"])
    f.loc[f.end >= "2026-04-01", "y"] += 100
    after = holdout(f, ["signal"])
    assert np.allclose(before.loc["2026-04", "augmented"], after.loc["2026-04", "augmented"])
    print("PASS: exact Shapley allocation, duplicate-factor symmetry, purged holdout")


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    scores, q = load()
    scores.to_csv(OUT / "score_inputs.csv")
    controls = pd.DataFrame({"nq1": q.diff(), "nq5": q.diff(5), "nq20": q.diff(20),
                             "vol20": q.diff().rolling(20).std()})
    coefficients, models, stability, predictions = [], [], [], []
    for group, columns in [("market_macro_7", FACTORS), ("axes_3", AXES)]:
        for mode in ["concurrent_change", "future_level", "future_change5"]:
            for h in [1, 5, 20]:
                x = scores[columns].diff(h if mode == "concurrent_change" else 5) if mode != "future_level" else scores[columns]
                y = q.diff(h) if mode == "concurrent_change" else future(q, h)
                end = pd.Series(q.index, index=q.index) if mode == "concurrent_change" else pd.Series(q.index, index=q.index).shift(-h)
                f = x.join(controls).assign(y=y, end=end).dropna()
                z = (f[columns] - f[columns].mean()) / f[columns].std()
                beta, se, pv = hac(f.y, z, max(20, h + 5))
                shares, r2 = shapley_r2(z, f.y)
                row = {"group": group, "mode": mode, "horizon": h, "n": len(f),
                       "start": str(f.index.min().date()), "end": str(f.index.max().date()),
                       "r2": r2, "unexplained": 1-r2, "condition_number": np.linalg.cond(z)}
                for i, col in enumerate(columns):
                    coefficients.append({**{k: row[k] for k in ["group", "mode", "horizon", "n"]},
                                         "factor": col, "beta_1sd": beta[i+1],
                                         "ci_low": beta[i+1]-1.96*se[i+1], "ci_high": beta[i+1]+1.96*se[i+1],
                                         "p_hac": pv[i+1], "r2_contribution": shares[i],
                                         "share_of_explained": shares[i]/r2,
                                         "univariate_corr": f[col].corr(f.y)})
                for period, sub in [("early", f.loc[:"2026-03-31"]), ("late", f.loc["2026-04-01":])]:
                    subshares, subr2 = shapley_r2(sub[columns], sub.y)
                    for col, share in zip(columns, subshares):
                        stability.append({"group": group, "mode": mode, "horizon": h, "period": period,
                                          "n": len(sub), "factor": col, "r2_contribution": share,
                                          "share_of_explained": share/subr2, "r2": subr2})
                if mode != "concurrent_change":
                    pred = holdout(f, columns)
                    mse = {k: ((pred.y - pred[k]) ** 2).mean() for k in ["mean", "factors", "baseline", "augmented"]}
                    row.update({"oos_n": len(pred), "oos_r2_mean": 1-mse["factors"]/mse["mean"],
                                "oos_increment_baseline": 1-mse["augmented"]/mse["baseline"],
                                "direction_accuracy": ((pred.factors>0) == (pred.y>0)).mean(),
                                "always_up_accuracy": (pred.y>0).mean()})
                    predictions.append(pred.assign(group=group, mode=mode, horizon=h))
                models.append(row)
    c = pd.DataFrame(coefficients)
    c["p_holm"] = holm(c.p_hac)
    c.to_csv(OUT / "factor_statistics.csv", index=False)
    pd.DataFrame(models).to_csv(OUT / "model_statistics.csv", index=False)
    pd.DataFrame(stability).to_csv(OUT / "subperiod_attribution.csv", index=False)
    pd.concat(predictions).to_csv(OUT / "holdout_predictions.csv")
    # Diagnostic: remove the two scores that explicitly contain NQ itself.
    external = [k for k in FACTORS if k not in ["equity_trend_score", "safe_haven_score"]]
    sensitivity = []
    for h in [1, 5, 20]:
        f = scores[FACTORS].diff(h).assign(y=q.diff(h)).dropna()
        shares, r2 = shapley_r2(f[external], f.y)
        for col, share in zip(external, shares):
            sensitivity.append({"horizon": h, "n": len(f), "factor": col,
                                "r2": r2, "r2_contribution": share,
                                "share_of_explained": share/r2})
    pd.DataFrame(sensitivity).to_csv(OUT / "mechanical_overlap_sensitivity.csv", index=False)
    metadata = {"version": VERSION, "score_start": str(scores.index.min().date()),
                "score_end": str(scores.index.max().date()), "score_rows": len(scores),
                "missing_scores": scores.isna().sum().to_dict(),
                "target": "NQ=F continuous futures adjusted close proxy, not executable futures return",
                "attribution": "Exact LMG/Shapley allocation of in-sample linear-model R2, not causal or portfolio weights",
                "p_values": "HAC maxlags=max(20,h+5); Holm over all 90 coefficients",
                "holdout": "2026-04-01 onward, monthly expanding, label end < month start, minimum 80 training rows",
                "caveats": ["short retrospective reconstructed sample", "overlapping 5/20-day labels",
                            "equity trend and gold relative strength mechanically contain NQ",
                            "FX/gold/breadth availability aligned by existing source contract",
                            "No new downloads, production writes, trades or weight changes"]}
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(pd.DataFrame(models).to_string(index=False))
    print(c.loc[(c.group == "market_macro_7") & (c.horizon == 5)].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test() if args.self_test else run()
