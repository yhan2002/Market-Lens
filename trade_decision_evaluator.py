from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from trading_log import _unit_value, infer_actions, read_holdings_workbook


HORIZONS = (1, 5, 20)
SCOPES = ("TOTAL", "BUY", "SELL")
BOOTSTRAP_BLOCK_DAYS = 20
DEFAULT_SEED = 1729


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _record_price(record: dict[str, object] | None) -> tuple[float | None, str | None]:
    if record:
        try:
            unit_value = _number(_unit_value(record))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            unit_value = None
        if unit_value is not None and unit_value > 0:
            return unit_value, "VALUATION_UNIT_PROXY"
        close = _number(record.get("close_local"))
        if close is not None and close > 0:
            return close, "LOCAL_CLOSE_FALLBACK"
    return None, None


def _snapshots(holdings: list[dict[str, object]]) -> dict[str, dict[str, dict[str, object]]]:
    result: dict[str, dict[str, dict[str, object]]] = {}
    for row in holdings:
        result.setdefault(str(row["date"]), {})[str(row["isin"])] = row
    return result


def infer_market_sessions(holdings: list[dict[str, object]]) -> list[str]:
    """Infer market sessions without treating stale weekend snapshots as sessions."""
    snapshots = _snapshots(holdings)
    observed_dates = sorted(snapshots)
    sessions: list[str] = []
    for previous_date, current_date in zip(observed_dates, observed_dates[1:]):
        try:
            if datetime.fromisoformat(current_date).weekday() >= 5:
                continue
        except ValueError:
            continue
        previous = snapshots[previous_date]
        current = snapshots[current_date]
        previous_isins = set(previous)
        current_isins = set(current)
        changed = previous_isins != current_isins
        for isin in previous_isins & current_isins:
            previous_price, _ = _record_price(previous[isin])
            current_price, _ = _record_price(current[isin])
            previous_quantity = _number(previous[isin].get("quantity"))
            current_quantity = _number(current[isin].get("quantity"))
            price_changed = (
                previous_price is not None
                and current_price is not None
                and not math.isclose(
                    previous_price, current_price, rel_tol=1e-12, abs_tol=1e-12
                )
            )
            quantity_changed = (
                previous_quantity != current_quantity
                if previous_quantity is None or current_quantity is None
                else not math.isclose(
                    previous_quantity,
                    current_quantity,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
            if price_changed or quantity_changed:
                changed = True
                break
        if changed:
            sessions.append(current_date)
    return sessions


def _future_session(session_dates: list[str], action_date: str, horizon: int) -> str | None:
    index = bisect_right(session_dates, action_date) + horizon - 1
    return session_dates[index] if index < len(session_dates) else None


def _evaluation_quality(action: dict[str, object], session_dates: set[str]) -> str:
    flags: list[str] = []
    source_flag = str(action.get("quality_flag") or "OK")
    if source_flag != "OK":
        flags.append(source_flag)
    if str(action["date"]) not in session_dates:
        flags.append("NON_SESSION_DATE")
    return "|".join(flags) if flags else "OK"


def _enrich_actions(
    holdings: list[dict[str, object]],
    actions: list[dict[str, object]],
    *,
    cost_bps: float,
    session_dates: list[str],
) -> list[dict[str, object]]:
    snapshots = _snapshots(holdings)
    session_set = set(session_dates)
    cost_rate = cost_bps / 10_000.0
    enriched: list[dict[str, object]] = []
    for source in sorted(actions, key=lambda row: (str(row["date"]), str(row["isin"]))):
        row = dict(source)
        action_date = str(row["date"])
        previous_date = str(row["previous_date"])
        isin = str(row["isin"])
        direction = 1 if row["side"] == "BUY" else -1
        evaluation_quality = _evaluation_quality(row, session_set)
        action_record = snapshots.get(action_date, {}).get(isin)
        base_price, base_price_source = _record_price(action_record)
        base_date_source = "ACTION_DATE"
        if base_price is None:
            previous_record = snapshots.get(previous_date, {}).get(isin)
            base_price, base_price_source = _record_price(previous_record)
            base_date_source = "PREVIOUS_CLOSE_PROXY" if base_price is not None else "MISSING"

        notional = _number(row.get("action_notional_base"))
        row.update(
            {
                "direction": direction,
                "source_quality_flag": str(row.get("quality_flag") or "OK"),
                "evaluation_quality_flag": evaluation_quality,
                "statistics_included": evaluation_quality == "OK",
                "abs_notional_base": abs(notional) if notional is not None else None,
                "base_price": base_price,
                "base_date_source": base_date_source,
                "base_price_source": base_price_source,
            }
        )
        for horizon in HORIZONS:
            future_date = _future_session(session_dates, action_date, horizon)
            future_record = snapshots.get(future_date, {}).get(isin) if future_date else None
            future_price, future_source = _record_price(future_record)
            underlying_return = None
            gross = None
            net = None
            if base_price is not None and future_price is not None:
                underlying_return = future_price / base_price - 1.0
                gross = direction * underlying_return
                net = gross - cost_rate
            row.update(
                {
                    f"future_date_{horizon}": future_date,
                    f"future_price_{horizon}": future_price,
                    f"future_price_source_{horizon}": future_source,
                    f"underlying_return_{horizon}": underlying_return,
                    f"gross_directional_return_{horizon}": gross,
                    f"net_markout_{horizon}": net,
                }
            )
        enriched.append(row)
    return enriched


def _weighted_mean(rows: list[dict[str, object]], field: str) -> float | None:
    weighted = [
        (_number(row.get(field)), _number(row.get("abs_notional_base"))) for row in rows
    ]
    usable = [(value, weight) for value, weight in weighted if value is not None and weight and weight > 0]
    if not usable:
        return None
    total_weight = sum(weight for _, weight in usable)
    return sum(value * weight for value, weight in usable) / total_weight


def _daily_markouts(
    rows: list[dict[str, object]], horizon: int
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if not row["statistics_included"]:
            continue
        if _number(row.get(f"net_markout_{horizon}")) is None:
            continue
        grouped.setdefault(str(row["date"]), []).append(row)
    daily: list[dict[str, object]] = []
    for day, day_rows in sorted(grouped.items()):
        weighted_rows = [
            row
            for row in day_rows
            if (_number(row.get("abs_notional_base")) or 0) > 0
        ]
        if not weighted_rows:
            continue
        daily.append(
            {
                "date": day,
                "gross": _weighted_mean(weighted_rows, f"gross_directional_return_{horizon}"),
                "net": _weighted_mean(weighted_rows, f"net_markout_{horizon}"),
                "notional": sum(float(row["abs_notional_base"]) for row in weighted_rows),
                "actions": len(weighted_rows),
            }
        )
    return daily


def _moving_block_ci(
    daily: list[dict[str, object]],
    session_dates: list[str],
    *,
    samples: int,
    seed: int,
) -> dict[str, list[float | None]]:
    if not daily or samples <= 0:
        return {"gross": [None, None], "net": [None, None]}
    by_date = {str(row["date"]): row for row in daily}
    first, last = min(by_date), max(by_date)
    calendar = [day for day in session_dates if first <= day <= last]
    if not calendar:
        return {"gross": [None, None], "net": [None, None]}
    block = min(BOOTSTRAP_BLOCK_DAYS, len(calendar))
    blocks_needed = math.ceil(len(calendar) / block)
    rng = np.random.default_rng(seed)
    gross_samples: list[float] = []
    net_samples: list[float] = []
    for _ in range(samples):
        sampled_dates: list[str] = []
        for _ in range(blocks_needed):
            start = int(rng.integers(0, len(calendar) - block + 1))
            sampled_dates.extend(calendar[start : start + block])
        sampled = [by_date[day] for day in sampled_dates[: len(calendar)] if day in by_date]
        if not sampled:
            continue
        gross_samples.append(float(np.mean([row["gross"] for row in sampled])))
        net_samples.append(float(np.mean([row["net"] for row in sampled])))
    if not gross_samples:
        return {"gross": [None, None], "net": [None, None]}
    return {
        "gross": [float(value) for value in np.percentile(gross_samples, [2.5, 97.5])],
        "net": [float(value) for value in np.percentile(net_samples, [2.5, 97.5])],
    }


def _summary_rows(
    actions: list[dict[str, object]],
    session_dates: list[str],
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for scope_index, scope in enumerate(SCOPES):
        scoped = actions if scope == "TOTAL" else [row for row in actions if row["side"] == scope]
        eligible = [row for row in scoped if row["statistics_included"]]
        for horizon in HORIZONS:
            gross_field = f"gross_directional_return_{horizon}"
            net_field = f"net_markout_{horizon}"
            complete = [row for row in eligible if _number(row.get(net_field)) is not None]
            gross_values = [float(row[gross_field]) for row in complete]
            net_values = [float(row[net_field]) for row in complete]
            daily = _daily_markouts(scoped, horizon)
            daily_weights = [float(row["notional"]) for row in daily]
            ess = (
                sum(daily_weights) ** 2 / sum(weight**2 for weight in daily_weights)
                if daily_weights
                else 0.0
            )
            ci = _moving_block_ci(
                daily,
                session_dates,
                samples=bootstrap_samples,
                seed=seed + scope_index * 100 + horizon,
            )
            total = len(eligible)
            coverage = len(complete) / total if total else 0.0
            verdict = None
            if horizon == 5:
                low, high = ci["net"]
                if coverage < 0.8 or len({row["date"] for row in complete}) < 30 or ess < 20 or low is None:
                    verdict = "INSUFFICIENT"
                elif low > 0:
                    verdict = "POSITIVE"
                elif high < 0:
                    verdict = "NEGATIVE"
                else:
                    verdict = "INCONCLUSIVE"
            summary.append(
                {
                    "scope": scope,
                    "horizon": horizon,
                    "all_actions": len(scoped),
                    "excluded_quality_actions": len(scoped) - total,
                    "total": total,
                    "complete": len(complete),
                    "coverage": coverage,
                    "unique_days": len({row["date"] for row in complete}),
                    "unique_tickers": len({row.get("ticker") or row["isin"] for row in complete}),
                    "equal_mean_gross": float(np.mean(gross_values)) if gross_values else None,
                    "equal_mean_net": float(np.mean(net_values)) if net_values else None,
                    "equal_median_gross": float(np.median(gross_values)) if gross_values else None,
                    "equal_median_net": float(np.median(net_values)) if net_values else None,
                    "hit_rate_gross": sum(value > 0 for value in gross_values) / len(gross_values) if gross_values else None,
                    "hit_rate_net": sum(value > 0 for value in net_values) / len(net_values) if net_values else None,
                    "pooled_abs_notional_weighted_mean_gross": _weighted_mean(complete, gross_field),
                    "pooled_abs_notional_weighted_mean_net": _weighted_mean(complete, net_field),
                    "daily_weighted_mean_gross": float(np.mean([row["gross"] for row in daily])) if daily else None,
                    "daily_weighted_mean_net": float(np.mean([row["net"] for row in daily])) if daily else None,
                    "weighted_unique_days": len(daily),
                    "ess_days": ess,
                    "bootstrap_ci_95_gross_low": ci["gross"][0],
                    "bootstrap_ci_95_gross_high": ci["gross"][1],
                    "bootstrap_ci_95_net_low": ci["net"][0],
                    "bootstrap_ci_95_net_high": ci["net"][1],
                    "primary_verdict": verdict,
                }
            )
    return summary


def _daily_rows(actions: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for action in actions:
        grouped.setdefault(str(action["date"]), []).append(action)
    result: list[dict[str, object]] = []
    for day, rows in sorted(grouped.items()):
        buy = sum(_number(row.get("abs_notional_base")) or 0.0 for row in rows if row["side"] == "BUY")
        sell = sum(_number(row.get("abs_notional_base")) or 0.0 for row in rows if row["side"] == "SELL")
        daily: dict[str, object] = {
            "date": day,
            "actions": len(rows),
            "included_actions": sum(bool(row["statistics_included"]) for row in rows),
            "excluded_actions": sum(not bool(row["statistics_included"]) for row in rows),
            "buy_notional": buy,
            "sell_notional": sell,
            "gross_notional": buy + sell,
            "net_notional": buy - sell,
        }
        for horizon in HORIZONS:
            marks = _daily_markouts(rows, horizon)
            mark = marks[0] if marks else None
            daily[f"complete_actions_{horizon}"] = sum(
                row["statistics_included"] and _number(row.get(f"net_markout_{horizon}")) is not None
                for row in rows
            )
            daily[f"gross_markout_{horizon}"] = mark["gross"] if mark else None
            daily[f"net_markout_{horizon}"] = mark["net"] if mark else None
            daily[f"complete_notional_{horizon}"] = mark["notional"] if mark else None
        result.append(daily)
    return result


def evaluate_holdings(
    holdings: list[dict[str, object]],
    *,
    cost_bps: float = 10.0,
    bootstrap_samples: int = 1000,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative")
    actions = infer_actions(holdings)
    sessions = infer_market_sessions(holdings)
    enriched = _enrich_actions(
        holdings, actions, cost_bps=cost_bps, session_dates=sessions
    )
    observed_dates = sorted({str(row["date"]) for row in holdings})
    return {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "horizons": list(HORIZONS),
            "horizon_unit": "inferred portfolio market sessions",
            "direction": "BUY=+1; SELL=-1",
            "price_proxy": "abs(market_value_base/quantity) first; close_local only as LOCAL_CLOSE_FALLBACK",
            "price_limitation": "Snapshot valuation proxy, not execution price or total-return price.",
            "base_price_policy": "Same-ISIN action-date price first; exact previous_date price as PREVIOUS_CLOSE_PROXY if absent.",
            "future_price_policy": "Exact inferred horizon date and same ISIN only; otherwise NA (including many EXIT observations).",
            "market_session_inference": "weekday<5 and (any valid common-ISIN held-price change or any ISIN quantity/composition change) versus the immediately previous observation; not an exact exchange calendar",
            "statistics_filter": "evaluation_quality_flag must be OK; NON_SESSION_DATE and source quality flags remain in action detail but are excluded",
            "cost_bps": cost_bps,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": seed,
            "bootstrap": "20-session moving-block bootstrap of daily abs-notional-weighted markout",
            "ess_unit": "effective decision days from daily absolute-notional weights",
            "primary_rule": "5-session net CI; requires coverage>=80%, unique_days>=30, ESS>=20",
            "sell_markout_interpretation": "Avoidance outcome only; SELL markout is not realized P&L.",
            "observed_dates": len(observed_dates),
            "inferred_market_sessions": len(sessions),
            "inferred_actions": len(enriched),
        },
        "summary": _summary_rows(
            enriched, sessions, bootstrap_samples=bootstrap_samples, seed=seed
        ),
        "daily": _daily_rows(enriched),
        "actions": enriched,
    }


def evaluate_workbook(
    workbook_path: Path,
    *,
    cost_bps: float = 10.0,
    bootstrap_samples: int = 1000,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    holdings, _ = read_holdings_workbook(workbook_path)
    result = evaluate_holdings(
        holdings,
        cost_bps=cost_bps,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    result["metadata"]["input_workbook"] = str(workbook_path.resolve())
    result["metadata"]["holdings_rows"] = len(holdings)
    return result


def _clean_json(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, np.generic):
        return _clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    records = list(rows)
    fieldnames = list(records[0]) if records else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_clean_json(records))


def write_outputs(result: dict[str, object], output_json: Path) -> dict[str, Path]:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_json,
        "summary_csv": output_json.with_name(f"{output_json.stem}_summary.csv"),
        "daily_csv": output_json.with_name(f"{output_json.stem}_daily.csv"),
        "actions_csv": output_json.with_name(f"{output_json.stem}_actions.csv"),
    }
    cleaned = _clean_json(result)
    output_json.write_text(
        json.dumps(cleaned, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_csv(paths["summary_csv"], cleaned["summary"])
    _write_csv(paths["daily_csv"], cleaned["daily"])
    _write_csv(paths["actions_csv"], cleaned["actions"])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate inferred net-holdings decisions from an SS workbook"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    result = evaluate_workbook(
        args.input,
        cost_bps=args.cost_bps,
        bootstrap_samples=args.bootstrap_samples,
    )
    paths = write_outputs(result, args.output)
    print("wrote " + ", ".join(str(path) for path in paths.values()))


if __name__ == "__main__":
    main()
