from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


GROUPS = {
    "S5CPGS Index": ("2010", "Capital Goods"),
    "S5BANKX Index": ("4010", "Banks"),
    "S5SFTW Index": ("4510", "Software & Services"),
}
REQUIRED_FIELDS = ("IN013", "IN014", "IN015", "IN049")
MODEL_FEATURES = (
    "breadth_net_rel_spx_5d",
    "breadth_net_rel_spx_accel_5d",
    "breadth_pct200_rel_spx",
    "breadth_pct200_rel_spx_change_5d",
)


def _group_map(history: pd.DataFrame) -> dict[str, tuple[str, str]]:
    if not {"industry_group_code", "industry_group_name"}.issubset(history.columns):
        return GROUPS
    mapping = history.loc[
        history["security"].ne("SPX Index"),
        ["security", "industry_group_code", "industry_group_name"],
    ].copy()
    mapping = mapping.astype("string").apply(lambda values: values.str.strip()).drop_duplicates()
    if mapping.empty or mapping["security"].duplicated().any():
        raise ValueError("breadth security must map to exactly one industry group")
    return {
        str(row.security): (str(row.industry_group_code), str(row.industry_group_name))
        for row in mapping.itertuples(index=False)
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def build_features(history: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    required_columns = {"date", "security", "field_id", "value"}
    missing_columns = required_columns - set(history.columns)
    if missing_columns:
        raise ValueError("missing columns: " + ", ".join(sorted(missing_columns)))

    groups = _group_map(history)
    rows = history.loc[history["field_id"].isin((*REQUIRED_FIELDS, "IX109"))].copy()
    rows["date"] = pd.to_datetime(rows["date"], errors="raise").dt.normalize()
    rows["value"] = pd.to_numeric(rows["value"], errors="raise")
    if not np.isfinite(rows["value"]).all():
        raise ValueError("non-finite breadth values")
    if rows.duplicated(["date", "security", "field_id"]).any():
        raise ValueError("duplicate date/security/field rows")

    pivot = rows.pivot(index=["date", "security"], columns="field_id", values="value")
    if set(groups) | {"SPX Index"} != set(pivot.index.get_level_values("security")):
        raise ValueError("unexpected breadth security universe")
    if pivot.loc[:, REQUIRED_FIELDS].isna().any().any():
        raise ValueError("required breadth field is missing")
    if "IX109" in pivot and not np.allclose(
        pivot["IX109"], pivot["IN013"] - pivot["IN014"], atol=1e-9
    ):
        raise ValueError("IX109 does not equal IN013 minus IN014")

    pivot["member_count"] = pivot["IN013"] + pivot["IN014"] + pivot["IN015"]
    if (pivot["member_count"] <= 0).any():
        raise ValueError("non-positive member count")
    if ((pivot["IN049"] < 0) | (pivot["IN049"] > 100)).any():
        raise ValueError("IN049 outside 0..100")

    spx = pivot.xs("SPX Index", level="security").sort_index()
    trading_dates = spx.index[(spx["IN013"] + spx["IN014"]) > 0]
    next_session = pd.Series(trading_dates[1:].append(pd.DatetimeIndex([pd.NaT])), index=trading_dates)

    derived: dict[str, pd.DataFrame] = {}
    for security in (*groups, "SPX Index"):
        group = pivot.xs(security, level="security").reindex(trading_dates)
        if group.loc[:, REQUIRED_FIELDS].isna().any().any():
            raise ValueError(f"{security}: missing required trading-session breadth")
        out = pd.DataFrame(index=trading_dates)
        out["member_count"] = group["member_count"]
        out["net_share"] = (group["IN013"] - group["IN014"]) / group["member_count"]
        out["decline_share"] = group["IN014"] / group["member_count"]
        out["pct_above_200dma"] = group["IN049"] / 100.0
        derived[security] = out

    parts: list[pd.DataFrame] = []
    benchmark = derived["SPX Index"]
    for security, (sector, sector_name) in groups.items():
        out = derived[security].copy()
        out["breadth_net_rel_spx"] = out["net_share"] - benchmark["net_share"]
        out["breadth_decline_rel_spx"] = out["decline_share"] - benchmark["decline_share"]
        out["breadth_pct200_rel_spx"] = (
            out["pct_above_200dma"] - benchmark["pct_above_200dma"]
        )
        out["breadth_net_rel_spx_5d"] = out["breadth_net_rel_spx"].rolling(5).mean()
        out["breadth_net_rel_spx_accel_5d"] = (
            out["breadth_net_rel_spx_5d"] - out["breadth_net_rel_spx_5d"].shift(5)
        )
        out["breadth_pct200_rel_spx_change_5d"] = out["breadth_pct200_rel_spx"].diff(5)
        out.insert(0, "date", out.index)
        out.insert(1, "usable_from_session", next_session.reindex(out.index).to_numpy())
        out.insert(2, "sector", sector)
        out.insert(3, "sector_name", sector_name)
        out.insert(4, "security", security)
        parts.append(out.reset_index(drop=True))

    result = pd.concat(parts, ignore_index=True).sort_values(["date", "sector"])
    complete = result.loc[:, MODEL_FEATURES].notna().all(axis=1)
    audit: dict[str, object] = {
        "pit_status": "PIT_PARTIAL",
        "feature_timing": "EOD_T_OBSERVED_ACTION_NO_EARLIER_THAN_T_PLUS_1",
        "raw_fill_or_interpolation": False,
        "calendar_rule": "drop dates where SPX IN013+IN014 equals zero",
        "source_rows": int(len(history)),
        "trading_sessions": int(len(trading_dates)),
        "dropped_nontrading_weekdays": int(history["date"].nunique() - len(trading_dates)),
        "feature_rows": int(len(result)),
        "complete_model_rows": int(complete.sum()),
        "groups": len(groups),
        "group_codes": sorted(code for code, _ in groups.values()),
        "first_date": str(result["date"].min().date()),
        "last_date": str(result["date"].max().date()),
        "model_features": list(MODEL_FEATURES),
        "ix109_is_exactly_redundant": True,
        "arms_ix116_excluded": "group coverage is incomplete, especially Banks",
    }
    return result.reset_index(drop=True), audit


def self_test() -> None:
    dates = pd.bdate_range("2026-01-01", periods=14)
    rows = []
    for date_no, date in enumerate(dates):
        holiday = date_no == 2
        for security, count in {
            "S5CPGS Index": 50,
            "S5BANKX Index": 13,
            "S5SFTW Index": 28,
            "SPX Index": 500,
        }.items():
            advance = 0 if holiday else count // 2 + date_no % 2
            decline = 0 if holiday else count - advance
            unchanged = count if holiday else 0
            for field, value in {
                "IN013": advance,
                "IN014": decline,
                "IN015": unchanged,
                "IN049": 50 + date_no,
                "IX109": advance - decline,
            }.items():
                rows.append({"date": date, "security": security, "field_id": field, "value": value})
    features, audit = build_features(pd.DataFrame(rows))
    assert audit["trading_sessions"] == 13
    assert audit["dropped_nontrading_weekdays"] == 1
    assert len(features) == 39
    assert features.loc[:, MODEL_FEATURES].notna().all(axis=1).sum() == 12
    print("self-test: PASS")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Prepare causal GICS breadth research features")
    result.add_argument("--history-csv", type=Path)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--self-test", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.self_test:
        self_test()
        return
    if args.history_csv is None or args.output_dir is None:
        raise ValueError("--history-csv and --output-dir are required")
    history_path = args.history_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    features, audit = build_features(
        pd.read_csv(history_path, dtype={"industry_group_code": "string"}, low_memory=False)
    )
    audit.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_file": history_path.name,
            "source_sha256": file_sha256(history_path),
        }
    )
    features.to_csv(output_dir / "sector_breadth_features.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
