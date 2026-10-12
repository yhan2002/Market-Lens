from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bloomberg_gics_l2_export import (
    BloombergClient,
    ERROR_COLUMNS,
    csv_data_rows,
    request_reference,
    sha256,
    write_csv,
)


GROUPS = {
    "2010": ("Capital Goods", "S5CPGS Index"),
    "4010": ("Banks", "S5BANKX Index"),
    "4510": ("Software & Services", "S5SFTW Index"),
}
DEFAULT_DATES = ("20240131", "20241231", "20251231", "20260902")
MEMBERSHIP_COLUMNS = (
    "requested_date",
    "industry_group_code",
    "industry_group_name",
    "index_security",
    "member_security",
    "weight_pct",
    "repeat_no",
    "source_field_keys",
)


def _bulk_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        rows: list[dict[str, Any]] = []
        for item in value:
            rows.extend(_bulk_rows(item))
        return rows
    if not isinstance(value, dict):
        return []
    scalar_count = sum(not isinstance(v, (dict, list)) for v in value.values())
    if scalar_count >= 2:
        return [value]
    rows = []
    for item in value.values():
        rows.extend(_bulk_rows(item))
    return rows


def _field(row: dict[str, Any], needles: tuple[str, ...]) -> Any:
    normalized = {
        "".join(ch for ch in str(key).lower() if ch.isalnum()): value
        for key, value in row.items()
    }
    for needle in needles:
        for key, value in normalized.items():
            if needle in key and value not in (None, ""):
                return value
    return None


def parse_membership(value: Any) -> list[tuple[str, float, str]]:
    parsed: list[tuple[str, float, str]] = []
    for row in _bulk_rows(value):
        member = _field(row, ("membertickerandexchangecode", "memberticker", "indexmember"))
        weight = _field(row, ("percentageweight", "percentweight", "weight"))
        if member in (None, "") or weight in (None, ""):
            continue
        try:
            number = float(str(weight).replace("%", "").replace(",", ""))
        except ValueError:
            continue
        parsed.append((str(member).strip(), number, "|".join(sorted(map(str, row)))))
    return parsed


def fingerprint(rows: list[tuple[str, float, str]]) -> list[tuple[str, float]]:
    return sorted((member, round(weight, 8)) for member, weight, _ in rows)


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Bloomberg GICS historical membership diagnostic",
        "",
        f"- status: **{summary['status']}**",
        f"- created_at_utc: {summary['created_at_utc']}",
        f"- requested dates: {', '.join(summary['requested_dates'])}",
        f"- membership rows: {summary['counts']['membership_rows']}",
        f"- repeat mismatches: {summary['counts']['repeat_mismatches']}",
        f"- errors: {summary['counts']['errors']}",
        "",
        "## Coverage",
        "",
        "| group | date | members | weight sum | repeat match |",
        "|---|---:|---:|---:|---|",
    ]
    for row in summary["coverage"]:
        lines.append(
            f"| {row['group_code']} {row['group_name']} | {row['date']} | "
            f"{row['members']} | {row['weight_sum']:.4f} | {row['repeat_match']} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "This only tests whether historical constituent snapshots can be reproduced. "
        "It does not prove point-in-time classification history or authorize model/DB/web use.",
    ]
    return "\n".join(lines) + "\n"


def self_test() -> None:
    sample = {
        "rows": [
            {"Member Ticker and Exchange Code": "MSFT UW", "Percentage Weight": "55.5"},
            {"Member Ticker and Exchange Code": "ORCL UN", "Percentage Weight": 44.5},
        ]
    }
    parsed = parse_membership(sample)
    assert fingerprint(parsed) == [("MSFT UW", 55.5), ("ORCL UN", 44.5)]
    assert parse_membership({"message": "not applicable"}) == []
    print("self-test: PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small Bloomberg diagnostic for historical GICS group membership."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8194)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--dates", nargs="+", default=list(DEFAULT_DATES))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    requested_dates = [datetime.strptime(value, "%Y%m%d").strftime("%Y%m%d") for value in args.dates]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"bloomberg_gics_breadth_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    errors: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []

    with BloombergClient(args.host, args.port, args.timeout) as client:
        for group_code, (group_name, security) in GROUPS.items():
            for requested_date in requested_dates:
                repeats: list[list[tuple[str, float, str]]] = []
                for repeat_no in (1, 2):
                    data = request_reference(
                        client,
                        [security],
                        ["INDX_MWEIGHT_HIST"],
                        f"membership_{group_code}_{requested_date}_r{repeat_no}",
                        errors,
                        overrides=[{"fieldId": "END_DT", "value": requested_date}],
                    )
                    fields = data.get(security, {})
                    parsed = parse_membership(fields.get("INDX_MWEIGHT_HIST"))
                    repeats.append(parsed)
                    for member, weight, keys in parsed:
                        membership_rows.append(
                            {
                                "requested_date": requested_date,
                                "industry_group_code": group_code,
                                "industry_group_name": group_name,
                                "index_security": security,
                                "member_security": member,
                                "weight_pct": weight,
                                "repeat_no": repeat_no,
                                "source_field_keys": keys,
                            }
                        )
                first = repeats[0]
                coverage.append(
                    {
                        "group_code": group_code,
                        "group_name": group_name,
                        "date": requested_date,
                        "members": len(first),
                        "weight_sum": sum(weight for _, weight, _ in first),
                        "repeat_match": fingerprint(first) == fingerprint(repeats[1]),
                    }
                )

    missing = sum(row["members"] == 0 for row in coverage)
    repeat_mismatches = sum(not row["repeat_match"] for row in coverage)
    status = "PASS" if not missing and not repeat_mismatches else "PARTIAL" if membership_rows else "FAIL"
    summary = {
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pit_status": "HISTORICAL_SNAPSHOT_AVAILABILITY_ONLY",
        "field": "INDX_MWEIGHT_HIST",
        "requested_dates": requested_dates,
        "groups": [
            {"code": code, "name": name, "security": security}
            for code, (name, security) in GROUPS.items()
        ],
        "counts": {
            "membership_rows": len(membership_rows),
            "missing_group_dates": missing,
            "repeat_mismatches": repeat_mismatches,
            "errors": len(errors),
        },
        "coverage": coverage,
        "notes": [
            "Each group/date is queried twice; exact member-weight fingerprints must match.",
            "No raw Bloomberg response, database, model, or website is written.",
            "Historical snapshots do not by themselves prove GICS point-in-time classification validity.",
        ],
    }

    shutil.copy2(Path(__file__).resolve(), output_dir / Path(__file__).name)
    write_csv(output_dir / "membership_sample.csv", membership_rows, MEMBERSHIP_COLUMNS)
    write_csv(output_dir / "errors.csv", errors, ERROR_COLUMNS)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "RUN_REPORT.md").write_text(render_report(summary), encoding="utf-8")

    manifest_rows = []
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name.lower()):
        if path.name == "manifest_sha256.csv" or not path.is_file():
            continue
        manifest_rows.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "data_rows": csv_data_rows(path),
                "sha256": sha256(path),
            }
        )
    write_csv(
        output_dir / "manifest_sha256.csv",
        manifest_rows,
        ("file", "bytes", "data_rows", "sha256"),
    )
    zip_path = args.output_root / f"bloomberg_gics_breadth_handoff_{stamp}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.iterdir(), key=lambda item: item.name.lower()):
            if path.is_file():
                archive.write(path, arcname=f"{output_dir.name}/{path.name}")
    result = {"status": status, "output_dir": str(output_dir.resolve()), "zip_path": str(zip_path.resolve())}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run(args)


if __name__ == "__main__":
    main()
