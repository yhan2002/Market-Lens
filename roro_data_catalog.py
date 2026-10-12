"""Validation contract for the future authoritative RORO data catalog CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


COLUMNS = (
    "block",
    "signal",
    "local_security",
    "bloomberg_security",
    "table",
    "field",
    "unit",
    "timezone",
    "observation_date_rule",
    "available_at_rule",
    "source",
    "coverage_start",
    "coverage_end",
    "rows",
    "pit_status",
    "freshness_rule",
    "transform",
    "risk_direction",
    "status",
    "action",
)

SIGNALS = (
    ("Market Trend", "NQ price, momentum, MA, drawdown"),
    ("Equity Volatility", "VIX level"),
    ("Equity Volatility", "VIX curve"),
    ("Rates Volatility", "MOVE level"),
    ("Credit", "HY OAS level/change"),
    ("FX / Carry", "Yen strength"),
    ("Safe Haven", "GLD relative to NQ"),
    ("Market Internals", "S&P 500 breadth"),
    ("Portfolio Internals", "Weighted correlation 20D"),
    ("Portfolio Internals", "Correlation trajectory"),
    ("Portfolio Internals", "NQ beta 60D"),
    ("Portfolio Internals", "Concentration"),
    ("Portfolio Internals", "Portfolio breadth 20D"),
    ("Portfolio Internals", "Portfolio breadth 50D"),
    ("Portfolio Internals", "Portfolio breadth 200D"),
    ("Portfolio Internals", "Negative participation 20D"),
    ("Portfolio Internals", "Portfolio drawdown 60D"),
    ("Portfolio Internals", "Cross-sectional dispersion 20D"),
    ("Sector Rotation", "Sector adjusted prices"),
    ("Sector Rotation", "Communication Services"),
    ("Sector Rotation", "Relative benchmark"),
    ("Sector Rotation", "Market context"),
)

_PRICE_ROWS = {
    SIGNALS[0],
    SIGNALS[1],
    SIGNALS[2],
    SIGNALS[3],
    SIGNALS[18],
    SIGNALS[19],
    SIGNALS[20],
}
_PORTFOLIO_ROWS = set(SIGNALS[8:18])
_BLOCKED_ROWS = {SIGNALS[5], SIGNALS[6], SIGNALS[7]}


def validate_catalog(path: str | Path) -> list[dict[str, str]]:
    """Read and validate the exact catalog grain without inventing vendor metadata."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError("RORO catalog columns or column order do not match the frozen contract")
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]

    keys = [(row["block"], row["signal"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("RORO catalog contains duplicate block/signal rows")
    missing = set(SIGNALS).difference(keys)
    extra = set(keys).difference(SIGNALS)
    if missing or extra:
        raise ValueError(f"RORO catalog signal set mismatch; missing={sorted(missing)!r}, extra={sorted(extra)!r}")

    for row, key in zip(rows, keys):
        if row["rows"]:
            try:
                count = int(row["rows"])
            except ValueError as error:
                raise ValueError(f"rows must be a non-negative integer or blank for {key!r}") from error
            if count < 0:
                raise ValueError(f"rows must be a non-negative integer or blank for {key!r}")
        if key in _PRICE_ROWS and row["pit_status"] != "CONDITIONAL_PRICE_NO_VINTAGE":
            raise ValueError(f"price PIT classification was weakened for {key!r}")
        if key == SIGNALS[4] and row["pit_status"] != "PIT_CAUSAL":
            raise ValueError("HY OAS must remain PIT_CAUSAL")
        if key in _PORTFOLIO_ROWS and row["pit_status"] != "CONDITIONAL_EVENT_PIT":
            raise ValueError(f"portfolio PIT classification was weakened for {key!r}")
        if key in _BLOCKED_ROWS and (
            row["pit_status"] != "DATA_BLOCKED" or row["status"] != "DATA_BLOCKED"
        ):
            raise ValueError(f"blocked input was promoted without evidence for {key!r}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate RORO_DATA_CATALOG.csv")
    parser.add_argument("catalog", type=Path)
    args = parser.parse_args()
    rows = validate_catalog(args.catalog)
    print(f"VALID: {len(rows)} RORO catalog rows")


if __name__ == "__main__":
    main()
