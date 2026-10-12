"""Preview, or explicitly execute, a missing-session breadth recovery."""

import argparse
import json
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

from breadth_metrics import SP500_BREADTH_GROUP, compute_breadth_metrics
from market_data import download_prices
from risk_signals import load_breadth_history


def recover_prices(database: Path, tickers: tuple[str, ...], target: date) -> list[str]:
    failed = []
    for offset in range(0, len(tickers), 25):
        batch = tickers[offset:offset + 25]
        try:
            download_prices(database, years=2, tickers=batch, require_date=target)
        except RuntimeError:
            for ticker in batch:
                try:
                    download_prices(database, years=2, tickers=(ticker,), require_date=target)
                except RuntimeError:
                    failed.append(ticker)
        print(json.dumps({"processed": offset + len(batch), "total": len(tickers), "failed": failed}), flush=True)
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("regime_router_publish.db"))
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    day = args.as_of.isoformat()
    with closing(sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        tickers = tuple(row[0] for row in connection.execute(
            """SELECT ticker FROM universe u WHERE group_name=? AND active=1
               AND NOT EXISTS (SELECT 1 FROM daily_prices p
                   WHERE p.ticker=u.ticker AND p.date=? AND p.adj_close>0)
               ORDER BY ticker""", (SP500_BREADTH_GROUP, day)
        ))
    if len(tickers) > 550:
        raise ValueError("Unexpected breadth recovery scope: more than 550 tickers")
    print(json.dumps({"as_of": day, "missing_tickers": len(tickers),
                      "execute": args.execute, "backfill": False, "upload": False}))
    if not args.execute:
        return
    if tickers:
        recover_prices(args.database, tickers, args.as_of)
    compute_breadth_metrics(args.database)
    rows = load_breadth_history(args.database, day, 1)
    if not rows or rows[0]["date"] != day:
        raise RuntimeError(f"No valid breadth aggregate for {day}; do not publish")
    print(json.dumps({"status": "VERIFIED", "breadth": rows[0]}))


if __name__ == "__main__":
    main()
