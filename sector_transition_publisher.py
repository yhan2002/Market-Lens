"""Publish the latest allowlisted sector transition snapshot."""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs" / "sector_turning_dashboard_scores_20260918" / "latest_snapshot.json"
SERIES = "sector-transition-price-multiscale-v1"


def payload(path: Path) -> dict:
    source = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        {
            "ticker": row["ticker"],
            "sector": row["sector"],
            "state": row["state"],
            "direction": row["transition_direction"],
            "score": row["transition_score"],
            "status": row["status"],
        }
        for row in source["sectors"]
    ]
    if len(rows) != 11 or len({row["ticker"] for row in rows}) != 11:
        raise ValueError("exactly 11 unique sector rows are required")
    return {
        "schema_version": 1,
        "series_version": SERIES,
        "as_of": source["as_of"],
        "score_semantics": "CAUSAL_TRAILING_PERCENTILE",
        "high_threshold": 80,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    outgoing = payload(args.source)
    if args.upload:
        config = json.loads((ROOT / "site_upload_config.json").read_text(encoding="utf-8-sig"))
        endpoint = config["site_url"].rstrip("/") + "/api/sector-transition"
        headers = {"Authorization": "Bearer " + config["upload_token"], "Content-Type": "application/json"}
        if config.get("sites_bypass_token"):
            headers["OAI-Sites-Authorization"] = "Bearer " + config["sites_bypass_token"]
        request = urllib.request.Request(
            endpoint, data=json.dumps(outgoing, separators=(",", ":")).encode(), headers=headers, method="POST"
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            acknowledgement = json.load(response)
        if not acknowledgement.get("ok"):
            raise RuntimeError("sector transition upload was not acknowledged")
    print(json.dumps({"as_of": outgoing["as_of"], "rows": len(outgoing["rows"]), "uploaded": args.upload}))


if __name__ == "__main__":
    main()
