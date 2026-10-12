"""Publish aggregate-only Sector RORO SHADOW history to its separate API."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from roro_shadow import SERIES_VERSION as RORO_SERIES_VERSION
from roro_shadow import (
    _calendar_date,
    _canonical_json,
    _canonical_provenance_encoding,
)
from sector_rotation import SECTOR_ETFS
from sector_roro_shadow import MODEL_VERSION, OUTPUT_START
from sector_shadow_store import (
    DEFAULT_TARGET as STORE_DEFAULT_TARGET,
    PAYLOAD_FIELDS,
    SCHEMA_COLUMNS,
    SERIES_VERSION,
    read_sector_history,
)


APP_DIR = Path(__file__).resolve().parent
DEFAULT_SHADOW_DATABASE = APP_DIR / STORE_DEFAULT_TARGET
DEFAULT_CONFIG = APP_DIR / "site_upload_config.json"
DEFAULT_CONSENT = APP_DIR / "roro_publish_consent.json"
DEFAULT_OUTPUT = APP_DIR / "outputs" / "sector_shadow" / "site_payload.json"
UPLOAD_BATCH_SIZE = 1
STATUS_CONTRACT = "SELECT/WATCH/AVOID; no forced Top3 fill"
WEB_SCHEMA_VERSION = 2
RESEARCH_SCHEMA_VERSION = 1
RESEARCH_SERIES_VERSION = f"{MODEL_VERSION}__weekly-research-2025-r2"

COMMON_FIELDS = (
    "forecast_target",
    "training_rows",
    "training_weeks",
    "training_start",
    "training_end",
    "latest_training_label_end",
    "alpha",
    "top_k",
    "context_type",
    "status_contract",
    "market_risk_score",
    "risk_regime",
    "roro_trajectory_5d",
    "roro_data_status",
)
ROW_FIELDS = tuple(field for field in PAYLOAD_FIELDS if field not in COMMON_FIELDS)
ENVELOPE_FIELDS = (
    "schema_version",
    "series_version",
    "as_of",
    "generated_at",
    "input_cutoff",
    "provenance_hash",
    "model_versions",
    "payload",
)
PAYLOAD_WEB_FIELDS = (
    "relative_benchmark",
    "market_context",
    *COMMON_FIELDS,
    "rows",
)
ROW_WEB_FIELDS = ("sector", *ROW_FIELDS)
OBSERVATION_FIELDS = tuple(
    field for field in SCHEMA_COLUMNS if field != "payload_json"
) + ("payload",)
FORBIDDEN_IDENTITY_FIELDS = {
    "ticker",
    "isin",
    "quantity",
    "holding_name",
    "position",
    "positions",
    "market_value",
    "daily_holdings",
}


def _assert_private_safe(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).casefold() in FORBIDDEN_IDENTITY_FIELDS:
                raise ValueError(f"holding identity cannot be published: {key}")
            _assert_private_safe(item)
    elif isinstance(value, list):
        for item in value:
            _assert_private_safe(item)


def _group_provenance(
    *,
    series_version: str,
    date: str,
    input_cutoff: str,
    model_versions: dict[str, str],
    payload: dict[str, Any],
) -> str:
    """Hash only aggregate fields present in the public envelope."""
    material = {
        "series_version": series_version,
        "date": date,
        "input_cutoff": input_cutoff,
        "model_versions": model_versions,
        "payload": payload,
    }
    return hashlib.sha256(
        _canonical_provenance_encoding(material).encode("utf-8")
    ).hexdigest()


def web_snapshot(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine one date's 11 immutable rows into the strict public envelope."""
    if not observations:
        raise ValueError("no Sector SHADOW observations")
    _assert_private_safe(observations)
    for observation in observations:
        if set(observation) != set(OBSERVATION_FIELDS):
            raise ValueError("Sector SHADOW observation fields are not exact")
        if set(observation["payload"]) != set(PAYLOAD_FIELDS):
            raise ValueError("Sector SHADOW payload fields are not exact")

    series = {str(row["series_version"]) for row in observations}
    dates = {str(row["date"]) for row in observations}
    cutoffs = {str(row["input_cutoff"]) for row in observations}
    models = {str(row["model_version"]) for row in observations}
    roro_series = {str(row["roro_series_version"]) for row in observations}
    sectors = [str(row["sector"]) for row in observations]
    if series != {SERIES_VERSION}:
        raise ValueError("exactly one supported Sector SHADOW series is required")
    if len(dates) != 1 or cutoffs != dates:
        raise ValueError("Sector input cutoff must equal its observation date")
    day = dates.pop()
    if _calendar_date(day, "as_of") != day or day < OUTPUT_START.date().isoformat():
        raise ValueError(f"Sector web history starts on {OUTPUT_START.date().isoformat()}")
    if models != {MODEL_VERSION} or roro_series != {RORO_SERIES_VERSION}:
        raise ValueError("Sector/RORO model provenance does not match the frozen series")
    if len(sectors) != len(SECTOR_ETFS) or set(sectors) != set(SECTOR_ETFS):
        raise ValueError("one complete row for each of the 11 sector ETFs is required")
    if len(set(sectors)) != len(sectors):
        raise ValueError("duplicate sector row")
    if any(
        len(str(row["provenance_hash"])) != 64
        or any(character not in "0123456789abcdef" for character in str(row["provenance_hash"]).lower())
        for row in observations
    ):
        raise ValueError("invalid Sector SHADOW provenance hash")

    first_payload = observations[0]["payload"]
    common = {field: first_payload[field] for field in COMMON_FIELDS}
    if common["status_contract"] != STATUS_CONTRACT:
        raise ValueError("unexpected Sector status contract")
    for observation in observations[1:]:
        candidate = {field: observation["payload"][field] for field in COMMON_FIELDS}
        if _canonical_json(candidate) != _canonical_json(common):
            raise ValueError("same-date Sector common metadata must agree")

    rows = []
    for observation in observations:
        payload = observation["payload"]
        sector = str(observation["sector"])
        if payload["sector_name"] != SECTOR_ETFS[sector]:
            raise ValueError(f"sector name does not match {sector}")
        rows.append({"sector": sector, **{field: payload[field] for field in ROW_FIELDS}})
    rows.sort(key=lambda row: (row["forecast_rank"], row["sector"]))
    payload = {
        "relative_benchmark": "SPY",
        "market_context": "NQ=F",
        **common,
        "rows": rows,
    }
    model_versions = {
        "sector_roro": MODEL_VERSION,
        "roro": RORO_SERIES_VERSION,
    }
    snapshot = {
        "schema_version": WEB_SCHEMA_VERSION,
        "series_version": SERIES_VERSION,
        "as_of": day,
        "generated_at": max(str(row["created_at"]) for row in observations),
        "input_cutoff": day,
        "provenance_hash": _group_provenance(
            series_version=SERIES_VERSION,
            date=day,
            input_cutoff=day,
            model_versions=model_versions,
            payload=payload,
        ),
        "model_versions": model_versions,
        "payload": payload,
    }
    if tuple(snapshot) != ENVELOPE_FIELDS or tuple(payload) != PAYLOAD_WEB_FIELDS:
        raise ValueError("Sector web envelope does not match the frozen contract")
    if any(tuple(row) != ROW_WEB_FIELDS for row in rows):
        raise ValueError("Sector web row does not match the frozen contract")
    _assert_private_safe(snapshot)
    return snapshot


def web_snapshots(
    observations: list[dict[str, Any]],
    *,
    latest_only: bool = False,
) -> list[dict[str, Any]]:
    if not observations:
        return []
    series = {str(row.get("series_version")) for row in observations}
    if series != {SERIES_VERSION}:
        raise ValueError("Sector upload must contain one immutable series")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        grouped[str(observation.get("date"))].append(observation)
    snapshots = [web_snapshot(grouped[day]) for day in sorted(grouped)]
    return snapshots[-1:] if latest_only else snapshots


def _headers(token: str, sites_bypass_token: str | None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    if sites_bypass_token:
        headers["OAI-Sites-Authorization"] = f"Bearer {sites_bypass_token}"
    return headers


def upload_snapshots(
    snapshots: list[dict[str, Any]],
    site_url: str,
    token: str,
    sites_bypass_token: str | None = None,
) -> dict[str, Any]:
    if not snapshots:
        raise ValueError("no Sector RORO snapshots to upload")
    if len({snapshot.get("series_version") for snapshot in snapshots}) != 1:
        raise ValueError("Sector upload must contain one immutable series")
    endpoint = site_url.rstrip("/") + "/api/sector-roro"
    totals = {"dates": 0, "rows": 0, "inserted": 0, "idempotent": 0}
    for offset in range(0, len(snapshots), UPLOAD_BATCH_SIZE):
        batch = snapshots[offset : offset + UPLOAD_BATCH_SIZE]
        batch_rows = sum(len(snapshot["payload"]["rows"]) for snapshot in batch)
        request = urllib.request.Request(
            endpoint,
            data=_canonical_json(batch).encode("utf-8"),
            headers=_headers(token, sites_bypass_token),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Sector RORO 업로드 실패 HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(f"Sector RORO 업로드 연결 실패: {error}") from error
        totals["dates"] += int(result.get("dates", len(batch)))
        totals["rows"] += int(result.get("rows", batch_rows))
        totals["inserted"] += int(result.get("inserted", 0))
        totals["idempotent"] += int(result.get("idempotent", 0))
    totals.update(
        {"series": snapshots[0]["series_version"], "as_of": snapshots[-1]["as_of"]}
    )
    return totals


def research_history_payload(source: str | Path) -> dict[str, Any]:
    path = Path(source)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    required = {
        "decision_date", "ticker", "sector", "forecast_rank", "predicted_excess_20d",
        "actual_excess_20d", "alpha_hit", "quadrant", "quadrant_transition",
        "tail_heading", "roro_fit", "roro_context", "fit_change", "sector_status",
    }
    if not source_rows or not required.issubset(source_rows[0]):
        raise ValueError("Sector research history columns are incomplete")

    def optional_float(value: str) -> float | None:
        return float(value) if value.strip() else None

    def optional_bool(value: str) -> bool | None:
        if not value.strip():
            return None
        if value not in {"True", "False"}:
            raise ValueError("alpha_hit must be True, False or empty")
        return value == "True"

    rows = [
        {
            "decision_date": row["decision_date"],
            "sector": row["ticker"],
            "sector_name": row["sector"],
            "forecast_rank": int(row["forecast_rank"]),
            "predicted_excess_20d": float(row["predicted_excess_20d"]),
            "actual_excess_20d": optional_float(row["actual_excess_20d"]),
            "alpha_hit": optional_bool(row["alpha_hit"]),
            "quadrant": row["quadrant"],
            "quadrant_transition": row["quadrant_transition"],
            "tail_heading": row["tail_heading"],
            "roro_fit": optional_float(row["roro_fit"]),
            "roro_context": row["roro_context"],
            "fit_change": row["fit_change"],
            "sector_status": row["sector_status"],
        }
        for row in source_rows
    ]
    dates = sorted({row["decision_date"] for row in rows})
    for day in dates:
        day_rows = [row for row in rows if row["decision_date"] == day]
        if len(day_rows) != len(SECTOR_ETFS) or {row["sector"] for row in day_rows} != set(SECTOR_ETFS):
            raise ValueError(f"Sector research history requires 11 sectors on {day}")
    rows.sort(key=lambda row: (row["decision_date"], row["forecast_rank"], row["sector"]))
    generated_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "schema_version": RESEARCH_SCHEMA_VERSION,
        "series_version": RESEARCH_SERIES_VERSION,
        "model_version": MODEL_VERSION,
        "start": dates[0],
        "as_of": dates[-1],
        "generated_at": generated_at,
        "cadence": "WEEKLY",
        "rows": rows,
    }


def upload_research_history(
    payload: dict[str, Any], site_url: str, token: str, sites_bypass_token: str | None = None
) -> dict[str, Any]:
    request = urllib.request.Request(
        site_url.rstrip("/") + "/api/sector-history",
        data=_canonical_json(payload).encode("utf-8"),
        headers=_headers(token, sites_bypass_token),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Sector research history 업로드 실패 HTTP {error.code}: {detail}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"Sector research history 업로드 연결 실패: {error}") from error


def build_and_publish(
    shadow_database: str | Path,
    *,
    as_of: str | None = None,
    latest_only: bool = False,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    cutoff = _calendar_date(as_of, "as_of") if as_of is not None else None
    observations = read_sector_history(
        shadow_database,
        series_version=SERIES_VERSION,
        start=cutoff if latest_only and cutoff else None,
        end=cutoff,
    )
    snapshots = web_snapshots(observations, latest_only=latest_only)
    if not snapshots:
        raise ValueError("no Sector RORO snapshots for the requested cutoff")
    rows = sum(len(snapshot["payload"]["rows"]) for snapshot in snapshots)
    summary = {
        "series_version": SERIES_VERSION,
        "dates": len(snapshots),
        "rows": rows,
        "as_of": snapshots[-1]["as_of"],
        "source_database_mode": "mode=ro; query_only=ON",
    }
    uploaded = None
    if config is not None:
        uploaded = upload_snapshots(
            snapshots,
            str(config["site_url"]),
            str(config["upload_token"]),
            config.get("sites_bypass_token"),
        )
    return summary, snapshots, uploaded


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-database", type=Path, default=DEFAULT_SHADOW_DATABASE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--consent", type=Path, default=DEFAULT_CONSENT)
    parser.add_argument("--as-of")
    parser.add_argument("--latest-only", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--research-history", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if not args.no_upload and not args.config.is_file():
        raise FileNotFoundError(args.config)
    consent = (
        json.loads(args.consent.read_text(encoding="utf-8-sig"))
        if args.consent.is_file()
        else {}
    )
    upload_allowed = consent.get("allow_portfolio_aggregate_upload") is True
    config = (
        json.loads(args.config.read_text(encoding="utf-8-sig"))
        if not args.no_upload and upload_allowed
        else None
    )
    if args.research_history is not None:
        payload = research_history_payload(args.research_history)
        uploaded = upload_research_history(
            payload,
            str(config["site_url"]),
            str(config["upload_token"]),
            config.get("sites_bypass_token"),
        ) if config is not None else None
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"build": {"weeks": len({row["decision_date"] for row in payload["rows"]}), "rows": len(payload["rows"]), "as_of": payload["as_of"]}, "upload": uploaded}, ensure_ascii=False, allow_nan=False))
        return
    summary, snapshots, uploaded = build_and_publish(
        args.shadow_database,
        as_of=args.as_of,
        latest_only=args.latest_only,
        config=config,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            snapshots,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if not args.no_upload and not upload_allowed:
        print("Sector RORO 외부 게시 보류: roro_publish_consent.json의 명시 동의가 필요합니다.")
    print(
        json.dumps(
            {"build": summary, "upload": uploaded},
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
