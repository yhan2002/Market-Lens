from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from pathlib import Path
from typing import Any


# Same price universe used by sector_rotation.py, expressed in Bloomberg syntax.
SECURITIES = {
    "SPY": "SPY US Equity",
    "XLB": "XLB US Equity",
    "XLC": "XLC US Equity",
    "XLE": "XLE US Equity",
    "XLF": "XLF US Equity",
    "XLI": "XLI US Equity",
    "XLK": "XLK US Equity",
    "XLP": "XLP US Equity",
    "XLRE": "XLRE US Equity",
    "XLU": "XLU US Equity",
    "XLV": "XLV US Equity",
    "XLY": "XLY US Equity",
    "^VIX": "VIX Index",
    "^MOVE": "MOVE Index",
}
REQUIRED_LOCAL_TICKERS = tuple(ticker for ticker in SECURITIES if not ticker.startswith("^"))
REFERENCE_FIELDS = ("NAME", "ID_BB_GLOBAL", "CRNCY", "SECURITY_TYP", "PX_LAST")
HISTORY_FIELDS = ("PX_LAST", "PX_VOLUME", "TOT_RETURN_INDEX_GROSS_DVDS")
FLOW_SECURITIES = {
    ticker: security for ticker, security in SECURITIES.items() if not ticker.startswith("^")
}
FLOW_FIELDS = (
    "FUND_FLOW",
    "FUND_FLOW_DATE",
    "FUND_FLOW_1_WK",
    "FUND_FLOW_1_MONTH",
    "FUND_FLOW_3_MONTH",
    "AGGREGATED_FUND_FLOW",
    "AGGREGATED_FUND_FLOW_DATE",
    "FUND_TOTAL_ASSETS",
)
FLOW_VALUE_FIELDS = (
    "FUND_FLOW",
    "FUND_FLOW_1_WK",
    "FUND_FLOW_1_MONTH",
    "FUND_FLOW_3_MONTH",
    "AGGREGATED_FUND_FLOW",
)
FIELD_SEARCH_TERMS = (
    "relative rotation graph",
    "RS ratio",
    "RS momentum",
    "earnings estimate revision",
    "fund flow",
)
BLPAPI_INSTALL = (
    "python -m pip install "
    "--index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (datetime, date, datetime_time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _as_list(value: Any) -> list[Any]:
    if value in (None, {}):
        return []
    return value if isinstance(value, list) else [value]


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], preferred: tuple[str, ...]) -> None:
    extra = sorted({key for row in rows for key in row} - set(preferred))
    fieldnames = [*preferred, *extra]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {key: _csv_value(value) for key, value in row.items()} for row in rows
        )


def _event_name(blpapi: Any, event_type: int) -> str:
    for name in (
        "ADMIN",
        "SESSION_STATUS",
        "SUBSCRIPTION_STATUS",
        "REQUEST_STATUS",
        "RESPONSE",
        "PARTIAL_RESPONSE",
        "TIMEOUT",
        "AUTHORIZATION_STATUS",
        "SERVICE_STATUS",
    ):
        if getattr(blpapi.Event, name, None) == event_type:
            return name
    return str(event_type)


def _send_request(
    session: Any,
    blpapi: Any,
    service_name: str,
    operation: str,
    payload: dict[str, Any],
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    service = session.getService(service_name)
    request = service.createRequest(operation)
    request.fromPy(payload)
    session.sendRequest(request)

    messages: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"{operation} timed out after {timeout_seconds}s")
        event = session.nextEvent(max(1, min(1000, math.ceil(remaining * 1000))))
        event_type = event.eventType()
        for message in event:
            message_type = str(message.messageType())
            messages.append(
                {
                    "event_type": _event_name(blpapi, event_type),
                    "message_type": message_type,
                    "data": _jsonable(message.asElement().toPy()),
                }
            )
            if event_type == blpapi.Event.REQUEST_STATUS and "RequestFailure" in message_type:
                return messages
        if event_type == blpapi.Event.RESPONSE:
            return messages


def _security_blocks(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for message in messages:
        data = message.get("data") or {}
        if isinstance(data, dict):
            blocks.extend(item for item in _as_list(data.get("securityData")) if isinstance(item, dict))
    return blocks


def _reference_rows(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reverse = {security: ticker for ticker, security in SECURITIES.items()}
    rows: list[dict[str, Any]] = []
    for block in _security_blocks(messages):
        security = str(block.get("security", ""))
        fields = block.get("fieldData") or {}
        if not isinstance(fields, dict):
            fields = {}
        rows.append(
            {
                "local_ticker": reverse.get(security, ""),
                "security": security,
                **fields,
            }
        )
    return rows


def _history_rows(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reverse = {security: ticker for ticker, security in SECURITIES.items()}
    rows: list[dict[str, Any]] = []
    for block in _security_blocks(messages):
        security = str(block.get("security", ""))
        for fields in _as_list(block.get("fieldData")):
            if isinstance(fields, dict):
                rows.append(
                    {
                        "local_ticker": reverse.get(security, ""),
                        "security": security,
                        **fields,
                    }
                )
    return rows


def _field_catalog_rows(
    request_type: str,
    search_term: str,
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for message in messages:
        data = message.get("data") or {}
        if not isinstance(data, dict):
            continue
        for item in _as_list(data.get("fieldData")):
            if not isinstance(item, dict):
                continue
            info = item.get("fieldInfo") or {}
            error = item.get("fieldError") or {}
            rows.append(
                {
                    "request_type": request_type,
                    "search_term": search_term,
                    "field_id": item.get("id", ""),
                    "mnemonic": info.get("mnemonic", "") if isinstance(info, dict) else "",
                    "description": info.get("description", "") if isinstance(info, dict) else "",
                    "datatype": info.get("datatype", "") if isinstance(info, dict) else "",
                    "field_type": info.get("ftype", "") if isinstance(info, dict) else "",
                    "category": info.get("categoryName", "") if isinstance(info, dict) else "",
                    "documentation": info.get("documentation", "")
                    if isinstance(info, dict)
                    else "",
                    "field_info_json": info if isinstance(info, dict) else {},
                    "status": "ERROR" if error else "OK",
                    "error_category": error.get("category", "") if isinstance(error, dict) else "",
                    "error_message": error.get("message", "") if isinstance(error, dict) else "",
                }
            )
    return rows


def _error_row(
    request_name: str,
    level: str,
    error: Any,
    *,
    security: str = "",
    field: str = "",
) -> dict[str, Any]:
    detail = error if isinstance(error, dict) else {"message": str(error)}
    return {
        "request": request_name,
        "level": level,
        "security": security,
        "field": field,
        "category": detail.get("category", ""),
        "subcategory": detail.get("subcategory", ""),
        "code": detail.get("code", ""),
        "message": detail.get("message", ""),
        "source": detail.get("source", ""),
    }


def _errors(request_name: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for message in messages:
        data = message.get("data") or {}
        if not isinstance(data, dict):
            continue
        for key in ("responseError", "fieldSearchError", "reason"):
            if data.get(key):
                rows.append(_error_row(request_name, "REQUEST", data[key]))
        for block in _as_list(data.get("securityData")):
            if not isinstance(block, dict):
                continue
            security = str(block.get("security", ""))
            if block.get("securityError"):
                rows.append(
                    _error_row(request_name, "SECURITY", block["securityError"], security=security)
                )
            for exception in _as_list(block.get("fieldExceptions")):
                if isinstance(exception, dict):
                    rows.append(
                        _error_row(
                            request_name,
                            "FIELD",
                            exception.get("errorInfo", {}),
                            security=security,
                            field=str(exception.get("fieldId", "")),
                        )
                    )
        for item in _as_list(data.get("fieldData")):
            if isinstance(item, dict) and item.get("fieldError"):
                rows.append(
                    _error_row(
                        request_name,
                        "FIELD_INFO",
                        item["fieldError"],
                        field=str(item.get("id", "")),
                    )
                )
    return rows


def _has_value(value: Any) -> bool:
    return value not in (None, "", {}) and not (
        isinstance(value, float) and math.isnan(value)
    )


def _as_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text[:10], fmt).date()
        except ValueError:
            pass
    return None


def _flow_coverage(
    reference: list[dict[str, Any]],
    history: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reference_by_ticker = {
        str(row.get("local_ticker")): row for row in reference if row.get("local_ticker")
    }
    history_by_ticker: dict[str, list[dict[str, Any]]] = {}
    for row in history:
        history_by_ticker.setdefault(str(row.get("local_ticker", "")), []).append(row)

    rows: list[dict[str, Any]] = []
    for field in FLOW_FIELDS:
        reference_tickers = {
            ticker
            for ticker, row in reference_by_ticker.items()
            if _has_value(row.get(field))
        }
        historical_tickers: set[str] = set()
        observations: list[tuple[date, str, Any]] = []
        change_gaps: list[int] = []
        for ticker, ticker_rows in history_by_ticker.items():
            previous: Any = object()
            previous_change_date: date | None = None
            for item in sorted(ticker_rows, key=lambda row: str(row.get("date", ""))):
                value = item.get(field)
                observation_date = _as_date(item.get("date"))
                if not _has_value(value) or observation_date is None:
                    continue
                historical_tickers.add(ticker)
                observations.append((observation_date, ticker, value))
                marker = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, default=str)
                if marker != previous:
                    if previous_change_date is not None:
                        change_gaps.append((observation_date - previous_change_date).days)
                    previous = marker
                    previous_change_date = observation_date

        value_dates = [
            parsed
            for ticker in reference_tickers
            if (parsed := _as_date(reference_by_ticker[ticker].get(field))) is not None
        ]
        field_errors = [row for row in errors if str(row.get("field", "")) == field]
        rows.append(
            {
                "field": field,
                "kind": (
                    "date"
                    if field.endswith("_DATE")
                    else "denominator"
                    if field == "FUND_TOTAL_ASSETS"
                    else "flow"
                ),
                "reference_populated": len(reference_tickers),
                "reference_missing_tickers": sorted(set(FLOW_SECURITIES) - reference_tickers),
                "reference_value_date_min": min(value_dates).isoformat() if value_dates else "",
                "reference_value_date_max": max(value_dates).isoformat() if value_dates else "",
                "reference_value_date_age_days": (
                    (date.today() - max(value_dates)).days if value_dates else ""
                ),
                "historical_populated_securities": len(historical_tickers),
                "historical_observations": len(observations),
                "historical_missing_tickers": sorted(set(FLOW_SECURITIES) - historical_tickers),
                "history_start": min(item[0] for item in observations).isoformat()
                if observations
                else "",
                "history_end": max(item[0] for item in observations).isoformat()
                if observations
                else "",
                "median_calendar_days_between_value_changes": (
                    statistics.median(change_gaps) if change_gaps else ""
                ),
                "field_error_count": len(field_errors),
                "field_error_securities": sorted(
                    {str(row.get("security", "")) for row in field_errors if row.get("security")}
                ),
            }
        )
    return rows


def _request_or_record(
    raw: dict[str, list[dict[str, Any]]],
    errors: list[dict[str, Any]],
    session: Any,
    blpapi: Any,
    label: str,
    service_name: str,
    operation: str,
    payload: dict[str, Any],
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    try:
        messages = _send_request(
            session, blpapi, service_name, operation, payload, timeout_seconds
        )
    except Exception as exc:  # Keep later diagnostic requests running.
        errors.append(_error_row(label, "CLIENT", exc))
        messages = []
    raw[label] = messages
    errors.extend(_errors(label, messages))
    return messages


def _output_dir(value: str | None, *, fund_flow: bool = False) -> Path:
    if value:
        return Path(value)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = "bloomberg_fund_flow_diagnostic" if fund_flow else "bloomberg_diagnostic"
    return Path("outputs") / f"{name}_{stamp}"


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    end = date.today()
    start = end - timedelta(days=args.days)
    securities = FLOW_SECURITIES if args.fund_flow else SECURITIES
    reference_fields = (
        (*REFERENCE_FIELDS[:-1], *FLOW_FIELDS) if args.fund_flow else REFERENCE_FIELDS
    )
    history_fields = FLOW_FIELDS if args.fund_flow else HISTORY_FIELDS
    search_terms = () if args.fund_flow else FIELD_SEARCH_TERMS
    return {
        "mode": "fund-flow" if args.fund_flow else "baseline",
        "dry_run": True,
        "host": args.host,
        "port": args.port,
        "date_range": {"start": start.isoformat(), "end": end.isoformat()},
        "securities": securities,
        "reference_fields": reference_fields,
        "history_fields": history_fields,
        "history_periodicity": "WEEKLY" if args.fund_flow else "DAILY",
        "field_search_terms": search_terms,
        "database_write": False,
        "estimated_scope": {
            "requests": 3 + len(search_terms),
            "securities": len(securities),
            "calendar_days": args.days,
        },
    }


def _self_test() -> None:
    reference_messages = [
        {
            "data": {
                "securityData": [
                    {
                        "security": "SPY US Equity",
                        "fieldData": {
                            "PX_LAST": 100.0,
                            "NAME": "SPDR S&P 500 ETF TRUST",
                            "FUND_FLOW": 1.0,
                            "FUND_FLOW_DATE": "2026-08-28",
                            "FUND_FLOW_1_WK": float("nan"),
                        },
                        "fieldExceptions": [],
                    }
                ]
            }
        }
    ]
    history_messages = [
        {
            "data": {
                "securityData": {
                    "security": "SPY US Equity",
                    "fieldData": [
                        {"date": "2026-08-21", "PX_LAST": 99.0, "FUND_FLOW": 1.0},
                        {"date": "2026-08-28", "PX_LAST": 100.0, "FUND_FLOW": 2.0},
                    ],
                }
            }
        }
    ]
    error_messages = [
        {
            "data": {
                "securityData": [
                    {
                        "security": "SPY US Equity",
                        "fieldExceptions": [
                            {
                                "fieldId": "BAD_FIELD",
                                "errorInfo": {"category": "BAD_FLD", "message": "Unknown field"},
                            }
                        ],
                    }
                ]
            }
        }
    ]
    assert _reference_rows(reference_messages)[0]["local_ticker"] == "SPY"
    assert _history_rows(history_messages)[1]["date"] == "2026-08-28"
    assert _errors("test", error_messages)[0]["field"] == "BAD_FIELD"
    coverage = _flow_coverage(
        _reference_rows(reference_messages), _history_rows(history_messages), []
    )
    fund_flow = next(row for row in coverage if row["field"] == "FUND_FLOW")
    one_week = next(row for row in coverage if row["field"] == "FUND_FLOW_1_WK")
    assert fund_flow["reference_populated"] == 1
    assert fund_flow["historical_observations"] == 2
    assert fund_flow["median_calendar_days_between_value_changes"] == 7
    assert one_week["reference_populated"] == 0
    print("self-test: PASS")


def run(args: argparse.Namespace) -> int:
    if args.self_test:
        _self_test()
        return 0
    if args.dry_run:
        print(json.dumps(_plan(args), ensure_ascii=False, indent=2))
        return 0

    try:
        import blpapi  # type: ignore[import-not-found]
    except ImportError:
        print("Bloomberg Python API(blpapi)가 설치되어 있지 않습니다.", file=sys.stderr)
        print(BLPAPI_INSTALL, file=sys.stderr)
        return 2

    fund_flow = args.fund_flow
    securities = FLOW_SECURITIES if fund_flow else SECURITIES
    reference_fields = (
        (*REFERENCE_FIELDS[:-1], *FLOW_FIELDS) if fund_flow else REFERENCE_FIELDS
    )
    history_fields = FLOW_FIELDS if fund_flow else HISTORY_FIELDS
    field_info_fields = (
        FLOW_FIELDS if fund_flow else sorted(set((*REFERENCE_FIELDS, *HISTORY_FIELDS)))
    )
    search_terms = () if fund_flow else FIELD_SEARCH_TERMS
    periodicity = "WEEKLY" if fund_flow else "DAILY"

    output_dir = _output_dir(args.output_dir, fund_flow=fund_flow)
    output_dir.mkdir(parents=True, exist_ok=False)
    end = date.today()
    start = end - timedelta(days=args.days)
    options = blpapi.SessionOptions()
    options.setServerHost(args.host)
    options.setServerPort(args.port)
    if hasattr(options, "setClientMode") and hasattr(blpapi.SessionOptions, "DAPI"):
        options.setClientMode(blpapi.SessionOptions.DAPI)
    session = blpapi.Session(options)

    raw: dict[str, list[dict[str, Any]]] = {}
    errors: list[dict[str, Any]] = []
    try:
        if not session.start():
            raise RuntimeError("Desktop API session start failed")
        for service_name in ("//blp/refdata", "//blp/apiflds"):
            if not session.openService(service_name):
                raise RuntimeError(f"service open failed: {service_name}")

        reference_messages = _request_or_record(
            raw,
            errors,
            session,
            blpapi,
            "reference",
            "//blp/refdata",
            "ReferenceDataRequest",
            {"securities": list(securities.values()), "fields": list(reference_fields)},
            args.timeout,
        )
        history_messages = _request_or_record(
            raw,
            errors,
            session,
            blpapi,
            "history",
            "//blp/refdata",
            "HistoricalDataRequest",
            {
                "securities": list(securities.values()),
                "fields": list(history_fields),
                "startDate": start.strftime("%Y%m%d"),
                "endDate": end.strftime("%Y%m%d"),
                "periodicitySelection": periodicity,
            },
            args.timeout,
        )
        field_info_messages = _request_or_record(
            raw,
            errors,
            session,
            blpapi,
            "field_info",
            "//blp/apiflds",
            "FieldInfoRequest",
            {
                "id": list(field_info_fields),
                "returnFieldDocumentation": fund_flow,
            },
            args.timeout,
        )
        search_messages: dict[str, list[dict[str, Any]]] = {}
        for index, term in enumerate(search_terms, 1):
            label = f"field_search_{index}"
            search_messages[term] = _request_or_record(
                raw,
                errors,
                session,
                blpapi,
                label,
                "//blp/apiflds",
                "FieldSearchRequest",
                {"searchSpec": term, "returnFieldDocumentation": False},
                args.timeout,
            )
    except Exception as exc:
        errors.append(_error_row("connection", "CLIENT", exc))
        reference_messages = []
        history_messages = []
        field_info_messages = []
        search_messages = {}
    finally:
        session.stop()

    reference = _reference_rows(reference_messages)
    history = _history_rows(history_messages)
    catalog = _field_catalog_rows("FieldInfoRequest", "", field_info_messages)
    for term, messages in search_messages.items():
        catalog.extend(_field_catalog_rows("FieldSearchRequest", term, messages))

    _write_csv(
        output_dir / "reference.csv",
        reference,
        ("local_ticker", "security", *reference_fields),
    )
    _write_csv(
        output_dir / "history.csv",
        history,
        ("local_ticker", "security", "date", *history_fields),
    )
    _write_csv(
        output_dir / "BLOOMBERG_FIELD_CATALOG.csv",
        catalog,
        (
            "request_type",
            "search_term",
            "field_id",
            "mnemonic",
            "description",
            "datatype",
            "field_type",
            "category",
            "documentation",
            "field_info_json",
            "status",
            "error_category",
            "error_message",
        ),
    )
    _write_csv(
        output_dir / "errors.csv",
        errors,
        (
            "request",
            "level",
            "security",
            "field",
            "category",
            "subcategory",
            "code",
            "message",
            "source",
        ),
    )
    coverage = _flow_coverage(reference, history, errors) if fund_flow else []
    if fund_flow:
        _write_csv(
            output_dir / "fund_flow_coverage.csv",
            coverage,
            (
                "field",
                "kind",
                "reference_populated",
                "reference_missing_tickers",
                "reference_value_date_min",
                "reference_value_date_max",
                "reference_value_date_age_days",
                "historical_populated_securities",
                "historical_observations",
                "historical_missing_tickers",
                "history_start",
                "history_end",
                "median_calendar_days_between_value_changes",
                "field_error_count",
                "field_error_securities",
            ),
        )
    (output_dir / "raw_responses.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    client_errors = [row for row in errors if row.get("level") == "CLIENT"]
    request_status = "FAIL" if any(row.get("request") == "connection" for row in client_errors) else (
        "PARTIAL" if client_errors else "PASS"
    )
    summary = {
        "status": request_status,
        "mode": "fund-flow" if fund_flow else "baseline",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": args.host,
        "port": args.port,
        "date_range": {"start": start.isoformat(), "end": end.isoformat()},
        "history_periodicity": periodicity,
        "counts": {
            "reference_rows": len(reference),
            "history_rows": len(history),
            "field_catalog_rows": len(catalog),
            "errors": len(errors),
        },
    }
    if fund_flow:
        reference_ok = {
            str(row.get("local_ticker"))
            for row in reference
            if any(_has_value(row.get(field)) for field in FLOW_VALUE_FIELDS)
        }
        history_ok = {
            str(row.get("local_ticker"))
            for row in history
            if any(_has_value(row.get(field)) for field in FLOW_VALUE_FIELDS)
        }
        aum_ok = {
            str(row.get("local_ticker"))
            for row in reference
            if _has_value(row.get("FUND_TOTAL_ASSETS"))
        }
        metadata_ok = {
            str(row.get(key, "")).upper()
            for row in catalog
            if row.get("status") == "OK"
            for key in ("field_id", "mnemonic")
            if row.get(key)
        }
        missing_reference = sorted(set(FLOW_SECURITIES) - reference_ok)
        missing_history = sorted(set(FLOW_SECURITIES) - history_ok)
        missing_aum = sorted(set(FLOW_SECURITIES) - aum_ok)
        missing_metadata = sorted(set(FLOW_FIELDS) - metadata_ok)
        dates_by_ticker = {ticker: set() for ticker in FLOW_SECURITIES}
        for row in history:
            ticker = str(row.get("local_ticker", ""))
            observation_date = _as_date(row.get("date"))
            if (
                ticker in dates_by_ticker
                and observation_date is not None
                and any(_has_value(row.get(field)) for field in FLOW_VALUE_FIELDS)
            ):
                dates_by_ticker[ticker].add(observation_date)
        common_dates = set.intersection(*dates_by_ticker.values()) if dates_by_ticker else set()
        data_readiness = (
            "READY"
            if not missing_reference and not missing_history and not missing_aum and not missing_metadata
            else "PARTIAL"
            if reference_ok or history_ok
            else "UNAVAILABLE"
        )
        status = (
            "PASS"
            if request_status == "PASS" and data_readiness == "READY"
            else "FAIL"
            if request_status == "FAIL"
            else "PARTIAL"
        )
        summary["status"] = status
        summary.update(
            {
                "request_status": request_status,
                "data_readiness": data_readiness,
                "flow_reference_fields_with_values": [
                    field
                    for field in FLOW_VALUE_FIELDS
                    if any(_has_value(row.get(field)) for row in reference)
                ],
                "flow_history_fields_with_values": [
                    field
                    for field in FLOW_VALUE_FIELDS
                    if any(_has_value(row.get(field)) for row in history)
                ],
                "missing_flow_reference_securities": missing_reference,
                "missing_flow_history_securities": missing_history,
                "missing_aum_reference_securities": missing_aum,
                "missing_field_metadata": missing_metadata,
                "common_latest_history_date": max(common_dates).isoformat()
                if common_dates
                else "",
                "notes": [
                    "No database, model, or website rows were written.",
                    "PASS describes request execution; data_readiness describes model usability.",
                    "Field metadata OK does not prove value entitlement or historical availability.",
                    "Do not import until units, aggregation scope, value dates, and PIT history are approved.",
                ],
            }
        )
    else:
        reference_ok = {
            str(row.get("local_ticker")) for row in reference if _has_value(row.get("PX_LAST"))
        }
        history_ok = {
            str(row.get("local_ticker")) for row in history if _has_value(row.get("PX_LAST"))
        }
        missing_reference = sorted(set(REQUIRED_LOCAL_TICKERS) - reference_ok)
        missing_history = sorted(set(REQUIRED_LOCAL_TICKERS) - history_ok)
        if request_status == "PASS" and (missing_reference or missing_history):
            summary["status"] = "PARTIAL"
        summary.update(
            {
                "missing_required_reference": missing_reference,
                "missing_required_history": missing_history,
                "notes": [
                    "No database rows were written.",
                    "PX_LAST is diagnostic only; adjustment and vintage semantics are not approved for import.",
                    "Zero RRG field-search results do not prove an API failure or create an RRG entitlement.",
                ],
            }
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({**summary, "output_dir": str(output_dir.resolve())}, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "PASS" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small, read-only Bloomberg Desktop API entitlement and field diagnostic."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8194)
    parser.add_argument("--days", type=int, default=21, help="calendar-day history window")
    parser.add_argument("--timeout", type=int, default=30, help="seconds per request")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--fund-flow",
        action="store_true",
        help="probe SPY + 11 sector ETF fund-flow/AUM fields with weekly history",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan without blpapi or API calls")
    parser.add_argument("--self-test", action="store_true", help="test local response parsing without API calls")
    args = parser.parse_args()
    if args.days < 7:
        parser.error("--days must be at least 7")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.timeout < 5:
        parser.error("--timeout must be at least 5 seconds")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
