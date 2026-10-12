from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any


class BloombergClient:
    def __init__(self, host: str, port: int, timeout: int):
        import blpapi
        self.blpapi = blpapi
        self.timeout = timeout
        self.next_request = 0
        options = blpapi.SessionOptions()
        options.setServerHost(host)
        options.setServerPort(port)
        if hasattr(options, "setClientMode") and hasattr(blpapi.SessionOptions, "DAPI"):
            options.setClientMode(blpapi.SessionOptions.DAPI)
        self.session = blpapi.Session(options)

    def __enter__(self):
        if not self.session.start() or not self.session.openService("//blp/refdata"):
            raise RuntimeError("Bloomberg Desktop API connection failed")
        return self

    def __exit__(self, *_):
        self.session.stop()

    def send(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        request = self.session.getService("//blp/refdata").createRequest("HistoricalDataRequest")
        request.fromPy(payload)
        self.next_request += 1
        correlation = self.blpapi.CorrelationId(self.next_request)
        self.session.sendRequest(request, correlationId=correlation)
        messages: list[dict[str, Any]] = []
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.session.cancel(correlation)
                raise TimeoutError("HistoricalDataRequest timed out")
            event = self.session.nextEvent(max(1, min(1000, math.ceil(remaining * 1000))))
            matched = False
            for message in event:
                if correlation not in message.correlationIds():
                    continue
                matched = True
                if str(message.messageType()) == "RequestFailure":
                    self.session.cancel(correlation)
                    raise RuntimeError("Bloomberg RequestFailure")
                messages.append(message.asElement().toPy())
            if matched and event.eventType() == self.blpapi.Event.RESPONSE:
                return messages


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def chunks(start: date, end: date) -> list[tuple[date, date]]:
    result = []
    cursor = start
    while cursor <= end:
        try:
            candidate = cursor.replace(year=cursor.year + 5) - timedelta(days=1)
        except ValueError:
            candidate = cursor.replace(year=cursor.year + 5, day=28) - timedelta(days=1)
        chunk_end = min(candidate, end)
        result.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return result


def errors_from(messages: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    rows = []
    for data in messages:
        if data.get("responseError"):
            rows.append({"request": label, "security": "", "field_id": "", "severity": "FATAL",
                         "message": str(data["responseError"])})
        for block in as_list(data.get("securityData")):
            security = str(block.get("security", ""))
            if block.get("securityError"):
                rows.append({"request": label, "security": security, "field_id": "", "severity": "FATAL",
                             "message": str(block["securityError"])})
            for exc in as_list(block.get("fieldExceptions")):
                info = exc.get("errorInfo", {})
                rows.append({"request": label, "security": security, "field_id": str(exc.get("fieldId", "")),
                             "severity": "NOT_APPLICABLE" if "NOT_APPLICABLE" in str(info.get("subcategory", "")) else "ERROR",
                             "message": str(info.get("message", info))})
    return rows


def history_from(messages: list[dict[str, Any]], group: str, fields: list[str], period: str,
                 retrieved: str) -> list[dict[str, Any]]:
    rows = []
    for data in messages:
        for block in as_list(data.get("securityData")):
            security = str(block.get("security", ""))
            for item in as_list(block.get("fieldData")):
                if not isinstance(item, dict):
                    continue
                observation = str(item.get("date", ""))[:10]
                for field_id in fields:
                    if item.get(field_id) not in (None, ""):
                        rows.append({"data_group": group, "security": security, "field_id": field_id,
                                     "target_period": period, "observation_period": observation,
                                     "observation_date": observation, "value": item[field_id],
                                     "first_release_datetime": "", "available_at": "", "unit": "", "currency": "",
                                     "missing_reason": "Publication timing and index units not inferred; retain diagnostic field_catalog.csv",
                                     "revision_status": "PIT_NOT_VERIFIED" if group == "EARNINGS" else
                                         "INITIAL_VALUE_CANDIDATE;PIT_NOT_VERIFIED" if field_id == "PR296" else
                                         "PRE_RELEASE_CONSENSUS;PIT_NOT_VERIFIED",
                                     "retrieved_at_utc": retrieved, "source": "Bloomberg Desktop API"})
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def complete_history(rows: list[dict[str, Any]], securities: list[str], fields: list[str]) -> bool:
    expected = {(s,f) for s in securities for f in fields}
    found = {(r['security'],r['field_id']) for r in rows}
    return expected <= found and all(math.isfinite(float(r['value'])) for r in rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepared Bloomberg earnings/economic full download; not run by diagnostic.")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8194)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--execute", action="store_true", help="Required acknowledgement for real Bloomberg requests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.execute:
        raise SystemExit("No requests sent. Re-run with --execute after reviewing request/time/quota estimates.")
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    if start > end or args.timeout <= 0:
        raise ValueError("Invalid date range or timeout")
    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    state_path = outdir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"completed": []}
    completed = set(state["completed"])
    history_path, error_path = outdir / "history.csv", outdir / "errors.csv"
    if completed and not history_path.is_file():
        raise ValueError("Checkpoint exists without history; use a new output directory")
    raw_dir = outdir / 'raw_responses'
    raw_dir.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    hist_cols = ["data_group", "security", "field_id", "target_period", "observation_period", "observation_date",
                 "value", "first_release_datetime", "available_at", "unit", "currency", "missing_reason",
                 "revision_status", "retrieved_at_utc", "source"]
    err_cols = ["request", "security", "field_id", "severity", "message"]
    if history_path.exists():
        with history_path.open(encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if error_path.exists():
        with error_path.open(encoding="utf-8-sig", newline="") as handle:
            errors.extend(csv.DictReader(handle))

    plans = []
    for left, right in chunks(start, end):
        for period in ("1TY", "1FY", "2FY"):
            plans.append((f"earnings_{period}_{left}_{right}", "EARNINGS", ["NDX Index", "SPX Index"],
                          ["BE008", "BE001"], period, "WEEKLY", left, right))
        plans.append((f"econ_{left}_{right}", "ECONOMIC_RELEASE", ["CPI YOY Index", "NFP TCH Index", "NAPMPMI Index"],
                      ["PR296", "PR291"], "", "DAILY", left, right))

    with BloombergClient(args.host, args.port, args.timeout) as client:
        for label, group, securities, fields, period, frequency, left, right in plans:
            if label in completed:
                continue
            payload: dict[str, Any] = {"securities": securities, "fields": fields,
                "startDate": left.strftime("%Y%m%d"), "endDate": right.strftime("%Y%m%d"),
                "periodicitySelection": frequency}
            if period:
                payload["overrides"] = [{"fieldId": "BE997", "value": period}]
            success = False
            for attempt in range(1, 4):
                try:
                    retrieved = datetime.now(timezone.utc).isoformat()
                    messages = client.send(payload)
                    (raw_dir/f'{label}_attempt{attempt}.json').write_text(
                        json.dumps({'payload':payload,'retrieved_at_utc':retrieved,'messages':messages},default=str,ensure_ascii=False,indent=2),encoding='utf-8')
                    new_errors = errors_from(messages, label)
                    errors.extend(new_errors)
                    if any(e["severity"] == "FATAL" for e in new_errors):
                        raise RuntimeError("fatal Bloomberg response")
                    received = history_from(messages, group, fields, period, retrieved)
                    rows.extend(received)
                    success = not new_errors and complete_history(received,securities,fields)
                    if not success:
                        errors.append({'request':label,'security':'','field_id':'','severity':'INCOMPLETE',
                                       'message':'Field errors or incomplete security/field coverage; request not checkpointed'})
                    break
                except Exception as exc:
                    if attempt == 3:
                        errors.append({"request": label, "security": "", "field_id": "", "severity": "FATAL",
                                       "message": f"retry exhausted: {type(exc).__name__}: {exc}"})
                    else:
                        time.sleep(attempt * 2)
            unique = {(r["data_group"], r["security"], r["field_id"], r["target_period"], r["observation_date"]): r for r in rows}
            rows = sorted(unique.values(), key=lambda r: (r["data_group"], r["security"], r["field_id"], r["target_period"], r["observation_date"]))
            write_csv(history_path, rows, hist_cols)
            write_csv(error_path, errors, err_cols)
            if success:
                completed.add(label)
                state["completed"] = sorted(completed)
                temporary = state_path.with_suffix('.json.tmp')
                temporary.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
                temporary.replace(state_path)
    pending = [p[0] for p in plans if p[0] not in completed]
    print(json.dumps({"status": "PARTIAL" if pending else "COMPLETE", "pending_requests":pending,"history_rows": len(rows), "errors": len(errors),
                      "output_dir": str(outdir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
