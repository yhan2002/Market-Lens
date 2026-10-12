"""Research-only daily SOXX/IGV internals beneath the production XLK sector."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import urllib.error
import urllib.request
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from market_data import download_prices, register_universe
from sector_rotation import SECTOR_ETFS


TICKERS = ("SOXX", "IGV", "XLK")
GROUP_NAME = "IT_INTERNALS_SHADOW"
SCHEMA_VERSION = 2
SERIES_VERSION = "it-internals-shadow-v3-soxx-igv-xlk-daily-r20-r60-f20"
SHORT_WINDOW = 20
LONG_WINDOW = 60
FORWARD_WINDOW = 20
DEFAULT_DATABASE = Path("outputs/it_internals_shadow/it_internals.db")
DEFAULT_OUTPUT = Path("outputs/it_internals_shadow/daily_history.csv")
DEFAULT_METADATA = Path("outputs/it_internals_shadow/metadata.json")
DEFAULT_WEB_OUTPUT = Path("outputs/it_internals_shadow/site_payload.json")
DEFAULT_CONFIG = Path("site_upload_config.json")
DEFAULT_CONSENT = Path("roro_publish_consent.json")
STATES = (
    "BROAD_STRENGTH",
    "BROAD_WEAKNESS",
    "SEMICONDUCTOR_LEAD",
    "SOFTWARE_LEAD",
)
STATE_CODES = {state: code for code, state in enumerate(STATES)}

OUTPUT_COLUMNS = [
    "schema_version",
    "series_version",
    "decision_date",
    "effective_date",
    "label_end",
    "actual_available",
    "soxx_price",
    "igv_price",
    "xlk_price",
    "soxx_return_20d",
    "igv_return_20d",
    "xlk_return_20d",
    "soxx_return_60d",
    "igv_return_60d",
    "xlk_return_60d",
    "soxx_vs_xlk_20d",
    "igv_vs_xlk_20d",
    "soxx_vs_xlk_60d",
    "igv_vs_xlk_60d",
    "soxx_minus_igv_20d",
    "soxx_minus_igv_60d",
    "state_20d",
    "state_60d",
    "state_confirmed",
    "prior_state_20d",
    "state_transition",
    "current_pair_leader_20d",
    "soxx_forward_20d",
    "igv_forward_20d",
    "xlk_forward_20d",
    "pair_forward_actual_20d",
    "pair_actual_winner",
    "soxx_forward_vs_xlk_20d",
    "igv_forward_vs_xlk_20d",
]


def refresh_prices(
    database: Path = DEFAULT_DATABASE,
    *,
    years: int = 20,
    backfill: bool = False,
    require_date: date | None = None,
) -> dict[str, object]:
    registered = register_universe(
        database,
        TICKERS,
        group_name=GROUP_NAME,
        source="IT internals research shadow",
    )
    ticker_count, row_count, start = download_prices(
        database,
        years=years,
        group_name=GROUP_NAME,
        backfill=backfill,
        require_date=require_date,
    )
    return {
        "database": str(database),
        "registered": registered,
        "downloaded_tickers": ticker_count,
        "upserted_rows": row_count,
        "requested_start": start,
    }


def load_it_price_matrix(
    database: Path, *, tickers: tuple[str, ...] = TICKERS, as_of: str | None = None
) -> pd.DataFrame:
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        prices = pd.read_sql_query(
            f"""
            SELECT ticker, date, adj_close AS price
            FROM daily_prices
            WHERE ticker IN ({','.join('?' for _ in tickers)})
              AND date <= ?
            ORDER BY date, ticker
            """,
            connection,
            params=(*tickers, as_of or "9999-12-31"),
            parse_dates=["date"],
        )
    if prices.empty:
        raise ValueError("adjusted-close history is missing: " + "/".join(tickers))
    if prices.duplicated(["ticker", "date"]).any():
        raise ValueError("IT internals prices contain duplicate ticker-date rows")
    prices["price"] = pd.to_numeric(prices["price"], errors="coerce")
    matrix = prices.pivot(index="date", columns="ticker", values="price").sort_index()
    missing = [ticker for ticker in tickers if ticker not in matrix]
    if missing:
        raise ValueError("missing adjusted-close history: " + ", ".join(missing))
    matrix = matrix.loc[:, list(tickers)].dropna(how="any")
    if len(matrix) <= LONG_WINDOW:
        raise ValueError(f"need more than {LONG_WINDOW} common {'/'.join(tickers)} sessions")
    if not np.isfinite(matrix.to_numpy()).all() or (matrix <= 0).any().any():
        raise ValueError("adjusted closes must be finite and positive")
    matrix.columns.name = None
    return matrix


def classify_it_state(
    soxx_vs_xlk: pd.Series, igv_vs_xlk: pd.Series
) -> pd.Series:
    if not soxx_vs_xlk.index.equals(igv_vs_xlk.index):
        raise ValueError("state inputs must share the same index")
    complete = soxx_vs_xlk.notna() & igv_vs_xlk.notna()
    state = pd.Series(pd.NA, index=soxx_vs_xlk.index, dtype="string")
    state.loc[complete & (soxx_vs_xlk >= 0) & (igv_vs_xlk >= 0)] = "BROAD_STRENGTH"
    state.loc[complete & (soxx_vs_xlk < 0) & (igv_vs_xlk < 0)] = "BROAD_WEAKNESS"
    state.loc[complete & (soxx_vs_xlk >= 0) & (igv_vs_xlk < 0)] = "SEMICONDUCTOR_LEAD"
    state.loc[complete & (soxx_vs_xlk < 0) & (igv_vs_xlk >= 0)] = "SOFTWARE_LEAD"
    return state


def _leader(spread: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [spread > 0, spread < 0],
            ["SEMICONDUCTOR", "SOFTWARE"],
            default="TIE",
        ),
        index=spread.index,
        dtype="string",
    ).where(spread.notna())


def build_it_daily_frame(prices: pd.DataFrame) -> pd.DataFrame:
    missing = [ticker for ticker in TICKERS if ticker not in prices]
    if missing:
        raise ValueError("missing price columns: " + ", ".join(missing))
    clean = prices.loc[:, list(TICKERS)].copy()
    clean.index = pd.to_datetime(clean.index)
    clean = clean.sort_index().apply(pd.to_numeric, errors="coerce")
    if clean.index.has_duplicates:
        raise ValueError("price dates must be unique")
    if clean.isna().any().any() or not np.isfinite(clean.to_numpy()).all() or (clean <= 0).any().any():
        raise ValueError("prices must form a complete, finite, positive panel")

    returns_20 = clean.pct_change(SHORT_WINDOW, fill_method=None)
    returns_60 = clean.pct_change(LONG_WINDOW, fill_method=None)
    forward = clean.shift(-FORWARD_WINDOW).div(clean).sub(1)
    future_dates = pd.Series(clean.index, index=clean.index).shift(-FORWARD_WINDOW)
    effective_dates = pd.Series(clean.index, index=clean.index).shift(-1)

    frame = pd.DataFrame(index=clean.index)
    for ticker, prefix in (("SOXX", "soxx"), ("IGV", "igv"), ("XLK", "xlk")):
        frame[f"{prefix}_price"] = clean[ticker]
        frame[f"{prefix}_return_20d"] = returns_20[ticker]
        frame[f"{prefix}_return_60d"] = returns_60[ticker]
        frame[f"{prefix}_forward_20d"] = forward[ticker]

    for window, returns in ((20, returns_20), (60, returns_60)):
        frame[f"soxx_vs_xlk_{window}d"] = returns["SOXX"] - returns["XLK"]
        frame[f"igv_vs_xlk_{window}d"] = returns["IGV"] - returns["XLK"]
        frame[f"soxx_minus_igv_{window}d"] = returns["SOXX"] - returns["IGV"]
        frame[f"state_{window}d"] = classify_it_state(
            frame[f"soxx_vs_xlk_{window}d"], frame[f"igv_vs_xlk_{window}d"]
        )

    frame["effective_date"] = effective_dates
    frame["label_end"] = future_dates
    frame["pair_forward_actual_20d"] = forward["SOXX"] - forward["IGV"]
    frame["pair_actual_winner"] = _leader(frame["pair_forward_actual_20d"])
    frame["soxx_forward_vs_xlk_20d"] = forward["SOXX"] - forward["XLK"]
    frame["igv_forward_vs_xlk_20d"] = forward["IGV"] - forward["XLK"]
    frame["actual_available"] = frame[
        ["soxx_forward_20d", "igv_forward_20d", "xlk_forward_20d"]
    ].notna().all(axis=1)
    return frame


def build_it_daily_history(
    daily: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    if daily.empty:
        raise ValueError("daily IT internals frame is empty")
    cutoff = pd.Timestamp(as_of).normalize() if as_of is not None else daily.index.max().normalize()
    eligible = daily.loc[daily.index <= cutoff]
    history = eligible.dropna(subset=["state_20d", "state_60d"]).copy()
    if history.empty:
        raise ValueError("no daily rows have both 20D and 60D state")

    actual_columns = [
        "soxx_forward_20d",
        "igv_forward_20d",
        "xlk_forward_20d",
        "pair_forward_actual_20d",
        "pair_actual_winner",
        "soxx_forward_vs_xlk_20d",
        "igv_forward_vs_xlk_20d",
    ]
    known_actual = history["label_end"].notna() & history["label_end"].le(cutoff)
    history.loc[~known_actual, actual_columns] = pd.NA
    history["actual_available"] = known_actual

    prior = history["state_20d"].shift()
    history["state_confirmed"] = history["state_20d"].eq(history["state_60d"])
    history["prior_state_20d"] = prior
    transition = pd.Series("N/A", index=history.index, dtype="string")
    same = prior.notna() & prior.eq(history["state_20d"]).fillna(False)
    changed = prior.notna() & ~same
    transition.loc[same] = "—"
    transition.loc[changed] = prior.loc[changed] + " → " + history.loc[changed, "state_20d"]
    history["state_transition"] = transition
    history["current_pair_leader_20d"] = _leader(history["soxx_minus_igv_20d"])
    history.insert(0, "decision_date", history.index)
    history.insert(0, "series_version", SERIES_VERSION)
    history.insert(0, "schema_version", SCHEMA_VERSION)
    return history.loc[:, OUTPUT_COLUMNS].reset_index(drop=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_history(
    database: Path = DEFAULT_DATABASE,
    output: Path = DEFAULT_OUTPUT,
    metadata_path: Path = DEFAULT_METADATA,
    *,
    as_of: str | pd.Timestamp | None = None,
) -> dict[str, object]:
    prices = load_it_price_matrix(database)
    history = build_it_daily_history(build_it_daily_frame(prices), as_of=as_of)
    if as_of is not None and history["decision_date"].max() != pd.Timestamp(as_of).normalize():
        raise ValueError(f"SOXX/IGV/XLK common close is stale for {as_of}; history was not published")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(output, index=False, date_format="%Y-%m-%d")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "series_version": SERIES_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "production_sector_series_unchanged": True,
        "database": str(database),
        "database_sha256": _sha256(database),
        "tickers": {
            "SOXX": "SEMICONDUCTORS_AND_SEMICONDUCTOR_EQUIPMENT",
            "IGV": "NORTH_AMERICAN_SOFTWARE_PROXY",
            "XLK": "PARENT_IT_BENCHMARK",
        },
        "common_start": prices.index.min().date().isoformat(),
        "common_end": prices.index.max().date().isoformat(),
        "common_sessions": len(prices),
        "decision_days": len(history),
        "labelled_days": int(history["actual_available"].sum()),
        "decision_start": history["decision_date"].min().date().isoformat(),
        "decision_end": history["decision_date"].max().date().isoformat(),
        "state_definition": "SOXX and IGV 20D/60D excess returns versus XLK; four sign combinations",
        "target_definition": "SOXX t+20 return minus IGV t+20 return on common sessions",
        "frequency": "daily; every common SOXX/IGV/XLK trading session",
        "price_basis": "adjusted close only; no interpolation or forward fill",
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {**metadata, "output": str(output), "metadata": str(metadata_path)}


def web_payload(
    history_path: Path = DEFAULT_OUTPUT, metadata_path: Path = DEFAULT_METADATA,
    *, sector_database: Path | None = None,
) -> dict[str, object]:
    history = pd.read_csv(history_path, keep_default_na=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    required = {
        "decision_date", "series_version", "state_20d", "state_60d", "soxx_vs_xlk_20d",
        "igv_vs_xlk_20d", "soxx_vs_xlk_60d", "igv_vs_xlk_60d",
        "soxx_minus_igv_20d", "pair_forward_actual_20d", "actual_available",
    }
    if history.empty or not required.issubset(history.columns):
        raise ValueError("IT internals web history columns are incomplete")

    def boolean(value: object, field: str) -> bool:
        text = str(value).strip().casefold()
        if text not in {"true", "false"}:
            raise ValueError(f"{field} must be true or false")
        return text == "true"

    rows = []
    for source in history.to_dict("records"):
        if source["series_version"] != SERIES_VERSION or source["state_20d"] not in STATES or source["state_60d"] not in STATES:
            raise ValueError("unexpected IT internals series or state")
        actual_available = boolean(source["actual_available"], "actual_available")
        pair_actual = str(source["pair_forward_actual_20d"]).strip()
        rows.append([
            str(source["decision_date"]),
            STATE_CODES[str(source["state_20d"])],
            STATE_CODES[str(source["state_60d"])],
            round(float(source["soxx_vs_xlk_20d"]), 8),
            round(float(source["igv_vs_xlk_20d"]), 8),
            round(float(source["soxx_vs_xlk_60d"]), 8),
            round(float(source["igv_vs_xlk_60d"]), 8),
            round(float(source["soxx_minus_igv_20d"]), 8),
            round(float(pair_actual), 8) if actual_available and pair_actual else None,
        ])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "series_version": SERIES_VERSION,
        "generated_at": metadata["generated_at_utc"],
        "start": rows[0][0],
        "as_of": rows[-1][0],
        "cadence": "DAILY",
        "benchmark": "XLK",
        "research_only": True,
        "proxies": {"semiconductor": "SOXX", "software": "IGV"},
        "rows": rows,
    }
    if sector_database is not None:
        # Existing sector prices are refreshed before this stage; no extra download.
        sector = load_it_price_matrix(
            sector_database, tickers=("XLC", "XLE", "SPY"), as_of=payload["as_of"]
        )
        if sector.index[-1].date().isoformat() != payload["as_of"]:
            raise ValueError("XLC/XLE/SPY common close is stale; daily comparison was not published")
        hw = history.loc[:, ["soxx_price", "igv_price", "xlk_price"]].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(hw.to_numpy()).all() or (hw <= 0).any().any():
            raise ValueError("daily pair prices must be finite and positive")
        payload["daily_pairs"] = {
            "hw_sw": [[str(day), *[round(float(p), 6) for p in prices]]
                      for day, prices in zip(history["decision_date"], hw.to_numpy())],
            "comm_energy": [[day.date().isoformat(), *[round(float(p), 6) for p in prices]]
                            for day, prices in zip(sector.index, sector.to_numpy())],
        }
        sectors = load_it_price_matrix(
            sector_database, tickers=(*SECTOR_ETFS, "SPY"), as_of=payload["as_of"]
        ).tail(1_384)  # 1,260 spread observations plus return/statistic warm-up.
        if sectors.index[-1].date().isoformat() != payload["as_of"]:
            raise ValueError("11-sector common close is stale; daily comparison was not published")
        payload["sector_prices"] = {
            "tickers": list(SECTOR_ETFS),
            "rows": [[day.date().isoformat(), *[round(float(p), 6) for p in prices]]
                     for day, prices in zip(sectors.index, sectors.to_numpy())],
        }
    json.dumps(payload, allow_nan=False)
    return payload


def publish_history(
    history_path: Path = DEFAULT_OUTPUT,
    metadata_path: Path = DEFAULT_METADATA,
    web_output: Path = DEFAULT_WEB_OUTPUT,
    *,
    config_path: Path = DEFAULT_CONFIG,
    consent_path: Path = DEFAULT_CONSENT,
    sector_database: Path | None = None,
) -> dict[str, object]:
    consent = json.loads(consent_path.read_text(encoding="utf-8-sig")) if consent_path.is_file() else {}
    if consent.get("allow_portfolio_aggregate_upload") is not True:
        raise PermissionError("IT internals upload requires roro_publish_consent.json")
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    payload = web_payload(history_path, metadata_path, sector_database=sector_database)
    web_output.parent.mkdir(parents=True, exist_ok=True)
    web_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    headers = {
        "Authorization": f"Bearer {config['upload_token']}",
        "Content-Type": "application/json",
    }
    if config.get("sites_bypass_token"):
        headers["OAI-Sites-Authorization"] = f"Bearer {config['sites_bypass_token']}"
    request = urllib.request.Request(
        str(config["site_url"]).rstrip("/") + "/api/it-internals",
        data=json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"IT internals upload failed HTTP {error.code}: {detail}") from error
    return {"payload": str(web_output), "upload": result}


def write_web_payload(
    history_path: Path = DEFAULT_OUTPUT,
    metadata_path: Path = DEFAULT_METADATA,
    web_output: Path = DEFAULT_WEB_OUTPUT,
    *, sector_database: Path | None = None,
) -> dict[str, object]:
    payload = web_payload(history_path, metadata_path, sector_database=sector_database)
    web_output.parent.mkdir(parents=True, exist_ok=True)
    web_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {"payload": str(web_output), "observations": len(payload["rows"]), "as_of": payload["as_of"]}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Research-only IT internals shadow")
    commands = parser.add_subparsers(dest="command", required=True)
    refresh = commands.add_parser("refresh", help="register and download SOXX/IGV/XLK")
    refresh.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    refresh.add_argument("--years", type=int, default=20)
    refresh.add_argument("--backfill", action="store_true")
    build = commands.add_parser("build", help="write daily history and metadata")
    build.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    build.add_argument("--as-of")
    build.add_argument("--upload", action="store_true")
    update = commands.add_parser("update", help="refresh then build")
    update.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    update.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    update.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    update.add_argument("--years", type=int, default=20)
    update.add_argument("--backfill", action="store_true")
    update.add_argument("--as-of")
    update.add_argument("--upload", action="store_true")
    publish = commands.add_parser("publish", help="publish an existing daily history")
    publish.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    publish.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    publish.add_argument("--web-output", type=Path, default=DEFAULT_WEB_OUTPUT)
    for command in (build, update, publish):
        command.add_argument("--sector-database", type=Path, help="existing 11-sector/SPY price DB; enables daily pair checks")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "refresh":
        result = refresh_prices(args.database, years=args.years, backfill=args.backfill)
    elif args.command == "build":
        result = build_history(
            args.database, args.output, args.metadata, as_of=args.as_of
        )
        result = {"build": result, "web": write_web_payload(args.output, args.metadata, args.output.parent / "site_payload.json", sector_database=args.sector_database)}
        if args.upload:
            result["publish"] = publish_history(args.output, args.metadata, sector_database=args.sector_database)
    elif args.command == "update":
        refresh = refresh_prices(
            args.database, years=args.years, backfill=args.backfill,
            require_date=date.fromisoformat(args.as_of) if args.as_of else None,
        )
        build = build_history(
            args.database, args.output, args.metadata, as_of=args.as_of
        )
        result = {
            "refresh": refresh,
            "build": build,
            "web": write_web_payload(args.output, args.metadata, args.output.parent / "site_payload.json", sector_database=args.sector_database),
        }
        if args.upload:
            result["publish"] = publish_history(args.output, args.metadata, sector_database=args.sector_database)
    else:
        result = publish_history(args.output, args.metadata, args.web_output, sector_database=args.sector_database)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
