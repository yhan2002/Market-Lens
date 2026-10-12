"""Single refresh pipeline shared by web, scheduled, and manual updates."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterator, TextIO

from market_data import probe_latest_completed_market_date
from risk_signals import load_breadth_history
from trading_log import read_holdings_workbook


BASE_DIR = Path(__file__).resolve().parent
PYTHON = BASE_DIR / ".venv" / "Scripts" / "python.exe"
PUBLISH_DATABASE = BASE_DIR / "regime_router_publish.db"
PRICE_OBSERVATIONS = BASE_DIR / "outputs" / "roro_shadow" / "roro_price_observations.db"
BREADTH_OBSERVATIONS = BASE_DIR / "outputs" / "roro_shadow" / "ndth_breadth_observations.db"
RORO_DATABASE = BASE_DIR / "outputs" / "roro_shadow" / "roro_shadow.db"
SECTOR_DATABASE = BASE_DIR / "outputs" / "sector_shadow" / "sector_shadow.db"
PRIVATE_DIR = Path(os.environ.get("REGIME_ROUTER_PRIVATE_DIR", BASE_DIR / "private"))
HOLDINGS_CACHE = PRIVATE_DIR / "holdings.xlsx"
HOLDINGS_SOURCE = Path(os.environ.get("REGIME_ROUTER_HOLDINGS_SOURCE", PRIVATE_DIR / "holdings-source.xlsx"))
HOLDINGS_DATABASE = BASE_DIR / "regime_router_shadow.db"
LOG_PATH = Path(os.environ.get("LOCALAPPDATA", BASE_DIR)) / "RegimeRouter" / "logs" / "site_update.log"
LOCK_PATH = BASE_DIR / "outputs" / ".refresh.lock"


class RefreshBusy(RuntimeError):
    pass


class RefreshFailed(RuntimeError):
    pass


@dataclass(frozen=True)
class RefreshResult:
    requested_as_of: str
    warnings: tuple[str, ...]
    holdings_as_of: str | None = None


@contextmanager
def refresh_lock(path: Path = LOCK_PATH) -> Iterator[None]:
    """Use one OS lock across web and Task Scheduler processes."""
    import msvcrt

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0)
        if handle.read(1) == b"":
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RefreshBusy("다른 갱신 작업이 이미 실행 중입니다.") from error
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def is_refresh_busy(path: Path = LOCK_PATH) -> bool:
    try:
        with refresh_lock(path):
            return False
    except RefreshBusy:
        return True


def _run(name: str, command: list[str], log: TextIO) -> int:
    log.write(f"\n[단계] {name}\n")
    log.flush()
    process = subprocess.Popen(
        command, cwd=BASE_DIR, stdout=log, stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUTF8": "1"},
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        start_new_session=os.name != "nt",
    )
    try:
        code = process.wait(timeout=900)
    except subprocess.TimeoutExpired as error:
        # Target only this stage's process tree; never terminate unrelated Excel sessions.
        try:
            if os.name == "nt":
                cleanup = subprocess.run(
                    [str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "taskkill.exe"),
                     "/PID", str(process.pid), "/T", "/F"],
                    stdout=log, stderr=subprocess.STDOUT, timeout=15, check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if cleanup.returncode:
                    log.write(f"[{name}] 하위 프로세스 종료 확인 실패: {cleanup.returncode}\n")
            else:
                import signal
                os.killpg(process.pid, signal.SIGKILL)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
        log.write(f"[{name}] 900초 제한 초과; 단계 중단\n")
        log.flush()
        raise RefreshFailed(f"{name}: 실행 시간 제한 초과") from error
    log.write(f"[{name}] exit {code}\n")
    log.flush()
    return code


def _export_holdings(log: TextIO) -> str | None:
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    cscript = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "cscript.exe"
    script = BASE_DIR / "EXPORT_CLOUDIUM_HOLDINGS.vbs"
    source, source_label = HOLDINGS_SOURCE, "MP 원본"

    for attempt in range(1, 4):
        try:
            source_stat = source.stat()
            if not stat.S_ISREG(source_stat.st_mode):
                raise IsADirectoryError("holdings input must be a file")
            break
        except OSError as error:
            log.write(
                f"[보유내역] {source_label} 접근 실패 {attempt}/3: "
                f"{type(error).__name__} / 오류 {error.errno} / Windows {getattr(error, 'winerror', None)}\n"
            )
            log.flush()
            if attempt == 3:
                return f"보유내역 {source_label} 원본 접근 실패(3회): 이전 확인 포지션을 사용했습니다."
            time.sleep(2)

    for executable, label in ((cscript, "Windows Script Host"), (script, "Excel 내보내기 스크립트")):
        if not executable.is_file():
            log.write(f"[보유내역] {label} 없음\n")
            return f"보유내역 {label} 없음: 이전 확인 포지션을 사용했습니다."
    modified = datetime.fromtimestamp(source_stat.st_mtime).date()
    log.write(f"[보유내역] 입력={source_label} / 파일 저장일={modified.isoformat()}\n")
    try:
        # Validate a separate export before replacing the last working private cache.
        with tempfile.TemporaryDirectory(prefix="holdings-", dir=PRIVATE_DIR) as directory:
            staged = Path(directory) / "holdings.xlsx"
            code = _run(
                "보유내역 확인",
                [str(cscript), "//B", "//NoLogo", str(script), str(source), str(staged)],
                log,
            )
            if code:
                log.write(f"[보유내역] Excel 내보내기 실패: exit {code}\n")
                return "보유내역 Excel 내보내기 실패: 이전 확인 포지션을 사용했습니다."
            holdings, _ = read_holdings_workbook(staged)
            if not holdings:
                raise ValueError("no equity holdings")
            latest = max(str(row["date"]) for row in holdings)
            if HOLDINGS_DATABASE.is_file():
                with closing(sqlite3.connect(HOLDINGS_DATABASE.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
                    exists = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='daily_holdings'").fetchone()
                    previous = connection.execute("SELECT MAX(date) FROM daily_holdings").fetchone()[0] if exists else None
                if previous and latest < str(previous):
                    log.write(f"[보유내역] 과거 파일 거부: 입력 {latest} < DB {previous}\n")
                    return "보유내역 입력 기준일이 저장된 내역보다 과거라 반영하지 않았습니다."
            staged.replace(HOLDINGS_CACHE)
            log.write(f"[보유내역] 원본 확인 완료 / 파일 내 최신 기준일={latest}\n")
    except Exception as error:
        # Keep parser/file errors private: they can contain position names or paths.
        log.write(f"[보유내역] 내보내기·검증·캐시 교체 실패: {type(error).__name__}\n")
        return "보유내역 파일 검증·캐시 교체 실패: 이전 확인 포지션을 사용했습니다."
    if (date.today() - modified).days > 10:
        return f"보유내역 원본 저장일 {modified.isoformat()}(10일 초과): 주간 파일을 확인하세요. 포지션 불변일 수도 있습니다."
    return None


def _holdings_status(as_of: str) -> tuple[str | None, str | None]:
    """Report the actual private DB position date, not the refreshed price date."""
    try:
        with closing(sqlite3.connect(HOLDINGS_DATABASE.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            latest = connection.execute("SELECT MAX(date) FROM daily_holdings WHERE date<=?", (as_of,)).fetchone()[0]
            row = connection.execute("SELECT value FROM metadata WHERE key='event_holdings_workbook_fingerprint'").fetchone()
        cache_stat = HOLDINGS_CACHE.stat()
        if not row or row[0] != f"{cache_stat.st_size}:{cache_stat.st_mtime_ns}":
            return latest, "보유내역 입력 파일의 DB 반영을 확인하지 못했습니다. 이전 포지션일 수 있습니다."
        return latest, None if latest else "해당 기준일에 사용할 보유내역이 없습니다."
    except (OSError, sqlite3.Error):
        return None, "보유내역 기준일·DB 반영 상태 확인 실패: 갱신 로그를 확인하세요."


def run_refresh_pipeline(as_of: date | None = None, *, log_path: Path = LOG_PATH) -> RefreshResult:
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    if not (BASE_DIR / "site_upload_config.json").is_file():
        raise FileNotFoundError(BASE_DIR / "site_upload_config.json")
    target = as_of or probe_latest_completed_market_date(
        tickers=("NQ=F",), fallback_database=PUBLISH_DATABASE
    )
    warnings: list[str] = []
    failures: list[str] = []
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with refresh_lock(), log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{date.today().isoformat()}] 통합 갱신 시작 · NQ {target.isoformat()}\n")
        warning = _export_holdings(log)
        if warning:
            warnings.append(warning)

        base_command = [
            str(PYTHON), "site_publisher.py", "--date", target.isoformat(),
            "--holdings-workbook", str(HOLDINGS_CACHE),
        ]
        if _run("기본 대시보드", base_command, log) != 0:
            failures.append("기본 대시보드")
        holdings_as_of, holdings_warning = _holdings_status(target.isoformat())
        log.write(f"[보유내역] DB 기준일={holdings_as_of or '없음'} / 시장 기준일={target.isoformat()}\n")
        if holdings_warning:
            warnings.append(holdings_warning)

        if _run("섹터 순환 진입 경보", [str(PYTHON), "reports/rotation_entry_publish.py", "--record", "--upload"], log) != 0:
            warnings.append("섹터 순환 진입 경보 갱신 실패: 화면의 기준일을 확인하세요.")

        optional_stages = (
            (
                "JPY·GLD",
                [str(PYTHON), "roro_price_observations.py", "refresh", str(PRICE_OBSERVATIONS), "--publish-database", str(PUBLISH_DATABASE)],
            ),
            (
                "시장 참여 폭",
                [str(PYTHON), "roro_market_breadth.py", "refresh", str(BREADTH_OBSERVATIONS),
                 "--required-as-of", target.isoformat(), "--publish-database", str(PUBLISH_DATABASE)],
            ),
        )
        for name, command in optional_stages:
            if _run(name, command, log) != 0:
                warnings.append(
                    "시장 참여 폭 수집 지연 또는 오류: 기준일 입력을 재확인하며, 미완료이면 Market Lens 점수를 게시하지 않습니다."
                    if name == "시장 참여 폭" else f"{name} 입력 갱신 실패: 기준일의 실제 가용 여부를 확인합니다."
                )

        roro_command = [
            str(PYTHON), "roro_site_publisher.py",
            "--price-observation-database", str(PRICE_OBSERVATIONS),
            "--breadth-observation-database", str(BREADTH_OBSERVATIONS),
            "--latest-only", "--require-as-of", "--repair-missing-breadth", "--as-of", target.isoformat(),
        ]
        if _run("Market Lens", roro_command, log) != 0:
            failures.append("Market Lens")

        if not failures:
            sector_build = [
                str(PYTHON), "sector_shadow_store.py", "build",
                str(PUBLISH_DATABASE), str(RORO_DATABASE), str(SECTOR_DATABASE),
                "--as-of", target.isoformat(), "--latest-only",
            ]
            if _run("섹터 계산", sector_build, log) == 0:
                sector_publish = [
                    str(PYTHON), "sector_site_publisher.py",
                    "--shadow-database", str(SECTOR_DATABASE),
                    "--as-of", target.isoformat(), "--latest-only",
                ]
                if _run("섹터 게시", sector_publish, log) != 0:
                    warnings.append("섹터 화면 게시가 보류됐습니다.")
            else:
                warnings.append("섹터 입력이 준비되지 않아 직전 주간 판단을 유지했습니다.")

        it_internals = [
            str(PYTHON), "it_internals_shadow.py", "update",
            "--as-of", target.isoformat(), "--upload", "--sector-database", str(PUBLISH_DATABASE),
        ]
        if _run("IT 내부 순환", it_internals, log) != 0:
            warnings.append("HW/SW·Communication/Energy 일간 비교 갱신이 보류됐습니다.")

        if _run("섹터 순환 새 엔진", [str(PYTHON), "sector_hybrid_publisher.py", "--as-of", target.isoformat(), "--upload"], log) != 0:
            warnings.append("섹터 순환 모델 갱신 실패: 주간 판단일과 데이터 상태를 확인하세요.")

        if failures:
            message = "실패 단계: " + ", ".join(failures)
            log.write(f"[실패] {message}\n")
            raise RefreshFailed(message)
        breadth = load_breadth_history(PUBLISH_DATABASE, target.isoformat(), 1)
        if not breadth or breadth[0]["date"] != target.isoformat():
            latest = breadth[0]["date"] if breadth else "없음"
            warnings.append(f"Fast 8 Breadth 미완료: 요청 {target.isoformat()}, 유효 {latest}")
        for warning in warnings:
            log.write(f"[주의] {warning}\n")
        log.write("[부분 완료] 미완료 항목 확인 필요\n" if warnings else "[완료] 통합 갱신 완료\n")
    return RefreshResult(target.isoformat(), tuple(warnings), holdings_as_of)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of")
    args = parser.parse_args(argv)
    try:
        target = date.fromisoformat(args.as_of) if args.as_of else None
        result = run_refresh_pipeline(target)
        print(json.dumps(asdict(result), ensure_ascii=False))
        print("[부분 완료] 경고 항목을 확인하세요." if result.warnings else "[완료] 통합 갱신 완료")
        return 0
    except RefreshBusy as error:
        print(str(error), file=sys.stderr)
        return 75
    except Exception as error:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as log:
            log.write("\n[통합 갱신 예외]\n")
            traceback.print_exc(file=log)
        print(f"갱신 실패: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
