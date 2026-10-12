from __future__ import annotations

import sqlite3
import threading
from contextlib import closing
from datetime import date
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable


APP_DIR = Path(__file__).resolve().parent
DATABASE = APP_DIR / "regime_router.db"
QUANTBOT_CONFIG = APP_DIR.parent / "QuantBot" / "QuantBotSystem" / "config.py"
HMM_MODEL_VERSION = (
    "hmm3-sticky-nq-independent-a2d-logvol-v1-diag-mc25-"
    "k25-m252-w1260-r20-s42-n3"
)
HAZARD_ALERT_CUTOFF = 0.72

STATE_LABELS = {
    "BULL": "상승장",
    "SIDEWAYS": "횡보장",
    "BEAR": "하락장",
}

BAND_COPY = {
    "ALERT_SHADOW": (
        "조기경보: 위험 축소 검토",
        "다음 거래일에 위험 축소 여부를 검토하세요. 자동 매도 신호는 아닙니다.",
        "alert",
    ),
    "NORMAL_SHADOW": (
        "조기경보 없음",
        "현재 조기경보 조건에 해당하지 않습니다. HMM 국면과 함께 참고하세요.",
        "normal",
    ),
    "DORMANT": (
        "Onset 적용 밖",
        "NQ가 252일 고점 대비 -3% 밖에 있어 Onset 적용 구간이 아닙니다. Crash Guard는 계속 감시합니다.",
        "caution",
    ),
    "WARMUP": (
        "조기경보 준비 중",
        "점수를 만들 과거 데이터가 아직 부족합니다. HMM 국면만 참고하세요.",
        "neutral",
    ),
}

CRASH_GUARD_COPY = {
    "CRASH_WATCH_SHADOW": (
        "급락 주의",
        "1일 급변 또는 5일 취약성이 높은 구간입니다. 다음 거래일 위험 축소나 헤지 여부를 검토하세요. 자동 주문 신호는 아닙니다.",
        "alert",
    ),
    "VULNERABLE_SHADOW": (
        "급락 취약성 증가",
        "급락 취약성이 평소보다 높습니다. 익스포저와 대응 계획을 점검하세요. 자동 주문 신호는 아닙니다.",
        "caution",
    ),
    "NORMAL_SHADOW": (
        "상시 감시 정상",
        "현재 급락 취약성 순위가 경계 구간보다 낮습니다. 갑작스러운 충격이 없다는 보장은 아닙니다.",
        "normal",
    ),
    "WARMUP": (
        "상시 감시 준비 중",
        "Crash Guard 계산에 필요한 과거 데이터가 아직 부족합니다.",
        "neutral",
    ),
}


def normalize_date(value: str) -> str:
    try:
        return date.fromisoformat(value.strip()).isoformat()
    except ValueError as error:
        raise ValueError("날짜를 YYYY-MM-DD 형식으로 입력하세요.") from error


def latest_nq_date(database: Path, as_of_date: str | None = None) -> str | None:
    query = "SELECT MAX(date) FROM daily_prices WHERE ticker='NQ=F'"
    params: tuple[str, ...] = ()
    if as_of_date is not None:
        query += " AND date<=?"
        params = (normalize_date(as_of_date),)
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        row = connection.execute(query, params).fetchone()
    return str(row[0]) if row and row[0] else None


def _current_model_date(
    database: Path,
    *,
    hmm_model_version: str,
    drawdown_model_version: str,
) -> str | None:
    try:
        with closing(sqlite3.connect(database)) as connection:
            row = connection.execute(
                """
                SELECT
                    (SELECT MAX(date) FROM daily_prices WHERE ticker='NQ=F'),
                    (SELECT MAX(date) FROM regime_model_probabilities WHERE model_version=?),
                    (SELECT MAX(date) FROM drawdown_hazard_scores WHERE model_version=?)
                """,
                (hmm_model_version, drawdown_model_version),
            ).fetchone()
    except sqlite3.OperationalError:
        return None
    return str(row[0]) if row and row[0] and len(set(row)) == 1 else None


def _percent(value: object) -> str:
    return f"{float(value):.1%}"


def _score(value: object) -> str:
    return "계산 전" if value is None else f"{float(value):.3f}"


def build_status_view(
    regime: dict[str, object],
    hazard: dict[str, object] | None,
    *,
    requested_date: str,
    price_date: str | None,
    crash_guard: dict[str, object] | None = None,
) -> tuple[str, str, str]:
    state = str(regime["state"])
    state_label = STATE_LABELS.get(state, state)
    regime_date = str(regime["date"])

    if hazard is None:
        onset_title = "조기경보 미계산"
        onset_interpretation = "Drawdown Onset 조기경보가 아직 계산되지 않았습니다. 전체 최신화를 실행하세요."
        onset_tone = "neutral"
        hazard_date = None
        hazard_lines = ["조기경보: 계산된 결과 없음"]
    else:
        band = str(hazard["risk_band"])
        band_title, onset_interpretation, onset_tone = BAND_COPY.get(
            band,
            (band, "알 수 없는 경보 상태입니다.", "neutral"),
        )
        onset_title = band_title
        hazard_date = str(hazard["date"])
        band_value = band_title.removeprefix("조기경보: ").removeprefix("조기경보 ")
        hazard_lines = [
            f"Drawdown Onset: {band_value}",
            f"Hazard 점수: {_score(hazard.get('score'))}  (경보 기준 {HAZARD_ALERT_CUTOFF:.2f})",
            f"경보 계산일: {hazard_date}",
        ]

    if crash_guard is None:
        guard_band = None
        guard_title = "계산 결과 없음"
        guard_interpretation = "가격 데이터가 있으면 상시 감시 점수를 계산합니다. 관리자에게 파일 구성을 확인해 달라고 요청하세요."
        guard_tone = "neutral"
        guard_date = None
        guard_lines = ["Crash Guard(상시 Shadow): 계산 결과 없음"]
    else:
        guard_band = str(crash_guard.get("risk_band", "UNKNOWN"))
        guard_title, guard_interpretation, guard_tone = CRASH_GUARD_COPY.get(
            guard_band,
            (guard_band, "알 수 없는 Crash Guard 상태입니다.", "neutral"),
        )
        guard_date_value = crash_guard.get("date")
        guard_date = str(guard_date_value) if guard_date_value else None
        guard_lines = [
            f"Crash Guard(상시 Shadow): {guard_title}",
            (
                f"1일 급변 {_score(crash_guard.get('score_1d'))}  ·  "
                f"5일 취약성 {_score(crash_guard.get('score_5d'))}  ·  "
                f"종합 {_score(crash_guard.get('score'))}"
            ),
            f"Guard 계산일: {guard_date or '계산 전'}",
        ]

    if guard_band == "CRASH_WATCH_SHADOW" and hazard is not None and str(hazard["risk_band"]) == "ALERT_SHADOW":
        headline = f"{state_label} · 복합 경보: 위험 축소 검토"
        tone = "alert"
    elif guard_band == "CRASH_WATCH_SHADOW":
        headline = f"{state_label} · 급락 주의"
        tone = "alert"
    elif hazard is not None and str(hazard["risk_band"]) == "ALERT_SHADOW":
        headline = f"{state_label} · {onset_title}"
        tone = "alert"
    elif guard_band == "VULNERABLE_SHADOW":
        headline = f"{state_label} · 급락 취약성 증가"
        tone = "caution"
    else:
        headline = f"{state_label} · {onset_title}"
        tone = onset_tone

    details = [
        f"요청일: {requested_date}",
        f"HMM 기준 거래일: {regime_date}",
        "",
        f"시장 국면: {state_label}",
        (
            f"상승 {_percent(regime['p_bull'])}  ·  "
            f"횡보 {_percent(regime['p_sideways'])}  ·  "
            f"하락 {_percent(regime['p_bear'])}"
        ),
        "",
        *guard_lines,
        f"Guard 해석: {guard_interpretation}",
        "",
        *hazard_lines,
        f"Onset 해석: {onset_interpretation}",
        "",
        "Shadow 점수는 하락 확률이 아니라 과거 데이터 내 위험 순위입니다. 즉, 급락 확률이 아니라 상대적 취약성입니다.",
    ]

    if price_date:
        details.extend(("", f"저장된 NQ 가격 최신일: {price_date}"))
        stale = []
        if price_date <= requested_date:
            if regime_date < price_date:
                stale.append("HMM")
            if hazard_date and hazard_date < price_date:
                stale.append("조기경보")
            if guard_date and guard_date < price_date:
                stale.append("Crash Guard")
        if stale:
            details.append(
                f"주의: {', '.join(stale)} 결과가 가격 데이터보다 오래됐습니다. 전체 최신화를 실행하세요."
            )

    return headline, "\n".join(details), tone


def load_status(database: Path, target_date: str) -> tuple[str, str, str]:
    target_date = normalize_date(target_date)
    if not database.exists():
        raise FileNotFoundError(f"데이터베이스를 찾을 수 없습니다: {database}")

    from regime_router import get_regime_status
    from crash_guard import get_crash_guard_status
    from research_drawdown_onset import get_shadow_hazard_status

    regime = get_regime_status(
        database,
        target_date,
        model_version=HMM_MODEL_VERSION,
    )
    try:
        hazard = get_shadow_hazard_status(database, target_date)
    except (ValueError, sqlite3.OperationalError):
        hazard = None
    try:
        crash_guard = get_crash_guard_status(database, target_date)
    except (ValueError, sqlite3.OperationalError):
        crash_guard = None
    return build_status_view(
        regime,
        hazard,
        requested_date=target_date,
        price_date=latest_nq_date(database, target_date),
        crash_guard=crash_guard,
    )


def update_everything(
    database: Path,
    progress: Callable[[str], None],
    config_path: Path = QUANTBOT_CONFIG,
    *,
    include_equity_universe: bool = True,
    force_models: bool = False,
    required_price_date: date | None = None,
) -> str:
    if not database.exists():
        raise FileNotFoundError(f"데이터베이스를 찾을 수 없습니다: {database}")

    from market_data import (
        CROSS_ASSET_TICKERS,
        EQUITY_UNIVERSE_GROUP,
        download_prices,
        initialize_database,
        register_universe,
    )
    from regime_router import fit_candidate_from_database
    from research_drawdown_onset import SHADOW_MODEL_VERSION, write_shadow_scores

    if include_equity_universe:
        progress("1/6  CSQ_MP·CSQ_CP 합집합 동기화 중…")
        universe_count, _ = initialize_database(database, config_path)
        progress("2/6  확장 주식 유니버스 가격 갱신 중…")
        equity_count, equity_rows, _ = download_prices(
            database,
            years=10,
            group_name=EQUITY_UNIVERSE_GROUP,
        )
    else:
        progress("1/6  공개 시장 벤치마크 등록 중…")
        register_universe(
            database,
            ("NQ=F",),
            group_name="REGIME_BENCHMARK",
            source="built-in",
        )
        progress("2/6  내부 종목 유니버스 갱신 생략")
        universe_count = equity_count = equity_rows = 0

    progress("3/6  시장 위험 데이터 갱신 중…")
    register_universe(database, CROSS_ASSET_TICKERS)
    cross_count, cross_rows, _ = download_prices(
        database,
        years=20,
        group_name="REGIME_CROSS_ASSET",
        require_date=required_price_date,
    )

    progress("4/6  NQ 데이터 갱신 중…")
    benchmark_count, benchmark_rows, _ = download_prices(
        database,
        years=20,
        tickers=("NQ=F",),
        require_date=required_price_date,
    )

    current_model_date = None if force_models else _current_model_date(
        database,
        hmm_model_version=HMM_MODEL_VERSION,
        drawdown_model_version=SHADOW_MODEL_VERSION,
    )
    if current_model_date:
        progress(f"5/6  HMM 국면 최신({current_model_date}) · 재계산 생략")
        progress(f"6/6  Drawdown 조기경보 최신({current_model_date}) · 재계산 생략")
        model_summary = f"HMM·조기경보 {current_model_date} 최신, 재계산 생략"
    else:
        progress("5/6  HMM 국면 전체 재계산 중… (가장 오래 걸립니다)")
        probabilities = fit_candidate_from_database(
            database,
            model_version=HMM_MODEL_VERSION,
            benchmark="NQ=F",
            feature_set="independent-a2c",
            covariance_type="diag",
            covariance_floor=0.25,
            sticky_strength=25.0,
            min_train=252,
            train_window=1260,
            refit_every=20,
            random_state=42,
            n_init=3,
        )

        progress("6/6  Drawdown 조기경보 재계산 중…")
        hazard_rows = write_shadow_scores(database)
        model_summary = f"HMM {len(probabilities):,}일, 조기경보 {hazard_rows:,}일 계산 완료"

    return (
        f"MP+CP 합집합 {universe_count}종목, "
        f"가격 {equity_count + cross_count + benchmark_count}종목 / "
        f"{equity_rows + cross_rows + benchmark_rows:,}행 수신, "
        f"{model_summary}"
    )


def friendly_error(error: Exception) -> str:
    if isinstance(error, FileNotFoundError):
        missing = getattr(error, "filename", None)
        if missing and Path(missing).name.lower() == "config.py":
            return f"QuantBot config.py를 찾을 수 없습니다.\n\n확인 경로: {missing}"
        return "regime_router.db가 없습니다. 관리자에게 초기 설치를 요청하세요."
    if isinstance(error, ModuleNotFoundError):
        return "필수 패키지가 없습니다. 관리자에게 requirements.txt 설치를 요청하세요."
    if isinstance(error, sqlite3.DatabaseError):
        return f"데이터베이스를 읽지 못했습니다.\n\n상세: {error}"
    return str(error) or error.__class__.__name__


class RegimeRouterApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Regime Router")
        self.root.geometry("800x720")
        self.root.minsize(700, 640)

        self.date_var = tk.StringVar(value=date.today().isoformat())
        self.headline_var = tk.StringVar(value="상태를 불러오는 중…")
        self.detail_var = tk.StringVar(value="")
        self.progress_var = tk.StringVar(value="준비됨")

        self._configure_styles()
        self._build()
        self.root.after(100, self.show_status)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.configure("Title.TLabel", font=("Malgun Gothic", 22, "bold"))
        style.configure("Subtitle.TLabel", font=("Malgun Gothic", 10), foreground="#4b5563")
        style.configure("Headline.TLabel", font=("Malgun Gothic", 18, "bold"))
        style.configure("Normal.Headline.TLabel", font=("Malgun Gothic", 18, "bold"), foreground="#166534")
        style.configure("Alert.Headline.TLabel", font=("Malgun Gothic", 18, "bold"), foreground="#b91c1c")
        style.configure("Caution.Headline.TLabel", font=("Malgun Gothic", 18, "bold"), foreground="#92400e")
        style.configure("Neutral.Headline.TLabel", font=("Malgun Gothic", 18, "bold"), foreground="#374151")
        style.configure("Detail.TLabel", font=("Malgun Gothic", 10), foreground="#1f2937")
        style.configure("Primary.TButton", font=("Malgun Gothic", 11, "bold"), padding=(16, 10))
        style.configure("Secondary.TButton", font=("Malgun Gothic", 10), padding=(12, 9))
        style.configure("Status.TLabel", font=("Malgun Gothic", 9), foreground="#4b5563")

    def _build(self) -> None:
        main = ttk.Frame(self.root, padding=24)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(4, weight=1)

        ttk.Label(main, text="Regime Router", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            main,
            text="시장 국면, 상시 Crash Guard, Drawdown Onset을 한 화면에서 확인합니다. 주문은 실행하지 않습니다.",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 18))

        controls = ttk.Frame(main)
        controls.grid(row=2, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="확인 날짜").grid(row=0, column=0, padx=(0, 8))
        self.date_entry = ttk.Entry(controls, textvariable=self.date_var, width=16, font=("Consolas", 11))
        self.date_entry.grid(row=0, column=1, sticky="w")
        self.date_entry.bind("<Return>", lambda _event: self.show_status())
        ttk.Button(controls, text="오늘", command=self.set_today).grid(row=0, column=2, padx=(8, 18))
        self.status_button = ttk.Button(
            controls,
            text="상태 확인",
            style="Primary.TButton",
            command=self.show_status,
        )
        self.status_button.grid(row=0, column=3, padx=(0, 8))
        self.update_button = ttk.Button(
            controls,
            text="전체 최신화",
            style="Secondary.TButton",
            command=self.confirm_update,
        )
        self.update_button.grid(row=0, column=4)

        ttk.Label(
            main,
            text="휴장일은 직전 거래일을 표시합니다. 전체 최신화는 인터넷 연결이 필요하며 수 분 이상 걸릴 수 있습니다.",
            style="Subtitle.TLabel",
        ).grid(row=3, column=0, sticky="w", pady=(9, 16))

        card = ttk.LabelFrame(main, text="판단 보조 결과", padding=20)
        card.grid(row=4, column=0, sticky="nsew")
        card.columnconfigure(0, weight=1)
        self.headline_label = ttk.Label(
            card,
            textvariable=self.headline_var,
            style="Headline.TLabel",
            wraplength=690,
        )
        self.headline_label.grid(row=0, column=0, sticky="w")
        ttk.Separator(card).grid(row=1, column=0, sticky="ew", pady=14)
        ttk.Label(
            card,
            textvariable=self.detail_var,
            style="Detail.TLabel",
            justify="left",
            anchor="nw",
            wraplength=690,
        ).grid(row=2, column=0, sticky="nw")

        footer = ttk.Frame(main)
        footer.grid(row=5, column=0, sticky="ew", pady=(14, 0))
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.progress_var, style="Status.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.progress = ttk.Progressbar(footer, mode="indeterminate", length=150)
        self.progress.grid(row=0, column=1, sticky="e")

    def set_today(self) -> None:
        self.date_var.set(date.today().isoformat())
        self.show_status()

    def show_status(self) -> None:
        try:
            headline, detail, tone = load_status(DATABASE, self.date_var.get())
        except Exception as error:
            self.headline_var.set("상태를 확인할 수 없습니다")
            self.detail_var.set(friendly_error(error))
            self.progress_var.set("확인 실패")
            return

        self.headline_var.set(headline)
        self.detail_var.set(detail)
        headline_styles = {
            "normal": "Normal.Headline.TLabel",
            "alert": "Alert.Headline.TLabel",
            "caution": "Caution.Headline.TLabel",
            "neutral": "Neutral.Headline.TLabel",
        }
        self.headline_label.configure(style=headline_styles.get(tone, "Headline.TLabel"))
        self.progress_var.set("상태 확인 완료")

    def confirm_update(self) -> None:
        approved = messagebox.askyesno(
            "전체 최신화",
            "다음 작업을 시작합니다.\n\n"
            "• 대상: regime_router.db\n"
            "• 유니버스: QuantBot CSQ_MP + CSQ_CP 합집합 동기화\n"
            "• 가격: 합집합 85종목 + 시장 위험 데이터 + NQ 증분 갱신\n"
            "• 계산: NQ 전체 이력 HMM + Drawdown Onset 조기경보\n"
            "• 상시 감시: Crash Guard는 갱신된 가격으로 상태 확인 시 즉시 계산\n"
            "• 예상 시간: 최초 CP 확장은 10~30분 이상, 이후에는 증분 갱신\n"
            "• 비용/외부 영향: Yahoo 데이터 조회와 로컬 CPU만 사용, 주문 없음\n\n"
            "진행할까요?",
            parent=self.root,
        )
        if not approved:
            return
        self._run_update()

    def _run_update(self) -> None:
        self._set_busy(True)
        self.progress_var.set("전체 최신화 시작…")

        def report(message: str) -> None:
            self.root.after(0, self.progress_var.set, message)

        def worker() -> None:
            try:
                summary = update_everything(DATABASE, report)
            except Exception as error:
                self.root.after(0, self._update_failed, error)
            else:
                self.root.after(0, self._update_finished, summary)

        threading.Thread(target=worker, daemon=True).start()

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        self.status_button.configure(state=state)
        self.update_button.configure(state=state)
        self.date_entry.configure(state=state)
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()

    def _update_failed(self, error: Exception) -> None:
        self._set_busy(False)
        failed_stage = self.progress_var.get()
        message = f"{failed_stage}\n\n{friendly_error(error)}"
        self.progress_var.set("전체 최신화 실패")
        messagebox.showerror("최신화 실패", message, parent=self.root)

    def _update_finished(self, summary: str) -> None:
        self._set_busy(False)
        self.date_var.set(date.today().isoformat())
        self.show_status()
        self.progress_var.set(summary)
        messagebox.showinfo("최신화 완료", summary, parent=self.root)


def main() -> None:
    root = tk.Tk()
    RegimeRouterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
