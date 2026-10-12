"""Run the Python unit suite without network access or operational subprocesses."""
from __future__ import annotations
import os
from pathlib import Path
import sys
import tempfile
import unittest


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.dont_write_bytecode = True

    def offline_guard(event: str, args: tuple[object, ...]) -> None:
        if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo", "socket.bind", "subprocess.Popen", "os.system", "os.startfile"}:
            raise RuntimeError(f"Offline tests blocked {event}")

    with tempfile.TemporaryDirectory(prefix="market-lens-test-") as directory:
        os.environ.update({"LOCALAPPDATA": directory, "APPDATA": directory,
                           "REGIME_ROUTER_PRIVATE_DIR": str(Path(directory) / "private"),
                           "REGIME_ROUTER_HOLDINGS_SOURCE": str(Path(directory) / "missing.xlsx"),
                           "PYTHONDONTWRITEBYTECODE": "1"})
        for key in ("FRED_API_KEY", "OPENROUTER_API_KEY", "UPLOAD_TOKEN", "VIEW_TOKEN", "NEWS_AUTH", "NEWS_SESSION_KEY", "NEWS_UPLOAD_TOKEN"):
            os.environ.pop(key, None)
        sys.addaudithook(offline_guard)
        names = sorted(p.stem for p in root.glob("test_*.py") if p.stem != "test_run_roro_shadow_local")
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in names)
        result = unittest.TextTestRunner(verbosity=1).run(suite)
        print("Excluded test_run_roro_shadow_local: launches PowerShell; inspect/run separately in an isolated environment.")
        return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
