import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "RUN_RORO_SHADOW_LOCAL.ps1"


REQUIRED_SITE_FILES = (
    ".openai/hosting.json",
    "build/sites-vite-plugin.ts",
    "cloudflare-env.d.ts",
    "drizzle.config.ts",
    "eslint.config.mjs",
    "next-env.d.ts",
    "next.config.ts",
    "package-lock.json",
    "package.json",
    "postcss.config.mjs",
    "tests/rendered-html.test.mjs",
    "tsconfig.json",
    "vite.config.ts",
    "worker/index.ts",
)


def create_execute_fixture(root: Path) -> Path:
    script = root / SCRIPT.name
    shutil.copy2(SCRIPT, script)
    (root / "regime_router_publish.db").write_bytes(b"publish")
    (root / "regime_router_shadow.db").write_bytes(b"private")
    (root / "requirements.txt").write_text("fixture==1\n", encoding="utf-8")

    site = root / "site"
    (site / "node_modules").mkdir(parents=True)
    for relative_path in REQUIRED_SITE_FILES:
        target = site / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"fixture: {relative_path}\n", encoding="utf-8")
    for relative_path in (
        "app/page.tsx",
        "lib/roro.ts",
        "db/schema.ts",
        "drizzle/0000_fixture.sql",
        "drizzle/meta/_journal.json",
        "public/fixture.svg",
    ):
        target = site / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"fixture: {relative_path}\n", encoding="utf-8")

    fake_python = root / ".venv" / "Scripts" / "python.exe"
    fake_python.parent.mkdir(parents=True)
    shutil.copy2(Path(os.environ["SystemRoot"]) / "System32" / "where.exe", fake_python)
    return script


def run_execute_fixture(root: Path, run_id: str) -> tuple[subprocess.CompletedProcess[str], dict]:
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(root / SCRIPT.name),
            "-Execute",
            "-RunId",
            run_id,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
        check=False,
    )
    manifest = json.loads((root / "outputs" / run_id / "run_manifest.json").read_text(encoding="utf-8-sig"))
    return completed, manifest


class RunRoroShadowLocalTest(unittest.TestCase):
    def test_default_is_a_side_effect_free_plan(self) -> None:
        run_id = f"plan_guard_{uuid.uuid4().hex[:12]}"
        stage = ROOT / "outputs" / run_id
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(SCRIPT),
                "-RunId",
                run_id,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("PLAN ONLY", completed.stdout)
        self.assertFalse(stage.exists())
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count("'--no-upload'"), 2)
        self.assertNotIn("wrangler", source.lower())

    def test_manifest_hashes_all_site_build_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = Path(temporary_directory)
            create_execute_fixture(fixture)
            _, manifest = run_execute_fixture(fixture, "manifest_coverage")

            hashes = {item["path"]: item["sha256"] for item in manifest["code_snapshot"]["files"]}
            expected = {
                "requirements.txt",
                *(str(Path("site") / path) for path in REQUIRED_SITE_FILES),
                str(Path("site/drizzle/0000_fixture.sql")),
                str(Path("site/drizzle/meta/_journal.json")),
                str(Path("site/public/fixture.svg")),
            }
            self.assertTrue(expected <= hashes.keys(), expected - hashes.keys())
            migration = fixture / "site" / "drizzle" / "0000_fixture.sql"
            self.assertEqual(hashes[str(Path("site/drizzle/0000_fixture.sql"))], hashlib.sha256(migration.read_bytes()).hexdigest().upper())

    def test_execute_stops_after_first_nonzero_step(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = Path(temporary_directory)
            create_execute_fixture(fixture)
            completed, manifest = run_execute_fixture(fixture, "nonzero_stop")

            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(manifest["status"], "FAILED")
            self.assertEqual([step["name"] for step in manifest["steps"]], ["environment_python_version"])
            self.assertEqual(manifest["steps"][0]["exit_code"], 1)
            self.assertNotIn("hy_oas_alfred_backfill", completed.stdout)


if __name__ == "__main__":
    unittest.main()
