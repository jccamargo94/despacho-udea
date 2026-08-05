"""Layer-4 check: the literal CLI command from the Fase 2B spec exit
criterion, run as a subprocess exactly as a Docker CMD would invoke it."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DD = REPO_ROOT / "tests" / "fixtures" / "xm_smoke"


def test_cli_run_against_fixture(tmp_path):
    out = tmp_path / "results"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "app",
            "run",
            "2024-04-18",
            "-t",
            "preideal",
            "--data-dir",
            str(DD),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Done: 1 ok, 0 failed." in result.stdout
    assert (out / "marginal_price-2024-04-18-preideal.csv").exists()
    assert (out / "dispatch_by_gen-2024-04-18-preideal.csv").exists()
