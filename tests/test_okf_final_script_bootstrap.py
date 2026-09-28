"""Every pipeline script must import this repository's packages, never an installed copy.

`python scripts/x.py` puts scripts/ (not the repo root) first on sys.path, so without the
bootstrap an editable install of another checkout that also provides `okf_rag` silently wins,
and the script talks to that checkout's database defaults.
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_bootstrap_precedes_first_package_import(script):
    lines = script.read_text(encoding="utf-8").splitlines()
    boot = next(i for i, line in enumerate(lines) if "_sys.path.insert(0" in line)
    first_pkg = next(i for i, line in enumerate(lines)
                     if line.startswith(("from okf_rag", "import okf_rag", "from scripts")))
    assert boot < first_pkg


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_loads_okf_rag_from_this_repo(script, tmp_path):
    """Run the script's --help from an unrelated directory, then report where
    `okf_rag` was imported from: it must be this repository."""
    code = (
        "import runpy, sys\n"
        f"sys.argv = [{str(script)!r}, '--help']\n"
        "try:\n"
        f"    runpy.run_path({str(script)!r}, run_name='__main__')\n"
        "except SystemExit:\n"
        "    pass\n"
        "import okf_rag\n"
        "print('OKF_RAG_FROM=' + okf_rag.__file__)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(tmp_path))
    assert out.returncode == 0, out.stderr[-2000:]
    origin = [line for line in out.stdout.splitlines() if line.startswith("OKF_RAG_FROM=")][-1]
    assert Path(origin.split("=", 1)[1]).resolve().is_relative_to(ROOT)
