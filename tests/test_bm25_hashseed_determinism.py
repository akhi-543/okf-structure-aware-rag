"""BM25 scores must not depend on the per-process string hash seed (set iteration order)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = (
    "from okf_rag.retrieve.control_arms_v4 import BM25Index\n"
    "docs = [{'chunk_id': str(i), 'doc_path': f'd{i}', 'text': ' '.join(f'w{(i * j) % 17} x{j % 5}' for j in range(40))}"
    " for i in range(60)]\n"
    "idx = BM25Index.build(docs)\n"
    "q = ' '.join(f'w{k} x{k % 5}' for k in range(17))\n"
    "print(repr(idx.search(q, 60)))\n"
)


def _run(seed: str) -> str:
    out = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True, cwd=str(ROOT),
                         env={**__import__("os").environ, "PYTHONHASHSEED": seed})
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


def test_bm25_search_is_identical_across_hash_seeds():
    assert _run("1") == _run("2") == _run("3")
