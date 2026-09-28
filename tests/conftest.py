"""Session-wide import-order guard for pytest.

On Windows, importing pyarrow before torch in the same process leaves
torch's own DLL loader unable to initialize its C extension --
`OSError: [WinError 1114]` loading `torch/lib/c10.dll` or one of its
dependencies -- not a catchable ImportError. See the module docstring of
`okf_rag/jobs/embed.py` and the matching note in `scripts/ingest_corpus.py`
for the mechanism and precedent: torch has to enter the process before
pyarrow does, once, and the safe order holds for the rest of the process.

pytest imports every conftest.py in a directory before it collects any test
module under that directory. `tests/jobs/test_embed.py` imports pyarrow at
module scope; without this, whichever test happens to run the first real
`import torch` in the session inherits whatever import order got there
first, and on this suite that's pyarrow. Importing torch here pins the safe
order before any test file gets the chance to import pyarrow first.

Guarded: a machine with no torch installed must still collect the suite.
"""
try:
    import torch  # noqa: F401
except ImportError:
    pass
