"""Shared edge-kind -> default-weight table (spec §4.3, arch §7).

Both transcoders (`github_docs.py`, `home_assistant.py`) read from here
rather than each defining their own copy (I3). Spec §12's typed-vs-uniform
edge-weight ablation mutates exactly these constants, and a value
duplicated across files (e.g. `prose = 0.6` previously lived in both) could
drift out of sync between corpora -- silently, since nothing would flag it.
"""

EDGE_WEIGHTS: dict[str, float] = {
    "child": 1.0,
    "curated": 1.0,
    "related": 0.8,
    "prose": 0.6,
    "sibling": 0.2,  # ablation only (spec §4.3); not emitted by either transcoder yet
}
