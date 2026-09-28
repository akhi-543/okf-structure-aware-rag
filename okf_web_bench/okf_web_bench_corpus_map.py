from __future__ import annotations

from okf_web_bench.okf_web_bench_compare import ALLOWED_CORPORA

__all__ = [
    "ALLOWED_CORPORA",
    "texts_overlap",
    "identical_chunk_pair",
    "overlap_ids",
    "document_row",
    "map_document",
]

_MIN_OVERLAP_LEN = 8


def texts_overlap(a: str, b: str) -> bool:
    if len(a) <= len(b):
        shorter, longer = a, b
    else:
        shorter, longer = b, a

    if len(shorter) < _MIN_OVERLAP_LEN:
        return bool(shorter) and shorter in longer

    for i in range(len(a) - _MIN_OVERLAP_LEN + 1):
        if a[i : i + _MIN_OVERLAP_LEN] in b:
            return True
    for i in range(len(b) - _MIN_OVERLAP_LEN + 1):
        if b[i : i + _MIN_OVERLAP_LEN] in a:
            return True
    return False


def identical_chunk_pair(flat: list[dict], struct: list[dict]) -> bool:
    return (
        len(flat) == 1
        and len(struct) == 1
        and flat[0]["text"] == struct[0]["text"]
    )


def overlap_ids(selected: dict, others: list[dict]) -> list:
    selected_text = selected["text"]
    return [
        other["chunk_id"]
        for other in others
        if texts_overlap(selected_text, other["text"])
    ]


def document_row(
    path: str,
    title: str,
    flat_chunks: list[dict],
    struct_chunks: list[dict],
) -> dict:
    return {
        "path": path,
        "title": title,
        "n_flat": len(flat_chunks),
        "n_struct": len(struct_chunks),
        "identical": identical_chunk_pair(flat_chunks, struct_chunks),
    }


def map_document(
    *,
    path: str,
    title: str,
    flat_chunks: list[dict],
    struct_chunks: list[dict],
    authored_edges: list[dict],
) -> dict:
    row = document_row(path, title, flat_chunks, struct_chunks)
    row["flat"] = flat_chunks
    row["struct"] = struct_chunks
    row["edges"] = authored_edges
    return row
