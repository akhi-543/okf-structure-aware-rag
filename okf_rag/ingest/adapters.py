"""Adapters between the two chunking policies' native shapes and the
`(ord, heading_path, text, n_tokens)` row tuple `load_pg.load_chunks`
expects.

Kept separate from `scripts/ingest_corpus.py` so they're unit-testable with
a fake `count_tokens` -- the script itself is integration-grade (real
tokenizer, real corpus, real Postgres) and is verified by running it, not
by a unit test.
"""
from typing import Callable

from okf_rag.ingest.chunk import StructChunk


def flat_rows(texts: list[str], count_tokens: Callable[[str], int]) -> list[tuple[int, list[str], str, int]]:
    """C-flat chunks carry no heading structure, so `heading_path` is always
    `[]`; `ord` is the chunk's position in `texts` (the order `chunk_flat`
    already emits them in)."""
    return [(i, [], t, count_tokens(t)) for i, t in enumerate(texts)]


def struct_rows(chunks: list[StructChunk], count_tokens: Callable[[str], int]) -> list[tuple[int, list[str], str, int]]:
    """`ord` is carried through from `StructChunk.ord` (assigned by
    `chunk_struct`) rather than re-derived from list position, so a caller
    that filters or reorders the chunk list first still stores correct
    ordinals."""
    return [(c.ord, c.heading_path, c.text, count_tokens(c.text)) for c in chunks]
