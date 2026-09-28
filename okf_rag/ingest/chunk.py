from dataclasses import dataclass
import re
from typing import Callable
from okf_rag.ingest.flatten import reduce_links
from okf_rag.transcode.fences import get_fenced_regions as _get_fenced_regions
from okf_rag.transcode.model import ConceptDoc

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

@dataclass
class StructChunk:
    heading_path: list[str]
    text: str
    ord: int

def _in_fenced_region(fenced_regions: list[tuple[int, int]], pos: int) -> bool:
    """Check if position is inside any fenced code block."""
    for start, end in fenced_regions:
        if start <= pos < end:
            return True
    return False

def _find_window_end(words: list[str], i: int, n: int, count_tokens: Callable[[str], int], size: int) -> int:
    """
    Largest j in (i, n] such that count_tokens(" ".join(words[i:j])) <= size,
    found by binary search (assumes count_tokens is non-decreasing as the
    window grows -- true of any real tokenizer). O(log(n - i)) calls.

    `best` is pre-seeded to i + 1 so that even when the single word at `i`
    alone exceeds `size`, the function still returns i + 1: the degenerate
    oversized-word case is emitted as its own one-word chunk rather than
    dropped or looped on.
    """
    lo, hi = i + 1, n
    best = i + 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if count_tokens(" ".join(words[i:mid])) <= size:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best

def _find_next_start(words: list[str], i: int, j: int, count_tokens: Callable[[str], int], overlap: int) -> int:
    """
    Smallest k in [i, j] such that the trailing span words[k:j] fits within
    the `overlap` token budget, found by binary search (the tail only grows,
    and its token count only rises, as k decreases toward i). O(log(j - i))
    calls. Returns j directly (no overlap) when overlap <= 0.
    """
    if overlap <= 0 or j <= i:
        return j
    lo, hi = i, j
    best = j
    while lo <= hi:
        mid = (lo + hi) // 2
        if count_tokens(" ".join(words[mid:j])) <= overlap:
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return best

def chunk_flat(text: str, count_tokens: Callable[[str], int], size: int = 512, overlap: int = 50) -> list[str]:
    """
    Create overlapping chunks from text, sized in `count_tokens` units (not
    whitespace words -- the spec budgets chunks at 512 *tokens*, and subword
    tokenizers are typically denser than word count).

    Each window is grown via binary search to the largest word span that
    fits within `size` tokens (see `_find_window_end`). The next window's
    start is chosen by walking back from the current window's end far enough
    to carry ~`overlap` tokens of trailing context, also via binary search
    (see `_find_next_start`) -- so the step tracks what was *actually*
    emitted rather than a nominal `size - overlap` word count. This is what
    keeps windows contiguous (no un-emitted words between chunks) even when
    count_tokens is far denser than word count.

    A single word that alone exceeds `size` tokens is still emitted as its
    own chunk (never dropped). The start index strictly increases every
    iteration, so the loop always terminates. Windows that would add no
    content beyond what earlier chunks already cover are skipped (redundant-
    chunk suppression), without ever discarding a window that carries
    genuinely new content.
    """
    words = text.split()
    if not words:
        return []
    n = len(words)
    size = max(size, 1)
    overlap = max(overlap, 0)

    chunks: list[str] = []
    covered_until = 0  # rightmost word index already covered by an emitted chunk
    i = 0
    while i < n:
        j = _find_window_end(words, i, n, count_tokens, size)
        if j > covered_until:
            chunks.append(" ".join(words[i:j]))
            covered_until = j
        if j >= n:
            break
        k = _find_next_start(words, i, j, count_tokens, overlap)
        i = max(k, i + 1)  # progress invariant: strictly advance past i
    return chunks

def chunk_struct(
    doc: ConceptDoc,
    count_tokens: Callable[[str], int],
    size: int = 512,
    overlap: int = 50,
    link_reducer: Callable[[str], str] | None = None,
) -> list[StructChunk]:
    """
    Split document into chunks by heading boundaries while preserving heading context.
    Oversized sections (exceeding `size` tokens) are split with chunk_flat.
    Ignores heading matches inside fenced code blocks.

    Chunk-text contract (C1): `text` is prose only, on par with C-flat, so
    the C-flat/C-struct ablation (spec S12) isolates segmentation and
    nothing else. That means two things versus the pre-fix behaviour:
    - the section's own heading line (e.g. "## Configuration") is now part
      of its chunk text, not just `heading_path` -- C-flat keeps headings
      in its text too (flatten_doc doesn't strip `#` lines), so dropping
      them here made C-struct embed strictly less than C-flat.
    - link targets are reduced to anchor text via the same `reduce_links`
      flatten_doc uses, so C-struct text doesn't carry raw URLs that
      C-flat's text never has.

    Heading-only sections (finding 3): since the heading line now travels
    in section text, a heading immediately followed by a subheading (no
    body of its own) produces a section whose text is just that heading
    line -- a near-content-free chunk that C-flat has no equivalent of.
    Rather than skip such a section (which would drop its title from
    C-struct text while C-flat keeps it, recreating the very heading/text
    asymmetry C1 removed), it is folded forward into the *next* section's
    chunk -- so "## Overview\n### Detail A\ntext a" becomes one chunk
    carrying both heading lines, under the *next* section's heading_path
    (the one whose content it now carries). A heading-only section with no
    following section (nothing left to fold into) is emitted as its own
    chunk instead.
    """
    reducer = reduce_links if link_reducer is None else link_reducer
    body = reducer(doc.body)
    # (heading_path, section_text, heading_only) -- heading_only is True iff
    # nothing but whitespace sits between this heading's own line and the
    # next heading (or the end of the document).
    sections: list[tuple[list[str], str, bool]] = []
    # Stack of (level, title) rather than a flat list sliced by absolute
    # level (finding 4): slicing by level assumes heading levels map onto
    # list depth, which only holds for a document rooted at "#". Popping
    # every entry whose level is >= the incoming heading's level -- instead
    # of `path[:level - 1]` -- is correct for any root level and any level
    # jump, because depth is tracked by what's actually on the stack, not
    # by the heading's absolute level number.
    stack: list[tuple[int, str]] = []

    # Find all heading positions and identify fenced regions to skip
    fenced_regions = _get_fenced_regions(body)
    matches = [m for m in _HEADING.finditer(body) if not _in_fenced_region(fenced_regions, m.start())]

    if not matches:
        sections.append(([], body, False))
    else:
        # Handle preamble before first heading
        if matches[0].start() > 0 and body[:matches[0].start()].strip():
            sections.append(([], body[:matches[0].start()], False))

        # Process each heading and its section. Sliced from m.start() (not
        # m.end()) so the heading line itself travels with its section text.
        for i, m in enumerate(matches):
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            heading_only = not body[m.end() : end].strip()
            sections.append(([t for _, t in stack], body[m.start() : end], heading_only))

    # Fold heading-only sections forward (finding 3). `StructChunk.ord` must
    # stay sequential and gap-free, which falls out naturally: this pass
    # produces exactly one (heading_path, text) entry per emitted chunk
    # base, in order, before the size-split loop below assigns `ord`.
    folded: list[tuple[list[str], str]] = []
    pending = ""
    last = len(sections) - 1
    for i, (hp, sec_text, heading_only) in enumerate(sections):
        combined = pending + sec_text
        if heading_only and i != last:
            pending = combined  # nothing to emit yet -- fold into the next section
            continue
        pending = ""
        folded.append((hp, combined))

    # Convert sections to chunks
    out: list[StructChunk] = []
    for hp, sec_text in folded:
        if not sec_text.strip():
            continue
        if count_tokens(sec_text) <= size:
            # I8: normalise whitespace the same way chunk_flat does for an
            # over-size section (" ".join(text.split())), not just strip().
            # Without this, chunks.text had two different shapes under one
            # policy -- multi-line for a section that fit under `size`,
            # single-line for one that didn't -- purely as a side effect of
            # section length, which the schema/consumers have no reason to
            # expect. Original line structure (and any code-block
            # indentation/fencing within it) is sacrificed here exactly as
            # it already was for oversized sections and for C-flat text
            # generally -- this is not a new loss, just a consistently
            # applied one.
            out.append(StructChunk(hp, " ".join(sec_text.split()), len(out)))
        else:
            # Split oversized section with chunk_flat
            for piece in chunk_flat(sec_text, count_tokens, size, overlap):
                out.append(StructChunk(hp, piece, len(out)))

    return out
