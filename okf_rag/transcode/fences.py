"""Shared fenced-code-block detection (I5).

Promoted from `okf_rag/ingest/chunk.py`, which used this to keep headings
inside fenced code blocks from being treated as section breaks. `extract_links`
(`okf_rag/transcode/links.py`) needs the identical logic to keep a
markdown-link-shaped string inside a fenced code block (sample code, not an
authored link) from being surfaced as a transcoder edge -- rather than
duplicate the fence-matching regex and CommonMark closing rules in two
places, both call sites share this one implementation.
"""

import re

_FENCE_PATTERN = re.compile(r"^(```+|~~~+)(?:.*?)$", re.MULTILINE)


def get_fenced_regions(text: str) -> list[tuple[int, int]]:
    """
    Return list of (start, end) byte positions for fenced code blocks.
    Respects CommonMark fence rules: a fence is closed only by the same marker
    character with at least the same count, at line start.
    """
    fences = []
    matches = list(_FENCE_PATTERN.finditer(text))
    i = 0
    while i < len(matches):
        fence_open = matches[i]
        fence_char = fence_open.group(1)[0]
        fence_len = len(fence_open.group(1))

        # Look for closing fence
        j = i + 1
        found_close = False
        while j < len(matches):
            fence_close = matches[j]
            close_char = fence_close.group(1)[0]
            close_len = len(fence_close.group(1))

            if close_char == fence_char and close_len >= fence_len:
                # Found matching closing fence
                fences.append((fence_open.start(), fence_close.end()))
                i = j + 1
                found_close = True
                break
            j += 1

        if not found_close:
            # No closing fence found, treat rest of document as fenced
            fences.append((fence_open.start(), len(text)))
            break

    return fences


def in_fenced_region(fenced_regions: list[tuple[int, int]], pos: int) -> bool:
    """Check if position is inside any fenced code block."""
    for start, end in fenced_regions:
        if start <= pos < end:
            return True
    return False


# A backtick inline-code span: a run of backticks, content, then a closing
# run of the SAME length (CommonMark's rule), with the negative lookarounds
# stopping a 2-backtick opener from closing against the first backtick of a
# longer run.
#
# Deliberately bounded to one line (`[^\n]`) even though CommonMark allows a
# code span to wrap. These corpora author inline code on a single line, and
# the bound is what makes an *unpaired* backtick harmless: without it, a
# stray backtick would pair with some backtick far later in the document and
# silently protect every real HTML tag in between, leaving raw link targets
# in chunk text (the opposite failure to finding 2, but still a failure).
_INLINE_CODE = re.compile(r"(?<!`)(`+)(?!`)([^\n]+?)(?<!`)\1(?!`)")


def get_inline_code_regions(
    text: str, fenced_regions: list[tuple[int, int]] | None = None
) -> list[tuple[int, int]]:
    """Return (start, end) positions of backtick inline-code spans (finding 2).

    Inline code is the second kind of "this is sample text, not markup"
    region in a markdown body -- the first being fenced blocks. The HTML
    strip in `okf_rag/ingest/flatten.py` protected fences but not inline
    code, so an identifier-shaped placeholder inside backticks
    (`jobs.<job_id>.steps`, `--target <dir>`) was deleted from `chunks.text`,
    the experiment's ground truth. Measured over the pinned github/docs
    checkout: 1,754 such occurrences.

    Scanned only in the gaps *between* fenced regions: a fence's own opening
    marker is itself a run of backticks, so scanning the whole text would
    manufacture spans that start inside a fence and end well outside it.
    """
    if fenced_regions is None:
        fenced_regions = get_fenced_regions(text)
    regions: list[tuple[int, int]] = []
    cursor = 0
    for fence_start, fence_end in sorted(fenced_regions) + [(len(text), len(text))]:
        if fence_start > cursor:
            segment = text[cursor:fence_start]
            regions.extend(
                (cursor + m.start(), cursor + m.end()) for m in _INLINE_CODE.finditer(segment)
            )
        cursor = max(cursor, fence_end)
    return regions
