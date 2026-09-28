from collections import Counter
from dataclasses import dataclass, field
import posixpath
import re
from typing import TYPE_CHECKING

from okf_rag.transcode.fences import get_fenced_regions as _get_fenced_regions
from okf_rag.transcode.fences import in_fenced_region as _in_fenced_region

if TYPE_CHECKING:
    # Task 2: type-only import to avoid a hard coupling from links.py (used
    # by every transcoder) to liquid.py (github_docs-specific rendering) --
    # ResolutionStats.liquid only needs LiquidStats for the annotation, and
    # this module never constructs one.
    from okf_rag.transcode.liquid import LiquidStats

# Matches markdown links [text](href) but NOT image markdown ![alt](src).
# Note: An href containing an unescaped ) will truncate at the first paren, e.g.
# [a](notes_(draft).md) truncates to notes_. This is rare in static docs.
_MD_LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")

# I6: cap on ResolutionStats.unresolved_samples so a corpus with thousands
# of broken links doesn't blow up memory or log output -- the `unresolved`
# counter itself is uncapped, only the sample list stops growing.
_MAX_UNRESOLVED_SAMPLES = 50

@dataclass
class ResolutionStats:
    resolved: int = 0
    unresolved: int = 0
    # I6: (src, href) pairs for a bounded sample of unresolved links, so a
    # human debugging a low resolution rate can see *which* links broke,
    # not just how many.
    unresolved_samples: list[tuple[str, str]] = field(default_factory=list)
    # Finding 6: an uncapped tally of every unresolved href, keyed by the
    # href itself. `unresolved_samples` is capped at 50 and so cannot
    # answer "how many unresolved links point *outside* the pinned corpus
    # subset vs at an in-scope page the resolver missed" -- the question
    # that would have caught finding 1 when the 79.63% rate was first
    # recorded. Bucketing is a corpus-specific prefix/shape test and lives
    # in the stats writer (`scripts/ingest_corpus.py`); this dataclass just
    # keeps the raw material, corpus-agnostic. Bounded by the number of
    # *distinct* broken hrefs (a few thousand on these corpora).
    unresolved_hrefs: Counter = field(default_factory=Counter)
    # Task 2: the LiquidStats a Liquid-rendering transcoder (github_docs)
    # already builds, surfaced here so a caller gets it from the return
    # value instead of only ever seeing it in a log line. Stays None for a
    # transcoder that isn't Liquid-rendered (home_assistant).
    liquid: "LiquidStats | None" = None

    @property
    def rate(self) -> float:
        total = self.resolved + self.unresolved
        return self.resolved / total if total else 1.0

    def record_unresolved(self, src: str, href: str) -> None:
        """Bump `unresolved` and append the (src, href) sample together, so
        the two can never drift apart -- the single call transcoders make
        on every unresolved increment."""
        self.unresolved += 1
        self.unresolved_hrefs[href] += 1
        if len(self.unresolved_samples) < _MAX_UNRESOLVED_SAMPLES:
            self.unresolved_samples.append((src, href))

def extract_links(body: str) -> list[str]:
    # I5: a link inside a fenced code block is sample code, not an authored
    # link -- skip matches whose start position falls inside a fenced
    # region (shared detection with okf_rag/ingest/chunk.py's heading skip,
    # via okf_rag/transcode/fences.py).
    fenced = _get_fenced_regions(body)
    return [m.group(1) for m in _MD_LINK.finditer(body) if not _in_fenced_region(fenced, m.start())]

def _candidates(path: str) -> list[str]:
    path = path.rstrip("/")
    if path.endswith(".md"):
        return [path]
    return [f"{path}.md", f"{path}/index.md"]

def is_external(href: str) -> bool:
    """Check if href is external (http, https, mailto) or anchor-only."""
    return href.startswith(("http://", "https://", "mailto:", "#"))

def resolve(src_path: str, href: str, known: set[str]) -> str | None:
    if is_external(href):
        return None
    href = href.split("#", 1)[0].split("?", 1)[0]
    if not href:
        return None
    if href.startswith("/"):
        base = href.lstrip("/")
    else:
        base = posixpath.normpath(posixpath.join(posixpath.dirname(src_path), href))
    for cand in _candidates(base):
        if cand in known:
            return cand
    return None


def resolve_dir_relative(src_path: str, href: str, known: set[str]) -> str | None:
    """Resolve `href` reading a leading `/` as relative to the *containing
    document's directory*, not the bundle root.

    Finding 1: github/docs authors frontmatter `children:` entries with a
    leading slash but means them directory-relative --
    `content/account-and-profile/concepts/index.md` lists
    `children: [/personal-profile, ...]` and the target is
    `content/account-and-profile/concepts/personal-profile.md`, not
    `content/personal-profile.md`. Read as bundle-root-relative (what
    `resolve` does), only 186 of that corpus's 3,784 `children:` entries
    resolved at all, against 3,733 read directory-relative -- so `G_auth`,
    the study's independent variable, was missing ~98% of its authored
    hierarchy, and `prose` edges outnumbered `child` edges 88:1.

    This is a *separate* entry point rather than a flag on `resolve`
    precisely because the convention is not universal: in the same corpus
    `introLinks`, `carousels` and prose body links ARE site-absolute
    (measured: 310/313 curated links resolve from the content root and 0
    resolve directory-relative), and home-assistant.io's `/integrations/x`
    is bundle-root-relative too. Only a caller that knows its field follows
    the directory-relative convention opts in.
    """
    if is_external(href):
        return None
    return resolve(src_path, href.lstrip("/"), known)
