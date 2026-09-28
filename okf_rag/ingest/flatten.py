import re
from okf_rag.transcode.fences import get_fenced_regions as _get_fenced_regions
from okf_rag.transcode.fences import get_inline_code_regions as _get_inline_code_regions
from okf_rag.transcode.fences import in_fenced_region as _in_fenced_region
from okf_rag.transcode.model import ConceptDoc

# Inline links and images: [text](url) or ![alt](url)
# Note: regex may truncate on unescaped ) in href; see links.py for discussion
_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]+\)")

# Reference definitions: [label]: url (and any trailing text)
_REF_DEF = re.compile(r"^\s*\[([^\]]+)\]:\s*\S+.*$", re.MULTILINE)

# HTML tags (I4): raw `<a href=...>text</a>` and `<img ...>` occasionally
# appear in authored markdown bodies alongside markdown-syntax links, and
# the markdown link regex above never touches them (different syntax
# entirely). A void/self-closing element (no closing tag reachable, e.g.
# `<img>`) has no inner text to keep, so the whole tag is dropped; a
# paired element (`<a>...</a>`, `<span>...</span>`, etc.) keeps its inner
# text and loses only the tags. This is a plain tag strip, not an HTML
# parser -- adequate for the well-formed inline HTML these corpora use.
#
# Fix round 1 (Critical): `_HTML_VOID` used to be `<[^>]+>` -- a negated
# character class matches across newlines regardless of `re.DOTALL`
# (DOTALL only changes what `.` matches), and it had no requirement that
# what follows `<` even look like a tag name. So a stray `<` from ordinary
# prose (a comparison operator, a generic-type bracket, a typo) had no
# bound on how far the match could run: it consumed everything up to the
# *next* literal `>` anywhere later in the document, silently deleting
# unrelated paragraphs. `_HTML_VOID` now requires a real tag shape --
# `<` or `</`, then a tag name, then attributes that may not contain `<`,
# `>`, or a newline -- so `x < 5` (no tag name after `<`) and multi-line
# spans can no longer match. XHTML self-closing form (`<br/>`, `<img ...
# />`) is still recognized via the trailing optional `/`.
#
# Finding 2 (Important): requiring a *tag shape* was still not enough --
# `\w+` accepts any word as a tag name, so `<job_id>`, `<dir>`, `<file>`,
# `<service_id>` and friends all matched. Those are placeholders in CLI and
# Actions expressions, not markup, and they cluster in exactly the
# identifier-dense content retrieval is judged on. Measured over the pinned
# github/docs checkout: 1,754 placeholder occurrences inside backtick
# inline-code spans were being deleted from `chunks.text` -- the
# experiment's ground truth -- with no warning of any kind.
#
# Two independent narrowings, because neither covers the other's cases:
#   1. `_HTML_TAG_NAMES` -- an allowlist of tag names.
#   2. inline-code protection in `_strip_html_outside_fences` -- a *real*
#      tag name inside backticks (`` `<img src=...>` ``) is documentation
#      about markup, not markup, and must survive too.
#
# Fix round 2 (this fix): round 1 sized `_HTML_TAG_NAMES` to the placeholder
# repro examples rather than to the corpora's actual markup vocabulary -- a
# hand-picked 26-name list. Measured by diffing pre-/post-round-1 output over
# the real transcoded bodies of both pinned corpora: 307 occurrences across
# 35/3,736 github/docs documents and 72 across 11/1,525 home-assistant
# documents of raw HTML (`<b>`, `<thead>`/`<tbody>`, `<script>` with inline
# JavaScript, and more) were leaking into `chunks.text` where round 1's
# smaller list no longer stripped it.
#
# An allowlist assembled from observed failures is incomplete by
# construction -- the next corpus finds the next gap, same mistake in a new
# shape. The fix is not a bigger hand-picked list; it's the complete HTML5
# element-name list (current + obsolete-but-conforming, per the WHATWG HTML
# Living Standard's own element index and obsolete-features chapter, plus
# `svg`/`math` as HTML-parser-recognized foreign-content roots) -- finite,
# stable, and independent of what any particular corpus happens to contain.
#
# This is safe *because of* protection 2, not instead of it. `<dir>` and
# `<file>` -- two of finding 2's placeholder examples -- are themselves
# genuine (deprecated, in `dir`'s case) HTML element-ish names, so a
# name-based list could never separate a real `<dir>` tag from a `<dir>`
# placeholder in `--target <dir>`; only context (is it inside backticks or
# a fence?) can. That's exactly what protection 2 already provides, so
# widening protection 1 to a complete vocabulary loses nothing protection 2
# was already carrying. Both protections stay exactly as they were.
_HTML_TAG_NAMES = (
    "a|abbr|acronym|address|applet|area|article|aside|audio|b|base|basefont|"
    "bdi|bdo|bgsound|big|blink|blockquote|body|br|button|canvas|caption|"
    "center|cite|code|col|colgroup|content|data|datalist|dd|del|details|dfn|"
    "dialog|dir|div|dl|dt|em|embed|fencedframe|fieldset|figcaption|figure|"
    "font|footer|form|frame|frameset|h1|h2|h3|h4|h5|h6|head|header|hgroup|"
    "hr|html|i|iframe|image|img|input|ins|isindex|kbd|keygen|label|legend|"
    "li|link|listing|main|map|mark|marquee|math|menu|menuitem|meta|meter|"
    "multicol|nav|nextid|nobr|noembed|noframes|noscript|object|ol|optgroup|"
    "option|output|p|param|picture|plaintext|pre|progress|q|rb|rp|rt|rtc|"
    "ruby|s|samp|script|search|section|select|selectedcontent|shadow|slot|"
    "small|source|spacer|span|strike|strong|style|sub|summary|sup|svg|"
    "table|tbody|td|template|textarea|tfoot|th|thead|time|title|tr|track|"
    "tt|u|ul|var|video|wbr|xmp"
)
_HTML_PAIRED = re.compile(
    rf"<({_HTML_TAG_NAMES})(?:\s[^>]*)?>(.*?)</\1>", re.DOTALL | re.IGNORECASE
)
_HTML_VOID = re.compile(rf"</?(?:{_HTML_TAG_NAMES})(?:\s[^<>\n]*)?/?>", re.IGNORECASE)

# `<script>`/`<style>` are the one exception to "paired tags keep their
# inner text": their content is code, not prose, and leaving it behind
# would put raw JavaScript/CSS into `chunks.text` as though it were
# authored documentation. Matched and removed (tag + content, whole match)
# in its own pass, before the general paired-tag pass ever gets a chance to
# keep that content -- real corpus example: an unfenced
# `<script>...JSON.stringify(...)...</script>` block in
# apps/sharing-github-apps/registering-a-github-app-from-a-manifest.md (28
# occurrences of its content leaking).
_HTML_SCRIPT_STYLE = re.compile(r"<(script|style)(?:\s[^>]*)?>.*?</\1>", re.DOTALL | re.IGNORECASE)


def _protected_regions(text: str) -> list[tuple[int, int]]:
    """Regions of `text` that are sample code, not markup: fenced blocks
    (fix round 1) plus backtick inline-code spans (finding 2). A tag-shaped
    string starting inside either is left exactly as authored."""
    fenced = _get_fenced_regions(text)
    return fenced + _get_inline_code_regions(text, fenced)


def _strip_html_outside_fences(text: str) -> str:
    """Strip HTML tags (I4), leaving code regions untouched: fenced blocks
    (fix round 1, Critical finding) and backtick inline-code spans
    (finding 2). A tag-shaped string inside either -- C++'s
    `std::vector<int>` in a fence, `jobs.<job_id>.steps` in inline code --
    is sample code, not markup, and stripping it corrupts the sample
    exactly like an authored link would (I5's same reasoning, applied
    here).

    Protected regions are recomputed fresh before each of the two passes
    (paired, then void) rather than reused from a single earlier
    computation: the paired-tag pass may shift character offsets earlier
    in the document (by removing tags outside the regions), which would
    invalidate a stale set of (start, end) offsets for the void pass.
    Recomputing on the pass's current `text` is cheap (these are short
    document bodies) and always correct, since protected content itself is
    never modified by a skipped match.
    """
    def strip_script_style(text: str) -> str:
        protected = _protected_regions(text)
        def repl(m: re.Match) -> str:
            return m.group(0) if _in_fenced_region(protected, m.start()) else ""
        return _HTML_SCRIPT_STYLE.sub(repl, text)

    def strip_paired(text: str) -> str:
        protected = _protected_regions(text)
        def repl(m: re.Match) -> str:
            return m.group(0) if _in_fenced_region(protected, m.start()) else m.group(2)
        return _HTML_PAIRED.sub(repl, text)

    def strip_void(text: str) -> str:
        protected = _protected_regions(text)
        def repl(m: re.Match) -> str:
            return m.group(0) if _in_fenced_region(protected, m.start()) else ""
        return _HTML_VOID.sub(repl, text)

    text = strip_script_style(text)
    text = strip_paired(text)
    text = strip_void(text)
    return text


def reduce_links(text: str) -> str:
    """
    Replace markdown links with anchor text only, discarding every link
    target. This is the chunk-text contract (C1): chunk `text` in both
    chunking policies is prose only, never a raw URL -- structure travels
    in the other columns (`heading_path`, frontmatter, edges), not inline.
    Shared by `flatten_doc` (C-flat) and `chunk_struct` (C-struct) so the
    spec S12 C-flat/C-struct ablation isolates segmentation rather than
    also differing in how link targets survive into embedded text.

    - Inline links [text](url) and images ![alt](url) -> anchor text
    - Reference-style links [text][ref], [text][], [text] (with definition) -> anchor text
    - Reference definitions [ref]: url are removed entirely
    - HTML tags (I4): `<a href=...>x</a>` -> `x`; `<img ...>` -> removed;
      a tag-shaped string inside a fenced code block is left untouched.
    """
    # Step 0: Strip HTML tags (I4), skipping fenced code regions (fix
    # round 1). Paired tags keep their inner text (`<a href=...>x</a>` ->
    # `x`); whatever's left (void elements like `<img ...>`, or any
    # unpaired tag) is removed outright. Runs before the markdown-specific
    # steps so markdown syntax nested inside an HTML element's inner text
    # (e.g. `<span>[a](b)</span>`) still gets reduced by them afterward.
    # Markdown-link reduction (Steps 1-4 below) is unprotected by fences,
    # same as before this fix -- only the HTML strip changed.
    text = _strip_html_outside_fences(text)

    # Step 1: Collect reference definition labels and remove definition lines
    ref_labels = set()
    for m in _REF_DEF.finditer(text):
        ref_labels.add(m.group(1))
    text = _REF_DEF.sub("", text)

    # Step 2: Replace reference-style links [text][ref] and [text][] with anchor text
    text = re.sub(r"\[([^\]]*)\]\[([^\]]*)\]", r"\1", text)

    # Step 3: Replace shortcut references [text] only if [text]: definition exists
    for label in ref_labels:
        # Negative lookahead ensures we don't match [label](...) which is handled later
        text = re.sub(r"\[" + re.escape(label) + r"\](?!\()", label, text)

    # Step 4: Replace inline links [text](url) and images ![alt](url)
    text = _LINK.sub(lambda m: m.group(1), text)

    return text


def flatten_doc(doc: ConceptDoc) -> str:
    """
    Strip frontmatter and replace markdown links with anchor text only.

    Frontmatter is not processed explicitly; `doc.body` already excludes it
    (it was split out at transcode time), so this is just `reduce_links`
    applied to the body.
    """
    return reduce_links(doc.body)
