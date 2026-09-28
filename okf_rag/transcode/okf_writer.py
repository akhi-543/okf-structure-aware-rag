"""Writer for OKF v0.2 bundles.

A bundle is a directory tree of markdown files: each `ConceptDoc` is written
verbatim (YAML frontmatter + body) to its bundle-relative path, and every
directory gets a generated `index.md` listing its members as
`* [Title](/path) - description` bullets. The bundle root's `index.md`
additionally carries `okf_version`.

Ruling F-A: a directory's `index.md` path (including the bundle root) may
already be claimed by an authored `ConceptDoc` (e.g. a landing page). In that
case the authored document is written as-is rather than being overwritten by
a generated index; the generated bullet-list entries for its siblings are
appended to its body instead.
"""

from collections import defaultdict
from pathlib import Path

import yaml

from .model import ConceptDoc


def _frontmatter(d: ConceptDoc) -> dict:
    fm = {"type": d.okf_type, "title": d.title}
    if d.description:
        fm["description"] = d.description
    if d.resource:
        fm["resource"] = d.resource
    if d.tags:
        fm["tags"] = d.tags
    if d.timestamp:
        fm["timestamp"] = d.timestamp
    fm["status"] = d.status
    if d.x_source:
        fm["x_source"] = d.x_source
    return fm


def _bullet(d: ConceptDoc) -> str:
    desc = d.description or d.title
    return f"* [{d.title}](/{d.path}) - {desc}"


def _rel_dir(path: str) -> str:
    """POSIX-style parent directory of a bundle-relative path; "" for the bundle root."""
    parent = str(Path(path).parent)
    return "" if parent == "." else parent.replace("\\", "/")


def _ancestor_dirs(path: str) -> list[str]:
    """Every non-root directory on `path`, from the shallowest to the
    immediate parent (I7). For "a/b/c.md" this is ["a", "a/b"] -- both
    need an index.md, not just the immediate parent "a/b"."""
    rel_dir = _rel_dir(path)
    if rel_dir == "":
        return []
    parts = rel_dir.split("/")
    return ["/".join(parts[: i + 1]) for i in range(len(parts))]


def _render(fm: dict, body: str) -> str:
    fm_text = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True)
    return f"---\n{fm_text}---\n{body}\n"


def write_bundle(docs: list[ConceptDoc], out_dir: Path, okf_version: str = "0.2") -> None:
    out_dir = Path(out_dir)
    by_dir: dict[str, list[ConceptDoc]] = defaultdict(list)
    by_path: dict[str, ConceptDoc] = {d.path: d for d in docs}
    for d in docs:
        by_dir[_rel_dir(d.path)].append(d)

    # Write every authored document. A doc occupying an index.md path (a
    # directory landing page, or the bundle root) is never overwritten by the
    # generated index below; its siblings' bullet entries are appended to its
    # own body instead (Ruling F-A).
    for d in docs:
        target = out_dir / d.path
        target.parent.mkdir(parents=True, exist_ok=True)
        fm = _frontmatter(d)
        body = d.body
        if Path(d.path).name == "index.md":
            rel_dir = _rel_dir(d.path)
            if rel_dir == "":
                fm["okf_version"] = okf_version
                siblings = [m for m in docs if m.path != d.path]
            else:
                siblings = [m for m in by_dir[rel_dir] if m.path != d.path]
            if siblings:
                entries = "\n".join(_bullet(m) for m in sorted(siblings, key=lambda x: x.path))
                body = f"{body}\n\n{entries}"
        target.write_text(_render(fm, body), encoding="utf-8")

    # Generate a directory index for every directory with no authored
    # index.md. This covers every directory on every doc's path (I7), not
    # just immediate parents: an intermediate ancestor directory (e.g. "a"
    # for a doc at "a/b/c.md") has no docs of its own in by_dir, but the
    # bundle still needs an index.md there for the tree to be navigable.
    all_dirs: set[str] = set()
    for d in docs:
        all_dirs.update(_ancestor_dirs(d.path))
    for rel_dir in sorted(all_dirs):
        if f"{rel_dir}/index.md" in by_path:
            continue  # authored landing page already written above
        members = by_dir.get(rel_dir, [])
        lines = ["# Index", ""] + [_bullet(m) for m in sorted(members, key=lambda x: x.path)]
        idx = out_dir / rel_dir / "index.md"
        idx.parent.mkdir(parents=True, exist_ok=True)
        idx.write_text(_render({"type": "Index"}, "\n".join(lines)), encoding="utf-8")

    # Bundle-root index.md: generated (lists every doc) unless a ConceptDoc
    # already claims that path, in which case it was written above.
    if "index.md" not in by_path:
        entries = [_bullet(d) for d in sorted(docs, key=lambda x: x.path)]
        body = "# Index\n\n" + "\n".join(entries)
        fm = {"type": "Index", "okf_version": okf_version}
        (out_dir / "index.md").write_text(_render(fm, body), encoding="utf-8")
