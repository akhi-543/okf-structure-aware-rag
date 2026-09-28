from pathlib import Path
import yaml

RESERVED = {"index.md", "log.md"}

def _frontmatter_of(text: str):
    if not text.startswith("---\n"):
        return None
    parts = text.split("---\n", 2)
    if len(parts) < 3:
        return None
    try:
        fm = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        return None
    return fm if isinstance(fm, dict) else None

def validate_bundle(bundle_dir: Path) -> list[str]:
    errs: list[str] = []
    dirs_seen: set[str] = set()
    for p in sorted(bundle_dir.rglob("*.md")):
        rel = p.relative_to(bundle_dir).as_posix()
        rel_dir = p.parent.relative_to(bundle_dir).as_posix()
        if rel_dir != ".":
            # Fix round 1: every directory on the *full* ancestor chain up
            # to (but excluding) the bundle root needs its own index.md --
            # not just this file's immediate parent. A pass-through
            # ancestor directory (one holding only a subdirectory, no
            # *.md of its own) would otherwise never enter dirs_seen at
            # all, since no file found by rglob has it as a direct parent
            # -- the exact multi-level gap I7 exists to close, and this
            # validator must catch it independently of write_bundle
            # (a bundle it's handed may not have come from write_bundle).
            parts = rel_dir.split("/")
            for i in range(1, len(parts) + 1):
                dirs_seen.add("/".join(parts[:i]))
        fm = _frontmatter_of(p.read_text(encoding="utf-8"))
        if fm is None:
            errs.append(f"{rel}: unparseable or missing YAML frontmatter")
            continue
        if p.name not in RESERVED and not str(fm.get("type") or "").strip():
            errs.append(f"{rel}: missing or empty required field 'type'")
    # I7: every directory that holds at least one markdown file, or is an
    # ancestor of one, must also have its own index.md -- flat
    # "type"/frontmatter checks above only catch a *malformed* index.md,
    # not a directory missing one entirely.
    for rel_dir in sorted(dirs_seen):
        if not (bundle_dir / rel_dir / "index.md").exists():
            errs.append(f"{rel_dir}: directory missing index.md")
    root = bundle_dir / "index.md"
    if not root.exists():
        errs.append("index.md: bundle root index missing")
    else:
        fm = _frontmatter_of(root.read_text(encoding="utf-8"))
        if not fm or "okf_version" not in fm:
            errs.append("index.md: root index missing okf_version")
    return errs
