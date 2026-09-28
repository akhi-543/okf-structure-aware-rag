"""Loader and use-site accessor for `configs/models.lock` (spec §16; arch §10
Global Constraint).

One `ModelPin` per role, even when several roles call the same checkpoint:
`benchmark_wording`, `graph_extraction` and `answer_generator` all currently
point at `Qwen/Qwen3-4B-Instruct-2507`, but they are versioned independently
-- repointing the answer generator must not silently invalidate the frozen
benchmark wording or the extracted graph baseline. Keeping them as separate
lock entries (rather than one entry aliased by three roles) means a future
edit that repoints one role cannot touch another's pin by construction.

This module deliberately imports no model library -- no torch, no
transformers, not even lazily. `okf_rag/jobs/*` is meant to be importable in
a Postgres-free, torch-free environment (Kaggle, a CI shape-check, a laptop
with no GPU); this module only parses YAML (`pyyaml`, already a project
dependency) and validates it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# Resolved from this file's location rather than the caller's cwd, so
# `model_pin("embedding")` works the same whether it's invoked from a
# script in `scripts/`, a job in `okf_rag/jobs/`, or a test running from
# the repo root.
MODEL_LOCK_PATH = Path(__file__).resolve().parents[2] / "configs" / "models.lock"

# The sentinel `configs/models.lock` ships with before a checkpoint's commit
# SHA has been resolved offline. Matched against `revision` verbatim -- not
# a mixed-case or partial match -- so a hand-edited near-miss (e.g.
# "unresolved") is caught by the hex-SHA validation in `_validate_entry`
# instead of silently passing as a pin.
UNRESOLVED = "UNRESOLVED"

_SHA_LENGTH = 40
_HEX_DIGITS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class ModelPin:
    """One role's pinned checkpoint, as recorded in `configs/models.lock`."""
    role: str
    model_id: str
    revision: str
    dtype: str | None
    max_new_tokens: int | None
    generation: dict | None
    role_note: str


def is_valid_sha(revision: str) -> bool:
    """Whether `revision` is a resolved commit SHA rather than a moving pointer.

    Public because a lock entry is not the only way a revision reaches a
    model load: a CLI `--revision` override bypasses `_validate_entry`
    entirely, and a run pinned to `main` recorded as if it were a pin is
    exactly what this file exists to prevent.
    """
    return len(revision) == _SHA_LENGTH and set(revision) <= _HEX_DIGITS


# Retained for the in-module call sites; `is_valid_sha` is the name callers use.
_is_valid_sha = is_valid_sha


def _require_nonblank_str(role: str, raw: dict, key: str) -> str:
    """Fetch `raw[key]` as a non-blank string, or raise ValueError.

    A missing-key check alone (`key not in raw`) only proves the key was
    typed; it does not prove the entry holds a usable value. `key:` with
    nothing after the colon, and `key: ""`, both parse to something other
    than a real string -- the first to YAML `null`, present but empty --
    and either would otherwise flow straight into a ModelPin's str-typed
    field, or crash a later step (e.g. `_is_valid_sha` calling `len(None)`)
    with the wrong exception type entirely. Checking presence, non-null-
    ness and non-blankness together, before any code reads the value, is
    what the missing-key check alone did not do.
    """
    if key not in raw:
        raise ValueError(f"models.lock role {role!r} is missing required key {key!r}")
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"models.lock role {role!r} key {key!r} is present but blank ({value!r}); "
            f"write a real value for it"
        )
    return value


def _validated_generation(role: str, raw: dict) -> dict | None:
    """Fetch and validate the optional `generation` block.

    `generation` is optional-but-typed: omitting the key, or setting it to
    `null`, both mean "this role isn't generative" and are legal. Anything
    else present must be a mapping -- `generation: true` is a hand-edit
    mistake, not a third way to say "not generative", and left unchecked it
    reaches `.get("do_sample")` below and crashes with AttributeError
    instead of naming the role.
    """
    generation = raw.get("generation")
    if generation is None:
        return None
    if not isinstance(generation, dict):
        raise ValueError(
            f"models.lock role {role!r} key 'generation' must be a mapping or null, "
            f"got {generation!r}"
        )
    if "do_sample" not in generation:
        # Distinct from the case below: a generation block that never
        # mentions do_sample is an incomplete lock entry, not one that
        # actively chose to sample -- the message must say so, since the
        # fix for each is different (add the key vs. flip its value).
        raise ValueError(
            f"models.lock role {role!r} generation block is missing 'do_sample': "
            "a generative role's lock entry must set do_sample: false explicitly"
        )
    if generation["do_sample"] is not False:
        # A role with a generation block is one whose output feeds a
        # benchmark metric (spec §16): that output must be reproducible
        # from this file, which a sampled decode is not.
        raise ValueError(
            f"models.lock role {role!r} sets generation.do_sample to "
            f"{generation['do_sample']!r}: a lock may not enable sampling for a "
            "metric-producing role"
        )
    return generation


def _validate_entry(role: str, raw: dict) -> ModelPin:
    model_id = _require_nonblank_str(role, raw, "model_id")
    revision = _require_nonblank_str(role, raw, "revision")
    role_note = _require_nonblank_str(role, raw, "role_note")

    # UNRESOLVED is not validated as a SHA -- it is the documented
    # placeholder, not a malformed one. Anything else that isn't a 40-char
    # lowercase hex string is a hand-edit mistake (truncated SHA, branch
    # name, wrong case) and must fail loudly rather than pin the wrong
    # commit silently.
    if revision != UNRESOLVED and not _is_valid_sha(revision):
        raise ValueError(
            f"models.lock role {role!r} has an invalid revision {revision!r}: "
            f"expected {UNRESOLVED!r} or a {_SHA_LENGTH}-char lowercase hex SHA"
        )

    generation = _validated_generation(role, raw)

    return ModelPin(
        role=role,
        model_id=model_id,
        revision=revision,
        dtype=raw.get("dtype"),
        max_new_tokens=raw.get("max_new_tokens"),
        generation=generation,
        role_note=role_note,
    )


def load_model_pins(path: Path = MODEL_LOCK_PATH) -> dict[str, ModelPin]:
    """Parse and validate every role entry in a models.lock file.

    Tolerates `revision: UNRESOLVED` -- inspecting the whole file (e.g. to
    confirm every role is present) must stay possible before any checkpoint
    has been resolved. Raises ValueError for a structurally bad entry: a
    missing required key, a revision that is neither UNRESOLVED nor a valid
    SHA, or sampling enabled on a generative role.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {role: _validate_entry(role, entry) for role, entry in raw.items()}


def model_pin(role: str, path: Path = MODEL_LOCK_PATH) -> ModelPin:
    """Look up one role's pin, refusing to hand out an unresolved checkpoint.

    Unlike `load_model_pins`, this is the use-site accessor: any code about
    to actually load a checkpoint calls this, not `load_model_pins`
    directly, so an unresolved pin fails at the call site instead of
    surfacing later as a confusing download of "UNRESOLVED" as a model id.
    """
    pins = load_model_pins(path)
    if role not in pins:
        known = ", ".join(sorted(pins))
        raise KeyError(f"no models.lock entry for role {role!r}; known roles: {known}")

    pin = pins[role]
    if pin.revision == UNRESOLVED:
        raise RuntimeError(
            f"models.lock role {role!r} has revision {UNRESOLVED} -- resolve it with "
            f"scripts/resolve_model_revisions.py and commit the resolved SHA before "
            f"using this pin (file: {path})"
        )
    return pin
