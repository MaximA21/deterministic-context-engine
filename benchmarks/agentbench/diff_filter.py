#!/usr/bin/env python3
"""Patch filtering compatible with AGENTbench's DiffFile.

Filters model patches to only include valid Python source changes,
excluding test files, hidden directories, and virtual environments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


DISALLOWED_DIRS = frozenset({
    ".env", ".nox", ".tox", ".venv", "__pypackages__",
    "site-packages", "dist-packages", "env", "venv",
    "conda", "miniconda", "pyenv", "poetry", "pipenv",
    "virtualenv", "examples", "docs", "sites",
})


@dataclass(frozen=True)
class FilteredPatch:
    original: str
    cleaned: str
    files_kept: tuple[str, ...]
    files_dropped: tuple[str, ...]


def filter_patch(patch: str) -> FilteredPatch:
    """Filter a git diff to keep only valid Python source changes.

    Matches AGENTbench evaluation rules:
    - Only .py files
    - No hidden directories
    - No venv/conda/etc directories
    - No root-level test files
    """
    if not patch or not patch.strip():
        return FilteredPatch(
            original=patch, cleaned="", files_kept=(), files_dropped=()
        )

    hunks = re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE)
    kept_hunks = []
    files_kept = []
    files_dropped = []

    for hunk in hunks:
        if not hunk.strip():
            continue

        # Extract file path from diff header
        match = re.search(r"^diff --git a/(.*?) b/(.*?)$", hunk, re.MULTILINE)
        if not match:
            continue

        filepath = match.group(2)

        if _should_keep(filepath):
            kept_hunks.append(hunk)
            files_kept.append(filepath)
        else:
            files_dropped.append(filepath)

    return FilteredPatch(
        original=patch,
        cleaned="".join(kept_hunks),
        files_kept=tuple(files_kept),
        files_dropped=tuple(files_dropped),
    )


def _should_keep(filepath: str) -> bool:
    """Check if a file should be included in the filtered patch."""
    # Must be Python
    if not filepath.endswith(".py"):
        return False

    parts = filepath.split("/")

    # No hidden directories
    if any(p.startswith(".") for p in parts[:-1]):
        return False

    # No disallowed directories
    for part in parts:
        for disallowed in DISALLOWED_DIRS:
            if part == disallowed or part.startswith(disallowed):
                return False

    # No root-level test files
    if len(parts) == 1 and parts[0].startswith("test"):
        return False

    return True
