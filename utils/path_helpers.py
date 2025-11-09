"""
Utilities for resolving the repository root and ensuring it is on sys.path.
This removes any need for hard-coded absolute user paths in notebooks/scripts.
"""
from __future__ import annotations

import sys
from pathlib import Path
from functools import lru_cache
from typing import Iterable, Optional


@lru_cache(maxsize=1)
def find_repo_root(start: Optional[Path] = None, markers: Iterable[str] = ("requirements.txt", "README.md", ".git")) -> Optional[Path]:
    """Walk up from start (or CWD) to locate the repository root.

    We consider a directory a repo root if it contains any of the marker files/dirs.
    Returns the Path if found, else None.
    """
    if start is None:
        start = Path.cwd().resolve()
    else:
        start = Path(start).resolve()

    for candidate in [start] + list(start.parents):
        if any((candidate / m).exists() for m in markers):
            return candidate
    return None


def add_repo_root_to_sys_path(start: Optional[Path] = None) -> Optional[Path]:
    """Find the repo root and insert it at the front of sys.path if missing.

    Returns the repo root Path or None if not found.
    """
    root = find_repo_root(start=start)
    if root is not None:
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
    return root


__all__ = [
    "find_repo_root",
    "add_repo_root_to_sys_path",
]
