"""Shared fixtures for ML learning repository tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


# Ensure repo root is always on sys.path regardless of invocation directory
def _ensure_repo_on_path() -> Path | None:
    """Walk up from CWD to find the repo root and add it to sys.path."""
    for candidate in [Path.cwd().resolve()] + list(Path.cwd().resolve().parents):
        if (candidate / "requirements.txt").exists():
            root_str = str(candidate)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            return candidate
    return None


_repo_root = _ensure_repo_on_path()


@pytest.fixture(autouse=True)
def repo_root():
    """Expose the repository root path to tests."""
    return _repo_root
