"""Shared fixtures for ML learning repository tests.

Sets up ``sys.path`` once so tests can ``from utils...`` and import
extracted project modules. We inline the path-walk here (rather than
calling ``utils.path_helpers.add_repo_root_to_sys_path``) because
``utils`` is not yet importable until ``sys.path`` is set up.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _resolve_repo_root() -> Path | None:
    for candidate in [Path.cwd().resolve()] + list(Path.cwd().resolve().parents):
        if (candidate / "requirements.txt").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    return None


_repo_root = _resolve_repo_root()


@pytest.fixture
def repo_root() -> Path | None:
    """Expose the repository root path to tests that need it."""
    return _repo_root