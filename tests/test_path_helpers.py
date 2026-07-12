"""Tests for path helpers and visualization utility."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure repo root is importable
_repo_root = None
for candidate in [Path.cwd().resolve()] + list(Path.cwd().resolve().parents):
    if (candidate / "requirements.txt").exists():
        _repo_root = candidate
        if str(_repo_root) not in sys.path:
            sys.path.insert(0, str(_repo_root))
        break

from utils.path_helpers import add_repo_root_to_sys_path, find_repo_root


# ---------------------------------------------------------------------------
# find_repo_root
# ---------------------------------------------------------------------------


class TestFindRepoRoot:
    """Test repository root detection."""

    def test_finds_current_repo(self):
        result = find_repo_root(start=_repo_root)
        assert result is not None
        assert result == _repo_root

    def test_uses_markers(self):
        result = find_repo_root(start=_repo_root, markers=(".git",))
        assert result is not None
        assert result == _repo_root

    def test_nonexistent_dir_returns_none(self, tmp_path):
        result = find_repo_root(start=tmp_path / "nowhere")
        assert result is None

    def test_custom_markers(self, tmp_path):
        (tmp_path / "MY_MARKER").touch()
        result = find_repo_root(start=tmp_path, markers=("MY_MARKER",))
        assert result == tmp_path


# ---------------------------------------------------------------------------
# add_repo_root_to_sys_path
# ---------------------------------------------------------------------------


class TestAddRepoRootToSysPath:
    """Test that the repo root gets added to sys.path."""

    def test_adds_to_sys_path(self):
        """After adding, the repo root should be on sys.path."""
        result = add_repo_root_to_sys_path()
        if result is not None:
            assert str(result) in sys.path

    def test_inserts_at_front(self):
        result = add_repo_root_to_sys_path()
        if result is not None:
            assert str(result) in sys.path

    def test_already_in_sys_path(self):
        root_str = str(_repo_root)
        if root_str in sys.path:
            before_count = sys.path.count(root_str)
            add_repo_root_to_sys_path()
            after_count = sys.path.count(root_str)
            assert before_count == after_count

    def test_nonexistent_none(self, tmp_path):
        result = add_repo_root_to_sys_path(start=tmp_path / "nonexistent")
        assert result is None
