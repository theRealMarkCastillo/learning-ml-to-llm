"""Tests for path helpers and visualization utility."""

from __future__ import annotations

import sys

import pytest

from utils.path_helpers import add_repo_root_to_sys_path, find_repo_root


@pytest.fixture
def _repo_root():
    """Resolve repo root once per test (re-uses the helper under test indirectly)."""
    return add_repo_root_to_sys_path()


# ---------------------------------------------------------------------------
# find_repo_root
# ---------------------------------------------------------------------------


class TestFindRepoRoot:
    """Test repository root detection."""

    def test_finds_current_repo(self, _repo_root):
        result = find_repo_root(start=_repo_root)
        assert result is not None
        assert result == _repo_root

    def test_uses_markers(self, _repo_root):
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
        """After calling, the repo root should be at sys.path[0]."""
        result = add_repo_root_to_sys_path()
        if result is not None:
            assert sys.path[0] == str(result)

    def test_no_duplicates(self):
        """Calling repeatedly should not duplicate the entry."""
        result = add_repo_root_to_sys_path()
        if result is not None:
            before_count = sys.path.count(str(result))
            add_repo_root_to_sys_path()
            after_count = sys.path.count(str(result))
            assert before_count == after_count

    def test_nonexistent_none(self, tmp_path):
        result = add_repo_root_to_sys_path(start=tmp_path / "nonexistent")
        assert result is None
