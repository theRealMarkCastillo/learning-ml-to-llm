"""Tests for the device/backend abstraction module."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_repo_root = None
for candidate in [Path.cwd().resolve()] + list(Path.cwd().resolve().parents):
    if (candidate / "requirements.txt").exists():
        _repo_root = candidate
        if str(_repo_root) not in sys.path:
            sys.path.insert(0, str(_repo_root))
        break

import pytest

from utils.device import (
    Backend,
    BackendDetails,
    _detect_backend,
    _env_override,
    _materialize,
    backend_info,
    backend_name,
    get_backend,
    get_device,
    move_to,
    tensor,
)


# ---------------------------------------------------------------------------
# Environment override
# ---------------------------------------------------------------------------


class TestEnvOverride:
    """Test the LEARNING_ML_BACKEND env-override logic."""

    def test_no_env_returns_none(self, monkeypatch):
        monkeypatch.delenv("LEARNING_ML_BACKEND", raising=False)
        assert _env_override() is None

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("mlx", Backend.MLX),
            ("cuda", Backend.TORCH_CUDA),
            ("mps", Backend.TORCH_MPS),
            ("cpu", Backend.CPU),
            ("MLX", Backend.MLX),  # case-insensitive
            ("CUDA", Backend.TORCH_CUDA),
        ],
    )
    def test_valid_overrides(self, monkeypatch, raw, expected):
        monkeypatch.setenv("LEARNING_ML_BACKEND", raw)
        assert _env_override() == expected

    def test_invalid_returns_none(self, monkeypatch):
        monkeypatch.setenv("LEARNING_ML_BACKEND", "invalid_backend")
        assert _env_override() is None


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


class TestBackendDetection:
    """Test automatic backend detection."""

    def test_backend_name_returns_string(self):
        name = backend_name()
        assert isinstance(name, str)
        assert name in ("mlx", "torch_cuda", "torch_mps", "cpu")

    def test_backend_info_returns_string(self):
        info = backend_info()
        assert isinstance(info, str)
        assert "Backend=" in info

    def test_get_backend_cached(self):
        """Second call should return the same cached result."""
        b1 = get_backend()
        b2 = get_backend()
        assert b1 is b2  # Same enum instance (cached)

    def test_get_device_returns_value(self):
        dev = get_device()
        # Should be either a torch.device, MLX handle, or "cpu"
        assert dev is not None


# ---------------------------------------------------------------------------
# tensor and move_to helpers
# ---------------------------------------------------------------------------


class TestTensorMoveTo:
    """Test tensor creation and device moving."""

    def test_tensor_basic_list(self):
        t = tensor([[1.0, 2.0], [3.0, 4.0]])
        assert t is not None

    def test_tensor_array(self):
        t = tensor([1, 2, 3])
        assert t is not None

    def test_move_to_noop_for_cpu_backends(self):
        bname = backend_name()
        if bname == "cpu":
            obj = {"key": "value"}
            assert move_to(obj) is obj  # Passthrough

    def test_move_to_returns_obj_if_no_method(self):
        obj = object()
        assert move_to(obj) is obj  # No .to() method → passthrough


# ---------------------------------------------------------------------------
# ensure_seed (integrated test)
# ---------------------------------------------------------------------------


class TestEnsureSeed:
    """Test reproducibility seeding."""

    def test_seed_np_consistency(self, monkeypatch):
        """Seeding should make np.random deterministic."""
        from utils.device import ensure_seed

        ensure_seed(42)
        r1 = __import__("numpy").random.randn(3)
        ensure_seed(42)
        r2 = __import__("numpy").random.randn(3)
        __import__("numpy").testing.assert_array_equal(r1, r2)
