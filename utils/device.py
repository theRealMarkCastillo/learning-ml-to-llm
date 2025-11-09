"""Device and backend auto-detection utility.

This module provides a unified way to select the best available compute backend
across different platforms:

Priority order:
1. MLX (Apple Silicon, "mlx") if import succeeds.
2. PyTorch CUDA if torch.cuda.is_available().
3. PyTorch MPS if torch.backends.mps.is_available().
4. CPU fallback.

It exposes:
- get_backend(): returns a Backend enum instance (MLX, TORCH_CUDA, TORCH_MPS, CPU)
- get_device(): returns a torch.device or string identifier ("cpu") or MLX device handle
- tensor(x, **kwargs): convenience to create a tensor on the selected backend (torch or mlx)
- move_to(x): moves an existing model/tensor to the detected device when supported
- backend_name(): short string for logging
- backend_info(): verbose diagnostic string

If MLX is selected, torch helpers will raise if called. Conversely if a torch
backend is selected MLX helpers are no-ops.

Override:
You can override auto-detection by setting the environment variable
LEARNING_ML_BACKEND to one of: "mlx", "cuda", "mps", "cpu".

This keeps notebooks simple: from utils.device import get_device, tensor

"""
from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from enum import Enum
from typing import Any, Optional

# Internal state caches
__BACKEND: Optional["Backend"] = None
__DEVICE: Any = None

class Backend(Enum):
    MLX = "mlx"
    TORCH_CUDA = "torch_cuda"
    TORCH_MPS = "torch_mps"
    CPU = "cpu"

@dataclass(frozen=True)
class BackendDetails:
    backend: Backend
    device_repr: str
    extra: str = ""


def _env_override() -> Optional[Backend]:
    raw = os.getenv("LEARNING_ML_BACKEND")
    if not raw:
        return None
    raw_lower = raw.lower()
    mapping = {
        "mlx": Backend.MLX,
        "cuda": Backend.TORCH_CUDA,
        "mps": Backend.TORCH_MPS,
        "cpu": Backend.CPU,
    }
    return mapping.get(raw_lower)


def _detect_backend() -> BackendDetails:
    # Env override first
    override = _env_override()
    if override:
        return _materialize(override)

    # Try MLX first (Apple Silicon)
    try:
        import mlx.core as mx  # type: ignore
        dev = mx.gpu if hasattr(mx, "gpu") else None
        return BackendDetails(Backend.MLX, f"MLX({dev})", extra=f"mx.version={getattr(mx, '__version__', 'unknown')}")
    except Exception:
        pass

    # Try torch variants
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return BackendDetails(Backend.TORCH_CUDA, f"cuda:{torch.cuda.current_device()}", extra=f"torch.version={torch.__version__}")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return BackendDetails(Backend.TORCH_MPS, "mps", extra=f"torch.version={torch.__version__}")
        # Fallback CPU torch
        return BackendDetails(Backend.CPU, "cpu", extra=f"torch.version={torch.__version__}")
    except Exception:
        # Pure CPU (no torch / no mlx)
        return BackendDetails(Backend.CPU, "cpu", extra="no torch/mlx")


def _materialize(backend: Backend) -> BackendDetails:
    if backend == Backend.MLX:
        try:
            import mlx.core as mx  # type: ignore
            dev = mx.gpu if hasattr(mx, "gpu") else None
            return BackendDetails(Backend.MLX, f"MLX({dev})", extra=f"mx.version={getattr(mx, '__version__', 'unknown')}")
        except Exception:
            # Fallback cascade
            return _materialize(Backend.CPU)
    if backend in (Backend.TORCH_CUDA, Backend.TORCH_MPS, Backend.CPU):
        try:
            import torch  # type: ignore
            if backend == Backend.TORCH_CUDA and torch.cuda.is_available():
                return BackendDetails(Backend.TORCH_CUDA, f"cuda:{torch.cuda.current_device()}", extra=f"torch.version={torch.__version__}")
            if backend == Backend.TORCH_MPS and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return BackendDetails(Backend.TORCH_MPS, "mps", extra=f"torch.version={torch.__version__}")
            if backend == Backend.CPU:
                return BackendDetails(Backend.CPU, "cpu", extra=f"torch.version={torch.__version__}")
        except Exception:
            pass
    # Final fallback
    return BackendDetails(Backend.CPU, "cpu", extra="no torch/mlx")


def get_backend() -> Backend:
    """Return detected backend (cached)."""
    global __BACKEND, __DEVICE
    if __BACKEND is None:
        details = _detect_backend()
        __BACKEND = details.backend
        __DEVICE = _init_device(__BACKEND)
    return __BACKEND


def _init_device(backend: Backend):
    if backend == Backend.MLX:
        try:
            import mlx.core as mx  # type: ignore
            return mx.gpu if hasattr(mx, "gpu") else None
        except Exception:
            return None
    if backend in (Backend.TORCH_CUDA, Backend.TORCH_MPS, Backend.CPU):
        try:
            import torch  # type: ignore
            if backend == Backend.TORCH_CUDA:
                return torch.device("cuda")
            if backend == Backend.TORCH_MPS:
                return torch.device("mps")
            return torch.device("cpu")
        except Exception:
            return "cpu"
    return "cpu"


def get_device():
    """Return a device object / identifier suitable for tensor/model placement."""
    get_backend()  # Ensure initialized
    return __DEVICE


def backend_name() -> str:
    return get_backend().value


def backend_info() -> str:
    b = get_backend()
    if b == Backend.MLX:
        try:
            import mlx.core as mx  # type: ignore
            return f"Backend=MLX version={getattr(mx, '__version__', 'unknown')} device={get_device()}"
        except Exception:
            return "Backend=MLX (unavailable after initial detection)"
    if b in (Backend.TORCH_CUDA, Backend.TORCH_MPS, Backend.CPU):
        try:
            import torch  # type: ignore
            return f"Backend={b.value} torch.version={torch.__version__} device={get_device()}"
        except Exception:
            return f"Backend={b.value} (torch import failed)"
    return f"Backend={b.value}"


def tensor(data, **kwargs):
    """Create a tensor on the active backend.

    For torch backends returns torch.Tensor.
    For MLX returns mlx.core.array.
    CPU fallback returns torch tensor if torch is present, else raises if MLX not present.
    """
    b = get_backend()
    if b == Backend.MLX:
        import mlx.core as mx  # type: ignore
        return mx.array(data, **kwargs)
    try:
        import torch  # type: ignore
        device = get_device()
        return torch.as_tensor(data, **kwargs).to(device)
    except Exception as e:
        raise RuntimeError("No tensor backend available (torch/mlx missing)") from e


def move_to(obj):
    """Move model or tensor to detected device if supported.

    - torch.nn.Module / torch.Tensor: calls .to(device)
    - MLX arrays are returned unchanged (already on device)
    - other types are passthrough
    """
    b = get_backend()
    if b == Backend.MLX:
        return obj  # MLX arrays are implicitly device-bound
    try:
        import torch  # type: ignore
        device = get_device()
        if hasattr(obj, "to"):
            return obj.to(device)
        return obj
    except Exception:
        return obj


def ensure_seed(seed: int = 42):
    """Set seeds for available backends to encourage reproducibility."""
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        pass
    b = get_backend()
    if b == Backend.MLX:
        try:
            import mlx.core as mx  # type: ignore
            mx.random.seed(seed)
        except Exception:
            pass
    else:
        try:
            import torch  # type: ignore
            torch.manual_seed(seed)
            if b == Backend.TORCH_CUDA and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

# Eager detection so early logs can use backend_info()
_get = get_backend()
if os.getenv("LEARNING_ML_VERBOSE"):
    print(f"[device] {backend_info()}")
