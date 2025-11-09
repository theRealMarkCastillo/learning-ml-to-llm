#!/usr/bin/env python3
"""Quick backend verification script.

Runs a tiny op on the detected backend and prints diagnostics.
Usage: python scripts/verify_device.py
Optionally set LEARNING_ML_BACKEND=mlx|cuda|mps|cpu to override.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure repo root on sys.path
cur = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(cur))

from utils.device import backend_info, get_backend, backend_name, tensor, ensure_seed  # noqa: E402


def main():
    print(f"Detected: {backend_info()}")
    ensure_seed(123)

    bname = backend_name()
    try:
        if bname == "mlx":
            import mlx.core as mx  # type: ignore
            a = tensor([[1.0, 2.0], [3.0, 4.0]])
            b = tensor([[5.0, 6.0], [7.0, 8.0]])
            c = a @ b
            mx.eval(c)
            print("MLX matmul result:", c)
        else:
            import torch  # type: ignore
            a = tensor([[1.0, 2.0], [3.0, 4.0]])
            b = tensor([[5.0, 6.0], [7.0, 8.0]])
            c = a @ b
            print("Torch matmul result:", c, "on", c.device)
    except Exception as e:
        print("Error running tiny op:", repr(e))
        sys.exit(1)

    print("OK")


if __name__ == "__main__":
    main()
