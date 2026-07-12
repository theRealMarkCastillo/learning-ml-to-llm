"""Tests for the Transformer implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def _resolve_transformer_path() -> Path:
    for candidate in [Path.cwd().resolve()] + list(Path.cwd().resolve().parents):
        candidate_path = (
            candidate / "projects" / "phase2_transformers"
            / "project12_transformer_architecture"
        )
        if (candidate_path / "transformer.py").exists():
            return candidate_path
    raise FileNotFoundError("Could not locate project12_transformer_architecture")


_proj_dir = _resolve_transformer_path()
_proj_str = str(_proj_dir)
if _proj_str not in sys.path:
    sys.path.insert(0, _proj_str)

from transformer import (
    FeedForward,
    MultiHeadAttention,
    PositionalEncoding,
    TransformerDecoder,
    TransformerBlock,
    scaled_dot_product_attention,
)


# ---------------------------------------------------------------------------
# scaled_dot_product_attention
# ---------------------------------------------------------------------------


class TestScaledDotProductAttention:
    """Test the core attention mechanism."""

    def test_basic_shape(self):
        batch, heads, seq_len, d_k = 2, 4, 10, 64
        Q = torch.randn(batch, heads, seq_len, d_k)
        K = torch.randn(batch, heads, seq_len, d_k)
        V = torch.randn(batch, heads, seq_len, d_k)
        output, weights = scaled_dot_product_attention(Q, K, V)
        assert output.shape == (batch, heads, seq_len, d_k)
        assert weights.shape == (batch, heads, seq_len, seq_len)

    def test_attention_weights_normalise(self):
        batch, heads, seq_len, d_k = 1, 2, 5, 32
        Q = torch.randn(batch, heads, seq_len, d_k)
        K = torch.randn(batch, heads, seq_len, d_k)
        V = torch.randn(batch, heads, seq_len, d_k)
        _, weights = scaled_dot_product_attention(Q, K, V)
        # Each row of attention weights sums to ~1.0
        np.testing.assert_allclose(
            weights.sum(dim=-1), torch.ones_like(weights[:, :, :, 0]), atol=1e-6
        )

    def test_causal_mask(self):
        batch, heads, seq_len, d_k = 1, 2, 6, 32
        Q = torch.randn(batch, heads, seq_len, d_k)
        K = torch.randn(batch, heads, seq_len, d_k)
        V = torch.randn(batch, heads, seq_len, d_k)
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        output, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        for i in range(seq_len):
            assert torch.allclose(
                weights[:, :, i, i + 1 :],
                torch.zeros_like(weights[:, :, i, i + 1 :]),
                atol=1e-3,
            )


# ---------------------------------------------------------------------------
# MultiHeadAttention
# ---------------------------------------------------------------------------


class TestMultiHeadAttention:
    """Test the multi-head attention module."""

    def test_output_shape(self):
        d_model, num_heads = 384, 6
        mha = MultiHeadAttention(d_model, num_heads)
        x = torch.randn(2, 10, d_model)
        output, _ = mha(x)
        assert output.shape == (2, 10, d_model)

    def test_invalid_heads_raises(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=100, num_heads=3)

    def test_parameter_count_reasonable(self):
        mha = MultiHeadAttention(d_model=64, num_heads=4)
        n_params = sum(p.numel() for p in mha.parameters())
        # 3×(64×64) for QKV + 64×64 for output (no bias) = 4×64²
        expected = 4 * (64 * 64 + 64)
        assert n_params == expected

    def test_self_attention_conserves_d_model(self):
        for d_model in [32, 64, 128, 256]:
            mha = MultiHeadAttention(d_model, num_heads=4)
            x = torch.randn(1, 5, d_model)
            output, _ = mha(x)
            assert output.shape[-1] == d_model


# ---------------------------------------------------------------------------
# FeedForward
# ---------------------------------------------------------------------------


class TestFeedForward:
    """Test the position-wise feed-forward module."""

    def test_output_shape(self):
        ff = FeedForward(d_model=64, d_ff=128)
        x = torch.randn(2, 5, 64)
        out = ff(x)
        assert out.shape == (2, 5, 64)

    def test_d_ff_can_differ(self):
        ff = FeedForward(d_model=32, d_ff=96)
        x = torch.randn(1, 3, 32)
        out = ff(x)
        assert out.shape == (1, 3, 32)


# ---------------------------------------------------------------------------
# PositionalEncoding
# ---------------------------------------------------------------------------


class TestPositionalEncoding:
    """Test sinusoidal positional encodings."""

    def test_output_shape(self):
        pe = PositionalEncoding(d_model=64, max_seq_len=100)
        x = torch.randn(2, 10, 64)
        out = pe(x)
        assert out.shape == (2, 10, 64)

    def test_adds_positional_info(self):
        pe = PositionalEncoding(d_model=16, max_seq_len=20)
        x = torch.zeros(1, 5, 16)
        out = pe(x)
        assert not torch.allclose(out, torch.zeros_like(out))


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------


class TestTransformerBlock:
    """Test a single transformer block."""

    def test_block_preserves_shape(self):
        block = TransformerBlock(d_model=64, num_heads=4, d_ff=128)
        x = torch.randn(2, 10, 64)
        out = block(x)
        assert out.shape == (2, 10, 64)

    def test_block_with_mask(self):
        block = TransformerBlock(d_model=64, num_heads=4, d_ff=128)
        x = torch.randn(1, 5, 64)
        out = block(x, mask=None)
        assert out.shape == (1, 5, 64)


# ---------------------------------------------------------------------------
# TransformerDecoder (full model)
# ---------------------------------------------------------------------------


class TestTransformerDecoder:
    """Test the complete transformer decoder."""

    def test_basic_forward(self):
        model = TransformerDecoder(
            vocab_size=100, d_model=64, num_heads=4,
            num_layers=2, d_ff=128, max_seq_len=20,
        )
        tokens = torch.randint(0, 100, (2, 10))
        logits = model(tokens)
        assert logits.shape == (2, 10, 100)

    def test_num_parameters_property(self):
        model = TransformerDecoder(vocab_size=50, d_model=32, num_heads=2, num_layers=1)
        assert model.num_parameters > 0

    def test_weight_sharing(self):
        model = TransformerDecoder(vocab_size=50, d_model=32, num_heads=2, num_layers=1)
        assert (
            model.output_proj.weight is model.token_embedding.weight
        ), "Output and embedding weights should be shared"

    def test_vocabulary_size_consistency(self):
        model = TransformerDecoder(
            vocab_size=1000, d_model=64, num_heads=4, num_layers=2,
        )
        tokens = torch.randint(0, 999, (1, 5))
        logits = model(tokens)
        assert logits.shape[-1] == 1000

    def test_different_sequence_lengths(self):
        model = TransformerDecoder(
            vocab_size=50, d_model=32, num_heads=2, num_layers=1,
            max_seq_len=30,
        )
        for seq_len in [1, 5, 15, 25]:
            tokens = torch.randint(0, 50, (1, seq_len))
            logits = model(tokens)
            assert logits.shape == (1, seq_len, 50)
