"""
Transformer Architecture — PyTorch implementation from scratch.

Implements a decoder-only transformer (like GPT) with:
- Multi-head self-attention
- Position-wise feed-forward (GELU activation)
- Layer normalisation + residual connections
- Sinusoidal positional encodings
- Causal (autoregressive) masking

This module is extracted from the Jupyter notebook so that:
- Notebooks can import it cleanly
- Automated tests can exercise the implementation
- The core logic is portable outside Jupyter

Usage
-----
    from project12_transformer_architecture.transformer import (
        TransformerDecoder,
    )

    model = TransformerDecoder(
        vocab_size=tokenizer.vocab_size,
        d_model=384,
        num_heads=6,
        num_layers=4,
        d_ff=1536,
        max_seq_len=256,
    )
    logits = model(tokens)  # (batch, seq_len, vocab_size)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Scaled Dot-Product Attention
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V.

    Parameters
    ----------
    query, key, value : (batch, num_heads, seq_len, d_k)
    mask : (batch, 1, 1, seq_len) or broadcastable — 1 where attention
        is allowed, 0 where it is blocked.

    Returns
    -------
    output : (batch, num_heads, seq_len, d_k)
    attn_weights : (batch, num_heads, seq_len, seq_len) — softmax-ed
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)

    return output, attn_weights


# ---------------------------------------------------------------------------
# Multi-Head Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """

        Parameters
        ----------
        x : (batch, seq_len, d_model)
        mask : optional causal mask

        Returns
        -------
        output : (batch, seq_len, d_model)
        attn_weights : (batch, num_heads, seq_len, seq_len) or None (if caller
            discards it)
        """
        batch_size = x.size(0)

        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        return self.W_o(attn_output), attn_weights


# ---------------------------------------------------------------------------
# Feed-Forward Network (GELU variant of the post-Llama choice)
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Position-wise feed-forward: linear → GELU → dropout → linear."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


# ---------------------------------------------------------------------------
# Positional Encoding (sinusoidal)
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encodings (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, seq_len, d_model)."""
        return self.dropout(x + self.pe[:, : x.size(1), :])


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Single transformer block: MHA → FFN with residual + layer norm."""

    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention residual
        a, _ = self.attn(self.norm1(x), mask)
        x = x + self.dropout(a)
        # FFN residual
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Complete Transformer Decoder
# ---------------------------------------------------------------------------

class TransformerDecoder(nn.Module):
    """Decoder-only transformer for causal language modelling.

    Parameters
    ----------
    vocab_size : int — number of unique tokens
    d_model : int — embedding dimension
    num_heads : int — attention heads
    num_layers : int — transformer blocks
    d_ff : int — feed-forward inner dimension (typically 4× d_model)
    max_seq_len : int — maximum sequence length
    dropout : float
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        num_heads: int = 6,
        num_layers: int = 4,
        d_ff: int = 1536,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        # Share weights between token embedding and output projection
        self.output_proj.weight = self.token_embedding.weight

        # Build causal mask (seq_len × seq_len)
        self.register_buffer(
            "_causal_mask",
            self._build_causal_mask(max_seq_len),
            persistent=False,
        )

    @staticmethod
    def _build_causal_mask(seq_len: int) -> torch.Tensor:
        """Lower-triangular mask: 1 on and below the diagonal."""
        return torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

    def forward(
        self, tokens: torch.Tensor
    ) -> torch.Tensor:
        """

        Parameters
        ----------
        tokens : (batch, seq_len) — token indices

        Returns
        -------
        logits : (batch, seq_len, vocab_size)
        """
        batch, seq_len = tokens.shape

        x = self.token_embedding(tokens) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Expand causal mask to batch dimension
        mask = self._causal_mask[:, :, :seq_len, :seq_len]

        for block in self.blocks:
            x = block(x, mask)

        x = self.norm_out(x)
        logits = self.output_proj(x)
        return logits

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
