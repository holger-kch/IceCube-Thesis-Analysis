"""Transformer for IceCube MC-vs-data and HLC classification.

Architecture adapted from the MuonTransformer in
``Inars_zenith_azimuth_transformer_recon/models/transformer.py``
(itself inspired by iceaggr's FlatTransformerV2):

  - RMSNorm everywhere (functional, no learnable params)
  - Multi-Head Self-Attention with QK-norm
  - ReLU² FFN
  - Pre-Norm residuals + per-layer learnable residual scaling + x0 skip
  - Zero-init for output projections (stable training)
  - CLS token for event-level pooling (BERT-style)
  - Permutation invariant (no positional encoding) — pulses are unordered

Two output modes:
  - ``event``: returns one logit per event (CLS embedding → ScalarHead)
  - ``pulse``: returns one logit per pulse (per-pulse hidden states → linear)

Designed as a drop-in replacement for the DynEdge GNN models in this
project. Same input features, same train/val/test split logic, same
weighted BCE loss.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.c_q = nn.Linear(d_model, d_model, bias=False)
        self.c_k = nn.Linear(d_model, d_model, bias=False)
        self.c_v = nn.Linear(d_model, d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = rms_norm(q)
        k = rms_norm(k)
        mask_4d = attn_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask_4d,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class FFN(nn.Module):
    """ReLU² FFN — sparser activation than GELU, works well in small models."""

    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.c_fc = nn.Linear(d_model, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        self.attn = Attention(d_model, num_heads, dropout)
        self.ffn = FFN(d_model, ffn_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(rms_norm(x), attn_mask)
        x = x + self.ffn(rms_norm(x))
        return x


class ScalarHead(nn.Module):
    """Deep MLP that maps embedding → 1 logit. Used for event-mode."""

    def __init__(self, embed_dim: int, hidden_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.xavier_normal_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PulseTransformer(nn.Module):
    """Permutation-invariant transformer over pulses.

    Forward inputs:
        dom_vectors: (B, T, F) — F input features per pulse
        padding_mask: (B, T) — bool, True for valid pulse, False for padding

    Forward output:
        mode == "event" → (B,) logits, one per event (CLS-pooled)
        mode == "pulse" → (B, T) logits, one per pulse (caller masks padding)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 384,
        head_hidden_dim: int = 256,
        dropout: float = 0.05,
        mode: str = "event",
        input_mode: str = "mlp",
    ):
        super().__init__()
        if mode not in ("event", "pulse"):
            raise ValueError(f"mode must be 'event' or 'pulse', got {mode}")
        self.mode = mode
        self.d_model = d_model
        self.num_layers = num_layers

        if input_mode == "linear":
            self.input_proj = nn.Linear(input_dim, d_model, bias=False)
        elif input_mode == "mlp":
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, d_model, bias=False),
                nn.GELU(),
                nn.Linear(d_model, d_model, bias=False),
            )
        else:
            raise ValueError(f"unknown input_mode: {input_mode}")

        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))

        self.blocks = nn.ModuleList([
            Block(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # Per-layer learnable residual + x0 skip scalars (stable deep transformers)
        self.resid_lambdas = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for _ in range(num_layers)
        ])
        self.x0_lambdas = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1)) for _ in range(num_layers)
        ])

        if mode == "event":
            self.head = ScalarHead(d_model, head_hidden_dim, dropout)
        else:
            # Per-pulse head: small MLP applied position-wise
            self.head = nn.Sequential(
                nn.Linear(d_model, head_hidden_dim, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, 1, bias=True),
            )
            nn.init.xavier_normal_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

        self._init_weights()

    def _init_weights(self):
        s = math.sqrt(3.0 / self.d_model)
        nn.init.normal_(self.cls_token, std=0.02)
        for block in self.blocks:
            nn.init.uniform_(block.attn.c_q.weight, -s, s)
            nn.init.uniform_(block.attn.c_k.weight, -s, s)
            nn.init.uniform_(block.attn.c_v.weight, -s, s)
            nn.init.zeros_(block.attn.c_proj.weight)
            nn.init.uniform_(block.ffn.c_fc.weight, -s, s)
            nn.init.zeros_(block.ffn.c_proj.weight)

    def forward(
        self, dom_vectors: torch.Tensor, padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        B = dom_vectors.size(0)

        x = self.input_proj(dom_vectors)
        x = rms_norm(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, padding_mask], dim=1)

        x0 = x
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = block(x, full_mask)

        x = rms_norm(x)

        if self.mode == "event":
            return self.head(x[:, 0, :])  # (B,)
        # Pulse mode: drop CLS, apply head per pulse
        per_pulse = x[:, 1:, :]  # (B, T, d_model)
        return self.head(per_pulse).squeeze(-1)  # (B, T)
