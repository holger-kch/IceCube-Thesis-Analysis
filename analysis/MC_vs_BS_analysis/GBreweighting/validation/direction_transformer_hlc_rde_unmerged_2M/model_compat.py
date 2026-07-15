"""Python-3.9-compatible copy of Inar's direction transformer components."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.c_q = nn.Linear(d_model, d_model, bias=False)
        self.c_k = nn.Linear(d_model, d_model, bias=False)
        self.c_v = nn.Linear(d_model, d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        b, t, c = x.size()
        q = self.c_q(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        q = rms_norm(q)
        k = rms_norm(k)
        mask_4d = attn_mask.unsqueeze(1).unsqueeze(2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask_4d,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.c_proj(y)


class FFN(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.c_fc = nn.Linear(d_model, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        return self.c_proj(F.relu(x).square())


class Block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = Attention(d_model, num_heads, dropout)
        self.ffn = FFN(d_model, ffn_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(rms_norm(x), attn_mask)
        x = x + self.ffn(rms_norm(x))
        return x


class DirectionalHead(nn.Module):
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, 3)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, event_embedding: torch.Tensor) -> torch.Tensor:
        x = self.fc1(event_embedding)
        x = self.activation(x)
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=1)


class MuonTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 256,
        head_hidden_dim: int = 128,
        input_mode: str = "linear",
        dropout: float = 0.0,
    ):
        super().__init__()
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
        self.blocks = nn.ModuleList(
            [Block(d_model, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.resid_lambdas = nn.ParameterList(
            [nn.Parameter(torch.tensor(1.0)) for _ in range(num_layers)]
        )
        self.x0_lambdas = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.1)) for _ in range(num_layers)]
        )
        self.head = DirectionalHead(embed_dim=d_model, hidden_dim=head_hidden_dim)
        self._init_weights()

    def _init_weights(self):
        s = 3**0.5 * self.d_model**-0.5
        nn.init.normal_(self.cls_token, std=0.02)
        for block in self.blocks:
            nn.init.uniform_(block.attn.c_q.weight, -s, s)
            nn.init.uniform_(block.attn.c_k.weight, -s, s)
            nn.init.uniform_(block.attn.c_v.weight, -s, s)
            nn.init.zeros_(block.attn.c_proj.weight)
            nn.init.uniform_(block.ffn.c_fc.weight, -s, s)
            nn.init.zeros_(block.ffn.c_proj.weight)

    def forward(self, dom_vectors: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        b = dom_vectors.size(0)
        x = self.input_proj(dom_vectors)
        x = rms_norm(x)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        cls_mask = torch.ones(b, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, padding_mask], dim=1)
        x0 = x
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = block(x, full_mask)
        x = rms_norm(x)
        return self.head(x[:, 0, :])


def angular_distance(
    pred_vectors: torch.Tensor,
    true_vectors: torch.Tensor,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    dot = torch.sum(pred_vectors * true_vectors, dim=1)
    dot = torch.clamp(dot, -1.0 + epsilon, 1.0 - epsilon)
    return torch.arccos(dot)
