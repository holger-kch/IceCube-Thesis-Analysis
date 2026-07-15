"""Transformer-Modell fuer Muon Direction Reconstruction.

Architektur angelehnt an iceaggr FlatTransformerV2 (github.com/timinar/iceaggr):
- RMSNorm (funktional, keine lernbaren Parameter)
- Multi-Head Self-Attention mit QK-Norm
- ReLU^2 FFN
- CLS-Token fuer Event-Aggregation
- Per-Layer Residual Scaling + x0 Skip Connection
- Zero-Init fuer Output-Projektionen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Bausteine
# ---------------------------------------------------------------------------

def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """Funktionale RMSNorm ohne lernbare Parameter."""
    return F.rms_norm(x, (x.size(-1),))


class Attention(nn.Module):
    """Multi-Head Self-Attention mit QK-Norm.

    - Alle Linear-Layer ohne Bias
    - Q und K werden vor dem Dot-Product mit RMSNorm normalisiert
    - Nutzt F.scaled_dot_product_attention (Flash Attention wenn verfuegbar)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model muss durch num_heads teilbar sein"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        self.c_q = nn.Linear(d_model, d_model, bias=False)
        self.c_k = nn.Linear(d_model, d_model, bias=False)
        self.c_v = nn.Linear(d_model, d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input, Shape (B, T, d_model)
            attn_mask: Bool-Mask (B, T), True = gueltig, False = Padding

        Returns:
            Output, Shape (B, T, d_model)
        """
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # QK-Norm fuer Training-Stabilitaet
        q = rms_norm(q)
        k = rms_norm(k)

        # Mask: (B, T) -> (B, 1, 1, T) fuer SDPA
        mask_4d = attn_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask_4d,
            dropout_p=self.dropout if self.training else 0.0,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class FFN(nn.Module):
    """Feed-Forward Network mit ReLU-squared Aktivierung.

    ReLU^2(x) = relu(x)^2 — sparsere Aktivierung als GELU,
    gute Ergebnisse in kleinen Transformer-Modellen.
    """

    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.c_fc = nn.Linear(d_model, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()  # ReLU^2
        return self.c_proj(x)


class Block(nn.Module):
    """Ein Transformer-Block: Pre-Norm Attention + Pre-Norm FFN."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = Attention(d_model, num_heads, dropout)
        self.ffn = FFN(d_model, ffn_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(rms_norm(x), attn_mask)
        x = x + self.ffn(rms_norm(x))
        return x


# ---------------------------------------------------------------------------
# Hauptmodell
# ---------------------------------------------------------------------------

class MuonTransformer(nn.Module):
    """Transformer fuer Muon Direction Reconstruction.

    Architektur-Ueberblick:
        1. Input-Projektion: dom_vectors (B, max_doms, input_dim) -> (B, max_doms, d_model)
        2. Post-Embedding RMSNorm
        3. CLS-Token prependen -> (B, max_doms+1, d_model)
        4. N Transformer-Blocks mit Residual Scaling + x0 Skip
        5. Final RMSNorm
        6. CLS-Token extrahieren -> (B, d_model)
        7. DirectionalHead -> (B, 3) Unit Vector

    Args:
        input_dim: Dimension der DOM-Vektoren (z.B. 4 + 3*K = 52 fuer K=16)
        d_model: Transformer Hidden-Dimension
        num_layers: Anzahl Transformer-Blocks
        num_heads: Anzahl Attention-Heads
        ffn_dim: Hidden-Dimension im FFN
        head_hidden_dim: Hidden-Dimension im DirectionalHead
        input_mode: "linear" oder "mlp" fuer die Input-Projektion
        dropout: Dropout-Rate in Attention
    """

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
        head: nn.Module | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        # --- Input Projection ---
        if input_mode == "linear":
            self.input_proj = nn.Linear(input_dim, d_model, bias=False)
        elif input_mode == "mlp":
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, d_model, bias=False),
                nn.GELU(),
                nn.Linear(d_model, d_model, bias=False),
            )
        else:
            raise ValueError(f"Unbekannter input_mode: {input_mode}")

        # --- CLS Token ---
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))

        # --- Transformer Blocks ---
        self.blocks = nn.ModuleList([
            Block(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # --- Residual Scaling + x0 Skip (lernbar, ein Wert pro Layer) ---
        self.resid_lambdas = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for _ in range(num_layers)
        ])
        self.x0_lambdas = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1)) for _ in range(num_layers)
        ])

        # --- Prediction Head ---
        if head is not None:
            self.head = head
        else:
            from .directional_head import DirectionalHead
            self.head = DirectionalHead(embed_dim=d_model, hidden_dim=head_hidden_dim)

        # --- Initialisierung ---
        self._init_weights()

    def _init_weights(self):
        """Nanochat-style Initialisierung: uniform fuer QKV/FC, zero fuer Output-Proj."""
        s = 3**0.5 * self.d_model**-0.5

        # CLS Token
        nn.init.normal_(self.cls_token, std=0.02)

        # Transformer Blocks
        for block in self.blocks:
            nn.init.uniform_(block.attn.c_q.weight, -s, s)
            nn.init.uniform_(block.attn.c_k.weight, -s, s)
            nn.init.uniform_(block.attn.c_v.weight, -s, s)
            nn.init.zeros_(block.attn.c_proj.weight)  # Zero-Init
            nn.init.uniform_(block.ffn.c_fc.weight, -s, s)
            nn.init.zeros_(block.ffn.c_proj.weight)   # Zero-Init

    def forward(
        self,
        dom_vectors: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            dom_vectors: DOM-Feature-Vektoren, Shape (B, max_doms, input_dim)
            padding_mask: Bool-Mask (B, max_doms), True = gueltiger DOM

        Returns:
            Predicted unit vectors, Shape (B, 3)
        """
        B = dom_vectors.size(0)

        # 1. Input Projection
        x = self.input_proj(dom_vectors)  # (B, max_doms, d_model)

        # 2. Post-Embedding RMSNorm
        x = rms_norm(x)

        # 3. CLS-Token prependen
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)          # (B, 1 + max_doms, d_model)

        # Mask erweitern: CLS ist immer gueltig
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, padding_mask], dim=1)  # (B, 1 + max_doms)

        # 4. x0 speichern fuer Skip Connection
        x0 = x

        # 5. Transformer Blocks mit Residual Scaling + x0 Skip
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = block(x, full_mask)

        # 6. Final RMSNorm
        x = rms_norm(x)

        # 7. CLS-Token extrahieren
        cls_output = x[:, 0, :]  # (B, d_model)

        # 8. Prediction Head
        return self.head(cls_output)  # (B, 3)
