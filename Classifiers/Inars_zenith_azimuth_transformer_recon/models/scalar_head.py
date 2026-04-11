"""Prediction Head fuer skalare Targets (Energy, Position Z).

Nimmt das Event-Embedding (CLS-Token Output) und predicted
einen einzelnen Skalar-Wert. Keine L2-Normalisierung.
"""

import torch
import torch.nn as nn


class ScalarHead(nn.Module):
    """Deep MLP for scalar regression.

    Architektur:
        Linear(embed_dim, hidden_dim) -> GELU -> Dropout
        -> Linear(hidden_dim, hidden_dim) -> GELU -> Dropout
        -> Linear(hidden_dim, 1)

    Deeper than DirectionalHead to compensate for lack of geometric constraints.
    """

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256,
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

        # Xavier init for last layer
        nn.init.xavier_normal_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, event_embedding: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            event_embedding: CLS-Token Output, Shape (B, embed_dim)

        Returns:
            Scalar predictions, Shape (B, 1)
        """
        return self.net(event_embedding)
