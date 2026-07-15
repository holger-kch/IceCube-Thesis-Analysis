"""Prediction Head fuer Muon-Richtung als 3D Unit Vector.

Nimmt das Event-Embedding (CLS-Token Output) und predicted
einen normalisierten 3D-Richtungsvektor. Architektur aus iceaggr.
"""

import torch
import torch.nn as nn


class DirectionalHead(nn.Module):
    """MLP: embed_dim -> hidden -> 3D unit vector.

    Architektur:
        Linear(embed_dim, hidden_dim) -> GELU -> Linear(hidden_dim, 3) -> L2-Norm

    Der Output ist immer ein normalisierter Einheitsvektor auf der Kugel.
    """

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, 3)

        # Xavier-Init fuer letzte Schicht (std=0.01 fuehrt zu Mode Collapse)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, event_embedding: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            event_embedding: CLS-Token Output, Shape (B, embed_dim)

        Returns:
            Unit vectors, Shape (B, 3)
        """
        x = self.fc1(event_embedding)
        x = self.activation(x)
        x = self.fc2(x)  # (B, 3)

        # L2-Normalisierung auf Einheitskugel
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x
