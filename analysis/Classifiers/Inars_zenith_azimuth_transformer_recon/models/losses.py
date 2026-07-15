"""Loss-Funktionen fuer Direction Reconstruction.

Angular Distance Loss zwischen predicted unit vectors und
true (azimuth, zenith) Winkeln. Basierend auf iceaggr (Inar Timiryasov).
"""

import torch
import torch.nn as nn


def angles_to_unit_vector(azimuth: torch.Tensor, zenith: torch.Tensor) -> torch.Tensor:
    """Konvertiert (azimuth, zenith) Winkel in 3D-Einheitsvektoren.

    Args:
        azimuth: Azimut-Winkel in Radiant, Shape (B,)
        zenith:  Zenit-Winkel in Radiant, Shape (B,)

    Returns:
        Unit vectors, Shape (B, 3) mit [x, y, z]
    """
    x = torch.cos(azimuth) * torch.sin(zenith)
    y = torch.sin(azimuth) * torch.sin(zenith)
    z = torch.cos(zenith)
    return torch.stack([x, y, z], dim=1)


def angular_distance(
    pred_vectors: torch.Tensor,
    true_vectors: torch.Tensor,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Berechnet den Winkelabstand (in Radiant) zwischen Unit-Vektoren.

    Args:
        pred_vectors: Predicted unit vectors, Shape (B, 3)
        true_vectors: True unit vectors, Shape (B, 3)
        epsilon: Clipping-Wert fuer numerische Stabilitaet bei arccos

    Returns:
        Winkelabstand pro Sample, Shape (B,)
    """
    dot = torch.sum(pred_vectors * true_vectors, dim=1)
    dot = torch.clamp(dot, -1.0 + epsilon, 1.0 - epsilon)
    return torch.arccos(dot)


def angular_distance_loss(
    pred_vectors: torch.Tensor,
    true_angles: torch.Tensor,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Angular Distance Loss: mean(arccos(dot(pred, true))).

    Args:
        pred_vectors: Predicted unit vectors, Shape (B, 3)
        true_angles:  True Winkel als (azimuth, zenith), Shape (B, 2)
        epsilon: Clipping fuer numerische Stabilitaet

    Returns:
        Skalar-Loss in Radiant. Zufalls-Baseline ~ 1.57 rad (90 Grad).
    """
    true_vectors = angles_to_unit_vector(
        azimuth=true_angles[:, 0],
        zenith=true_angles[:, 1],
    )
    return angular_distance(pred_vectors, true_vectors, epsilon).mean()


class AngularDistanceLoss(nn.Module):
    """nn.Module-Wrapper fuer angular_distance_loss."""

    def __init__(self, epsilon: float = 1e-4):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        pred_vectors: torch.Tensor,
        true_angles: torch.Tensor,
    ) -> torch.Tensor:
        return angular_distance_loss(pred_vectors, true_angles, self.epsilon)
