"""Force-velocity relationship for concentric and eccentric contractions."""

from __future__ import annotations

import torch
import torch.nn as nn


class ForceVelocityRelationship(nn.Module):
    """Smooth force-velocity relationship bounded in shortening and lengthening."""

    def __init__(
        self,
        shape_factor: float = 0.25,
        max_eccentric_multiplier: float = 1.4,
    ) -> None:
        super().__init__()
        self.shape_factor = shape_factor
        self.max_eccentric_multiplier = max_eccentric_multiplier

    def forward(self, normalized_velocity: torch.Tensor) -> torch.Tensor:
        """Return the normalized force-velocity multiplier."""

        velocity = torch.clamp(normalized_velocity, -1.0, 1.0)
        concentric = (1.0 + velocity) / (1.0 - velocity / self.shape_factor)

        eccentric_velocity = torch.clamp(velocity, min=0.0, max=1.0)
        scale = 1.0 - torch.exp(
            torch.tensor(
                -1.0 / self.shape_factor,
                dtype=velocity.dtype,
                device=velocity.device,
            )
        )
        eccentric = 1.0 + (self.max_eccentric_multiplier - 1.0) * (
            (1.0 - torch.exp(-eccentric_velocity / self.shape_factor)) / scale
        )

        return torch.where(velocity < 0.0, concentric, eccentric)
