"""Active and passive force-length relationships."""

from __future__ import annotations

import torch
import torch.nn as nn


class ForceLengthRelationship(nn.Module):
    """Active and passive fiber force-length curves."""

    def __init__(self, passive_strain: float = 0.6) -> None:
        super().__init__()
        self.passive_strain = passive_strain

    def active(self, normalized_fiber_length: torch.Tensor) -> torch.Tensor:
        """Return active force-length scaling."""

        return torch.exp(-(((normalized_fiber_length - 1.0) / 0.25) ** 2))

    def passive(self, normalized_fiber_length: torch.Tensor) -> torch.Tensor:
        """Return passive force-length scaling."""

        stretch = torch.clamp(normalized_fiber_length - 1.0, min=0.0)
        numerator = torch.exp(4.0 * stretch / self.passive_strain) - 1.0
        denominator = (
            torch.exp(
                torch.tensor(
                    4.0,
                    dtype=normalized_fiber_length.dtype,
                    device=normalized_fiber_length.device,
                )
            )
            - 1.0
        )
        passive_force = numerator / denominator
        return torch.where(
            normalized_fiber_length > 1.0,
            passive_force,
            torch.zeros_like(normalized_fiber_length),
        )

    def total(
        self,
        normalized_fiber_length: torch.Tensor,
        activation: torch.Tensor,
    ) -> torch.Tensor:
        """Return total normalized fiber force without velocity effects."""

        return activation * self.active(normalized_fiber_length) + self.passive(
            normalized_fiber_length
        )
