"""Tendon force-strain relationship."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TendonModel(nn.Module):
    """Exponential compliant tendon model with an optional rigid mode."""

    def __init__(
        self,
        tendon_slack_length: float,
        strain_at_one_norm_force: float = 0.033,
        rigid: bool = False,
    ) -> None:
        super().__init__()
        self.tendon_slack_length = tendon_slack_length
        self.strain_at_one_norm_force = strain_at_one_norm_force
        self.rigid = rigid
        self.k_t = 35.0
        self._force_scale = math.exp(self.k_t) - 1.0

    def strain(self, tendon_length: torch.Tensor) -> torch.Tensor:
        """Return tendon strain ``epsilon = (l_t - l_ts) / l_ts``."""

        return (tendon_length - self.tendon_slack_length) / self.tendon_slack_length

    def force(self, tendon_length: torch.Tensor) -> torch.Tensor:
        """Return normalized tendon force from tendon length."""

        strain = self.strain(tendon_length)
        if self.rigid:
            return torch.where(
                strain > 0.0,
                torch.full_like(strain, torch.inf),
                torch.zeros_like(strain),
            )

        normalized_strain = torch.clamp(strain, min=0.0) / self.strain_at_one_norm_force
        normalized_force = (
            torch.exp(self.k_t * normalized_strain) - 1.0
        ) / self._force_scale
        return torch.where(strain > 0.0, normalized_force, torch.zeros_like(strain))

    def inverse(self, normalized_force: torch.Tensor) -> torch.Tensor:
        """Return tendon length corresponding to a normalized tendon force."""

        if self.rigid:
            positive_extension = torch.where(
                normalized_force > 0.0,
                torch.full_like(normalized_force, 1e-6),
                torch.zeros_like(normalized_force),
            )
            return self.tendon_slack_length * (1.0 + positive_extension)

        clamped_force = torch.clamp(normalized_force, min=0.0)
        normalized_strain = (
            torch.log(clamped_force * self._force_scale + 1.0) / self.k_t
        )
        strain = normalized_strain * self.strain_at_one_norm_force
        return self.tendon_slack_length * (1.0 + strain)
