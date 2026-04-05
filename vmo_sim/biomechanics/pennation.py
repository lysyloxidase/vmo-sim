"""Pennation angle kinematics under a constant-thickness assumption."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PennationModel(nn.Module):
    """Map fiber length changes to pennation angle and force projection."""

    def __init__(self, pennation_at_optimal: float) -> None:
        super().__init__()
        self.pennation_at_optimal = pennation_at_optimal
        self.sin_alpha_opt = math.sin(pennation_at_optimal)

    def angle(self, normalized_fiber_length: torch.Tensor) -> torch.Tensor:
        """Return the current pennation angle in radians."""

        safe_length = torch.clamp(
            normalized_fiber_length, min=self.sin_alpha_opt + 1e-4
        )
        sin_alpha = torch.clamp(self.sin_alpha_opt / safe_length, max=0.999)
        return torch.asin(sin_alpha)

    def projection_factor(self, normalized_fiber_length: torch.Tensor) -> torch.Tensor:
        """Return ``cos(alpha)``, the tendon-axis force projection factor."""

        return torch.cos(self.angle(normalized_fiber_length))
