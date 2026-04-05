"""Simplified patellofemoral joint mechanics suitable for real-time studies."""

from __future__ import annotations

import torch
import torch.nn as nn


class PatellofemoralModel(nn.Module):
    """Analytical model of mediolateral patellar position, tilt, and pressure."""

    def __init__(
        self,
        trochlear_depth: float = 0.005,
        patellar_width: float = 0.045,
        lateral_retinaculum_stiffness: float = 200.0,
        medial_retinaculum_stiffness: float = 150.0,
    ) -> None:
        super().__init__()
        self.trochlear_depth = trochlear_depth
        self.patellar_width = patellar_width
        self.lateral_retinaculum_stiffness = lateral_retinaculum_stiffness
        self.medial_retinaculum_stiffness = medial_retinaculum_stiffness

    def _constraint_stiffness(self, knee_angle: torch.Tensor) -> torch.Tensor:
        engagement = 0.2 + 2.5 * torch.clamp(
            torch.sin(torch.clamp(knee_angle, min=0.0)), min=0.0
        )
        groove_stiffness = 10_000.0 * self.trochlear_depth * engagement
        return (
            self.lateral_retinaculum_stiffness
            + self.medial_retinaculum_stiffness
            + groove_stiffness
        )

    def patellar_displacement(
        self,
        mediolateral_force: torch.Tensor,
        knee_angle: torch.Tensor,
    ) -> torch.Tensor:
        """Return mediolateral patellar displacement, positive toward the lateral facet."""

        stiffness = self._constraint_stiffness(knee_angle)
        max_excursion = self.patellar_width / 4.0
        return max_excursion * torch.tanh(-mediolateral_force / stiffness)

    def patellar_tilt(
        self,
        mediolateral_force: torch.Tensor,
        knee_angle: torch.Tensor,
    ) -> torch.Tensor:
        """Return patellar tilt angle, positive for lateral tilt."""

        displacement = self.patellar_displacement(mediolateral_force, knee_angle)
        lever_arm = self.patellar_width / 2.0
        return 0.5 * torch.atan2(displacement, torch.full_like(displacement, lever_arm))

    def contact_pressure(
        self,
        total_quadriceps_force: torch.Tensor,
        knee_angle: torch.Tensor,
        patellar_displacement: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return estimated medial and lateral contact pressures."""

        contact_area = 2.0e-4 + 3.5e-4 * torch.clamp(
            torch.sin(torch.clamp(knee_angle, min=0.0)),
            min=0.0,
        )
        total_pressure = total_quadriceps_force / contact_area
        bias = torch.clamp(
            patellar_displacement / (self.patellar_width / 4.0),
            min=-1.0,
            max=1.0,
        )
        medial_pressure = total_pressure * (1.0 - 0.35 * bias)
        lateral_pressure = total_pressure * (1.0 + 0.35 * bias)
        return medial_pressure, lateral_pressure
