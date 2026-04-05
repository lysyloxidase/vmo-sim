"""Full quadriceps model with explicit VMO and VML separation."""

from __future__ import annotations

import torch
import torch.nn as nn

from vmo_sim.biomechanics.hill_muscle import HillMuscle
from vmo_sim.biomechanics.parameters import MuscleParameters, get_default_quadriceps

_LENGTH_COEFFICIENTS: dict[str, tuple[float, float, float]] = {
    "VMO": (0.220, 0.036, 0.004),
    "VML": (0.226, 0.037, 0.004),
    "VL": (0.235, 0.041, 0.005),
    "RF": (0.410, 0.039, 0.003),
    "VI": (0.230, 0.040, 0.004),
}

_PATELLAR_FORCE_SCALE: dict[str, float] = {
    "VMO": 1.0,
    "VML": 0.25,
    "VL": 1.4,
    "RF": 0.0,
    "VI": 0.0,
}


class QuadricepsModel(nn.Module):
    """Five-muscle quadriceps model acting on the patella and knee."""

    def __init__(self, params: dict[str, MuscleParameters] | None = None) -> None:
        super().__init__()
        self.params = params if params is not None else get_default_quadriceps()
        self.muscles = nn.ModuleDict(
            {
                name: HillMuscle(muscle_params)
                for name, muscle_params in self.params.items()
            }
        )

    def forward(
        self,
        excitations: dict[str, torch.Tensor],
        knee_angle: torch.Tensor,
        knee_angular_velocity: torch.Tensor,
        dt: float = 0.001,
        state: dict[str, dict[str, torch.Tensor]] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
        """Advance the full quadriceps model by one timestep."""

        forces: dict[str, torch.Tensor] = {}
        new_state: dict[str, dict[str, torch.Tensor]] = {}

        for muscle_name, muscle in self.muscles.items():
            musculotendon_length = self.musculotendon_length(knee_angle, muscle_name)
            moment_arm = self.moment_arm(knee_angle, muscle_name)
            musculotendon_velocity = -moment_arm * knee_angular_velocity
            muscle_state = None if state is None else state.get(muscle_name)
            force, updated_state = muscle(
                excitations[muscle_name],
                musculotendon_length,
                musculotendon_velocity,
                dt=dt,
                state=muscle_state,
            )
            forces[muscle_name] = force
            new_state[muscle_name] = updated_state

        forces["total_force"] = torch.stack(
            [forces[name] for name in self.params], dim=0
        ).sum(dim=0)
        forces["knee_extension_moment"] = self.knee_extension_moment(forces, knee_angle)
        forces["mediolateral_force"] = self.patellar_mediolateral_force(forces)
        forces["vmo_vl_ratio"] = self.vmo_vl_ratio(forces)
        return forces, new_state

    def knee_extension_moment(
        self,
        forces: dict[str, torch.Tensor],
        knee_angle: torch.Tensor,
    ) -> torch.Tensor:
        """Return net knee extension moment from all quadriceps actuators."""

        moment = torch.zeros_like(knee_angle)
        for muscle_name in self.params:
            moment = moment + forces[muscle_name] * self.moment_arm(
                knee_angle, muscle_name
            )
        return moment

    def patellar_mediolateral_force(
        self, forces: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Return net mediolateral patellar force, positive toward the medial side."""

        total = torch.zeros_like(forces["VMO"])
        for muscle_name, muscle_params in self.params.items():
            angle = float(getattr(muscle_params, "patellar_force_angle", 0.0))
            scale = _PATELLAR_FORCE_SCALE.get(muscle_name, 0.0)
            total = total + scale * forces[muscle_name] * torch.sin(
                torch.tensor(
                    angle,
                    dtype=forces[muscle_name].dtype,
                    device=forces[muscle_name].device,
                )
            )
        return total

    def vmo_vl_ratio(self, forces: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the clinically meaningful VMO-to-VL force ratio."""

        return forces["VMO"] / torch.clamp(forces["VL"], min=1e-6)

    @staticmethod
    def musculotendon_length(
        knee_angle: torch.Tensor, muscle_name: str
    ) -> torch.Tensor:
        """Return musculotendon length as a smooth polynomial of knee angle."""

        base_length, linear_term, quadratic_term = _LENGTH_COEFFICIENTS[muscle_name]
        return (
            base_length
            - linear_term * knee_angle
            + 0.5 * quadratic_term * knee_angle**2
        )

    @staticmethod
    def moment_arm(knee_angle: torch.Tensor, muscle_name: str) -> torch.Tensor:
        """Return knee extension moment arm as a function of knee angle."""

        _, linear_term, quadratic_term = _LENGTH_COEFFICIENTS[muscle_name]
        return linear_term - quadratic_term * knee_angle
