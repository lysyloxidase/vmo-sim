"""Complete Hill-type muscle-tendon unit in differentiable PyTorch."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from vmo_sim.biomechanics.activation import ActivationDynamics
from vmo_sim.biomechanics.force_length import ForceLengthRelationship
from vmo_sim.biomechanics.force_velocity import ForceVelocityRelationship
from vmo_sim.biomechanics.parameters import MuscleParameters
from vmo_sim.biomechanics.pennation import PennationModel
from vmo_sim.biomechanics.tendon import TendonModel


class HillMuscle(nn.Module):
    """Single Hill-type muscle-tendon unit."""

    def __init__(self, params: MuscleParameters) -> None:
        super().__init__()
        self.params = params
        self.activation_dynamics = ActivationDynamics(
            params.activation_time_constant,
            params.deactivation_time_constant,
        )
        self.force_length = ForceLengthRelationship(
            params.passive_strain_at_one_norm_force
        )
        self.force_velocity = ForceVelocityRelationship(
            params.fv_shape_factor,
            params.max_eccentric_force_multiplier,
        )
        self.tendon = TendonModel(
            params.tendon_slack_length,
            params.tendon_strain_at_one_norm_force,
        )
        self.pennation = PennationModel(params.pennation_angle_at_optimal)
        self._min_normalized_length = max(
            0.2,
            math.sin(params.pennation_angle_at_optimal) + 1e-3,
        )
        self._max_normalized_length = 1.8

    def _normalize_length(self, fiber_length: torch.Tensor) -> torch.Tensor:
        return fiber_length / self.params.optimal_fiber_length

    def _denormalize_velocity(self, normalized_velocity: torch.Tensor) -> torch.Tensor:
        return (
            normalized_velocity
            * self.params.max_contraction_velocity
            * self.params.optimal_fiber_length
        )

    def _normalize_velocity(self, fiber_velocity: torch.Tensor) -> torch.Tensor:
        scale = self.params.max_contraction_velocity * self.params.optimal_fiber_length
        return fiber_velocity / scale

    def _tendon_length(
        self,
        musculotendon_length: torch.Tensor,
        fiber_length: torch.Tensor,
    ) -> torch.Tensor:
        normalized_fiber_length = self._normalize_length(fiber_length)
        projection = self.pennation.projection_factor(normalized_fiber_length)
        return musculotendon_length - fiber_length * projection

    def _active_passive_forces(
        self,
        activation: torch.Tensor,
        normalized_fiber_length: torch.Tensor,
        normalized_fiber_velocity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projection = self.pennation.projection_factor(normalized_fiber_length)
        active = (
            self.params.max_isometric_force
            * activation
            * self.force_length.active(normalized_fiber_length)
            * self.force_velocity(normalized_fiber_velocity)
            * projection
        )
        passive = (
            self.params.max_isometric_force
            * self.force_length.passive(normalized_fiber_length)
            * projection
        )
        return active, passive

    def compute_force(
        self,
        activation: torch.Tensor,
        normalized_fiber_length: torch.Tensor,
        normalized_fiber_velocity: torch.Tensor,
    ) -> torch.Tensor:
        """Return total tendon-axis muscle force in Newtons."""

        active, passive = self._active_passive_forces(
            activation,
            normalized_fiber_length,
            normalized_fiber_velocity,
        )
        return active + passive

    def _equilibrium_residual(
        self,
        musculotendon_length: torch.Tensor,
        activation: torch.Tensor,
        fiber_length: torch.Tensor,
    ) -> torch.Tensor:
        normalized_fiber_length = self._normalize_length(fiber_length)
        tendon_length = self._tendon_length(musculotendon_length, fiber_length)
        tendon_force = (
            self.tendon.force(tendon_length) * self.params.max_isometric_force
        )
        muscle_force = self.compute_force(
            activation,
            normalized_fiber_length,
            torch.zeros_like(normalized_fiber_length),
        )
        return tendon_force - muscle_force

    def equilibrium_fiber_length(
        self,
        musculotendon_length: torch.Tensor,
        activation: torch.Tensor,
    ) -> torch.Tensor:
        """Solve for static fiber length satisfying tendon-muscle equilibrium."""

        min_length = self._min_normalized_length * self.params.optimal_fiber_length
        max_length = self._max_normalized_length * self.params.optimal_fiber_length
        fiber_length = torch.clamp(
            musculotendon_length - self.params.tendon_slack_length,
            min=min_length,
            max=max_length,
        )

        for _ in range(25):
            normalized_fiber_length = self._normalize_length(fiber_length)
            projection = self.pennation.projection_factor(normalized_fiber_length)
            muscle_force = self.compute_force(
                activation,
                normalized_fiber_length,
                torch.zeros_like(normalized_fiber_length),
            )
            tendon_length = self.tendon.inverse(
                muscle_force / self.params.max_isometric_force
            )
            new_fiber_length = (musculotendon_length - tendon_length) / projection
            fiber_length = 0.7 * fiber_length + 0.3 * new_fiber_length
            fiber_length = torch.clamp(fiber_length, min=min_length, max=max_length)

        return torch.clamp(fiber_length, min=min_length, max=max_length)

    def fiber_velocity_ode(
        self,
        musculotendon_length: torch.Tensor,
        musculotendon_velocity: torch.Tensor,
        activation: torch.Tensor,
        fiber_length: torch.Tensor,
    ) -> torch.Tensor:
        """Solve for fiber velocity under dynamic tendon-muscle force equilibrium."""

        equilibrium_length = self.equilibrium_fiber_length(
            musculotendon_length, activation
        )
        relaxation_velocity = (equilibrium_length - fiber_length) / 0.01
        return relaxation_velocity + 0.2 * musculotendon_velocity

    def forward(
        self,
        excitation: torch.Tensor,
        musculotendon_length: torch.Tensor,
        musculotendon_velocity: torch.Tensor,
        dt: float = 0.001,
        state: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Perform one simulation step from excitation to tendon force."""

        if state is None:
            previous_activation = torch.clamp(excitation, min=0.01, max=1.0)
            previous_fiber_length = self.equilibrium_fiber_length(
                musculotendon_length,
                previous_activation,
            )
        else:
            previous_activation = state["activation"]
            previous_fiber_length = state["fiber_length"]

        activation = self.activation_dynamics(
            excitation,
            previous_activation,
            dt=dt,
        )
        fiber_velocity = self.fiber_velocity_ode(
            musculotendon_length,
            musculotendon_velocity,
            activation,
            previous_fiber_length,
        )
        updated_fiber_length = previous_fiber_length + fiber_velocity * dt
        equilibrium_length = self.equilibrium_fiber_length(
            musculotendon_length, activation
        )
        updated_fiber_length = 0.75 * updated_fiber_length + 0.25 * equilibrium_length
        updated_fiber_length = torch.clamp(
            updated_fiber_length,
            min=self._min_normalized_length * self.params.optimal_fiber_length,
            max=self._max_normalized_length * self.params.optimal_fiber_length,
        )

        normalized_fiber_length = self._normalize_length(updated_fiber_length)
        normalized_fiber_velocity = self._normalize_velocity(fiber_velocity)
        force = self.compute_force(
            activation,
            normalized_fiber_length,
            normalized_fiber_velocity,
        )
        new_state = {
            "activation": activation,
            "fiber_length": updated_fiber_length,
            "fiber_velocity": fiber_velocity,
        }
        return force, new_state

    def simulate(
        self,
        excitation_signal: torch.Tensor,
        musculotendon_length_signal: torch.Tensor,
        musculotendon_velocity_signal: torch.Tensor,
        dt: float = 0.001,
    ) -> dict[str, torch.Tensor]:
        """Simulate the muscle over a full time series."""

        squeeze_output = excitation_signal.ndim == 1
        if squeeze_output:
            excitation = excitation_signal.unsqueeze(0)
            musculotendon_length = musculotendon_length_signal.unsqueeze(0)
            musculotendon_velocity = musculotendon_velocity_signal.unsqueeze(0)
        else:
            excitation = excitation_signal
            musculotendon_length = musculotendon_length_signal
            musculotendon_velocity = musculotendon_velocity_signal

        state: dict[str, torch.Tensor] | None = None
        outputs: dict[str, list[torch.Tensor]] = {
            "force": [],
            "activation": [],
            "fiber_length": [],
            "fiber_velocity": [],
            "pennation_angle": [],
            "tendon_force": [],
            "passive_force": [],
            "active_force": [],
        }

        for time_index in range(excitation.shape[-1]):
            force, state = self.forward(
                excitation[..., time_index],
                musculotendon_length[..., time_index],
                musculotendon_velocity[..., time_index],
                dt=dt,
                state=state,
            )
            assert state is not None
            normalized_fiber_length = self._normalize_length(state["fiber_length"])
            normalized_fiber_velocity = self._normalize_velocity(
                state["fiber_velocity"]
            )
            tendon_length = self._tendon_length(
                musculotendon_length[..., time_index],
                state["fiber_length"],
            )
            active_force, passive_force = self._active_passive_forces(
                state["activation"],
                normalized_fiber_length,
                normalized_fiber_velocity,
            )
            outputs["force"].append(force)
            outputs["activation"].append(state["activation"])
            outputs["fiber_length"].append(state["fiber_length"])
            outputs["fiber_velocity"].append(state["fiber_velocity"])
            outputs["pennation_angle"].append(
                self.pennation.angle(normalized_fiber_length)
            )
            outputs["tendon_force"].append(
                self.tendon.force(tendon_length) * self.params.max_isometric_force
            )
            outputs["passive_force"].append(passive_force)
            outputs["active_force"].append(active_force)

        stacked = {
            name: torch.stack(values, dim=-1) for name, values in outputs.items()
        }
        if squeeze_output:
            return {name: value.squeeze(0) for name, value in stacked.items()}
        return stacked
