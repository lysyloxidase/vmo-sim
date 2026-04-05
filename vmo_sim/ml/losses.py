"""Physics-informed loss functions for scientific muscle modeling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from vmo_sim.biomechanics.activation import ActivationDynamics
from vmo_sim.biomechanics.force_length import ForceLengthRelationship
from vmo_sim.biomechanics.force_velocity import ForceVelocityRelationship
from vmo_sim.biomechanics.parameters import MuscleParameters
from vmo_sim.biomechanics.pennation import PennationModel


class PhysicsLoss(nn.Module):
    """Physics-informed losses encoding Hill-type muscle mechanics."""

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _force_length_model(params: MuscleParameters) -> ForceLengthRelationship:
        return ForceLengthRelationship(params.passive_strain_at_one_norm_force)

    @staticmethod
    def _force_velocity_model(params: MuscleParameters) -> ForceVelocityRelationship:
        return ForceVelocityRelationship(
            params.fv_shape_factor,
            params.max_eccentric_force_multiplier,
        )

    @staticmethod
    def _pennation_model(params: MuscleParameters) -> PennationModel:
        return PennationModel(params.pennation_angle_at_optimal)

    def force_length_consistency(
        self,
        predicted_force: torch.Tensor,
        fiber_length: torch.Tensor,
        activation: torch.Tensor,
        params: MuscleParameters,
    ) -> torch.Tensor:
        """Compute force-length consistency loss."""

        force_length = self._force_length_model(params)
        pennation = self._pennation_model(params)
        projection = pennation.projection_factor(fiber_length)
        expected_force = (
            params.max_isometric_force
            * (
                activation * force_length.active(fiber_length)
                + force_length.passive(fiber_length)
            )
            * projection
        )
        return functional.mse_loss(predicted_force, expected_force)

    def force_velocity_consistency(
        self,
        predicted_force: torch.Tensor,
        fiber_length: torch.Tensor,
        fiber_velocity: torch.Tensor,
        activation: torch.Tensor,
        params: MuscleParameters,
    ) -> torch.Tensor:
        """Compute force-velocity consistency loss."""

        force_length = self._force_length_model(params)
        force_velocity = self._force_velocity_model(params)
        pennation = self._pennation_model(params)
        projection = pennation.projection_factor(fiber_length)
        expected_force = (
            params.max_isometric_force
            * activation
            * force_length.active(fiber_length)
            * force_velocity(fiber_velocity)
            * projection
        )
        return functional.mse_loss(predicted_force, expected_force)

    def activation_dynamics_consistency(
        self,
        predicted_activation: torch.Tensor,
        excitation: torch.Tensor,
        dt: float,
        params: MuscleParameters,
    ) -> torch.Tensor:
        """Compute excitation-to-activation dynamics consistency loss."""

        if predicted_activation.shape != excitation.shape:
            raise ValueError(
                "predicted_activation and excitation must share the same shape."
            )
        if predicted_activation.shape[-1] < 2:
            return torch.zeros(
                (),
                dtype=predicted_activation.dtype,
                device=predicted_activation.device,
            )

        activation_dynamics = ActivationDynamics(
            params.activation_time_constant,
            params.deactivation_time_constant,
        )
        activation_prev = torch.clamp(predicted_activation[..., :-1], 0.01, 1.0)
        excitation_prev = torch.clamp(excitation[..., :-1], 0.0, 1.0)
        predicted_derivative = (
            predicted_activation[..., 1:] - predicted_activation[..., :-1]
        ) / dt
        target_derivative = (
            excitation_prev - activation_prev
        ) / activation_dynamics.time_constant(
            excitation_prev,
            activation_prev,
        )
        return functional.mse_loss(predicted_derivative, target_derivative)

    def tendon_equilibrium(
        self,
        muscle_force: torch.Tensor,
        tendon_force: torch.Tensor,
    ) -> torch.Tensor:
        """Compute muscle-tendon equilibrium loss."""

        return functional.mse_loss(muscle_force, tendon_force)

    def non_negativity(self, forces: torch.Tensor) -> torch.Tensor:
        """Penalize negative force predictions."""

        return torch.mean(torch.relu(-forces) ** 2)

    def combined(
        self,
        components: dict[str, torch.Tensor],
        weights: dict[str, float] | None = None,
    ) -> torch.Tensor:
        """Return a weighted sum of loss components."""

        if not components:
            raise ValueError("components must not be empty.")
        weights = weights or {}
        total = torch.zeros_like(next(iter(components.values())))
        for name, value in components.items():
            total = total + weights.get(name, 1.0) * value
        return total


def data_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return the supervised data-fitting loss."""

    return functional.mse_loss(predictions, targets)


def boundary_condition_loss(
    predictions: torch.Tensor,
    boundary_values: torch.Tensor,
) -> torch.Tensor:
    """Return the boundary-condition matching loss."""

    return functional.mse_loss(predictions, boundary_values)


def physics_informed_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    physics_residuals: torch.Tensor,
    physics_weight: float,
) -> torch.Tensor:
    """Blend supervised loss with physics residual regularization."""

    return data_loss(predictions, targets) + physics_weight * torch.mean(
        physics_residuals**2
    )
