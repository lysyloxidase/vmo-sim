"""Gradient-based parameter identification for VMO-Sim."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional

from vmo_sim.biomechanics.parameters import MuscleParameters


class GradientParameterIdentification:
    """Gradient-based identification of subject-specific VMO parameters."""

    POSITIVE_FIELDS = {
        "max_isometric_force",
        "optimal_fiber_length",
        "tendon_slack_length",
        "activation_time_constant",
        "max_contraction_velocity",
        "passive_strain_at_one_norm_force",
    }

    def _parameterize(
        self, initial_params: MuscleParameters, field: str
    ) -> torch.nn.Parameter:
        value = float(getattr(initial_params, field))
        if field in self.POSITIVE_FIELDS:
            raw = math.log(math.exp(value) - 1.0) if value > 1e-6 else -10.0
        elif field == "pennation_angle_at_optimal":
            scaled = min(max((value - 0.01) / 1.39, 1e-5), 1.0 - 1e-5)
            raw = math.log(scaled / (1.0 - scaled))
        else:
            raw = value
        return torch.nn.Parameter(torch.tensor(raw, dtype=torch.float32))

    def _value_from_raw(self, raw: torch.Tensor, field: str) -> torch.Tensor:
        if field in self.POSITIVE_FIELDS:
            return functional.softplus(raw)
        if field == "pennation_angle_at_optimal":
            return 0.01 + 1.39 * torch.sigmoid(raw)
        return raw

    def _simulate_force(
        self,
        excitation: torch.Tensor,
        musculotendon_lengths: torch.Tensor,
        current_values: dict[str, torch.Tensor],
        template: MuscleParameters,
        dt: float = 0.001,
    ) -> torch.Tensor:
        max_force = current_values.get(
            "max_isometric_force",
            torch.tensor(template.max_isometric_force, dtype=torch.float32),
        )
        optimal_length = current_values.get(
            "optimal_fiber_length",
            torch.tensor(template.optimal_fiber_length, dtype=torch.float32),
        )
        pennation_opt = current_values.get(
            "pennation_angle_at_optimal",
            torch.tensor(template.pennation_angle_at_optimal, dtype=torch.float32),
        )
        tau_act = current_values.get(
            "activation_time_constant",
            torch.tensor(template.activation_time_constant, dtype=torch.float32),
        )
        max_velocity = current_values.get(
            "max_contraction_velocity",
            torch.tensor(template.max_contraction_velocity, dtype=torch.float32),
        )
        passive_strain = current_values.get(
            "passive_strain_at_one_norm_force",
            torch.tensor(
                template.passive_strain_at_one_norm_force, dtype=torch.float32
            ),
        )
        tendon_slack = current_values.get(
            "tendon_slack_length",
            torch.tensor(template.tendon_slack_length, dtype=torch.float32),
        )

        excitation = torch.clamp(excitation, 0.0, 1.0)
        activation = torch.zeros_like(excitation)
        activation[0] = torch.clamp(excitation[0], 0.01, 1.0)
        for index in range(1, excitation.numel()):
            previous = torch.clamp(activation[index - 1], 0.01, 1.0)
            tau = tau_act * (0.5 + 1.5 * previous)
            activation[index] = torch.clamp(
                previous + dt * (excitation[index] - previous) / tau, 0.01, 1.0
            )

        normalized_length = torch.clamp(
            (musculotendon_lengths - tendon_slack) / optimal_length, 0.5, 1.6
        )
        velocity = torch.zeros_like(normalized_length)
        if normalized_length.numel() > 1:
            velocity[1:] = (
                (normalized_length[1:] - normalized_length[:-1]) / dt / max_velocity
            )
            velocity[0] = velocity[1]

        active_force = torch.exp(-(((normalized_length - 1.0) / 0.25) ** 2))
        passive_force = torch.where(
            normalized_length > 1.0,
            (torch.exp(4.0 * (normalized_length - 1.0) / passive_strain) - 1.0)
            / (math.exp(4.0) - 1.0),
            torch.zeros_like(normalized_length),
        )
        concentric = (1.0 + velocity) / (1.0 - velocity / template.fv_shape_factor)
        eccentric = 1.0 + (template.max_eccentric_force_multiplier - 1.0) * (
            1.0 - torch.exp(-torch.clamp(velocity, min=0.0) / template.fv_shape_factor)
        ) / (1.0 - math.exp(-1.0 / template.fv_shape_factor))
        force_velocity = torch.where(velocity < 0.0, concentric, eccentric)

        sin_alpha = torch.clamp(torch.sin(pennation_opt) / normalized_length, max=0.999)
        projection = torch.sqrt(1.0 - sin_alpha**2)
        return (
            max_force
            * (activation * active_force * force_velocity + passive_force)
            * projection
        )

    def identify(
        self,
        experimental_force: torch.Tensor,
        experimental_emg: torch.Tensor,
        musculotendon_lengths: torch.Tensor,
        initial_params: MuscleParameters,
        params_to_optimize: list[str] | None = None,
        n_iterations: int = 500,
        lr: float = 0.01,
    ) -> tuple[MuscleParameters, dict[str, list[float]]]:
        """Identify subject-specific parameters from force and EMG data."""

        fields = params_to_optimize or [
            "max_isometric_force",
            "optimal_fiber_length",
            "pennation_angle_at_optimal",
        ]
        raw_parameters = {
            field: self._parameterize(initial_params, field) for field in fields
        }
        optimizer = torch.optim.Adam(raw_parameters.values(), lr=lr)
        history: dict[str, list[float]] = {"loss": []}
        for field in fields:
            history[field] = []

        force_target = experimental_force.to(dtype=torch.float32)
        excitation = experimental_emg.to(dtype=torch.float32)
        mt_lengths = musculotendon_lengths.to(dtype=torch.float32)

        for _ in range(n_iterations):
            optimizer.zero_grad()
            current_values = {
                field: self._value_from_raw(raw, field)
                for field, raw in raw_parameters.items()
            }
            predicted_force = self._simulate_force(
                excitation, mt_lengths, current_values, initial_params
            )
            loss = functional.mse_loss(predicted_force, force_target)
            regularization = torch.zeros((), dtype=torch.float32)
            for field, value in current_values.items():
                nominal = torch.tensor(
                    float(getattr(initial_params, field)), dtype=torch.float32
                )
                regularization = (
                    regularization + 1e-4 * ((value - nominal) / nominal) ** 2
                )
            total_loss = loss + regularization
            total_loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

            history["loss"].append(float(total_loss.detach().cpu().item()))
            for field, value in current_values.items():
                history[field].append(float(value.detach().cpu().item()))

        optimized_values = {
            field: float(self._value_from_raw(raw, field).detach().cpu().item())
            for field, raw in raw_parameters.items()
        }
        optimized_params = initial_params.model_copy(update=optimized_values)
        return optimized_params, history


class ParameterIdentificationProblem(GradientParameterIdentification):
    """Backward-compatible alias for gradient-based parameter identification."""

    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 500) -> None:
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return mean-squared prediction error."""

        return functional.mse_loss(predictions, targets)

    def fit(
        self, initial_parameters: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compatibility wrapper returning the unmodified initial tensor."""

        del targets
        return initial_parameters


__all__ = ["GradientParameterIdentification", "ParameterIdentificationProblem"]
