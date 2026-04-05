"""Physics-informed surrogate models for VMO muscle mechanics."""

from __future__ import annotations

import torch
import torch.nn as nn

from vmo_sim.biomechanics.force_length import ForceLengthRelationship
from vmo_sim.biomechanics.force_velocity import ForceVelocityRelationship
from vmo_sim.biomechanics.hill_muscle import HillMuscle
from vmo_sim.biomechanics.parameters import MuscleParameters, VMOParameters
from vmo_sim.biomechanics.pennation import PennationModel
from vmo_sim.ml.losses import PhysicsLoss


class VMOPINNSurrogate(nn.Module):
    """Physics-Informed Neural Network surrogate for the Hill-type VMO model."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        muscle_params: MuscleParameters | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.muscle_params = muscle_params or VMOParameters()
        self.force_length = ForceLengthRelationship(
            self.muscle_params.passive_strain_at_one_norm_force
        )
        self.force_velocity = ForceVelocityRelationship(
            self.muscle_params.fv_shape_factor,
            self.muscle_params.max_eccentric_force_multiplier,
        )
        self.pennation = PennationModel(self.muscle_params.pennation_angle_at_optimal)
        self.physics = PhysicsLoss()

        layers: list[nn.Module] = []
        in_features = 3
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.SiLU())
            in_features = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _baseline_outputs(
        self,
        activation: torch.Tensor,
        fiber_length: torch.Tensor,
        fiber_velocity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pennation_angle = self.pennation.angle(fiber_length)
        projection = torch.cos(pennation_angle)
        normalized_force = (
            activation
            * self.force_length.active(fiber_length)
            * self.force_velocity(fiber_velocity)
            + self.force_length.passive(fiber_length)
        ) * projection
        return normalized_force, pennation_angle

    def forward(
        self,
        activation: torch.Tensor,
        fiber_length: torch.Tensor,
        fiber_velocity: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict normalized force and pennation angle."""

        activation, fiber_length, fiber_velocity = torch.broadcast_tensors(  # type: ignore[no-untyped-call]
            activation,
            fiber_length,
            fiber_velocity,
        )
        features = torch.stack(
            [activation, fiber_length, fiber_velocity],
            dim=-1,
        )
        residuals = self.head(self.backbone(features))
        baseline_force, baseline_angle = self._baseline_outputs(
            activation,
            fiber_length,
            fiber_velocity,
        )
        normalized_force = torch.clamp(
            baseline_force + 0.2 * torch.tanh(residuals[..., 0]),
            min=0.0,
        )
        pennation_angle = torch.clamp(
            baseline_angle + 0.2 * torch.tanh(residuals[..., 1]),
            min=0.0,
            max=1.55,
        )
        return {"force": normalized_force, "pennation_angle": pennation_angle}

    @staticmethod
    def _latin_hypercube(
        n_samples: int,
        n_dims: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        samples = torch.empty(n_samples, n_dims, dtype=torch.float32)
        for dim_index in range(n_dims):
            interval_samples = (
                torch.arange(n_samples, dtype=torch.float32)
                + torch.rand(n_samples, generator=generator)
            ) / float(n_samples)
            permutation = torch.randperm(n_samples, generator=generator)
            samples[:, dim_index] = interval_samples[permutation]
        return samples

    def generate_training_data(
        self,
        hill_muscle: HillMuscle,
        n_samples: int = 50000,
        seed: int = 42,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate surrogate training data using Latin hypercube sampling."""

        generator = torch.Generator().manual_seed(seed)
        samples = self._latin_hypercube(n_samples, 3, generator)
        activation = 0.01 + 0.99 * samples[:, 0]
        fiber_length = 0.5 + 1.1 * samples[:, 1]
        fiber_velocity = -1.0 + 2.0 * samples[:, 2]

        with torch.no_grad():
            force = (
                hill_muscle.compute_force(
                    activation,
                    fiber_length,
                    fiber_velocity,
                )
                / hill_muscle.params.max_isometric_force
            )
            pennation_angle = hill_muscle.pennation.angle(fiber_length)

        inputs = torch.stack([activation, fiber_length, fiber_velocity], dim=-1)
        outputs = torch.stack([force, pennation_angle], dim=-1)
        return inputs, outputs

    def physics_loss(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor | dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute physics-informed regularization loss."""

        if inputs.shape[-1] != 3:
            raise ValueError("inputs must have shape (..., 3).")

        if isinstance(outputs, dict):
            predicted_force = outputs["force"]
        else:
            predicted_force = outputs[..., 0]

        activation = inputs[..., 0]
        fiber_length = inputs[..., 1]
        fiber_velocity = inputs[..., 2]
        force_newtons = predicted_force * self.muscle_params.max_isometric_force

        components = {
            "force_length": self.physics.force_length_consistency(
                force_newtons,
                fiber_length,
                activation,
                self.muscle_params,
            ),
            "force_velocity": self.physics.force_velocity_consistency(
                force_newtons,
                fiber_length,
                fiber_velocity,
                activation,
                self.muscle_params,
            ),
            "non_negativity": self.physics.non_negativity(force_newtons),
        }
        return self.physics.combined(components)


class PINNSurrogate(VMOPINNSurrogate):
    """Backward-compatible alias for the VMO PINN surrogate."""

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 4,
        muscle_params: MuscleParameters | None = None,
    ) -> None:
        if input_dim != 3:
            raise ValueError("PINNSurrogate expects exactly 3 input features.")
        super().__init__(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            muscle_params=muscle_params,
        )
