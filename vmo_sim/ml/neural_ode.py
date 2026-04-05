"""Neural ODE formulations for VMO muscle dynamics."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn as nn
from torchdiffeq import odeint  # type: ignore[import-untyped]

from vmo_sim.biomechanics.hill_muscle import HillMuscle
from vmo_sim.biomechanics.parameters import MuscleParameters, VMOParameters


class MuscleNeuralODE(nn.Module):
    """Neural ODE formulation of VMO muscle dynamics."""

    def __init__(
        self,
        state_dim: int = 3,
        hidden_dim: int = 64,
        muscle_params: MuscleParameters | None = None,
        physics_weight: float = 0.8,
    ) -> None:
        super().__init__()
        if state_dim != 3:
            raise ValueError("MuscleNeuralODE currently supports a 3D state only.")
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.muscle_params = muscle_params or VMOParameters()
        self.physics_weight = physics_weight
        self.hill_muscle = HillMuscle(self.muscle_params)
        self.default_musculotendon_length = (
            self.muscle_params.tendon_slack_length
            + self.muscle_params.optimal_fiber_length
            * torch.cos(
                torch.tensor(self.muscle_params.pennation_angle_at_optimal)
            ).item()
        )
        self.min_fiber_length = 0.5 * self.muscle_params.optimal_fiber_length
        self.max_fiber_length = 1.6 * self.muscle_params.optimal_fiber_length
        self.neural_rhs = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )
        final_linear = self.neural_rhs[-1]
        assert isinstance(final_linear, nn.Linear)
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def physics_rhs(
        self,
        state: torch.Tensor,
        excitation: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the purely physics-based right-hand side."""

        fiber_length = torch.clamp(
            state[..., 0],
            min=self.min_fiber_length,
            max=self.max_fiber_length,
        )
        activation = torch.clamp(state[..., 1], min=0.01, max=1.0)
        fatigue_state = torch.clamp(state[..., 2], min=0.0, max=1.0)
        effective_activation = activation * (1.0 - fatigue_state)

        musculotendon_length = torch.full_like(
            fiber_length,
            self.default_musculotendon_length,
        )
        musculotendon_velocity = torch.zeros_like(fiber_length)
        fiber_velocity = self.hill_muscle.fiber_velocity_ode(
            musculotendon_length,
            musculotendon_velocity,
            effective_activation,
            fiber_length,
        )
        tau = self.hill_muscle.activation_dynamics.time_constant(
            excitation,
            activation,
        )
        activation_dot = (excitation - activation) / tau
        fatigue_dot = (
            self.muscle_params.fatigue_rate
            * effective_activation
            * (1.0 - fatigue_state)
            - self.muscle_params.recovery_rate
            * (1.0 - effective_activation)
            * fatigue_state
        )
        return torch.stack([fiber_velocity, activation_dot, fatigue_dot], dim=-1)

    def neural_correction(
        self,
        state: torch.Tensor,
        excitation: torch.Tensor,
    ) -> torch.Tensor:
        """Return the learned discrepancy correction."""

        if excitation.ndim < state.ndim - 1:
            excitation = torch.broadcast_to(excitation, state.shape[:-1])
        inputs = torch.cat([state, excitation.unsqueeze(-1)], dim=-1)
        return 0.1 * torch.tanh(self.neural_rhs(inputs))

    def _excitation_function(
        self,
        excitation_signal: torch.Tensor,
        t_span: torch.Tensor,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        def excitation_fn(time_point: torch.Tensor) -> torch.Tensor:
            index = torch.bucketize(time_point.detach(), t_span) - 1
            index = torch.clamp(index, min=0, max=t_span.numel() - 1)
            return excitation_signal[..., int(index.item())]

        return excitation_fn

    def ode_func(
        self,
        t: torch.Tensor,
        state: torch.Tensor,
        excitation_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Compute the combined physics and neural right-hand side."""

        excitation = excitation_fn(t)
        if excitation.ndim < state.ndim - 1:
            excitation = torch.broadcast_to(excitation, state.shape[:-1])
        physics_term = self.physics_rhs(state, excitation)
        correction = self.neural_correction(state, excitation)
        return physics_term + (1.0 - self.physics_weight) * correction

    def forward(
        self,
        initial_state: torch.Tensor,
        excitation_signal: torch.Tensor,
        t_span: torch.Tensor,
    ) -> torch.Tensor:
        """Integrate the combined dynamics with `torchdiffeq.odeint`."""

        excitation_fn = self._excitation_function(excitation_signal, t_span)
        return cast(
            torch.Tensor,
            odeint(
                lambda time, state: self.ode_func(time, state, excitation_fn),
                initial_state,
                t_span,
                method="rk4",
            ),
        )

    def physics_only(
        self,
        initial_state: torch.Tensor,
        excitation_signal: torch.Tensor,
        t_span: torch.Tensor,
    ) -> torch.Tensor:
        """Integrate the physics-only component of the Neural ODE."""

        excitation_fn = self._excitation_function(excitation_signal, t_span)
        return cast(
            torch.Tensor,
            odeint(
                lambda time, state: self.physics_rhs(state, excitation_fn(time)),
                initial_state,
                t_span,
                method="rk4",
            ),
        )


class NeuralMuscleODE(MuscleNeuralODE):
    """Backward-compatible alias for the muscle Neural ODE."""

    def __init__(
        self,
        state_dim: int = 3,
        control_dim: int = 1,
        hidden_dim: int = 64,
    ) -> None:
        del control_dim
        super().__init__(state_dim=state_dim, hidden_dim=hidden_dim)
