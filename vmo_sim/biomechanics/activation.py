"""First-order excitation-to-activation dynamics."""

from __future__ import annotations

import torch
import torch.nn as nn


class ActivationDynamics(nn.Module):
    """Differentiable first-order activation dynamics."""

    def __init__(self, tau_act: float = 0.010, tau_deact: float = 0.040) -> None:
        super().__init__()
        self.tau_act = tau_act
        self.tau_deact = tau_deact
        self.min_activation = 0.01

    def forward(
        self,
        excitation: torch.Tensor,
        activation: torch.Tensor,
        dt: float = 0.001,
    ) -> torch.Tensor:
        """Advance activation by one timestep."""

        excitation = torch.clamp(excitation, 0.0, 1.0)
        activation = torch.clamp(activation, self.min_activation, 1.0)
        tau = self.time_constant(excitation, activation)
        derivative = (excitation - activation) / tau
        updated = activation + dt * derivative
        return torch.clamp(updated, self.min_activation, 1.0)

    def integrate(
        self,
        excitation_signal: torch.Tensor,
        dt: float = 0.001,
        a0: float = 0.0,
    ) -> torch.Tensor:
        """Integrate an excitation time series."""

        squeeze_output = excitation_signal.ndim == 1
        if squeeze_output:
            signal = excitation_signal.unsqueeze(0)
        elif excitation_signal.ndim == 2:
            signal = excitation_signal
        else:
            raise ValueError("excitation_signal must have shape (T,) or (batch, T)")

        activation = torch.full(
            signal.shape[:-1],
            fill_value=max(a0, self.min_activation),
            dtype=signal.dtype,
            device=signal.device,
        )
        outputs: list[torch.Tensor] = []
        for time_index in range(signal.shape[-1]):
            activation = self.forward(signal[..., time_index], activation, dt=dt)
            outputs.append(activation)

        stacked = torch.stack(outputs, dim=-1)
        return stacked.squeeze(0) if squeeze_output else stacked

    def time_constant(
        self,
        excitation: torch.Tensor,
        activation: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the instantaneous activation or deactivation time constant."""

        activation = torch.clamp(activation, self.min_activation, 1.0)
        activation_tau = 0.7 * self.tau_act * (0.5 + 1.5 * activation)
        deactivation_tau = 0.85 * self.tau_deact / (0.5 + 1.5 * activation)
        return torch.where(excitation > activation, activation_tau, deactivation_tau)
