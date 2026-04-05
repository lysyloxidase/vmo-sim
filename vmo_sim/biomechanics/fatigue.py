"""Three-compartment fatigue model."""

from __future__ import annotations

import torch
import torch.nn as nn


class FatigueModel(nn.Module):
    """Three-compartment fatigue and recovery dynamics."""

    def __init__(
        self,
        fatigue_rate: float = 0.01,
        recovery_rate: float = 0.002,
        activation_rate: float = 0.05,
    ) -> None:
        super().__init__()
        self.fatigue_rate = fatigue_rate
        self.recovery_rate = recovery_rate
        self.activation_rate = activation_rate

    def forward(
        self,
        target_activation: torch.Tensor,
        state: dict[str, torch.Tensor],
        dt: float = 0.001,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Advance fatigue states by one timestep."""

        ma = state["MA"]
        mf = state["MF"]
        mr = torch.clamp(1.0 - ma - mf, min=0.0, max=1.0)
        target = torch.clamp(target_activation, min=0.0, max=1.0)

        recruitment = self.activation_rate * torch.clamp(target - ma, min=0.0) * mr
        fatigue = self.fatigue_rate * ma
        recovery = self.recovery_rate * mf

        d_ma = recruitment - fatigue
        d_mf = fatigue - recovery

        new_ma = torch.clamp(ma + d_ma * dt, min=0.0, max=1.0)
        new_mf = torch.clamp(mf + d_mf * dt, min=0.0, max=1.0)
        new_mr = torch.clamp(1.0 - new_ma - new_mf, min=0.0, max=1.0)
        effective_capacity = torch.clamp(1.0 - new_mf, min=0.0, max=1.0)

        new_state = {"MA": new_ma, "MF": new_mf, "MR": new_mr}
        return effective_capacity, new_state

    def simulate(
        self,
        target_activation_signal: torch.Tensor,
        dt: float = 0.001,
    ) -> dict[str, torch.Tensor]:
        """Simulate fatigue dynamics over a full time series."""

        squeeze_output = target_activation_signal.ndim == 1
        signal = (
            target_activation_signal.unsqueeze(0)
            if squeeze_output
            else target_activation_signal
        )
        state = {
            "MA": torch.zeros(
                signal.shape[:-1], dtype=signal.dtype, device=signal.device
            ),
            "MF": torch.zeros(
                signal.shape[:-1], dtype=signal.dtype, device=signal.device
            ),
            "MR": torch.ones(
                signal.shape[:-1], dtype=signal.dtype, device=signal.device
            ),
        }
        outputs: dict[str, list[torch.Tensor]] = {
            "MA": [],
            "MF": [],
            "MR": [],
            "effective_capacity": [],
        }

        for time_index in range(signal.shape[-1]):
            effective_capacity, state = self.forward(
                signal[..., time_index], state, dt=dt
            )
            outputs["MA"].append(state["MA"])
            outputs["MF"].append(state["MF"])
            outputs["MR"].append(state["MR"])
            outputs["effective_capacity"].append(effective_capacity)

        stacked = {
            name: torch.stack(values, dim=-1) for name, values in outputs.items()
        }
        if squeeze_output:
            return {name: value.squeeze(0) for name, value in stacked.items()}
        return stacked
