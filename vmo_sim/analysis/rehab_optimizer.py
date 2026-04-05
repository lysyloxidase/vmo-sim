"""Optimization tools for rehabilitation stimulation protocols."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
import pandas as pd  # type: ignore[import-untyped]
import torch

from vmo_sim.biomechanics.patellofemoral import PatellofemoralModel
from vmo_sim.rl.vmo_env import VMORehabEnv


class FESProtocol(BaseModel):
    """Definition of a functional electrical stimulation protocol."""

    model_config = ConfigDict(frozen=True)

    muscle_name: str
    onset_time: float
    duration: float
    amplitude: float
    frequency: float
    pulse_width: float


class RehabProtocolOptimizer:
    """Optimize rehabilitation stimulation protocols using the differentiable model."""

    def __init__(self, scenario: str = "pfps_moderate", dt: float = 0.01) -> None:
        self.scenario = scenario
        self.dt = dt
        self.patellofemoral = PatellofemoralModel()

    def _protocol_signal(
        self,
        time_points: torch.Tensor,
        onset: torch.Tensor,
        duration: torch.Tensor,
        amplitude: torch.Tensor,
    ) -> torch.Tensor:
        sharpness = torch.tensor(
            35.0, dtype=time_points.dtype, device=time_points.device
        )
        start = torch.sigmoid(sharpness * (time_points - onset))
        stop = torch.sigmoid(sharpness * (time_points - (onset + duration)))
        return torch.clamp(amplitude * (start - stop), 0.0, 1.0)

    def optimize_fes_pattern(
        self,
        target: str = "vmo_vl_balance",
        motion: str = "sit_to_stand",
        n_iterations: int = 200,
    ) -> FESProtocol:
        """Optimize an FES pattern for VMO stimulation."""

        env = VMORehabEnv(
            scenario=self.scenario,
            target_motion=motion,
            dt=self.dt,
            enable_fatigue=False,
        )
        time_points = torch.linspace(
            0.0, self.dt * (env.episode_length - 1), env.episode_length
        )
        knee_angles = torch.tensor(env.target_trajectory, dtype=torch.float32)
        knee_velocity = torch.zeros_like(knee_angles)
        knee_velocity[1:] = (knee_angles[1:] - knee_angles[:-1]) / self.dt

        onset_raw = torch.nn.Parameter(torch.tensor(0.0))
        duration_raw = torch.nn.Parameter(torch.tensor(0.0))
        amplitude_raw = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.Adam([onset_raw, duration_raw, amplitude_raw], lr=0.05)

        for _ in range(n_iterations):
            optimizer.zero_grad()
            onset = 1.5 * torch.sigmoid(onset_raw)
            duration = 0.1 + 0.8 * torch.sigmoid(duration_raw)
            amplitude = torch.sigmoid(amplitude_raw)
            signal = self._protocol_signal(time_points, onset, duration, amplitude)

            quadriceps = env._build_scenario(self.scenario)
            state: dict[str, dict[str, torch.Tensor]] | None = None
            ratios: list[torch.Tensor] = []
            displacements: list[torch.Tensor] = []
            for index in range(env.episode_length):
                excitations = {
                    "VMO": torch.clamp(0.15 + signal[index], 0.0, 1.0),
                    "VML": torch.tensor(0.35),
                    "VL": torch.tensor(0.55),
                    "RF": torch.tensor(0.15),
                    "VI": torch.tensor(0.20),
                }
                forces, state = quadriceps(
                    excitations,
                    knee_angles[index],
                    knee_velocity[index],
                    dt=self.dt,
                    state=state,
                )
                ratios.append(forces["vmo_vl_ratio"])
                displacements.append(
                    self.patellofemoral.patellar_displacement(
                        forces["mediolateral_force"], knee_angles[index]
                    )
                )

            ratio_tensor = torch.stack(ratios)
            displacement_tensor = torch.stack(displacements)
            if target == "vmo_vl_balance":
                loss = -torch.mean(torch.log(ratio_tensor + 1e-6))
            elif target == "minimize_lateral_tracking":
                loss = torch.mean(torch.relu(displacement_tensor))
            else:
                raise ValueError(f"Unsupported target: {target}")
            loss = (
                loss
                + 50.0 * torch.mean(torch.relu(displacement_tensor))
                + 0.1 * torch.mean(signal**2)
            )
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

        return FESProtocol(
            muscle_name="VMO",
            onset_time=float((1.5 * torch.sigmoid(onset_raw)).detach().cpu().item()),
            duration=float(
                (0.1 + 0.8 * torch.sigmoid(duration_raw)).detach().cpu().item()
            ),
            amplitude=float(torch.sigmoid(amplitude_raw).detach().cpu().item()),
            frequency=35.0,
            pulse_width=0.30,
        )

    def compare_protocols(
        self,
        protocols: list[FESProtocol],
        scenario: str = "pfps_moderate",
    ) -> pd.DataFrame:
        """Compare multiple FES protocols across key metrics."""

        records: list[dict[str, float | str]] = []
        for protocol in protocols:
            env = VMORehabEnv(
                scenario=scenario,
                target_motion="sit_to_stand",
                dt=self.dt,
                enable_fatigue=False,
            )
            observation, _ = env.reset()
            del observation
            rewards: list[float] = []
            ratios: list[float] = []
            displacements: list[float] = []
            time_points = torch.linspace(
                0.0, self.dt * (env.episode_length - 1), env.episode_length
            )
            signal = (
                self._protocol_signal(
                    time_points,
                    torch.tensor(protocol.onset_time),
                    torch.tensor(protocol.duration),
                    torch.tensor(protocol.amplitude),
                )
                .detach()
                .cpu()
                .numpy()
            )
            for index in range(env.episode_length):
                action = torch.tensor(
                    [min(1.0, 0.1 + signal[index]), 0.35, 0.55],
                    dtype=torch.float32,
                ).numpy()
                _, reward, terminated, truncated, info = env.step(action)
                rewards.append(reward)
                reward_state = info["reward_state"]
                ratios.append(float(reward_state["vmo_vl_ratio"]))
                displacements.append(float(reward_state["lateral_displacement"]))
                if terminated or truncated:
                    break
            records.append(
                {
                    "protocol": f"{protocol.muscle_name}@{protocol.onset_time:.2f}s",
                    "mean_reward": float(sum(rewards) / len(rewards)),
                    "mean_vmo_vl_ratio": float(sum(ratios) / len(ratios)),
                    "max_lateral_displacement_mm": 1000.0 * float(max(displacements)),
                    "amplitude": protocol.amplitude,
                }
            )
            env.close()
        return pd.DataFrame.from_records(records)


__all__ = ["FESProtocol", "RehabProtocolOptimizer"]
