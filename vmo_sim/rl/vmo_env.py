"""Gymnasium environment for VMO rehabilitation protocol optimization."""

from __future__ import annotations

from typing import Any, cast

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium.core import RenderFrame
from gymnasium import spaces
from matplotlib.backends.backend_agg import FigureCanvasAgg
from numpy.typing import NDArray

from vmo_sim.biomechanics.fatigue import FatigueModel
from vmo_sim.biomechanics.parameters import get_default_quadriceps
from vmo_sim.biomechanics.patellofemoral import PatellofemoralModel
from vmo_sim.biomechanics.quadriceps import QuadricepsModel
from vmo_sim.rl.reward import RehabReward

Float32Array = NDArray[np.float32]


class VMORehabEnv(gym.Env[Float32Array, Float32Array]):
    """Gymnasium environment for VMO rehabilitation protocol optimization."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(
        self,
        scenario: str = "healthy",
        target_motion: str = "sit_to_stand",
        render_mode: str | None = None,
        dt: float = 0.01,
        custom_modifications: dict[str, dict[str, float]] | None = None,
        enable_fatigue: bool = True,
    ) -> None:
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        self.dt = dt
        self.scenario = scenario
        self.target_motion = target_motion
        self.render_mode = render_mode
        self.custom_modifications = custom_modifications or {}
        self.enable_fatigue = enable_fatigue
        self.episode_length = int(round(2.0 / dt))
        self.reward_model = RehabReward()
        self.contact_pressure_scale = 4.0e6
        self.knee_inertia = 2.5
        self.gravity_scale = 6.0
        self.knee_damping = 1.2
        self.passive_stiffness = 1.5
        self.rest_angle = 0.2
        self.vmo_delay_steps = 0
        self.vmo_inhibition_scale = 1.0

        self.quadriceps = self._build_scenario(scenario)
        self.patellofemoral = PatellofemoralModel()
        self.fatigue_model = FatigueModel()

        low = np.array(
            [0.0, -10.0, 0.0, 0.0, 0.0, 0.5, 0.0, -0.015, -0.5, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        high = np.array(
            [2.2, 10.0, 1.0, 1.0, 1.0, 1.8, 2.5, 0.015, 0.5, 2.2, 1.0, 5.0, 5.0],
            dtype=np.float32,
        )
        self._obs_low = low
        self._obs_high = high
        self._action_low = np.zeros(3, dtype=np.float32)
        self._action_high = np.ones(3, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=self._obs_low, high=self._obs_high, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=self._action_low, high=self._action_high, dtype=np.float32
        )

        self.target_trajectory = self._generate_target(target_motion)
        self.step_count = 0
        self.knee_angle = float(self.target_trajectory[0])
        self.knee_angular_velocity = 0.0
        self.previous_action: Float32Array | None = None
        self.current_action = np.zeros(3, dtype=np.float32)
        self.quadriceps_state: dict[str, dict[str, torch.Tensor]] | None = None
        self.fatigue_state = {
            "MA": torch.tensor(0.0),
            "MF": torch.tensor(0.0),
            "MR": torch.tensor(1.0),
        }
        self.latest_forces: dict[str, torch.Tensor] = {}
        self.latest_patellar_displacement = 0.0
        self.latest_patellar_tilt = 0.0
        self.latest_medial_pressure = 0.0
        self.latest_lateral_pressure = 0.0
        self._vmo_buffer = np.zeros(self.vmo_delay_steps + 1, dtype=np.float32)

    def _build_scenario(self, scenario: str) -> QuadricepsModel:
        """Modify quadriceps parameters based on the clinical scenario."""

        params = get_default_quadriceps()

        def scale_muscle(name: str, factor: float) -> None:
            muscle = params[name]
            params[name] = muscle.model_copy(
                update={"max_isometric_force": muscle.max_isometric_force * factor}
            )

        if scenario == "healthy":
            self.vmo_delay_steps = 0
        elif scenario == "pfps_mild":
            scale_muscle("VMO", 0.8)
            self.vmo_delay_steps = int(round(0.015 / self.dt))
        elif scenario == "pfps_moderate":
            scale_muscle("VMO", 0.6)
            self.vmo_delay_steps = int(round(0.030 / self.dt))
        elif scenario == "post_acl":
            for name in params:
                scale_muscle(name, 0.85)
            scale_muscle("VMO", 0.74)
            self.vmo_delay_steps = int(round(0.010 / self.dt))
        elif scenario == "post_surgical":
            for name in params:
                scale_muscle(name, 0.9)
            scale_muscle("VMO", 0.75)
            self.vmo_delay_steps = int(round(0.020 / self.dt))
            self.vmo_inhibition_scale = 0.8
            self.knee_damping = 1.5
            self.passive_stiffness = 2.0
        elif scenario == "custom":
            for muscle_name, updates in self.custom_modifications.items():
                if muscle_name in params:
                    params[muscle_name] = params[muscle_name].model_copy(update=updates)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        return QuadricepsModel(params=params)

    def _generate_target(self, motion: str) -> np.ndarray:
        """Generate a target knee-angle trajectory."""

        time_points = np.linspace(0.0, 2.0, self.episode_length, dtype=np.float32)
        if motion == "sit_to_stand":
            trajectory = 0.2 + 1.3 * 0.5 * (1.0 + np.cos(np.pi * time_points / 2.0))
        elif motion == "squat":
            trajectory = 0.2 + 1.2 * np.sin(np.pi * time_points / 2.0) ** 2
        elif motion == "stair_climb":
            trajectory = 0.7 + 0.35 * np.sin(
                2.0 * np.pi * time_points / 2.0 - np.pi / 2.0
            )
            trajectory = np.clip(trajectory, 0.2, 1.3)
        elif motion == "isometric":
            trajectory = np.full(self.episode_length, 0.9, dtype=np.float32)
        elif motion == "isokinetic":
            phase = np.linspace(0.0, 1.0, self.episode_length, dtype=np.float32)
            trajectory = 0.4 + 0.8 * np.where(
                phase < 0.5, 2.0 * phase, 2.0 * (1.0 - phase)
            )
        else:
            raise ValueError(f"Unsupported target_motion: {motion}")
        return np.asarray(trajectory, dtype=np.float32)

    def _delayed_vmo_excitation(self, excitation: float) -> float:
        if self.vmo_delay_steps == 0:
            return excitation
        self._vmo_buffer = np.roll(self._vmo_buffer, -1)
        self._vmo_buffer[-1] = excitation
        return float(self._vmo_buffer[0])

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Float32Array, dict[str, Any]]:
        """Reset the environment to the initial rehabilitation state."""

        super().reset(seed=seed)
        del options
        self.step_count = 0
        self.target_trajectory = self._generate_target(self.target_motion)
        self.knee_angle = float(self.target_trajectory[0])
        self.knee_angular_velocity = 0.0
        self.previous_action = None
        self.current_action = np.zeros(3, dtype=np.float32)
        self.quadriceps_state = None
        self.fatigue_state = {
            "MA": torch.tensor(0.0),
            "MF": torch.tensor(0.0),
            "MR": torch.tensor(1.0),
        }
        self.latest_forces = {
            "VMO": torch.tensor(0.0),
            "VML": torch.tensor(0.0),
            "VL": torch.tensor(0.0),
            "total_force": torch.tensor(0.0),
            "knee_extension_moment": torch.tensor(0.0),
            "mediolateral_force": torch.tensor(0.0),
            "vmo_vl_ratio": torch.tensor(1.0),
        }
        self.latest_patellar_displacement = 0.0
        self.latest_patellar_tilt = 0.0
        self.latest_medial_pressure = 0.0
        self.latest_lateral_pressure = 0.0
        self._vmo_buffer = np.zeros(self.vmo_delay_steps + 1, dtype=np.float32)
        return self._compute_observation(), {
            "scenario": self.scenario,
            "target_motion": self.target_motion,
        }

    def _controlled_excitations(self, action: Float32Array) -> dict[str, torch.Tensor]:
        """Compute scenario- and fatigue-adjusted excitations."""

        effective_capacity = 1.0
        if self.enable_fatigue:
            cap_tensor, self.fatigue_state = self.fatigue_model(
                torch.tensor(float(np.mean(action, dtype=np.float32))),
                self.fatigue_state,
                dt=self.dt,
            )
            effective_capacity = float(cap_tensor.item())
        vmo_excitation = (
            self._delayed_vmo_excitation(float(action[0])) * self.vmo_inhibition_scale
        )
        scale = effective_capacity
        return {
            "VMO": torch.tensor(np.clip(vmo_excitation * scale, 0.0, 1.0)),
            "VML": torch.tensor(np.clip(float(action[1]) * scale, 0.0, 1.0)),
            "VL": torch.tensor(np.clip(float(action[2]) * scale, 0.0, 1.0)),
            "RF": torch.tensor(0.15 * scale),
            "VI": torch.tensor(0.20 * scale),
        }

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[Float32Array, float, bool, bool, dict[str, Any]]:
        """Advance the environment by one timestep."""

        clipped_action = np.clip(
            np.asarray(action, dtype=np.float32), self._action_low, self._action_high
        )
        self.current_action = clipped_action.astype(np.float32)
        forces, self.quadriceps_state = self.quadriceps(
            self._controlled_excitations(self.current_action),
            torch.tensor(self.knee_angle, dtype=torch.float32),
            torch.tensor(self.knee_angular_velocity, dtype=torch.float32),
            dt=self.dt,
            state=self.quadriceps_state,
        )
        self.latest_forces = forces
        current_target = float(self.target_trajectory[self.step_count])

        extension_moment = float(forces["knee_extension_moment"].item())
        gravity_torque = self.gravity_scale * np.sin(self.knee_angle + 0.15)
        passive_torque = self.passive_stiffness * (self.knee_angle - self.rest_angle)
        angular_acc = (
            gravity_torque
            - 0.08 * extension_moment
            - self.knee_damping * self.knee_angular_velocity
            - passive_torque
        ) / self.knee_inertia
        self.knee_angular_velocity = float(
            np.clip(self.knee_angular_velocity + self.dt * angular_acc, -8.0, 8.0)
        )
        self.knee_angle = float(
            np.clip(self.knee_angle + self.dt * self.knee_angular_velocity, 0.0, 2.2)
        )

        ml_force = forces["mediolateral_force"]
        knee_tensor = torch.tensor(self.knee_angle, dtype=torch.float32)
        self.latest_patellar_displacement = float(
            self.patellofemoral.patellar_displacement(ml_force, knee_tensor).item()
        )
        self.latest_patellar_tilt = float(
            self.patellofemoral.patellar_tilt(ml_force, knee_tensor).item()
        )
        medial_pressure, lateral_pressure = self.patellofemoral.contact_pressure(
            forces["total_force"],
            knee_tensor,
            torch.tensor(self.latest_patellar_displacement, dtype=torch.float32),
        )
        self.latest_medial_pressure = float(
            medial_pressure.item() / self.contact_pressure_scale
        )
        self.latest_lateral_pressure = float(
            lateral_pressure.item() / self.contact_pressure_scale
        )

        reward_state = {
            "current_angle": self.knee_angle,
            "target_angle": current_target,
            "lateral_displacement": max(0.0, self.latest_patellar_displacement),
            "contact_pressure": max(
                self.latest_medial_pressure, self.latest_lateral_pressure
            ),
            "vmo_vl_ratio": float(forces["vmo_vl_ratio"].item()),
        }
        reward = self.reward_model.compute(
            reward_state, self.current_action, self.previous_action
        )
        self.previous_action = self.current_action.copy()

        terminated = (
            self.latest_patellar_displacement > 0.010
            or reward_state["contact_pressure"] > 1.5
        )
        self.step_count += 1
        truncated = self.step_count >= self.episode_length
        info = {
            "target_angle": current_target,
            "reward_state": reward_state,
            "extension_moment": extension_moment,
        }
        return self._compute_observation(), float(reward), terminated, truncated, info

    def _compute_observation(self) -> Float32Array:
        """Assemble the 13-dimensional observation vector."""

        target_index = min(self.step_count, self.episode_length - 1)
        if self.quadriceps_state is None:
            activations = (0.01, 0.01, 0.01)
            vmo_fiber_length = 1.0
            vmo_force = 0.0
        else:
            activations = (
                float(self.quadriceps_state["VMO"]["activation"].item()),
                float(self.quadriceps_state["VML"]["activation"].item()),
                float(self.quadriceps_state["VL"]["activation"].item()),
            )
            vmo_fiber_length = float(
                self.quadriceps_state["VMO"]["fiber_length"].item()
                / self.quadriceps.params["VMO"].optimal_fiber_length
            )
            vmo_force = float(
                self.latest_forces["VMO"].item()
                / self.quadriceps.params["VMO"].max_isometric_force
            )

        observation = np.array(
            [
                self.knee_angle,
                self.knee_angular_velocity,
                activations[0],
                activations[1],
                activations[2],
                vmo_fiber_length,
                vmo_force,
                self.latest_patellar_displacement,
                self.latest_patellar_tilt,
                float(self.target_trajectory[target_index]),
                float(self.fatigue_state["MF"].item()),
                self.latest_medial_pressure,
                self.latest_lateral_pressure,
            ],
            dtype=np.float32,
        )
        return np.clip(observation, self._obs_low, self._obs_high).astype(np.float32)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Render the current state as a matplotlib figure or RGB array."""

        if self.render_mode is None:
            return None
        figure, axes = plt.subplots(1, 2, figsize=(8, 3))
        target = float(
            self.target_trajectory[min(self.step_count, self.episode_length - 1)]
        )
        axes[0].bar(
            ["Current", "Target"],
            [self.knee_angle, target],
            color=["steelblue", "darkorange"],
        )
        axes[0].set_ylim(0.0, 2.2)
        axes[0].set_title("Knee Angle")
        axes[1].bar(
            ["VMO", "VML", "VL"],
            self.current_action,
            color=["firebrick", "seagreen", "slateblue"],
        )
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_title("Excitations")
        figure.tight_layout()

        canvas = FigureCanvasAgg(figure)
        canvas.draw()  # type: ignore[no-untyped-call]
        width, height = canvas.get_width_height()
        image = np.frombuffer(
            canvas.buffer_rgba(),  # type: ignore[no-untyped-call]
            dtype=np.uint8,
        ).reshape(height, width, 4)[..., :3]

        if self.render_mode == "human":
            plt.show(block=False)
            plt.pause(0.001)
            plt.close(figure)
            return None

        plt.close(figure)
        return cast(RenderFrame, image)

    def close(self) -> None:
        """Release rendering resources."""

        plt.close("all")
