"""Reward functions for VMO rehabilitation control."""

from __future__ import annotations

import numpy as np


class RehabReward:
    """Multi-objective reward function for VMO rehabilitation."""

    def __init__(
        self,
        w_track: float = 1.0,
        w_energy: float = -0.1,
        w_safety: float = -5.0,
        w_balance: float = 0.5,
        w_smooth: float = -0.05,
    ) -> None:
        self.w_track = abs(w_track)
        self.w_energy = abs(w_energy)
        self.w_safety = abs(w_safety)
        self.w_balance = abs(w_balance)
        self.w_smooth = abs(w_smooth)

    def tracking(self, current_angle: float, target_angle: float) -> float:
        """Return negative squared tracking error."""

        error = current_angle - target_angle
        return -(error**2)

    def energy(self, excitations: np.ndarray) -> float:
        """Return a negative metabolic-cost proxy."""

        return -float(np.sum(np.square(excitations)))

    def safety(self, lateral_displacement: float, contact_pressure: float) -> float:
        """Return a large negative penalty for unsafe tracking or contact."""

        displacement_excess = max(0.0, lateral_displacement - 0.005) / 0.005
        pressure_excess = max(0.0, contact_pressure - 1.0)
        if displacement_excess == 0.0 and pressure_excess == 0.0:
            return 0.0
        return -10.0 * (1.0 + displacement_excess + pressure_excess)

    def balance(self, vmo_vl_ratio: float) -> float:
        """Return a bonus for maintaining a healthy VMO:VL ratio."""

        deviation = (vmo_vl_ratio - 1.0) / 0.2
        return float(max(0.0, 1.0 - deviation**2))

    def smoothness(
        self,
        current_action: np.ndarray,
        previous_action: np.ndarray,
    ) -> float:
        """Return a penalty for abrupt action changes."""

        return -float(np.mean(np.square(current_action - previous_action)))

    def compute(
        self,
        state: dict[str, float],
        action: np.ndarray,
        previous_action: np.ndarray | None = None,
    ) -> float:
        """Compute the total rehabilitation reward."""

        tracking_term = self.tracking(state["current_angle"], state["target_angle"])
        energy_term = self.energy(action)
        safety_term = self.safety(
            state["lateral_displacement"],
            state["contact_pressure"],
        )
        balance_term = self.balance(state["vmo_vl_ratio"])
        smooth_term = (
            self.smoothness(action, previous_action)
            if previous_action is not None
            else 0.0
        )

        return (
            self.w_track * tracking_term
            + self.w_energy * energy_term
            + self.w_safety * safety_term
            + self.w_balance * balance_term
            + self.w_smooth * smooth_term
        )


def tracking_reward(current_value: float, target_value: float) -> float:
    """Functional wrapper for the tracking reward."""

    return RehabReward().tracking(current_value, target_value)


def energy_penalty(muscle_activations: np.ndarray) -> float:
    """Functional wrapper for the energy penalty."""

    return RehabReward().energy(muscle_activations)


def safety_penalty(patellar_displacement: float, contact_pressure: float) -> float:
    """Functional wrapper for the safety penalty."""

    return RehabReward().safety(patellar_displacement, contact_pressure)


def combined_reward(
    tracking: float,
    energy: float,
    safety: float,
    energy_weight: float,
    safety_weight: float,
) -> float:
    """Combine the main reward terms with explicit weights."""

    return tracking + abs(energy_weight) * energy + abs(safety_weight) * safety
