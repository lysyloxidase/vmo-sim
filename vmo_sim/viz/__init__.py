"""Visualization utilities for VMO-Sim."""

from __future__ import annotations

from vmo_sim.viz.activation_plots import (
    plot_emg_signals,
    plot_excitation_vs_activation,
    plot_vmo_vl_ratio_over_time,
)
from vmo_sim.viz.interactive_dashboard import build_dashboard, main
from vmo_sim.viz.muscle_plots import (
    plot_activation_dynamics,
    plot_force_length_curve,
    plot_force_velocity_curve,
    plot_muscle_force_profile,
)
from vmo_sim.viz.patellar_plots import plot_contact_pressure_map, plot_patellar_tracking
from vmo_sim.viz.rl_plots import (
    plot_curriculum_progression,
    plot_policy_comparison,
    plot_training_curve,
    plot_training_history,
)

__all__ = [
    "build_dashboard",
    "main",
    "plot_activation_dynamics",
    "plot_contact_pressure_map",
    "plot_curriculum_progression",
    "plot_emg_signals",
    "plot_excitation_vs_activation",
    "plot_force_length_curve",
    "plot_force_velocity_curve",
    "plot_muscle_force_profile",
    "plot_patellar_tracking",
    "plot_policy_comparison",
    "plot_training_curve",
    "plot_training_history",
    "plot_vmo_vl_ratio_over_time",
]
