"""Publication-quality plots for muscle mechanics."""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # type: ignore[import-untyped]
import torch

from vmo_sim.biomechanics.activation import ActivationDynamics
from vmo_sim.biomechanics.force_length import ForceLengthRelationship
from vmo_sim.biomechanics.force_velocity import ForceVelocityRelationship
from vmo_sim.biomechanics.parameters import MuscleParameters


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")


def _finalize_figure(fig: Figure, save_path: str | Path | None) -> Figure:
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_force_length_curve(
    params: MuscleParameters,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot active, passive, and total force-length relationships."""

    _setup_style()
    lengths = torch.linspace(0.5, 1.6, 300)
    relationship = ForceLengthRelationship(params.passive_strain_at_one_norm_force)
    active = relationship.active(lengths).detach().cpu().numpy()
    passive = relationship.passive(lengths).detach().cpu().numpy()
    total = relationship.total(lengths, torch.ones_like(lengths)).detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    x = lengths.detach().cpu().numpy()
    ax.plot(x, active, label="Active", linewidth=2.5)
    ax.plot(x, passive, label="Passive", linewidth=2.5)
    ax.plot(x, total, label="Total", linewidth=2.5)
    ax.set_xlabel("Normalized fiber length")
    ax.set_ylabel("Normalized force")
    ax.set_title(f"{params.name} Force-Length Curve")
    ax.legend(frameon=False)
    return _finalize_figure(fig, save_path)


def plot_force_velocity_curve(
    params: MuscleParameters,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot concentric and eccentric force-velocity behavior."""

    _setup_style()
    velocities = torch.linspace(-1.0, 1.0, 300)
    relationship = ForceVelocityRelationship(
        params.fv_shape_factor,
        params.max_eccentric_force_multiplier,
    )
    forces = relationship(velocities).detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    x = velocities.detach().cpu().numpy()
    ax.plot(x, forces, linewidth=2.5, color="teal")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Normalized fiber velocity")
    ax.set_ylabel("Normalized force")
    ax.set_title(f"{params.name} Force-Velocity Curve")
    ax.annotate("Shortening", xy=(-0.8, 0.4), fontsize=10)
    ax.annotate("Lengthening", xy=(0.3, 1.2), fontsize=10)
    return _finalize_figure(fig, save_path)


def plot_activation_dynamics(
    tau_act: float,
    tau_deact: float,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot the activation-dynamics step response."""

    _setup_style()
    dt = 0.001
    excitation = torch.zeros(300)
    excitation[50:180] = 1.0
    activation = ActivationDynamics(tau_act=tau_act, tau_deact=tau_deact).integrate(
        excitation, dt=dt
    )
    time = np.arange(excitation.numel(), dtype=np.float64) * dt

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.plot(time, excitation.detach().cpu().numpy(), label="Excitation", linewidth=2.0)
    ax.plot(time, activation.detach().cpu().numpy(), label="Activation", linewidth=2.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Level")
    ax.set_title("Activation Dynamics Step Response")
    ax.legend(frameon=False)
    return _finalize_figure(fig, save_path)


def plot_muscle_force_profile(
    simulation_result: dict[str, torch.Tensor],
    save_path: str | Path | None = None,
) -> Figure:
    """Plot muscle force over time from a simulation result."""

    _setup_style()
    force = simulation_result["force"].detach().cpu().numpy()
    if "time" in simulation_result:
        time = simulation_result["time"].detach().cpu().numpy()
    else:
        time = np.arange(force.shape[-1], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.plot(time, force, linewidth=2.5, color="crimson")
    ax.set_xlabel("Time")
    ax.set_ylabel("Force (N)")
    ax.set_title("Muscle Force Profile")
    return _finalize_figure(fig, save_path)


__all__ = [
    "plot_activation_dynamics",
    "plot_force_length_curve",
    "plot_force_velocity_curve",
    "plot_muscle_force_profile",
]
