"""Plots for patellar tracking and contact pressure."""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # type: ignore[import-untyped]


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")


def _finalize_figure(fig: Figure, save_path: str | Path | None) -> Figure:
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_patellar_tracking(
    displacement: np.ndarray,
    tilt: np.ndarray,
    knee_angle: np.ndarray,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot patellar displacement and tilt against knee angle."""

    _setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), dpi=150, sharex=True)
    axes[0].plot(knee_angle, 1000.0 * displacement, linewidth=2.5, color="firebrick")
    axes[0].set_ylabel("Displacement (mm)")
    axes[0].set_title("Patellar Tracking")
    axes[1].plot(knee_angle, np.degrees(tilt), linewidth=2.5, color="steelblue")
    axes[1].set_xlabel("Knee angle (rad)")
    axes[1].set_ylabel("Tilt (deg)")
    return _finalize_figure(fig, save_path)


def plot_contact_pressure_map(
    medial: np.ndarray,
    lateral: np.ndarray,
    knee_angle: np.ndarray,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot medial and lateral contact pressures as a heatmap."""

    _setup_style()
    pressure_map = np.vstack([medial, lateral])
    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=150)
    image = ax.imshow(pressure_map, aspect="auto", cmap="mako", origin="lower")
    ax.set_yticks([0, 1], labels=["Medial", "Lateral"])
    tick_positions = np.linspace(0, knee_angle.shape[-1] - 1, 5, dtype=int)
    ax.set_xticks(
        tick_positions, labels=[f"{knee_angle[idx]:.2f}" for idx in tick_positions]
    )
    ax.set_xlabel("Knee angle (rad)")
    ax.set_title("Patellofemoral Contact Pressure")
    fig.colorbar(image, ax=ax, label="Pressure (normalized)")
    return _finalize_figure(fig, save_path)


__all__ = ["plot_contact_pressure_map", "plot_patellar_tracking"]
