"""Plots for excitation, activation, and EMG signals."""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # type: ignore[import-untyped]
import torch


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")


def _finalize_figure(fig: Figure, save_path: str | Path | None) -> Figure:
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_excitation_vs_activation(
    sim_result: dict[str, torch.Tensor | np.ndarray],
    save_path: str | Path | None = None,
) -> Figure:
    """Plot excitation and activation on a shared timeline."""

    _setup_style()
    excitation = np.asarray(sim_result["excitation"])
    activation = np.asarray(sim_result["activation"])
    time = np.asarray(
        sim_result.get("time", np.arange(excitation.shape[-1], dtype=np.float64))
    )

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.plot(time, excitation, label="Excitation", linewidth=2.0)
    ax.plot(time, activation, label="Activation", linewidth=2.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Level")
    ax.set_title("Excitation vs Activation")
    ax.legend(frameon=False)
    return _finalize_figure(fig, save_path)


def plot_emg_signals(
    vmo_emg: np.ndarray,
    vl_emg: np.ndarray,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot dual-channel VMO and VL EMG traces."""

    _setup_style()
    time = np.arange(vmo_emg.shape[-1], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot(time, vmo_emg, label="VMO EMG", linewidth=1.8, color="firebrick")
    ax.plot(time, vl_emg, label="VL EMG", linewidth=1.8, color="navy")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.set_title("VMO and VL EMG Signals")
    ax.legend(frameon=False)
    return _finalize_figure(fig, save_path)


def plot_vmo_vl_ratio_over_time(
    sim_result: dict[str, torch.Tensor | np.ndarray],
    save_path: str | Path | None = None,
) -> Figure:
    """Plot the VMO:VL ratio over time with a healthy target band."""

    _setup_style()
    if "VMO_force" in sim_result and "VL_force" in sim_result:
        vmo = np.asarray(sim_result["VMO_force"])
        vl = np.asarray(sim_result["VL_force"])
    elif "VMO_activation" in sim_result and "VL_activation" in sim_result:
        vmo = np.asarray(sim_result["VMO_activation"])
        vl = np.asarray(sim_result["VL_activation"])
    else:
        raise KeyError(
            "sim_result must contain VMO/VL force or activation time series."
        )
    ratio = vmo / np.maximum(vl, 1e-6)
    time = np.asarray(
        sim_result.get("time", np.arange(ratio.shape[-1], dtype=np.float64))
    )

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.fill_between(time, 0.9, 1.1, color="seagreen", alpha=0.15, label="Healthy band")
    ax.plot(time, ratio, linewidth=2.5, color="darkorange", label="VMO:VL ratio")
    ax.set_xlabel("Time")
    ax.set_ylabel("Ratio")
    ax.set_title("VMO:VL Ratio Over Time")
    ax.legend(frameon=False)
    return _finalize_figure(fig, save_path)


__all__ = [
    "plot_emg_signals",
    "plot_excitation_vs_activation",
    "plot_vmo_vl_ratio_over_time",
]
