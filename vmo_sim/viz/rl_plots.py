"""Visualization helpers for reinforcement-learning experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

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


def _as_numpy(values: Sequence[float] | np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().astype(np.float64)
    return np.asarray(values, dtype=np.float64)


def _load_training_rewards(log_dir: str | Path) -> np.ndarray:
    log_path = Path(log_dir)
    history_path = log_path / "training_history.json"
    if history_path.exists():
        payload = json.loads(history_path.read_text(encoding="utf-8"))
        rewards = payload.get("episode_rewards", [])
        if isinstance(rewards, list) and rewards:
            return np.asarray(rewards, dtype=np.float64)
    metrics_path = log_path / "training_metrics.json"
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        timesteps = float(payload.get("timesteps", 0.0))
        episodes = max(1, int(payload.get("episodes", 1)))
        return np.linspace(0.0, timesteps / 1000.0, episodes, dtype=np.float64)
    raise FileNotFoundError(f"No RL metrics found in {log_path}.")


def plot_training_curve(
    log_dir: str | Path, save_path: str | Path | None = None
) -> Figure:
    """Plot episode reward over training from saved RL logs."""

    _setup_style()
    rewards = _load_training_rewards(log_dir)
    episodes = np.arange(1, rewards.size + 1, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.plot(
        episodes, rewards, linewidth=2.0, color="darkorange", label="Episode reward"
    )
    if rewards.size >= 5:
        window = min(10, rewards.size)
        kernel = np.ones(window, dtype=np.float64) / float(window)
        smoothed = np.convolve(rewards, kernel, mode="same")
        ax.plot(episodes, smoothed, linewidth=2.5, color="navy", label="Moving average")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("RL Training Curve")
    ax.legend(frameon=False)
    return _finalize_figure(fig, save_path)


def plot_training_history(
    steps: Sequence[float] | np.ndarray | torch.Tensor,
    rewards: Sequence[float] | np.ndarray | torch.Tensor,
    safety: Sequence[float] | np.ndarray | torch.Tensor,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot reward and safety trajectories for backward compatibility."""

    _setup_style()
    x_values = _as_numpy(steps)
    reward_values = _as_numpy(rewards)
    safety_values = _as_numpy(safety)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.plot(x_values, reward_values, linewidth=2.2, color="darkorange", label="Reward")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax2 = ax.twinx()
    ax2.plot(x_values, safety_values, linewidth=2.0, color="firebrick", label="Safety")
    ax2.set_ylabel("Safety penalty")
    handles_1, labels_1 = ax.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(handles_1 + handles_2, labels_1 + labels_2, frameon=False, loc="best")
    ax.set_title("Training Diagnostics")
    return _finalize_figure(fig, save_path)


def _metric_value(
    result: Mapping[str, object], key: str, default: float = 0.0
) -> float:
    value = result.get(key, default)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, np.ndarray):
        return float(np.mean(value))
    if isinstance(value, list):
        return float(np.mean(np.asarray(value, dtype=np.float64)))
    return float(default)


def plot_policy_comparison(
    healthy_result: Mapping[str, object],
    pfps_result: Mapping[str, object],
    save_path: str | Path | None = None,
) -> Figure:
    """Compare policy outcomes between healthy and pathological scenarios."""

    _setup_style()
    metrics = [
        ("Mean reward", "mean_reward", 1.0),
        ("Max displacement (mm)", "mean_max_lateral_displacement", 1000.0),
        ("VMO:VL ratio", "mean_vmo_vl_ratio", 1.0),
    ]
    healthy_values = np.asarray(
        [_metric_value(healthy_result, key) * scale for _, key, scale in metrics],
        dtype=np.float64,
    )
    pfps_values = np.asarray(
        [_metric_value(pfps_result, key) * scale for _, key, scale in metrics],
        dtype=np.float64,
    )
    labels = [label for label, _, _ in metrics]
    positions = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.bar(
        positions - width / 2.0,
        healthy_values,
        width=width,
        label="Healthy",
        color="seagreen",
    )
    ax.bar(
        positions + width / 2.0,
        pfps_values,
        width=width,
        label="PFPS",
        color="firebrick",
    )
    ax.set_xticks(positions, labels=labels)
    ax.set_title("Policy Comparison Across Scenarios")
    ax.legend(frameon=False)
    return _finalize_figure(fig, save_path)


def plot_curriculum_progression(
    stage_rewards: Mapping[str, Sequence[float]] | Sequence[Sequence[float]],
    save_path: str | Path | None = None,
) -> Figure:
    """Plot mean reward achieved at each curriculum stage."""

    _setup_style()
    if isinstance(stage_rewards, Mapping):
        labels = list(stage_rewards.keys())
        series = [
            np.asarray(values, dtype=np.float64) for values in stage_rewards.values()
        ]
    else:
        series = [np.asarray(values, dtype=np.float64) for values in stage_rewards]
        labels = [f"Stage {index + 1}" for index in range(len(series))]
    means = np.asarray(
        [float(np.mean(values)) if values.size > 0 else 0.0 for values in series],
        dtype=np.float64,
    )
    stds = np.asarray(
        [float(np.std(values)) if values.size > 0 else 0.0 for values in series],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.errorbar(
        labels,
        means,
        yerr=stds,
        marker="o",
        linewidth=2.5,
        capsize=4,
        color="slateblue",
    )
    ax.set_xlabel("Curriculum stage")
    ax.set_ylabel("Reward")
    ax.set_title("Curriculum Progression")
    return _finalize_figure(fig, save_path)


__all__ = [
    "plot_curriculum_progression",
    "plot_policy_comparison",
    "plot_training_curve",
    "plot_training_history",
]
