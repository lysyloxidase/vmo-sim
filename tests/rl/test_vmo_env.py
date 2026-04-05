"""Tests for the VMO rehabilitation environment."""

from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from vmo_sim.rl.vmo_env import VMORehabEnv


def test_reset() -> None:
    env = VMORehabEnv()
    observation, info = env.reset()
    assert observation.shape == (13,)
    assert observation.dtype == np.float32
    assert "scenario" in info


def test_step() -> None:
    env = VMORehabEnv()
    env.reset()
    action = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    observation, reward, terminated, truncated, info = env.step(action)
    assert observation.shape == (13,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_observation_range() -> None:
    env = VMORehabEnv()
    observation, _ = env.reset()
    assert np.all(np.isfinite(observation))
    assert env.observation_space.contains(observation)


def test_scenario_healthy() -> None:
    env = VMORehabEnv(scenario="healthy")
    observation, _ = env.reset()
    assert observation.shape == (13,)


def test_scenario_pfps() -> None:
    healthy_env = VMORehabEnv(scenario="healthy")
    pfps_env = VMORehabEnv(scenario="pfps_mild")
    assert (
        pfps_env.quadriceps.params["VMO"].max_isometric_force
        < healthy_env.quadriceps.params["VMO"].max_isometric_force
    )


def test_episode_length() -> None:
    env = VMORehabEnv()
    env.reset()
    action = np.array([0.4, 0.4, 0.4], dtype=np.float32)
    terminated = False
    truncated = False
    steps = 0
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(action)
        steps += 1
    assert steps == env.episode_length
    assert truncated


def test_gymnasium_check() -> None:
    env = VMORehabEnv()
    check_env(env)
