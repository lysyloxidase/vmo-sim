"""Tests for rehabilitation reward functions."""

from __future__ import annotations

import numpy as np

from vmo_sim.rl.reward import RehabReward


def test_tracking_zero_error() -> None:
    reward = RehabReward()
    assert reward.tracking(1.0, 1.0) == 0.0
    assert reward.tracking(1.2, 1.0) < 0.0


def test_safety_penalty() -> None:
    reward = RehabReward()
    assert reward.safety(0.008, 1.2) < -10.0


def test_balance_bonus() -> None:
    reward = RehabReward()
    assert reward.balance(1.0) > 0.0
    assert reward.balance(0.5) == 0.0


def test_smoothness_penalty() -> None:
    reward = RehabReward()
    current = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    previous = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert reward.smoothness(current, previous) < 0.0
