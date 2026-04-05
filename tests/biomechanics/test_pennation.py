"""Tests for pennation kinematics."""

from __future__ import annotations

import torch

from vmo_sim.biomechanics.pennation import PennationModel


def test_angle_at_optimal_length() -> None:
    model = PennationModel(0.5)
    angle = model.angle(torch.tensor(1.0))
    assert torch.isclose(angle, torch.tensor(0.5), atol=1e-4)


def test_shortening_increases_angle() -> None:
    model = PennationModel(0.5)
    short_angle = model.angle(torch.tensor(0.8))
    long_angle = model.angle(torch.tensor(1.2))
    assert short_angle > long_angle


def test_projection_decreases_with_shortening() -> None:
    model = PennationModel(0.5)
    short_projection = model.projection_factor(torch.tensor(0.8))
    long_projection = model.projection_factor(torch.tensor(1.2))
    assert short_projection < long_projection
