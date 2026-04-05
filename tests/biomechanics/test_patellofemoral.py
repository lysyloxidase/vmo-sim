"""Tests for patellofemoral joint mechanics."""

from __future__ import annotations

import torch

from vmo_sim.biomechanics.patellofemoral import PatellofemoralModel


def test_balanced() -> None:
    model = PatellofemoralModel()
    displacement = model.patellar_displacement(torch.tensor(0.0), torch.tensor(0.6))
    assert torch.isclose(displacement, torch.tensor(0.0))


def test_vmo_weakness() -> None:
    model = PatellofemoralModel()
    displacement = model.patellar_displacement(torch.tensor(-50.0), torch.tensor(0.6))
    assert displacement > 0.0


def test_knee_angle_dependency() -> None:
    model = PatellofemoralModel()
    displacement_low = model.patellar_displacement(
        torch.tensor(-50.0), torch.tensor(0.2)
    )
    displacement_high = model.patellar_displacement(
        torch.tensor(-50.0), torch.tensor(1.0)
    )
    assert not torch.isclose(displacement_low, displacement_high)
    assert torch.abs(displacement_high) < torch.abs(displacement_low)
