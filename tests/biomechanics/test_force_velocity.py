"""Tests for the force-velocity relationship."""

from __future__ import annotations

import torch

from vmo_sim.biomechanics.force_velocity import ForceVelocityRelationship


def test_isometric() -> None:
    relationship = ForceVelocityRelationship()
    value = relationship(torch.tensor(0.0))
    assert torch.isclose(value, torch.tensor(1.0), atol=1e-4)


def test_shortening() -> None:
    relationship = ForceVelocityRelationship()
    value = relationship(torch.tensor(-0.5))
    assert value < 1.0


def test_lengthening() -> None:
    relationship = ForceVelocityRelationship()
    value = relationship(torch.tensor(0.5))
    assert value > 1.0


def test_max_eccentric() -> None:
    relationship = ForceVelocityRelationship(max_eccentric_multiplier=1.4)
    value = relationship(torch.tensor(1.0))
    assert torch.isclose(value, torch.tensor(1.4), atol=1e-4)
