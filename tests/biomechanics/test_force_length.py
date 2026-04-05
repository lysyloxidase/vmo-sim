"""Tests for active and passive force-length relationships."""

from __future__ import annotations

import torch

from vmo_sim.biomechanics.force_length import ForceLengthRelationship


def test_active_peak() -> None:
    relationship = ForceLengthRelationship()
    value = relationship.active(torch.tensor(1.0))
    assert torch.isclose(value, torch.tensor(1.0), atol=1e-4)


def test_active_zero() -> None:
    relationship = ForceLengthRelationship()
    values = relationship.active(torch.tensor([0.5, 1.5]))
    assert values[0] < 0.05
    assert values[1] < 0.05


def test_passive_zero_below_optimal() -> None:
    relationship = ForceLengthRelationship()
    value = relationship.passive(torch.tensor(0.9))
    assert torch.isclose(value, torch.tensor(0.0))


def test_passive_nonzero_above() -> None:
    relationship = ForceLengthRelationship()
    value = relationship.passive(torch.tensor(1.6))
    assert value > 0.0


def test_symmetry() -> None:
    relationship = ForceLengthRelationship()
    left = relationship.active(torch.tensor(0.9))
    right = relationship.active(torch.tensor(1.1))
    assert abs(float(left - right)) < 0.1
