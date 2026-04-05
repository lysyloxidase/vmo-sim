"""Tests for tendon mechanics."""

from __future__ import annotations

import torch

from vmo_sim.biomechanics.tendon import TendonModel


def test_zero_force_at_slack() -> None:
    model = TendonModel(tendon_slack_length=0.2)
    assert torch.isclose(model.force(torch.tensor(0.2)), torch.tensor(0.0))


def test_force_at_reference_strain() -> None:
    model = TendonModel(tendon_slack_length=0.2, strain_at_one_norm_force=0.033)
    tendon_length = torch.tensor(0.2 * (1.0 + 0.033))
    assert torch.isclose(model.force(tendon_length), torch.tensor(1.0), atol=1e-4)


def test_inverse_consistency() -> None:
    model = TendonModel(tendon_slack_length=0.2, strain_at_one_norm_force=0.033)
    recovered_length = model.inverse(torch.tensor(1.0))
    expected = torch.tensor(0.2 * (1.0 + 0.033))
    assert torch.isclose(recovered_length, expected, atol=1e-4)
