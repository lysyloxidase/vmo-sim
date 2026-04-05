"""Tests for physics-informed loss functions."""

from __future__ import annotations

import torch

from vmo_sim.biomechanics.parameters import VMOParameters
from vmo_sim.ml.losses import PhysicsLoss


def test_perfect_prediction_zero_loss() -> None:
    params = VMOParameters()
    loss_fn = PhysicsLoss()
    activation = torch.tensor(0.6)
    fiber_length = torch.tensor(1.0)
    fiber_velocity = torch.tensor(0.0)
    projection = torch.cos(torch.tensor(params.pennation_angle_at_optimal))
    predicted_force = params.max_isometric_force * activation * projection

    force_length_loss = loss_fn.force_length_consistency(
        predicted_force,
        fiber_length,
        activation,
        params,
    )
    force_velocity_loss = loss_fn.force_velocity_consistency(
        predicted_force,
        fiber_length,
        fiber_velocity,
        activation,
        params,
    )

    assert force_length_loss < 1e-8
    assert force_velocity_loss < 1e-8


def test_wrong_prediction_positive_loss() -> None:
    params = VMOParameters()
    loss_fn = PhysicsLoss()
    activation = torch.tensor(0.6)
    fiber_length = torch.tensor(1.0)
    fiber_velocity = torch.tensor(0.0)
    predicted_force = torch.tensor(0.0)

    components = {
        "force_length": loss_fn.force_length_consistency(
            predicted_force,
            fiber_length,
            activation,
            params,
        ),
        "force_velocity": loss_fn.force_velocity_consistency(
            predicted_force,
            fiber_length,
            fiber_velocity,
            activation,
            params,
        ),
    }

    assert loss_fn.combined(components) > 0.0


def test_non_negativity() -> None:
    loss_fn = PhysicsLoss()
    assert loss_fn.non_negativity(torch.tensor([-1.0, 0.5])) > 0.0
    assert torch.isclose(
        loss_fn.non_negativity(torch.tensor([0.1, 0.5])), torch.tensor(0.0)
    )
