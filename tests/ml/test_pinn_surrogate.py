"""Tests for the VMO PINN surrogate."""

from __future__ import annotations

import torch

from vmo_sim.biomechanics.hill_muscle import HillMuscle
from vmo_sim.biomechanics.parameters import VMOParameters
from vmo_sim.ml.pinn_surrogate import VMOPINNSurrogate
from vmo_sim.ml.surrogate_trainer import SurrogateTrainer


def test_forward_shape() -> None:
    model = VMOPINNSurrogate(hidden_dim=32, num_layers=2)
    outputs = model(torch.rand(8), torch.rand(8) + 0.5, torch.rand(8) - 0.5)
    assert outputs["force"].shape == (8,)
    assert outputs["pennation_angle"].shape == (8,)


def test_training_decreases_loss() -> None:
    params = VMOParameters()
    hill = HillMuscle(params)
    model = VMOPINNSurrogate(hidden_dim=32, num_layers=2, muscle_params=params)
    trainer = SurrogateTrainer()
    history = trainer.train_pinn(model, hill, epochs=100, physics_weight=0.1, lr=3e-3)
    assert history["train_loss"][0] > history["train_loss"][-1]


def test_physics_loss_nonzero() -> None:
    model = VMOPINNSurrogate(hidden_dim=32, num_layers=2)
    inputs = torch.tensor(
        [
            [0.2, 0.8, -0.4],
            [0.7, 1.3, 0.5],
            [0.9, 1.5, -0.1],
        ]
    )
    outputs = model(inputs[:, 0], inputs[:, 1], inputs[:, 2])
    assert model.physics_loss(inputs, outputs) > 0.0


def test_accuracy_vs_hill() -> None:
    params = VMOParameters()
    hill = HillMuscle(params)
    model = VMOPINNSurrogate(hidden_dim=64, num_layers=3, muscle_params=params)
    trainer = SurrogateTrainer()
    trainer.train_pinn(model, hill, epochs=150, physics_weight=0.1, lr=3e-3)
    inputs, targets = model.generate_training_data(hill, n_samples=256, seed=7)
    with torch.no_grad():
        predictions = model(inputs[:, 0], inputs[:, 1], inputs[:, 2])["force"]
    rmse = torch.sqrt(torch.mean((predictions - targets[:, 0]) ** 2))
    assert rmse < 0.05
