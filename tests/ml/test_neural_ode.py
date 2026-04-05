"""Tests for the muscle Neural ODE model."""

from __future__ import annotations

import torch

from vmo_sim.biomechanics.hill_muscle import HillMuscle
from vmo_sim.biomechanics.parameters import VMOParameters
from vmo_sim.ml.neural_ode import MuscleNeuralODE


def test_forward_shape() -> None:
    model = MuscleNeuralODE(hidden_dim=16)
    initial_state = torch.tensor([[0.07, 0.1, 0.0], [0.07, 0.2, 0.05]])
    excitation_signal = torch.ones(2, 20) * 0.4
    t_span = torch.linspace(0.0, 0.019, 20)
    trajectory = model(initial_state, excitation_signal, t_span)
    assert trajectory.shape == (20, 2, 3)


def test_physics_only_matches_hill() -> None:
    params = VMOParameters()
    model = MuscleNeuralODE(hidden_dim=16, muscle_params=params)
    hill = HillMuscle(params)
    t_span = torch.linspace(0.0, 0.039, 40)
    excitation = torch.zeros(40)
    excitation[10:] = 0.5
    initial_state = torch.tensor([params.optimal_fiber_length, 0.01, 0.0])
    node_trajectory = model.physics_only(initial_state, excitation, t_span)

    musculotendon_length = torch.full(
        (40,),
        params.tendon_slack_length
        + params.optimal_fiber_length
        * torch.cos(torch.tensor(params.pennation_angle_at_optimal)).item(),
    )
    hill_results = hill.simulate(
        excitation,
        musculotendon_length,
        torch.zeros_like(musculotendon_length),
        dt=0.001,
    )

    activation_error = torch.mean(
        torch.abs(node_trajectory[:, 1] - hill_results["activation"])
    )
    fiber_length_error = torch.mean(
        torch.abs(node_trajectory[:, 0] - hill_results["fiber_length"])
    )
    assert activation_error < 0.05
    assert fiber_length_error < 0.01


def test_neural_correction_small() -> None:
    model = MuscleNeuralODE(hidden_dim=16)
    state = torch.tensor([[0.07, 0.2, 0.1], [0.065, 0.4, 0.0]])
    excitation = torch.tensor([0.3, 0.6])
    correction = model.neural_correction(state, excitation)
    physics = model.physics_rhs(state, excitation)
    assert torch.norm(correction) <= 0.2 * torch.norm(physics) + 1e-6
