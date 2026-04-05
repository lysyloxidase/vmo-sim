"""Tests for the differentiable Hill muscle model."""

from __future__ import annotations

import torch

from vmo_sim.biomechanics.hill_muscle import HillMuscle
from vmo_sim.biomechanics.parameters import VLParameters, VMOParameters


def test_isometric_force(default_vmo_params: VMOParameters) -> None:
    muscle = HillMuscle(default_vmo_params)
    force = muscle.compute_force(
        activation=torch.tensor(0.6),
        normalized_fiber_length=torch.tensor(1.0),
        normalized_fiber_velocity=torch.tensor(0.0),
    )
    expected = (
        default_vmo_params.max_isometric_force
        * 0.6
        * torch.cos(torch.tensor(default_vmo_params.pennation_angle_at_optimal))
    )
    assert torch.isclose(force, expected, rtol=0.05, atol=1e-3)


def test_passive_stretch(default_vmo_params: VMOParameters) -> None:
    muscle = HillMuscle(default_vmo_params)
    force = muscle.compute_force(
        activation=torch.tensor(0.0),
        normalized_fiber_length=torch.tensor(1.4),
        normalized_fiber_velocity=torch.tensor(0.0),
    )
    assert force > 0.0


def test_simulate_twitch(
    default_vmo_params: VMOParameters,
    sample_excitation_signal: torch.Tensor,
    sample_musculotendon_length: torch.Tensor,
    sample_velocity_signal: torch.Tensor,
) -> None:
    muscle = HillMuscle(default_vmo_params)
    results = muscle.simulate(
        sample_excitation_signal,
        sample_musculotendon_length,
        sample_velocity_signal,
        dt=0.001,
    )
    peak_index = int(torch.argmax(results["force"]).item())
    assert results["force"].max() > results["force"][0]
    assert peak_index >= 10
    assert results["force"][-1] < results["force"].max()


def test_differentiable(default_vmo_params: VMOParameters) -> None:
    muscle = HillMuscle(default_vmo_params)
    excitation = torch.full((30,), 0.4, requires_grad=True)
    optimal_projection = torch.cos(
        torch.tensor(default_vmo_params.pennation_angle_at_optimal)
    ).item()
    length = default_vmo_params.tendon_slack_length + (
        default_vmo_params.optimal_fiber_length * optimal_projection
    )
    musculotendon_length = torch.full((30,), length)
    musculotendon_velocity = torch.zeros(30)
    results = muscle.simulate(
        excitation, musculotendon_length, musculotendon_velocity, dt=0.001
    )
    gradient = torch.autograd.grad(results["force"].sum(), excitation)[0]
    assert gradient is not None
    assert torch.all(torch.isfinite(gradient))


def test_vmo_vs_vl() -> None:
    vmo = HillMuscle(VMOParameters())
    vl = HillMuscle(VLParameters())
    vmo_force = vmo.compute_force(
        activation=torch.tensor(0.7),
        normalized_fiber_length=torch.tensor(1.0),
        normalized_fiber_velocity=torch.tensor(0.0),
    )
    vl_force = vl.compute_force(
        activation=torch.tensor(0.7),
        normalized_fiber_length=torch.tensor(1.0),
        normalized_fiber_velocity=torch.tensor(0.0),
    )
    vmo_angle = vmo.pennation.angle(torch.tensor(1.0))
    vl_angle = vl.pennation.angle(torch.tensor(1.0))
    assert vl_force > vmo_force
    assert vmo_angle > vl_angle
