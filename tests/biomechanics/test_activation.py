"""Tests for activation dynamics."""

from __future__ import annotations

import torch

from vmo_sim.biomechanics.activation import ActivationDynamics


def _first_true_index(mask: torch.Tensor) -> int:
    matches = torch.nonzero(mask, as_tuple=False)
    return int(matches[0].item()) if matches.numel() > 0 else -1


def test_rise_time() -> None:
    dynamics = ActivationDynamics()
    excitation = torch.ones(200)
    activation = dynamics.integrate(excitation, dt=0.001, a0=0.01)
    rise_index = _first_true_index(activation >= 0.95)
    assert rise_index != -1
    assert rise_index <= 30


def test_fall_time() -> None:
    dynamics = ActivationDynamics()
    excitation = torch.zeros(400)
    activation = dynamics.integrate(excitation, dt=0.001, a0=1.0)
    fall_index = _first_true_index(activation <= 0.05)
    assert fall_index != -1
    assert fall_index <= 120


def test_steady_state() -> None:
    dynamics = ActivationDynamics()
    excitation = torch.full((1500,), 0.6)
    activation = dynamics.integrate(excitation, dt=0.001, a0=0.01)
    assert torch.isclose(activation[-1], torch.tensor(0.6), atol=1e-2)


def test_clamp() -> None:
    dynamics = ActivationDynamics()
    excitation = torch.cat([torch.ones(50), torch.zeros(100)])
    activation = dynamics.integrate(excitation, dt=0.001, a0=0.0)
    assert torch.all(activation >= 0.01)
    assert torch.all(activation <= 1.0)


def test_batch() -> None:
    dynamics = ActivationDynamics()
    excitation = torch.stack(
        [torch.linspace(0.0, 1.0, 50), torch.linspace(1.0, 0.0, 50)],
        dim=0,
    )
    batched = dynamics.integrate(excitation, dt=0.001)
    sequential = torch.stack(
        [
            dynamics.integrate(excitation[0], dt=0.001),
            dynamics.integrate(excitation[1], dt=0.001),
        ],
        dim=0,
    )
    assert torch.allclose(batched, sequential)
