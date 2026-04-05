"""Shared fixtures for the VMO-Sim test suite."""

from __future__ import annotations

import pytest
import torch

from vmo_sim.biomechanics.parameters import VMOParameters


@pytest.fixture
def default_vmo_params() -> VMOParameters:
    """Return the default published VMO parameter set."""

    return VMOParameters()


@pytest.fixture
def sample_excitation_signal() -> torch.Tensor:
    """Return a short excitation pulse for twitch-style testing."""

    signal = torch.zeros(120)
    signal[10:20] = 1.0
    return signal


@pytest.fixture
def sample_musculotendon_length(default_vmo_params: VMOParameters) -> torch.Tensor:
    """Return a near-optimal constant musculotendon length."""

    optimal_projection = torch.cos(
        torch.tensor(default_vmo_params.pennation_angle_at_optimal)
    ).item()
    length = default_vmo_params.tendon_slack_length + (
        default_vmo_params.optimal_fiber_length * optimal_projection
    )
    return torch.full((120,), length)


@pytest.fixture
def sample_velocity_signal() -> torch.Tensor:
    """Return a zero musculotendon velocity signal."""

    return torch.zeros(120)
