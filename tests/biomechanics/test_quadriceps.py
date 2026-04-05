"""Tests for the full quadriceps model."""

from __future__ import annotations

import torch

from vmo_sim.biomechanics.quadriceps import QuadricepsModel


def _uniform_excitations(value: float) -> dict[str, torch.Tensor]:
    return {
        "VMO": torch.tensor(value),
        "VML": torch.tensor(value),
        "VL": torch.tensor(value),
        "RF": torch.tensor(value),
        "VI": torch.tensor(value),
    }


def test_full_activation() -> None:
    model = QuadricepsModel()
    forces, _ = model(_uniform_excitations(1.0), torch.tensor(0.6), torch.tensor(0.0))
    assert forces["knee_extension_moment"] > 0.0


def test_vmo_medial_pull() -> None:
    model = QuadricepsModel()
    excitations = _uniform_excitations(0.0)
    excitations["VMO"] = torch.tensor(1.0)
    forces, _ = model(excitations, torch.tensor(0.6), torch.tensor(0.0))
    assert forces["mediolateral_force"] > 0.0


def test_vl_lateral_pull() -> None:
    model = QuadricepsModel()
    excitations = _uniform_excitations(0.0)
    excitations["VL"] = torch.tensor(1.0)
    forces, _ = model(excitations, torch.tensor(0.6), torch.tensor(0.0))
    assert forces["mediolateral_force"] < 0.0


def test_vmo_vl_ratio() -> None:
    model = QuadricepsModel()
    forces, _ = model(_uniform_excitations(0.7), torch.tensor(0.6), torch.tensor(0.0))
    expected = forces["VMO"] / torch.clamp(forces["VL"], min=1e-6)
    assert torch.isclose(forces["vmo_vl_ratio"], expected)
