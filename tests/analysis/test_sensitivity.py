"""Tests for Sobol sensitivity analysis."""

from __future__ import annotations

import torch

from vmo_sim.analysis.sensitivity import MuscleSensitivityAnalysis, run_sobol_analysis


def test_run_sobol_analysis() -> None:
    samples = torch.rand(18, 3)
    outputs = samples[:, 0] + 0.5 * samples[:, 1]
    result = run_sobol_analysis(samples, outputs)
    assert result.output_variable == "custom"
    assert len(result.parameter_names) == 3


def test_muscle_sensitivity_analysis_peak_force() -> None:
    analysis = MuscleSensitivityAnalysis()
    result = analysis.run(output_variable="peak_force", n_samples=16)
    assert result.output_variable == "peak_force"
    assert "max_isometric_force" in result.first_order
    assert len(result.parameter_names) == 7
