"""Global sensitivity analysis for VMO biomechanics."""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel, ConfigDict
from SALib.analyze import sobol  # type: ignore[import-untyped]
from SALib.sample import sobol as sobol_sample  # type: ignore[import-untyped]
import numpy as np
import torch

from vmo_sim.biomechanics.fatigue import FatigueModel
from vmo_sim.biomechanics.hill_muscle import HillMuscle
from vmo_sim.biomechanics.parameters import VMOParameters, get_default_quadriceps
from vmo_sim.biomechanics.patellofemoral import PatellofemoralModel
from vmo_sim.biomechanics.quadriceps import QuadricepsModel


class SensitivityResult(BaseModel):
    """Container for Sobol sensitivity indices."""

    model_config = ConfigDict(frozen=True)

    parameter_names: list[str]
    first_order: dict[str, float]
    total_order: dict[str, float]
    confidence: dict[str, float]
    output_variable: str


class MuscleSensitivityAnalysis:
    """Sobol global sensitivity analysis on Hill-type VMO parameters."""

    PARAMETER_NAMES = [
        "max_isometric_force",
        "optimal_fiber_length",
        "pennation_angle_at_optimal",
        "activation_time_constant",
        "tendon_slack_length",
        "max_contraction_velocity",
        "passive_strain_at_one_norm_force",
    ]

    def __init__(self, nominal_params: VMOParameters | None = None) -> None:
        self.nominal_params = nominal_params or VMOParameters()
        self.patellofemoral = PatellofemoralModel()

    def _problem_definition(self) -> dict[str, object]:
        bounds: list[list[float]] = []
        for name in self.PARAMETER_NAMES:
            nominal = float(getattr(self.nominal_params, name))
            bounds.append([0.7 * nominal, 1.3 * nominal])
        return {
            "num_vars": len(self.PARAMETER_NAMES),
            "names": self.PARAMETER_NAMES,
            "bounds": bounds,
        }

    def _params_from_sample(self, sample: np.ndarray) -> VMOParameters:
        updates = {
            name: float(value)
            for name, value in zip(self.PARAMETER_NAMES, sample, strict=True)
        }
        return self.nominal_params.model_copy(update=updates)

    def _evaluate_output(self, params: VMOParameters, output_variable: str) -> float:
        hill = HillMuscle(params)
        optimal_mt_length = (
            params.tendon_slack_length
            + params.optimal_fiber_length * np.cos(params.pennation_angle_at_optimal)
        )
        dt = 0.001
        steps = 200
        excitation = torch.zeros(steps)
        excitation[20:] = 1.0
        mt_length = torch.full((steps,), float(optimal_mt_length))
        mt_velocity = torch.zeros(steps)
        simulation = hill.simulate(excitation, mt_length, mt_velocity, dt=dt)

        if output_variable == "peak_force":
            return float(simulation["force"].max().item())
        if output_variable == "time_to_peak":
            peak_index = int(torch.argmax(simulation["force"]).item())
            return float(max(peak_index - 20, 0) * dt)

        quad_params = get_default_quadriceps()
        quad_params["VMO"] = params
        quadriceps = QuadricepsModel(params=quad_params)
        excitations = {
            "VMO": torch.tensor(0.7),
            "VML": torch.tensor(0.6),
            "VL": torch.tensor(0.7),
            "RF": torch.tensor(0.25),
            "VI": torch.tensor(0.30),
        }
        forces, _ = quadriceps(excitations, torch.tensor(0.8), torch.tensor(0.0), dt=dt)

        if output_variable == "vmo_vl_ratio":
            return float(forces["vmo_vl_ratio"].item())
        if output_variable == "patellar_displacement":
            displacement = self.patellofemoral.patellar_displacement(
                forces["mediolateral_force"],
                torch.tensor(0.8),
            )
            return float(displacement.item())
        if output_variable == "fatigue_rate":
            fatigue = FatigueModel()
            fatigue_result = fatigue.simulate(torch.ones(300), dt=dt)
            fatigued_force = (
                1.0 - float(fatigue_result["effective_capacity"][-1].item())
            ) * float(forces["VMO"].item())
            return fatigued_force

        raise ValueError(f"Unsupported output_variable: {output_variable}")

    def run(
        self,
        output_variable: str = "peak_force",
        n_samples: int = 1024,
    ) -> SensitivityResult:
        """Run Sobol sensitivity analysis on a selected output variable."""

        problem = self._problem_definition()
        parameter_samples = sobol_sample.sample(
            problem,
            n_samples,
            calc_second_order=False,
            seed=42,
        )
        outputs = np.asarray(
            [
                self._evaluate_output(self._params_from_sample(sample), output_variable)
                for sample in parameter_samples
            ],
            dtype=np.float64,
        )
        analysis = sobol.analyze(problem, outputs, calc_second_order=False)
        names = cast(list[str], problem["names"])
        return SensitivityResult(
            parameter_names=names,
            first_order={
                name: float(value)
                for name, value in zip(names, analysis["S1"], strict=True)
            },
            total_order={
                name: float(value)
                for name, value in zip(names, analysis["ST"], strict=True)
            },
            confidence={
                name: float(value)
                for name, value in zip(names, analysis["S1_conf"], strict=True)
            },
            output_variable=output_variable,
        )


def run_sobol_analysis(
    parameter_samples: torch.Tensor, model_outputs: torch.Tensor
) -> SensitivityResult:
    """Run Sobol analysis from precomputed samples and outputs."""

    sample_array = parameter_samples.detach().cpu().numpy()
    output_array = model_outputs.detach().cpu().numpy().reshape(-1)
    parameter_names = [f"p{i}" for i in range(sample_array.shape[1])]
    if sample_array.ndim != 2:
        raise ValueError("parameter_samples must have shape (n_samples, n_parameters).")
    if sample_array.shape[0] != output_array.shape[0]:
        raise ValueError(
            "parameter_samples and model_outputs must contain the same number of samples."
        )

    # Prefer exact Sobol analysis when the samples follow SALib's Saltelli layout.
    n_parameters = sample_array.shape[1]
    if output_array.shape[0] % (n_parameters + 2) == 0:
        bounds = [
            [float(sample_array[:, idx].min()), float(sample_array[:, idx].max())]
            for idx in range(n_parameters)
        ]
        problem = {"num_vars": n_parameters, "names": parameter_names, "bounds": bounds}
        analysis = sobol.analyze(problem, output_array, calc_second_order=False)
        return SensitivityResult(
            parameter_names=parameter_names,
            first_order={
                name: float(value)
                for name, value in zip(parameter_names, analysis["S1"], strict=True)
            },
            total_order={
                name: float(value)
                for name, value in zip(parameter_names, analysis["ST"], strict=True)
            },
            confidence={
                name: float(value)
                for name, value in zip(
                    parameter_names, analysis["S1_conf"], strict=True
                )
            },
            output_variable="custom",
        )

    # Fall back to a normalized correlation-based proxy for arbitrary sample sets.
    correlations = []
    centered_output = output_array - np.mean(output_array)
    for index in range(n_parameters):
        centered_parameter = sample_array[:, index] - np.mean(sample_array[:, index])
        denominator = np.linalg.norm(centered_parameter) * np.linalg.norm(
            centered_output
        )
        if denominator <= 1e-12:
            correlations.append(0.0)
        else:
            correlations.append(
                abs(float(np.dot(centered_parameter, centered_output) / denominator))
            )
    weights = np.asarray(correlations, dtype=np.float64)
    if np.allclose(weights.sum(), 0.0):
        weights = np.full_like(weights, 1.0 / float(n_parameters))
    else:
        weights = weights / weights.sum()
    proxy_indices = {
        name: float(value) for name, value in zip(parameter_names, weights, strict=True)
    }
    return SensitivityResult(
        parameter_names=parameter_names,
        first_order=proxy_indices,
        total_order=proxy_indices.copy(),
        confidence={name: 0.0 for name in parameter_names},
        output_variable="custom",
    )


__all__ = ["MuscleSensitivityAnalysis", "SensitivityResult", "run_sobol_analysis"]
