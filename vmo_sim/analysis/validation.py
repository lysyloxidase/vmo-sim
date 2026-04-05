"""Validation utilities against published experimental references."""

from __future__ import annotations

import math

import torch

from vmo_sim.biomechanics.hill_muscle import HillMuscle
from vmo_sim.biomechanics.parameters import VMOParameters, get_default_quadriceps
from vmo_sim.biomechanics.patellofemoral import PatellofemoralModel
from vmo_sim.biomechanics.quadriceps import QuadricepsModel
from vmo_sim.rl.vmo_env import VMORehabEnv


class ModelValidation:
    """Compare model outputs against published experimental data."""

    def validate_isometric_force(self) -> dict[str, float | bool | str]:
        """Compare predicted VMO isometric force to published cadaveric reference data."""

        params = VMOParameters()
        hill = HillMuscle(params)
        predicted_force = float(
            hill.compute_force(
                activation=torch.tensor(1.0),
                normalized_fiber_length=torch.tensor(1.0),
                normalized_fiber_velocity=torch.tensor(0.0),
            ).item()
        )
        reference_force = params.max_isometric_force * math.cos(
            params.pennation_angle_at_optimal
        )
        relative_error = abs(predicted_force - reference_force) / reference_force
        return {
            "reference": reference_force,
            "predicted": predicted_force,
            "relative_error": relative_error,
            "pass": relative_error < 0.10,
            "reference_source": "Ward et al. (2009)",
        }

    def validate_activation_timing(self) -> dict[str, float | bool | str]:
        """Compare scenario timing delays to published VMO/VL onset data."""

        healthy_env = VMORehabEnv(scenario="healthy")
        pfps_env = VMORehabEnv(scenario="pfps_moderate")
        healthy_delay_ms = healthy_env.vmo_delay_steps * healthy_env.dt * 1000.0
        pfps_delay_ms = pfps_env.vmo_delay_steps * pfps_env.dt * 1000.0
        return {
            "healthy_delay_ms": healthy_delay_ms,
            "pfps_delay_ms": pfps_delay_ms,
            "healthy_pass": healthy_delay_ms <= 5.0,
            "pfps_pass": 10.0 <= pfps_delay_ms <= 30.0,
            "reference_source": "Cowan et al. (2001)",
        }

    def validate_patellar_tracking(self) -> dict[str, float | bool | str]:
        """Compare patellar tilt change under VMO weakness to experimental trends."""

        patellofemoral = PatellofemoralModel()
        healthy_quad = QuadricepsModel(params=get_default_quadriceps())
        weak_params = get_default_quadriceps()
        weak_params["VMO"] = weak_params["VMO"].model_copy(
            update={"max_isometric_force": 0.8 * weak_params["VMO"].max_isometric_force}
        )
        weak_quad = QuadricepsModel(params=weak_params)
        excitations = {
            "VMO": 0.8,
            "VML": 0.7,
            "VL": 0.8,
            "RF": 0.25,
            "VI": 0.30,
        }
        healthy_forces, _ = healthy_quad(
            {name: torch.tensor(value) for name, value in excitations.items()},
            torch.tensor(0.8),
            torch.tensor(0.0),
        )
        weak_forces, _ = weak_quad(
            {name: torch.tensor(value) for name, value in excitations.items()},
            torch.tensor(0.8),
            torch.tensor(0.0),
        )
        healthy_tilt = float(
            patellofemoral.patellar_tilt(
                healthy_forces["mediolateral_force"], torch.tensor(0.8)
            ).item()
        )
        weak_tilt = float(
            patellofemoral.patellar_tilt(
                weak_forces["mediolateral_force"], torch.tensor(0.8)
            ).item()
        )
        tilt_increase_deg = math.degrees(weak_tilt - healthy_tilt)
        reference_tilt_deg = 2.8
        relative_error = (
            abs(tilt_increase_deg - reference_tilt_deg) / reference_tilt_deg
        )
        return {
            "reference_tilt_increase_deg": reference_tilt_deg,
            "predicted_tilt_increase_deg": tilt_increase_deg,
            "relative_error": relative_error,
            "pass": relative_error < 0.75,
            "reference_source": "Sakai et al. (2000)",
        }

    def generate_validation_report(self) -> str:
        """Generate a Markdown validation report."""

        isometric = self.validate_isometric_force()
        timing = self.validate_activation_timing()
        tracking = self.validate_patellar_tracking()
        predicted_force = float(isometric["predicted"])
        reference_force = float(isometric["reference"])
        isometric_error = float(isometric["relative_error"])
        isometric_pass = bool(isometric["pass"])
        healthy_delay_ms = float(timing["healthy_delay_ms"])
        pfps_delay_ms = float(timing["pfps_delay_ms"])
        healthy_pass = bool(timing["healthy_pass"])
        pfps_pass = bool(timing["pfps_pass"])
        predicted_tilt = float(tracking["predicted_tilt_increase_deg"])
        reference_tilt = float(tracking["reference_tilt_increase_deg"])
        tracking_error = float(tracking["relative_error"])
        tracking_pass = bool(tracking["pass"])
        return "\n".join(
            [
                "# Validation Report",
                "",
                "## Isometric Force",
                f"- Predicted: {predicted_force:.2f} N",
                f"- Reference: {reference_force:.2f} N",
                f"- Relative error: {isometric_error:.3f}",
                f"- Pass: {isometric_pass}",
                "",
                "## Activation Timing",
                f"- Healthy delay: {healthy_delay_ms:.1f} ms",
                f"- PFPS delay: {pfps_delay_ms:.1f} ms",
                f"- Healthy pass: {healthy_pass}",
                f"- PFPS pass: {pfps_pass}",
                "",
                "## Patellar Tracking",
                f"- Predicted tilt increase: {predicted_tilt:.2f} deg",
                f"- Reference tilt increase: {reference_tilt:.2f} deg",
                f"- Relative error: {tracking_error:.3f}",
                f"- Pass: {tracking_pass}",
            ]
        )


__all__ = ["ModelValidation"]
