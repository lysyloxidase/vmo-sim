"""Published biomechanical parameters for VMO, VML, VL, RF, and VI.

Sources:
- Rajagopal et al. (2016) full-body musculoskeletal model
- Ward et al. (2009) lower limb cadaveric measurements
- Castanov et al. (2019) VMO vs VML architecture study
- Benjafield et al. (2015) VMO fiber angle ultrasound study
- Arnold et al. (2010) lower limb muscle parameters
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class MuscleParameters(BaseModel):
    """Hill-type muscle-tendon unit parameters."""

    model_config = ConfigDict(frozen=True)

    name: str
    max_isometric_force: float
    optimal_fiber_length: float
    tendon_slack_length: float
    pennation_angle_at_optimal: float
    max_contraction_velocity: float = 10.0
    activation_time_constant: float = 0.010
    deactivation_time_constant: float = 0.040
    passive_strain_at_one_norm_force: float = 0.6
    tendon_strain_at_one_norm_force: float = 0.033
    fv_shape_factor: float = 0.25
    max_eccentric_force_multiplier: float = 1.4
    slow_twitch_fraction: float = 0.5
    fatigue_rate: float = 0.01
    recovery_rate: float = 0.002


class VMOParameters(MuscleParameters):
    """Vastus Medialis Oblique, the primary medial patellar stabilizer."""

    name: str = "VMO"
    max_isometric_force: float = 450.0
    optimal_fiber_length: float = 0.070
    tendon_slack_length: float = 0.126
    pennation_angle_at_optimal: float = 0.872
    slow_twitch_fraction: float = 0.52
    patellar_force_angle: float = 0.872


class VMLParameters(MuscleParameters):
    """Vastus Medialis Longus, the proximal portion of vastus medialis."""

    name: str = "VML"
    max_isometric_force: float = 844.0
    optimal_fiber_length: float = 0.089
    tendon_slack_length: float = 0.126
    pennation_angle_at_optimal: float = 0.262
    slow_twitch_fraction: float = 0.48


class VLParameters(MuscleParameters):
    """Vastus Lateralis, the dominant lateral quadriceps contributor."""

    name: str = "VL"
    max_isometric_force: float = 1871.0
    optimal_fiber_length: float = 0.084
    tendon_slack_length: float = 0.157
    pennation_angle_at_optimal: float = 0.087
    patellar_force_angle: float = -0.244
    slow_twitch_fraction: float = 0.46


class RFParameters(MuscleParameters):
    """Rectus Femoris, the biarticular quadriceps component."""

    name: str = "RF"
    max_isometric_force: float = 1169.0
    optimal_fiber_length: float = 0.084
    tendon_slack_length: float = 0.346
    pennation_angle_at_optimal: float = 0.087
    slow_twitch_fraction: float = 0.45


class VIParameters(MuscleParameters):
    """Vastus Intermedius, the deep quadriceps component beneath RF."""

    name: str = "VI"
    max_isometric_force: float = 1365.0
    optimal_fiber_length: float = 0.087
    tendon_slack_length: float = 0.136
    pennation_angle_at_optimal: float = 0.052
    slow_twitch_fraction: float = 0.50


def get_default_quadriceps() -> dict[str, MuscleParameters]:
    """Return default parameters for the five modeled quadriceps actuators."""

    return {
        "VMO": VMOParameters(),
        "VML": VMLParameters(),
        "VL": VLParameters(),
        "RF": RFParameters(),
        "VI": VIParameters(),
    }
