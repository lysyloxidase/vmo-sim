"""EMG processing utilities for VMO-Sim."""

from __future__ import annotations

from vmo_sim.emg.activation_dynamics import EMGActivationDynamics, EMGToActivation
from vmo_sim.emg.electrode_config import ElectrodePlacement, default_vmo_configuration
from vmo_sim.emg.preprocessing import (
    EMGPreprocessor,
    bandpass_filter,
    linear_envelope,
    normalize_emg,
    preprocess_pipeline,
    rectify,
)
from vmo_sim.emg.vmo_vl_ratio import VMOVLRatioAnalyzer, compute_vmo_vl_ratio

__all__ = [
    "EMGActivationDynamics",
    "EMGPreprocessor",
    "EMGToActivation",
    "ElectrodePlacement",
    "VMOVLRatioAnalyzer",
    "bandpass_filter",
    "compute_vmo_vl_ratio",
    "default_vmo_configuration",
    "linear_envelope",
    "normalize_emg",
    "preprocess_pipeline",
    "rectify",
]
