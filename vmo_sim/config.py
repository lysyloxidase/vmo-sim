"""Application settings and muscle parameter presets."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from vmo_sim.biomechanics.parameters import (
    MuscleParameters,
    VMLParameters,
    VMOParameters,
    VLParameters,
    get_default_quadriceps,
)


class Settings(BaseSettings):
    """Runtime configuration for simulation, training, and analysis."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DEVICE: str = "cpu"
    RANDOM_SEED: int = 42
    DEFAULT_DT: float = 0.001
    DEFAULT_DURATION: float = 2.0

    PINN_HIDDEN_DIM: int = 128
    PINN_NUM_LAYERS: int = 4
    PINN_LEARNING_RATE: float = 1e-3
    PINN_EPOCHS: int = 5000
    PINN_PHYSICS_WEIGHT: float = 1.0

    NODE_HIDDEN_DIM: int = 64
    NODE_LEARNING_RATE: float = 1e-3
    NODE_EPOCHS: int = 2000

    RL_ALGORITHM: str = "PPO"
    RL_TOTAL_TIMESTEPS: int = 500_000
    RL_LEARNING_RATE: float = 3e-4

    DATA_DIR: str = "data"
    MODEL_DIR: str = "models"
    RESULTS_DIR: str = "results"


def get_parameter_presets() -> dict[str, MuscleParameters]:
    """Return named muscle parameter presets used across the package."""

    presets = get_default_quadriceps()
    presets["VMO_clinical_weak"] = VMOParameters(max_isometric_force=315.0)
    presets["VML_hypertrophy"] = VMLParameters(max_isometric_force=950.0)
    presets["VL_dominant"] = VLParameters(max_isometric_force=2100.0)
    return presets


__all__ = ["Settings", "get_parameter_presets"]
