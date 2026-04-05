"""Machine learning layers for VMO-Sim."""

from __future__ import annotations

from vmo_sim.ml.emg_net import EMGForceLSTM, EMGForceNet
from vmo_sim.ml.losses import PhysicsLoss
from vmo_sim.ml.neural_ode import MuscleNeuralODE, NeuralMuscleODE
from vmo_sim.ml.pinn_surrogate import PINNSurrogate, VMOPINNSurrogate
from vmo_sim.ml.surrogate_trainer import SurrogateTrainer

__all__ = [
    "EMGForceLSTM",
    "EMGForceNet",
    "MuscleNeuralODE",
    "NeuralMuscleODE",
    "PhysicsLoss",
    "PINNSurrogate",
    "SurrogateTrainer",
    "VMOPINNSurrogate",
]
