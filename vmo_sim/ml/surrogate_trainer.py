"""Training utilities for surrogate, Neural ODE, and EMG models."""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass

import pandas as pd  # type: ignore[import-untyped]
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

from vmo_sim.biomechanics.hill_muscle import HillMuscle
from vmo_sim.ml.neural_ode import MuscleNeuralODE
from vmo_sim.ml.pinn_surrogate import VMOPINNSurrogate


@dataclass(slots=True)
class TrainingBatch:
    """Container for supervised and physics-informed training data."""

    inputs: torch.Tensor
    targets: torch.Tensor
    metadata: dict[str, torch.Tensor]


class SurrogateTrainer:
    """Training loop for PINN, Neural ODE, and EMG models."""

    def __init__(
        self,
        learning_rate: float = 1e-3,
        physics_weight: float = 1.0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.learning_rate = learning_rate
        self.physics_weight = physics_weight
        self.device = torch.device(device)

    def train_pinn(
        self,
        pinn: VMOPINNSurrogate,
        hill_muscle: HillMuscle,
        epochs: int = 5000,
        physics_weight: float = 1.0,
        lr: float = 1e-3,
    ) -> dict[str, list[float]]:
        """Train a PINN surrogate against Hill-model-generated data."""

        pinn = pinn.to(self.device)
        n_samples = min(4096, max(768, epochs * 16))
        inputs, targets = pinn.generate_training_data(hill_muscle, n_samples=n_samples)
        permutation = torch.randperm(
            inputs.shape[0], generator=torch.Generator().manual_seed(42)
        )
        split_index = int(0.8 * inputs.shape[0])
        train_idx = permutation[:split_index]
        val_idx = permutation[split_index:]

        train_inputs = inputs[train_idx].to(self.device)
        train_targets = targets[train_idx].to(self.device)
        val_inputs = inputs[val_idx].to(self.device)
        val_targets = targets[val_idx].to(self.device)

        optimizer = torch.optim.Adam(pinn.parameters(), lr=lr)
        history: dict[str, list[float]] = {
            "train_loss": [],
            "data_loss": [],
            "physics_loss": [],
            "val_rmse": [],
        }
        best_state = deepcopy(pinn.state_dict())
        best_val_rmse = float("inf")

        for _ in range(epochs):
            pinn.train()
            optimizer.zero_grad()
            train_outputs = pinn(
                train_inputs[:, 0],
                train_inputs[:, 1],
                train_inputs[:, 2],
            )
            stacked_outputs = torch.stack(
                [train_outputs["force"], train_outputs["pennation_angle"]],
                dim=-1,
            )
            data_term = functional.mse_loss(stacked_outputs, train_targets)
            physics_term = pinn.physics_loss(train_inputs, train_outputs)
            loss = data_term + physics_weight * physics_term
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

            pinn.eval()
            with torch.no_grad():
                val_outputs = pinn(
                    val_inputs[:, 0],
                    val_inputs[:, 1],
                    val_inputs[:, 2],
                )
                val_rmse = torch.sqrt(
                    functional.mse_loss(val_outputs["force"], val_targets[:, 0])
                )
                val_rmse_value = float(val_rmse.detach().cpu().item())
                if val_rmse_value < best_val_rmse:
                    best_val_rmse = val_rmse_value
                    best_state = deepcopy(pinn.state_dict())

            history["train_loss"].append(float(loss.detach().cpu().item()))
            history["data_loss"].append(float(data_term.detach().cpu().item()))
            history["physics_loss"].append(float(physics_term.detach().cpu().item()))
            history["val_rmse"].append(val_rmse_value)

        pinn.load_state_dict(best_state)
        return history

    def train_neural_ode(
        self,
        node: MuscleNeuralODE,
        reference_trajectories: torch.Tensor,
        epochs: int = 2000,
        lr: float = 1e-3,
    ) -> dict[str, list[float]]:
        """Train a Neural ODE against reference state trajectories.

        The expected input shape is ``(batch, T, 4)`` or ``(T, 4)``, where the
        last dimension stores ``[fiber_length, activation, fatigue_state, excitation]``.
        """

        node = node.to(self.device)
        if reference_trajectories.ndim == 2:
            trajectories = reference_trajectories.unsqueeze(0)
        elif reference_trajectories.ndim == 3:
            trajectories = reference_trajectories
        else:
            raise ValueError(
                "reference_trajectories must have shape (T, 4) or (batch, T, 4)."
            )

        if trajectories.shape[-1] < 4:
            raise ValueError(
                "reference_trajectories must contain state and excitation columns."
            )

        trajectories = trajectories.to(self.device)
        reference_states = trajectories[..., :3]
        excitation_signal = trajectories[..., 3]
        initial_state = reference_states[:, 0, :]
        time_steps = trajectories.shape[1]
        t_span = torch.linspace(
            0.0,
            0.001 * (time_steps - 1),
            time_steps,
            device=self.device,
            dtype=trajectories.dtype,
        )

        optimizer = torch.optim.Adam(node.parameters(), lr=lr)
        history: dict[str, list[float]] = {"train_loss": []}

        for _ in range(epochs):
            node.train()
            optimizer.zero_grad()
            predicted = node(initial_state, excitation_signal, t_span).transpose(0, 1)
            trajectory_loss = functional.mse_loss(predicted, reference_states)
            excitation_flat = excitation_signal.reshape(-1)
            predicted_flat = predicted.reshape(-1, 3)
            correction_penalty = torch.mean(
                node.neural_correction(predicted_flat, excitation_flat) ** 2
            )
            loss = trajectory_loss + 1e-3 * correction_penalty
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            history["train_loss"].append(float(loss.detach().cpu().item()))

        return history

    def _prepare_emg_batch(self, model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
        layout = getattr(model, "input_layout", "channels_first")
        n_channels = int(getattr(model, "n_channels", 1))
        if batch.ndim != 3:
            raise ValueError(
                "EMG batches must have shape (batch, channels, T) or (batch, T, channels)."
            )

        if layout == "channels_first":
            if batch.shape[1] == n_channels:
                return batch
            if batch.shape[-1] == n_channels:
                return batch.transpose(1, 2)
        if layout == "sequence_first":
            if batch.shape[-1] == n_channels:
                return batch
            if batch.shape[1] == n_channels:
                return batch.transpose(1, 2)
        raise ValueError("Unable to align EMG batch with model input layout.")

    def train_emg_net(
        self,
        model: nn.Module,
        emg_data: torch.Tensor,
        force_data: torch.Tensor,
        epochs: int = 200,
        batch_size: int = 32,
    ) -> dict[str, list[float]]:
        """Train an EMG-to-force model."""

        model = model.to(self.device)
        dataset = TensorDataset(emg_data.to(self.device), force_data.to(self.device))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        history: dict[str, list[float]] = {"train_loss": []}

        for _ in range(epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch_emg, batch_force in loader:
                optimizer.zero_grad()
                prepared_emg = self._prepare_emg_batch(model, batch_emg)
                predicted_force = model(prepared_emg)
                target_force = batch_force.reshape_as(predicted_force)
                loss = functional.mse_loss(predicted_force, target_force)
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
                epoch_loss += float(loss.detach().cpu().item())
                n_batches += 1

            history["train_loss"].append(epoch_loss / max(n_batches, 1))

        return history

    @staticmethod
    def _r2_score(reference: torch.Tensor, prediction: torch.Tensor) -> float:
        residual_sum = torch.sum((reference - prediction) ** 2)
        total_sum = torch.sum((reference - torch.mean(reference)) ** 2)
        if torch.isclose(total_sum, torch.tensor(0.0, device=reference.device)):
            return 1.0
        return float((1.0 - residual_sum / total_sum).detach().cpu().item())

    @staticmethod
    def _node_force_from_trajectory(
        hill: HillMuscle,
        trajectory: torch.Tensor,
        t_span: torch.Tensor,
    ) -> torch.Tensor:
        fiber_length = trajectory[..., 0]
        activation = torch.clamp(trajectory[..., 1], min=0.01, max=1.0)
        fatigue_state = torch.clamp(trajectory[..., 2], min=0.0, max=1.0)
        effective_activation = activation * (1.0 - fatigue_state)
        dt = (
            float((t_span[1] - t_span[0]).detach().cpu().item())
            if t_span.numel() > 1
            else 1.0
        )
        fiber_velocity = torch.zeros_like(fiber_length)
        if trajectory.shape[0] > 1:
            fiber_velocity[1:] = (fiber_length[1:] - fiber_length[:-1]) / dt
            fiber_velocity[0] = fiber_velocity[1]
        normalized_length = fiber_length / hill.params.optimal_fiber_length
        normalized_velocity = fiber_velocity / (
            hill.params.optimal_fiber_length * hill.params.max_contraction_velocity
        )
        return (
            hill.compute_force(
                effective_activation,
                normalized_length,
                normalized_velocity,
            )
            / hill.params.max_isometric_force
        )

    @staticmethod
    def compare_models(
        hill: HillMuscle,
        pinn: VMOPINNSurrogate,
        node: MuscleNeuralODE,
        test_conditions: dict[str, torch.Tensor],
    ) -> pd.DataFrame:
        """Compare Hill, PINN, and Neural ODE models on common test conditions."""

        activation = test_conditions.get("activation")
        fiber_length = test_conditions.get("fiber_length")
        fiber_velocity = test_conditions.get("fiber_velocity")
        if activation is None or fiber_length is None or fiber_velocity is None:
            activation = torch.linspace(0.1, 1.0, 128)
            fiber_length = torch.linspace(0.7, 1.3, 128)
            fiber_velocity = torch.linspace(-0.5, 0.5, 128)

        hill_start = time.perf_counter()
        hill_force = (
            hill.compute_force(activation, fiber_length, fiber_velocity)
            / hill.params.max_isometric_force
        )
        hill_time = (time.perf_counter() - hill_start) * 1000.0

        pinn_start = time.perf_counter()
        with torch.no_grad():
            pinn_force = pinn(activation, fiber_length, fiber_velocity)["force"]
        pinn_time = (time.perf_counter() - pinn_start) * 1000.0

        excitation_signal = test_conditions.get("excitation_signal")
        t_span = test_conditions.get("t_span")
        if excitation_signal is None or t_span is None:
            t_span = torch.linspace(0.0, 0.099, 100)
            excitation_signal = torch.full((100,), float(torch.mean(activation).item()))
        initial_state = test_conditions.get(
            "initial_state",
            torch.tensor(
                [hill.params.optimal_fiber_length, 0.1, 0.0],
                dtype=t_span.dtype,
            ),
        )
        node_start = time.perf_counter()
        with torch.no_grad():
            node_trajectory = node(initial_state, excitation_signal, t_span)
        node_time = (time.perf_counter() - node_start) * 1000.0

        musculotendon_length = torch.full(
            (t_span.numel(),),
            hill.params.tendon_slack_length
            + hill.params.optimal_fiber_length
            * torch.cos(torch.tensor(hill.params.pennation_angle_at_optimal)).item(),
            dtype=t_span.dtype,
        )
        hill_results = hill.simulate(
            excitation_signal,
            musculotendon_length,
            torch.zeros_like(musculotendon_length),
            dt=float((t_span[1] - t_span[0]).item()) if t_span.numel() > 1 else 0.001,
        )
        hill_dynamic_force = hill_results["force"] / hill.params.max_isometric_force
        node_force = SurrogateTrainer._node_force_from_trajectory(
            hill, node_trajectory, t_span
        )

        records = [
            {
                "model": "Hill",
                "rmse": 0.0,
                "mae": 0.0,
                "r2": 1.0,
                "inference_ms": hill_time,
            },
            {
                "model": "PINN",
                "rmse": float(
                    torch.sqrt(functional.mse_loss(pinn_force, hill_force)).item()
                ),
                "mae": float(torch.mean(torch.abs(pinn_force - hill_force)).item()),
                "r2": SurrogateTrainer._r2_score(hill_force, pinn_force),
                "inference_ms": pinn_time,
            },
            {
                "model": "NeuralODE",
                "rmse": float(
                    torch.sqrt(
                        functional.mse_loss(node_force, hill_dynamic_force)
                    ).item()
                ),
                "mae": float(
                    torch.mean(torch.abs(node_force - hill_dynamic_force)).item()
                ),
                "r2": SurrogateTrainer._r2_score(hill_dynamic_force, node_force),
                "inference_ms": node_time,
            },
        ]
        return pd.DataFrame.from_records(records)
