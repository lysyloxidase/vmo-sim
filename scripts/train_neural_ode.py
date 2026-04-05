"""CLI entrypoint for Neural ODE training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from vmo_sim.biomechanics.parameters import VMOParameters
from vmo_sim.ml.neural_ode import MuscleNeuralODE
from vmo_sim.ml.surrogate_trainer import SurrogateTrainer


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for Neural ODE training."""

    parser = argparse.ArgumentParser(description="Train a Neural ODE for vmo-sim.")
    parser.add_argument(
        "--epochs", type=int, default=2000, help="Number of training epochs."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Optimizer learning rate."
    )
    return parser


def main() -> int:
    """Run Neural ODE training on a synthetic reference trajectory."""

    args = build_parser().parse_args()
    params = VMOParameters()
    model = MuscleNeuralODE(hidden_dim=64, muscle_params=params)
    time_steps = 100
    dt = 0.001
    time = torch.linspace(0.0, dt * (time_steps - 1), time_steps)
    excitation = torch.zeros(time_steps)
    excitation[20:] = 0.6
    reference = torch.zeros(time_steps, 4)
    reference[:, 0] = params.optimal_fiber_length
    reference[0, 1] = 0.01
    for index in range(1, time_steps):
        previous = reference[index - 1, 1]
        tau = params.activation_time_constant * (0.5 + 1.5 * previous)
        reference[index, 1] = torch.clamp(
            previous + dt * (excitation[index] - previous) / tau, 0.01, 1.0
        )
        reference[index, 0] = params.optimal_fiber_length * (
            1.0 - 0.03 * reference[index, 1]
        )
        reference[index, 2] = min(
            0.2, float(reference[index - 1, 2] + dt * 0.05 * reference[index, 1])
        )
    reference[:, 3] = excitation

    trainer = SurrogateTrainer(learning_rate=args.learning_rate)
    history = trainer.train_neural_ode(
        model, reference, epochs=args.epochs, lr=args.learning_rate
    )
    output_dir = Path("results") / "neural_ode"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "vmo_node.pt")
    summary = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "final_train_loss": history["train_loss"][-1],
        "duration_s": float(time[-1].item()),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
