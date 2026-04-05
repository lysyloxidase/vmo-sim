"""CLI entrypoint for PINN surrogate training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from vmo_sim.biomechanics.hill_muscle import HillMuscle
from vmo_sim.biomechanics.parameters import VMOParameters
from vmo_sim.ml.pinn_surrogate import VMOPINNSurrogate
from vmo_sim.ml.surrogate_trainer import SurrogateTrainer


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for PINN training."""

    parser = argparse.ArgumentParser(description="Train a PINN surrogate for vmo-sim.")
    parser.add_argument(
        "--epochs", type=int, default=5000, help="Number of training epochs."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Optimizer learning rate."
    )
    return parser


def main() -> int:
    """Run PINN training and save a compact metrics summary."""

    args = build_parser().parse_args()
    params = VMOParameters()
    hill = HillMuscle(params)
    pinn = VMOPINNSurrogate(
        hidden_dim=128,
        num_layers=4,
        muscle_params=params,
    )
    trainer = SurrogateTrainer(learning_rate=args.learning_rate)
    history = trainer.train_pinn(
        pinn,
        hill,
        epochs=args.epochs,
        physics_weight=1.0,
        lr=args.learning_rate,
    )
    output_dir = Path("results") / "pinn"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(pinn.state_dict(), output_dir / "vmo_pinn.pt")
    summary = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "final_train_loss": history["train_loss"][-1],
        "final_val_rmse": history["val_rmse"][-1],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
