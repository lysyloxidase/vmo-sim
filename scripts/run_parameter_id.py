"""CLI entrypoint for parameter identification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from vmo_sim.analysis.parameter_id import GradientParameterIdentification
from vmo_sim.biomechanics.parameters import VMOParameters


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for parameter identification."""

    parser = argparse.ArgumentParser(
        description="Run parameter identification for vmo-sim."
    )
    parser.add_argument(
        "--iterations", type=int, default=1000, help="Maximum optimization iterations."
    )
    return parser


def main() -> int:
    """Run gradient-based parameter identification on a synthetic example."""

    args = build_parser().parse_args()
    initial_params = VMOParameters()
    time = torch.linspace(0.0, 0.119, 120)
    experimental_emg = torch.zeros(120)
    experimental_emg[20:80] = 0.65
    musculotendon_lengths = torch.full(
        (120,), initial_params.tendon_slack_length + initial_params.optimal_fiber_length
    )
    target_params = initial_params.model_copy(
        update={
            "max_isometric_force": 520.0,
            "optimal_fiber_length": 0.074,
            "pennation_angle_at_optimal": 0.82,
        }
    )
    identifier = GradientParameterIdentification()
    target_force = identifier._simulate_force(  # type: ignore[attr-defined]
        experimental_emg,
        musculotendon_lengths,
        {
            "max_isometric_force": torch.tensor(target_params.max_isometric_force),
            "optimal_fiber_length": torch.tensor(target_params.optimal_fiber_length),
            "pennation_angle_at_optimal": torch.tensor(
                target_params.pennation_angle_at_optimal
            ),
        },
        target_params,
    )
    optimized_params, history = identifier.identify(
        experimental_force=target_force,
        experimental_emg=experimental_emg,
        musculotendon_lengths=musculotendon_lengths,
        initial_params=initial_params,
        n_iterations=args.iterations,
        lr=0.01,
    )
    output_dir = Path("results") / "parameter_id"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "iterations": args.iterations,
        "time_points": int(time.numel()),
        "optimized_params": optimized_params.model_dump(),
        "final_loss": history["loss"][-1],
    }
    (output_dir / "parameter_identification.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
