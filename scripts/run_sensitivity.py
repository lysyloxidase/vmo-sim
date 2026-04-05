"""CLI entrypoint for Sobol sensitivity analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vmo_sim.analysis.sensitivity import MuscleSensitivityAnalysis


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for sensitivity analysis."""

    parser = argparse.ArgumentParser(
        description="Run Sobol sensitivity analysis for vmo-sim."
    )
    parser.add_argument(
        "--samples", type=int, default=256, help="Number of parameter samples."
    )
    return parser


def main() -> int:
    """Run Sobol sensitivity analysis and save the resulting indices."""

    args = build_parser().parse_args()
    analysis = MuscleSensitivityAnalysis()
    result = analysis.run(output_variable="peak_force", n_samples=args.samples)
    output_dir = Path("results") / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = result.model_dump()
    (output_dir / "sensitivity_peak_force.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
