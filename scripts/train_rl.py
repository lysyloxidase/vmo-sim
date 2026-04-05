"""CLI entrypoint for reinforcement learning training."""

from __future__ import annotations

import argparse
import json

from vmo_sim.rl.agents import RLTrainer
from vmo_sim.rl.curriculum import RehabCurriculum


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for RL training."""

    parser = argparse.ArgumentParser(
        description="Train an RL agent for vmo-sim rehab tasks."
    )
    parser.add_argument(
        "--algorithm", type=str, default="PPO", help="RL algorithm to use."
    )
    parser.add_argument(
        "--scenario", type=str, default="healthy", help="Clinical scenario."
    )
    parser.add_argument(
        "--motion", type=str, default="sit_to_stand", help="Target motion."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Number of environment timesteps.",
    )
    parser.add_argument(
        "--curriculum", action="store_true", help="Enable staged curriculum learning."
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="results/rl",
        help="Directory for logs and metrics.",
    )
    return parser


def main() -> int:
    """Run the RL training CLI."""

    args = build_parser().parse_args()
    curriculum = RehabCurriculum() if args.curriculum else None
    trainer = RLTrainer(
        algorithm=args.algorithm,
        env_kwargs={"scenario": args.scenario, "target_motion": args.motion},
        curriculum=curriculum,
    )
    train_metrics = trainer.train(total_timesteps=args.timesteps, log_dir=args.log_dir)
    eval_metrics = trainer.evaluate(n_episodes=5)
    print(json.dumps({"train": train_metrics, "eval": eval_metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
