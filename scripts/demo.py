"""End-to-end command-line demo for VMO-Sim."""

from __future__ import annotations

import argparse
from typing import Iterable, cast

import torch

from vmo_sim.analysis.sensitivity import MuscleSensitivityAnalysis
from vmo_sim.biomechanics.hill_muscle import HillMuscle
from vmo_sim.biomechanics.parameters import VMOParameters, get_default_quadriceps
from vmo_sim.biomechanics.quadriceps import QuadricepsModel
from vmo_sim.ml.neural_ode import MuscleNeuralODE
from vmo_sim.ml.pinn_surrogate import VMOPINNSurrogate
from vmo_sim.ml.surrogate_trainer import SurrogateTrainer
from vmo_sim.rl.agents import RLTrainer


def build_parser() -> argparse.ArgumentParser:
    """Build the demo CLI parser."""

    parser = argparse.ArgumentParser(
        description="Run the full VMO-Sim demonstration workflow."
    )
    parser.add_argument(
        "--quick", action="store_true", help="Use reduced epochs and sample counts."
    )
    parser.add_argument("--no-rl", action="store_true", help="Skip RL training.")
    return parser


def _section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _rows(items: Iterable[tuple[str, object]]) -> None:
    pairs = [(label, str(value)) for label, value in items]
    width = max(len(label) for label, _ in pairs)
    for label, value in pairs:
        print(f"{label:<{width}} : {value}")


def main() -> int:
    """Run the final end-to-end project demo."""

    args = build_parser().parse_args()
    pinn_epochs = 100 if args.quick else 500
    rl_timesteps = 5_000 if args.quick else 50_000
    sensitivity_samples = 32 if args.quick else 128

    params = VMOParameters()
    muscle = HillMuscle(params)

    _section("1. Hill-Type VMO Model")
    isometric_force = muscle.compute_force(
        torch.tensor(1.0), torch.tensor(1.0), torch.tensor(0.0)
    ).item()
    excitation = torch.zeros(180)
    excitation[20:35] = 1.0
    musculotendon_length = torch.full(
        (180,),
        params.tendon_slack_length
        + params.optimal_fiber_length
        * torch.cos(torch.tensor(params.pennation_angle_at_optimal)).item(),
    )
    twitch = muscle.simulate(
        excitation,
        musculotendon_length,
        torch.zeros_like(musculotendon_length),
        dt=0.001,
    )
    twitch_peak_index = int(torch.argmax(twitch["force"]).item())
    _rows(
        [
            ("Isometric force", f"{isometric_force:.2f} N"),
            ("Twitch peak force", f"{float(twitch['force'].max().item()):.2f} N"),
            ("Time to peak", f"{twitch_peak_index:.1f} ms"),
            ("Peak activation", f"{float(twitch['activation'].max().item()):.3f}"),
        ]
    )

    _section("2. Quadriceps Force Balance: Healthy vs PFPS")
    healthy_model = QuadricepsModel(params=get_default_quadriceps())
    pfps_params = get_default_quadriceps()
    pfps_params["VMO"] = pfps_params["VMO"].model_copy(
        update={"max_isometric_force": 0.6 * pfps_params["VMO"].max_isometric_force}
    )
    pfps_model = QuadricepsModel(params=pfps_params)
    common_excitations = {
        "VMO": torch.tensor(0.75),
        "VML": torch.tensor(0.65),
        "VL": torch.tensor(0.75),
        "RF": torch.tensor(0.25),
        "VI": torch.tensor(0.30),
    }
    healthy_force, _ = healthy_model(
        common_excitations, torch.tensor(0.8), torch.tensor(0.0)
    )
    pfps_force, _ = pfps_model(common_excitations, torch.tensor(0.8), torch.tensor(0.0))
    _rows(
        [
            (
                "Healthy VMO:VL ratio",
                f"{float(healthy_force['vmo_vl_ratio'].item()):.3f}",
            ),
            ("PFPS VMO:VL ratio", f"{float(pfps_force['vmo_vl_ratio'].item()):.3f}"),
            (
                "Healthy ML force",
                f"{float(healthy_force['mediolateral_force'].item()):.2f} N",
            ),
            (
                "PFPS ML force",
                f"{float(pfps_force['mediolateral_force'].item()):.2f} N",
            ),
        ]
    )

    _section("3. PINN Surrogate Training")
    pinn = VMOPINNSurrogate(hidden_dim=64, num_layers=3, muscle_params=params)
    node = MuscleNeuralODE(hidden_dim=32, muscle_params=params)
    trainer = SurrogateTrainer()
    history = trainer.train_pinn(
        pinn, muscle, epochs=pinn_epochs, physics_weight=0.1, lr=3e-3
    )
    comparison = SurrogateTrainer.compare_models(muscle, pinn, node, {})
    _rows(
        [
            ("PINN epochs", pinn_epochs),
            ("Final train loss", f"{history['train_loss'][-1]:.6f}"),
            ("Final validation RMSE", f"{history['val_rmse'][-1]:.6f}"),
            (
                "PINN RMSE vs Hill",
                f"{float(comparison.loc[comparison['model'] == 'PINN', 'rmse'].iloc[0]):.6f}",
            ),
        ]
    )

    if not args.no_rl:
        _section("4. RL Rehabilitation Training")
        rl_trainer = RLTrainer(
            algorithm="PPO",
            env_kwargs={"scenario": "pfps_mild", "target_motion": "sit_to_stand"},
        )
        train_metrics = rl_trainer.train(
            total_timesteps=rl_timesteps, log_dir="results/demo_rl"
        )
        eval_metrics = rl_trainer.evaluate(n_episodes=5)
        mean_reward = cast(float, eval_metrics["mean_reward"])
        mean_max_lateral_displacement = cast(
            float, eval_metrics["mean_max_lateral_displacement"]
        )
        _rows(
            [
                ("Timesteps", rl_timesteps),
                ("Backend", train_metrics["backend"]),
                ("Mean reward", f"{mean_reward:.3f}"),
                (
                    "Mean max lateral displacement",
                    f"{1000.0 * mean_max_lateral_displacement:.3f} mm",
                ),
            ]
        )

    _section("5. Sobol Sensitivity Analysis")
    sensitivity = MuscleSensitivityAnalysis().run(
        output_variable="peak_force", n_samples=sensitivity_samples
    )
    ranked = sorted(
        sensitivity.total_order.items(), key=lambda item: item[1], reverse=True
    )[:5]
    _rows([(name, f"ST={value:.3f}") for name, value in ranked])

    _section("Demo Complete")
    print("All requested subsystems executed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
