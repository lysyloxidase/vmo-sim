"""Streamlit dashboard for interactive VMO-Sim exploration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import streamlit as st
import torch

from vmo_sim.analysis.sensitivity import MuscleSensitivityAnalysis
from vmo_sim.analysis.validation import ModelValidation
from vmo_sim.biomechanics.hill_muscle import HillMuscle
from vmo_sim.biomechanics.parameters import VMOParameters
from vmo_sim.ml.neural_ode import MuscleNeuralODE
from vmo_sim.ml.pinn_surrogate import VMOPINNSurrogate
from vmo_sim.ml.surrogate_trainer import SurrogateTrainer
from vmo_sim.rl.agents import HeuristicPolicy, RLTrainer
from vmo_sim.rl.vmo_env import VMORehabEnv
from vmo_sim.viz.activation_plots import (
    plot_excitation_vs_activation,
    plot_vmo_vl_ratio_over_time,
)
from vmo_sim.viz.muscle_plots import (
    plot_force_length_curve,
    plot_force_velocity_curve,
    plot_muscle_force_profile,
)
from vmo_sim.viz.patellar_plots import plot_contact_pressure_map
from vmo_sim.viz.rl_plots import plot_policy_comparison, plot_training_curve


def _excitation_pattern(
    pattern: str, time_points: np.ndarray, custom_text: str
) -> np.ndarray:
    if pattern == "step":
        return np.asarray(np.where(time_points >= 0.2, 1.0, 0.0), dtype=np.float64)
    if pattern == "ramp":
        return np.asarray(
            np.clip(time_points / max(time_points[-1], 1e-6), 0.0, 1.0),
            dtype=np.float64,
        )
    if pattern == "sinusoidal":
        return np.asarray(
            0.5
            * (
                1.0
                + np.sin(
                    2.0 * np.pi * time_points / max(time_points[-1], 1e-6) - np.pi / 2.0
                )
            ),
            dtype=np.float64,
        )
    values = np.fromstring(custom_text, sep=",", dtype=np.float64)
    if values.size == 0:
        return np.asarray(np.zeros_like(time_points), dtype=np.float64)
    xp = np.linspace(0.0, time_points[-1], values.size, dtype=np.float64)
    return np.asarray(
        np.clip(np.interp(time_points, xp, values), 0.0, 1.0),
        dtype=np.float64,
    )


def _simulate_hill(
    params: VMOParameters,
    pattern: str,
    duration: float = 1.0,
    dt: float = 0.001,
    custom_text: str = "",
) -> dict[str, torch.Tensor]:
    time_points = np.arange(0.0, duration, dt, dtype=np.float64)
    excitation_np = _excitation_pattern(pattern, time_points, custom_text)
    muscle = HillMuscle(params)
    excitation = torch.tensor(excitation_np, dtype=torch.float32)
    musculotendon_length = torch.full(
        (excitation.shape[0],),
        params.tendon_slack_length
        + params.optimal_fiber_length
        * torch.cos(torch.tensor(params.pennation_angle_at_optimal)).item(),
        dtype=torch.float32,
    )
    results = muscle.simulate(
        excitation,
        musculotendon_length,
        torch.zeros_like(musculotendon_length),
        dt=dt,
    )
    results["time"] = torch.tensor(time_points, dtype=torch.float32)
    results["excitation"] = excitation
    return results


def _manual_protocol(step_index: int, episode_length: int) -> np.ndarray:
    phase = step_index / max(episode_length - 1, 1)
    vmo = 0.35 + 0.45 * np.exp(-(((phase - 0.35) / 0.18) ** 2))
    vml = 0.45 + 0.20 * np.sin(np.pi * phase) ** 2
    vl = 0.50 + 0.18 * np.sin(np.pi * phase) ** 2
    return np.asarray([vmo, vml, vl], dtype=np.float32)


def _rollout_env(
    env: VMORehabEnv,
    policy: str = "heuristic",
    agent_path: str | None = None,
) -> dict[str, np.ndarray]:
    trainer = RLTrainer(
        algorithm="PPO",
        env_kwargs={
            "scenario": env.scenario,
            "target_motion": env.target_motion,
            "dt": env.dt,
        },
    )
    if agent_path:
        model_path = Path(agent_path)
        if model_path.exists():
            trainer.load(str(model_path))
        else:
            trainer.model = HeuristicPolicy()
    else:
        trainer.model = HeuristicPolicy()

    observation, _ = env.reset()
    rewards: list[float] = []
    actions: list[np.ndarray] = []
    knee_angles: list[float] = [float(observation[0])]
    targets: list[float] = [float(observation[9])]
    displacements: list[float] = [float(observation[7])]
    tilts: list[float] = [float(observation[8])]
    ratios: list[float] = [1.0]
    medial_pressures: list[float] = [float(observation[11])]
    lateral_pressures: list[float] = [float(observation[12])]
    vmo_force: list[float] = [0.0]
    vl_force: list[float] = [0.0]

    terminated = False
    truncated = False
    while not (terminated or truncated):
        if policy == "constant":
            action = np.asarray([0.45, 0.45, 0.45], dtype=np.float32)
        elif policy == "manual":
            action = _manual_protocol(env.step_count, env.episode_length)
        elif trainer.model is not None:
            action, _ = trainer.model.predict(observation)
        else:
            action, _ = HeuristicPolicy().predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        actions.append(np.asarray(action, dtype=np.float32))
        rewards.append(float(reward))
        knee_angles.append(float(observation[0]))
        targets.append(float(observation[9]))
        displacements.append(float(observation[7]))
        tilts.append(float(observation[8]))
        reward_state = info["reward_state"]
        ratios.append(float(reward_state["vmo_vl_ratio"]))
        medial_pressures.append(float(observation[11]))
        lateral_pressures.append(float(observation[12]))
        vmo_force.append(float(env.latest_forces["VMO"].item()))
        vl_force.append(float(env.latest_forces["VL"].item()))

    time = np.arange(len(knee_angles), dtype=np.float64) * env.dt
    return {
        "time": time,
        "reward": np.asarray(rewards, dtype=np.float64),
        "action": np.asarray(actions, dtype=np.float32),
        "knee_angle": np.asarray(knee_angles, dtype=np.float64),
        "target_angle": np.asarray(targets, dtype=np.float64),
        "patellar_displacement": np.asarray(displacements, dtype=np.float64),
        "patellar_tilt": np.asarray(tilts, dtype=np.float64),
        "vmo_vl_ratio": np.asarray(ratios, dtype=np.float64),
        "medial_pressure": np.asarray(medial_pressures, dtype=np.float64),
        "lateral_pressure": np.asarray(lateral_pressures, dtype=np.float64),
        "VMO_force": np.asarray(vmo_force, dtype=np.float64),
        "VL_force": np.asarray(vl_force, dtype=np.float64),
    }


@st.cache_data(show_spinner=False)
def _cached_rollout(
    scenario: str,
    target_motion: str,
    enable_fatigue: bool,
    policy: str,
    agent_path: str,
) -> dict[str, np.ndarray]:
    env = VMORehabEnv(
        scenario=scenario,
        target_motion=target_motion,
        enable_fatigue=enable_fatigue,
    )
    try:
        normalized_agent_path = agent_path if agent_path else None
        return _rollout_env(env, policy=policy, agent_path=normalized_agent_path)
    finally:
        env.close()


def _policy_summary(results: dict[str, np.ndarray]) -> dict[str, float]:
    reward = results["reward"]
    displacement = results["patellar_displacement"]
    ratio = results["vmo_vl_ratio"]
    return {
        "mean_reward": float(np.mean(reward)) if reward.size > 0 else 0.0,
        "mean_max_lateral_displacement": float(np.max(np.maximum(displacement, 0.0))),
        "mean_vmo_vl_ratio": float(np.mean(ratio)),
    }


def _line_plot(
    x_values: np.ndarray,
    y_series: dict[str, np.ndarray],
    title: str,
    x_label: str,
    y_label: str,
) -> Figure:
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    for name, values in y_series.items():
        ax.plot(x_values[: values.shape[0]], values, linewidth=2.2, label=name)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _analysis_snapshot(
    n_samples: int,
    output_variable: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sensitivity = MuscleSensitivityAnalysis().run(
        output_variable=output_variable,
        n_samples=n_samples,
    )
    sensitivity_frame = pd.DataFrame(
        {
            "parameter": sensitivity.parameter_names,
            "S1": [
                sensitivity.first_order[name] for name in sensitivity.parameter_names
            ],
            "ST": [
                sensitivity.total_order[name] for name in sensitivity.parameter_names
            ],
        }
    )

    params = VMOParameters()
    hill = HillMuscle(params)
    pinn = VMOPINNSurrogate(muscle_params=params)
    node = MuscleNeuralODE(muscle_params=params)
    comparison = SurrogateTrainer.compare_models(hill, pinn, node, test_conditions={})
    return sensitivity_frame, comparison


@st.cache_data(show_spinner=False)
def _cached_analysis_snapshot(
    n_samples: int,
    output_variable: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    sensitivity_frame, comparison_frame = _analysis_snapshot(
        n_samples=n_samples,
        output_variable=output_variable,
    )
    validation = ModelValidation()
    validation_frame = pd.DataFrame.from_records(
        [
            validation.validate_isometric_force(),
            validation.validate_activation_timing(),
            validation.validate_patellar_tracking(),
        ]
    )
    return (
        sensitivity_frame,
        comparison_frame,
        validation_frame,
        validation.generate_validation_report(),
    )


def _request_matches(payload: object, request: Mapping[str, object]) -> bool:
    if not isinstance(payload, dict):
        return False
    return payload.get("request") == request


def build_dashboard() -> None:
    """Assemble the four-tab Streamlit dashboard."""

    st.set_page_config(page_title="VMO-Sim Dashboard", layout="wide")
    st.title("VMO-Sim")
    st.caption(
        "Interactive VMO-specific biomechanics, machine learning, rehabilitation, and analysis dashboard."
    )

    tab_muscle, tab_pathology, tab_rl, tab_analysis = st.tabs(
        ["Muscle Simulator", "Pathology Simulator", "RL Rehabilitation", "Analysis"]
    )

    with tab_muscle:
        left, right = st.columns([1, 1])
        with left:
            max_force = st.slider(
                "VMO max force (N)",
                min_value=250.0,
                max_value=800.0,
                value=450.0,
                step=5.0,
            )
            pennation_deg = st.slider(
                "Pennation angle (deg)",
                min_value=10.0,
                max_value=65.0,
                value=50.0,
                step=0.5,
            )
            tau_act = st.slider(
                "Activation tau (s)",
                min_value=0.005,
                max_value=0.050,
                value=0.010,
                step=0.001,
            )
            tau_deact = st.slider(
                "Deactivation tau (s)",
                min_value=0.010,
                max_value=0.100,
                value=0.040,
                step=0.001,
            )
            pattern = st.selectbox(
                "Excitation pattern",
                ["step", "ramp", "sinusoidal", "custom"],
            )
            custom_text = st.text_input(
                "Custom excitation values",
                value="0,0,0.2,0.6,1.0,0.8,0.3,0",
            )
            run_simulation = st.button("Simulate", key="muscle_simulate")

        params = VMOParameters(
            max_isometric_force=max_force,
            pennation_angle_at_optimal=np.deg2rad(pennation_deg),
            activation_time_constant=tau_act,
            deactivation_time_constant=tau_deact,
        )
        with right:
            st.pyplot(plot_force_length_curve(params))
            st.pyplot(plot_force_velocity_curve(params))

        if run_simulation:
            results = _simulate_hill(params, pattern=pattern, custom_text=custom_text)
            st.pyplot(
                plot_excitation_vs_activation(
                    {
                        "time": results["time"].detach().cpu().numpy(),
                        "excitation": results["excitation"].detach().cpu().numpy(),
                        "activation": results["activation"].detach().cpu().numpy(),
                    }
                )
            )
            st.pyplot(plot_muscle_force_profile(results))
            st.pyplot(
                _line_plot(
                    results["time"].detach().cpu().numpy(),
                    {
                        "Pennation angle (deg)": np.degrees(
                            results["pennation_angle"].detach().cpu().numpy()
                        )
                    },
                    "Pennation Dynamics",
                    "Time (s)",
                    "Angle (deg)",
                )
            )

    with tab_pathology:
        scenario = st.selectbox(
            "Scenario",
            ["healthy", "pfps_mild", "pfps_moderate", "post_acl", "post_surgical"],
        )
        motion = st.selectbox(
            "Target motion",
            ["sit_to_stand", "squat", "stair_climb", "isometric", "isokinetic"],
        )
        pathology_request = {
            "scenario": scenario,
            "motion": motion,
        }
        run_pathology = st.button("Run pathology comparison", key="pathology_run")
        pathology_payload = st.session_state.get("pathology_payload")

        if run_pathology:
            with st.spinner("Running healthy vs pathological comparison..."):
                healthy_result = _cached_rollout(
                    scenario="healthy",
                    target_motion=motion,
                    enable_fatigue=False,
                    policy="heuristic",
                    agent_path="",
                )
                pathology_result = _cached_rollout(
                    scenario=scenario,
                    target_motion=motion,
                    enable_fatigue=False,
                    policy="heuristic",
                    agent_path="",
                )
            st.session_state["pathology_payload"] = {
                "request": pathology_request,
                "healthy_result": healthy_result,
                "pathology_result": pathology_result,
            }
            pathology_payload = st.session_state.get("pathology_payload")

        if _request_matches(pathology_payload, pathology_request):
            pathology_data = cast(dict[str, Any], pathology_payload)
            healthy_result = cast(
                dict[str, np.ndarray],
                pathology_data["healthy_result"],
            )
            pathology_result = cast(
                dict[str, np.ndarray],
                pathology_data["pathology_result"],
            )
            st.pyplot(
                plot_vmo_vl_ratio_over_time(
                    {
                        "time": pathology_result["time"],
                        "VMO_force": pathology_result["VMO_force"],
                        "VL_force": pathology_result["VL_force"],
                    }
                )
            )
            st.pyplot(
                _line_plot(
                    pathology_result["knee_angle"],
                    {
                        "Healthy displacement (mm)": 1000.0
                        * healthy_result["patellar_displacement"],
                        "Pathology displacement (mm)": 1000.0
                        * pathology_result["patellar_displacement"],
                    },
                    "Patellar Tracking Comparison",
                    "Knee angle (rad)",
                    "Displacement (mm)",
                )
            )
            st.pyplot(
                plot_contact_pressure_map(
                    pathology_result["medial_pressure"],
                    pathology_result["lateral_pressure"],
                    pathology_result["knee_angle"],
                )
            )
            st.pyplot(
                plot_policy_comparison(
                    _policy_summary(healthy_result),
                    _policy_summary(pathology_result),
                )
            )
        else:
            st.info(
                "Click `Run pathology comparison` to generate healthy vs pathology plots."
            )

    with tab_rl:
        rl_scenario = st.selectbox(
            "RL scenario",
            ["healthy", "pfps_mild", "pfps_moderate", "post_acl", "post_surgical"],
            key="rl_scenario",
        )
        agent_path = st.text_input("Agent path", value="results/rl/model")
        log_dir = st.text_input("Training log dir", value="results/rl")
        rl_request = {
            "scenario": rl_scenario,
            "agent_path": agent_path,
            "log_dir": log_dir,
        }
        run_rl = st.button("Run RL replay", key="rl_run")
        rl_payload = st.session_state.get("rl_payload")

        if run_rl:
            with st.spinner("Running RL and baseline policy comparisons..."):
                rl_result = _cached_rollout(
                    scenario=rl_scenario,
                    target_motion="sit_to_stand",
                    enable_fatigue=True,
                    policy="agent",
                    agent_path=agent_path,
                )
                constant_result = _cached_rollout(
                    scenario=rl_scenario,
                    target_motion="sit_to_stand",
                    enable_fatigue=True,
                    policy="constant",
                    agent_path="",
                )
                manual_result = _cached_rollout(
                    scenario=rl_scenario,
                    target_motion="sit_to_stand",
                    enable_fatigue=True,
                    policy="manual",
                    agent_path="",
                )
            st.session_state["rl_payload"] = {
                "request": rl_request,
                "rl_result": rl_result,
                "constant_result": constant_result,
                "manual_result": manual_result,
            }
            rl_payload = st.session_state.get("rl_payload")

        if _request_matches(rl_payload, rl_request):
            rl_data = cast(dict[str, Any], rl_payload)
            rl_result = cast(dict[str, np.ndarray], rl_data["rl_result"])
            constant_result = cast(
                dict[str, np.ndarray],
                rl_data["constant_result"],
            )
            manual_result = cast(dict[str, np.ndarray], rl_data["manual_result"])

            if Path(log_dir).exists():
                try:
                    st.pyplot(plot_training_curve(log_dir))
                except FileNotFoundError:
                    st.info(
                        "No training curve found in the selected log directory yet."
                    )

            comparison = pd.DataFrame.from_records(
                [
                    {"policy": "Agent", **_policy_summary(rl_result)},
                    {"policy": "Constant", **_policy_summary(constant_result)},
                    {"policy": "Manual", **_policy_summary(manual_result)},
                ]
            )
            st.dataframe(comparison, use_container_width=True)
            st.pyplot(
                _line_plot(
                    rl_result["time"][:-1],
                    {
                        "Reward": rl_result["reward"],
                        "Lateral displacement (mm)": 1000.0
                        * np.maximum(rl_result["patellar_displacement"][:-1], 0.0),
                    },
                    "Episode Replay",
                    "Time (s)",
                    "Value",
                )
            )
            st.pyplot(
                _line_plot(
                    rl_result["time"][:-1],
                    {
                        "VMO": rl_result["action"][:, 0],
                        "VML": rl_result["action"][:, 1],
                        "VL": rl_result["action"][:, 2],
                    },
                    "Control Policy",
                    "Time (s)",
                    "Excitation",
                )
            )
        else:
            st.info("Click `Run RL replay` to generate policy visualizations.")

    with tab_analysis:
        output_variable = st.selectbox(
            "Sobol output variable",
            [
                "peak_force",
                "patellar_displacement",
                "vmo_vl_ratio",
                "time_to_peak",
                "fatigue_rate",
            ],
        )
        n_samples = st.slider(
            "Sobol base sample count",
            min_value=16,
            max_value=256,
            value=32,
            step=16,
        )
        analysis_request = {
            "output_variable": output_variable,
            "n_samples": n_samples,
        }
        run_analysis = st.button("Run analysis snapshot", key="analysis_snapshot")
        analysis_payload = st.session_state.get("analysis_payload")

        if run_analysis:
            with st.spinner("Running sensitivity and validation analysis..."):
                (
                    sensitivity_frame,
                    comparison_frame,
                    validation_frame,
                    validation_report,
                ) = _cached_analysis_snapshot(
                    n_samples=n_samples,
                    output_variable=output_variable,
                )
            st.session_state["analysis_payload"] = {
                "request": analysis_request,
                "sensitivity_frame": sensitivity_frame,
                "comparison_frame": comparison_frame,
                "validation_frame": validation_frame,
                "validation_report": validation_report,
            }
            analysis_payload = st.session_state.get("analysis_payload")

        if _request_matches(analysis_payload, analysis_request):
            analysis_data = cast(dict[str, Any], analysis_payload)
            sensitivity_frame = cast(
                pd.DataFrame,
                analysis_data["sensitivity_frame"],
            )
            comparison_frame = cast(
                pd.DataFrame,
                analysis_data["comparison_frame"],
            )
            validation_frame = cast(
                pd.DataFrame,
                analysis_data["validation_frame"],
            )
            validation_report = cast(str, analysis_data["validation_report"])
            st.bar_chart(sensitivity_frame.set_index("parameter")[["S1", "ST"]])
            st.dataframe(comparison_frame, use_container_width=True)
            st.dataframe(validation_frame, use_container_width=True)
            st.markdown(validation_report)
        else:
            st.info(
                "Click `Run analysis snapshot` to generate Sobol indices and validation tables."
            )


def main() -> None:
    """Entry point for `streamlit run` or direct execution."""

    build_dashboard()


__all__ = ["build_dashboard", "main"]


if __name__ == "__main__":
    main()
