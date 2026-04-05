"""Training utilities for RL-based rehabilitation policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from vmo_sim.rl.curriculum import RehabCurriculum
from vmo_sim.rl.vmo_env import VMORehabEnv

try:
    from stable_baselines3 import PPO, SAC  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    PPO = None
    SAC = None


class SupportsPredict(Protocol):
    """Protocol for policies exposing a predict method."""

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, Any]:
        """Return an action prediction."""


class HeuristicPolicy:
    """Fallback policy used when stable-baselines3 is unavailable."""

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
        del deterministic
        angle_error = float(observation[9] - observation[0])
        velocity = float(observation[1])
        displacement = float(observation[7])
        drive = np.clip(0.45 + 1.2 * angle_error - 0.08 * velocity, 0.0, 1.0)
        medial_bias = np.clip(0.5 - 30.0 * displacement, 0.0, 1.0)
        action = np.array(
            [
                np.clip(drive * (0.8 + 0.4 * medial_bias), 0.0, 1.0),
                np.clip(drive * 0.9, 0.0, 1.0),
                np.clip(drive * (1.2 - 0.4 * medial_bias), 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        return action, None


class RLTrainer:
    """Wrapper around stable-baselines3 for rehabilitation-agent training."""

    def __init__(
        self,
        algorithm: str = "PPO",
        env_kwargs: dict[str, object] | None = None,
        curriculum: RehabCurriculum | None = None,
    ) -> None:
        self.algorithm = algorithm.upper()
        self.env_kwargs = env_kwargs or {}
        self.curriculum = curriculum
        self.env = self._make_env(self._current_env_kwargs())
        self.model: SupportsPredict | None = None
        self.training_history: dict[str, list[float]] = {"episode_rewards": []}

    def _current_env_kwargs(self) -> dict[str, object]:
        if self.curriculum is None:
            return self.env_kwargs.copy()
        merged = self.env_kwargs.copy()
        merged.update(self.curriculum.get_env_kwargs())
        return merged

    @staticmethod
    def _make_env(kwargs: dict[str, object]) -> VMORehabEnv:
        return VMORehabEnv(
            scenario=cast(str, kwargs.get("scenario", "healthy")),
            target_motion=cast(str, kwargs.get("target_motion", "sit_to_stand")),
            render_mode=cast(str | None, kwargs.get("render_mode")),
            dt=cast(float, kwargs.get("dt", 0.01)),
            custom_modifications=cast(
                dict[str, dict[str, float]] | None,
                kwargs.get("custom_modifications"),
            ),
            enable_fatigue=cast(bool, kwargs.get("enable_fatigue", True)),
        )

    def _build_model(self) -> SupportsPredict:
        if self.algorithm == "PPO" and PPO is not None:
            return cast(SupportsPredict, PPO("MlpPolicy", self.env, verbose=0))
        if self.algorithm == "SAC" and SAC is not None:
            return cast(SupportsPredict, SAC("MlpPolicy", self.env, verbose=0))
        return HeuristicPolicy()

    def train(
        self,
        total_timesteps: int = 500_000,
        log_dir: str = "results/rl",
    ) -> dict[str, object]:
        """Train an RL agent with optional curriculum learning."""

        self.model = self._build_model()
        metrics: dict[str, object] = {
            "algorithm": self.algorithm,
            "timesteps": total_timesteps,
        }

        if hasattr(self.model, "learn"):
            learnable_model = cast(Any, self.model)
            learnable_model.learn(total_timesteps=total_timesteps)
            metrics["backend"] = "stable_baselines3"
        else:
            metrics["backend"] = "heuristic"
            n_episodes = max(1, total_timesteps // self.env.episode_length)
            recent_rewards: list[float] = []
            for _ in range(n_episodes):
                observation, _ = self.env.reset()
                episode_reward = 0.0
                terminated = False
                truncated = False
                while not (terminated or truncated):
                    action, _ = self.model.predict(observation)
                    observation, reward, terminated, truncated, _ = self.env.step(
                        action
                    )
                    episode_reward += reward
                recent_rewards.append(episode_reward)
                self.training_history["episode_rewards"].append(episode_reward)
                if self.curriculum is not None and self.curriculum.should_advance(
                    recent_rewards
                ):
                    self.curriculum.advance()
                    self.env.close()
                    self.env = self._make_env(self._current_env_kwargs())
            metrics["episodes"] = len(self.training_history["episode_rewards"])

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        (Path(log_dir) / "training_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        (Path(log_dir) / "training_history.json").write_text(
            json.dumps(self.training_history, indent=2),
            encoding="utf-8",
        )
        return metrics

    def evaluate(self, n_episodes: int = 50) -> dict[str, object]:
        """Evaluate the trained policy over multiple episodes."""

        if self.model is None:
            self.model = self._build_model()
        rewards: list[float] = []
        max_displacements: list[float] = []
        for _ in range(n_episodes):
            observation, _ = self.env.reset()
            terminated = False
            truncated = False
            episode_reward = 0.0
            episode_displacement = 0.0
            while not (terminated or truncated):
                action, _ = self.model.predict(observation)
                observation, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                reward_state = cast(dict[str, float], info["reward_state"])
                episode_displacement = max(
                    episode_displacement, reward_state["lateral_displacement"]
                )
            rewards.append(episode_reward)
            max_displacements.append(episode_displacement)
        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_max_lateral_displacement": float(np.mean(max_displacements)),
            "episode_rewards": rewards,
        }

    def compare_scenarios(
        self,
        scenarios: list[str],
        n_episodes: int = 20,
    ) -> pd.DataFrame:
        """Evaluate the same policy across multiple clinical scenarios."""

        if self.model is None:
            self.model = self._build_model()
        records: list[dict[str, float | str]] = []
        base_kwargs = {
            key: value for key, value in self.env_kwargs.items() if key != "scenario"
        }
        for scenario in scenarios:
            scenario_kwargs = base_kwargs.copy()
            scenario_kwargs["scenario"] = scenario
            env = self._make_env(scenario_kwargs)
            scenario_rewards: list[float] = []
            for _ in range(n_episodes):
                observation, _ = env.reset()
                terminated = False
                truncated = False
                episode_reward = 0.0
                while not (terminated or truncated):
                    action, _ = self.model.predict(observation)
                    observation, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                scenario_rewards.append(episode_reward)
            records.append(
                {
                    "scenario": scenario,
                    "mean_reward": float(np.mean(scenario_rewards)),
                    "std_reward": float(np.std(scenario_rewards)),
                }
            )
            env.close()
        return pd.DataFrame.from_records(records)

    def save(self, path: str) -> None:
        """Save the current agent or heuristic metadata."""

        if self.model is None:
            raise RuntimeError("No model available to save.")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, "save"):
            saveable = cast(Any, self.model)
            saveable.save(str(output))
            return
        output.write_text(
            json.dumps({"algorithm": self.algorithm, "backend": "heuristic"}, indent=2),
            encoding="utf-8",
        )

    def load(self, path: str) -> None:
        """Load a saved RL model or heuristic metadata."""

        model_path = Path(path)
        if self.algorithm == "PPO" and PPO is not None and model_path.suffix != ".json":
            self.model = cast(SupportsPredict, PPO.load(str(model_path), env=self.env))
            return
        if self.algorithm == "SAC" and SAC is not None and model_path.suffix != ".json":
            self.model = cast(SupportsPredict, SAC.load(str(model_path), env=self.env))
            return
        if model_path.exists():
            self.model = HeuristicPolicy()
            return
        raise FileNotFoundError(path)
