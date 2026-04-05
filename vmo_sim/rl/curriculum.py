"""Curriculum learning utilities for staged rehabilitation training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import cast


@dataclass(slots=True)
class CurriculumStage:
    """Definition of a single rehabilitation curriculum stage."""

    name: str
    target_motion: str
    scenario: str
    fatigue_enabled: bool
    advance_threshold: float


class RehabCurriculum:
    """Progressive difficulty curriculum for RL training."""

    def __init__(self, stages: list[dict[str, object]] | None = None) -> None:
        default_stages = stages or [
            {
                "name": "isometric",
                "target_motion": "isometric",
                "scenario": "healthy",
                "fatigue_enabled": False,
                "advance_threshold": -0.10,
            },
            {
                "name": "slow_isokinetic",
                "target_motion": "isokinetic",
                "scenario": "healthy",
                "fatigue_enabled": False,
                "advance_threshold": 0.00,
            },
            {
                "name": "sit_to_stand",
                "target_motion": "sit_to_stand",
                "scenario": "healthy",
                "fatigue_enabled": False,
                "advance_threshold": 0.10,
            },
            {
                "name": "stair_climb",
                "target_motion": "stair_climb",
                "scenario": "healthy",
                "fatigue_enabled": False,
                "advance_threshold": 0.15,
            },
            {
                "name": "fatigue",
                "target_motion": "sit_to_stand",
                "scenario": "healthy",
                "fatigue_enabled": True,
                "advance_threshold": 0.20,
            },
            {
                "name": "pathological",
                "target_motion": "sit_to_stand",
                "scenario": "pfps_moderate",
                "fatigue_enabled": True,
                "advance_threshold": 0.25,
            },
        ]
        self.stages = [
            CurriculumStage(
                name=str(stage["name"]),
                target_motion=str(stage["target_motion"]),
                scenario=str(stage["scenario"]),
                fatigue_enabled=bool(stage["fatigue_enabled"]),
                advance_threshold=float(cast(float | int, stage["advance_threshold"])),
            )
            for stage in default_stages
        ]
        self.stage_index = 0

    def current_stage(self) -> dict[str, object]:
        """Return the current stage configuration."""

        return asdict(self.stages[self.stage_index])

    def should_advance(self, recent_rewards: list[float]) -> bool:
        """Return whether the curriculum should progress."""

        if self.stage_index >= len(self.stages) - 1:
            return False
        if len(recent_rewards) < 50:
            return False
        mean_reward = sum(recent_rewards[-50:]) / 50.0
        return mean_reward > self.stages[self.stage_index].advance_threshold

    def advance(self) -> dict[str, object]:
        """Advance to the next stage and return its configuration."""

        if self.stage_index < len(self.stages) - 1:
            self.stage_index += 1
        return self.current_stage()

    def get_env_kwargs(self) -> dict[str, object]:
        """Return environment keyword arguments for the current stage."""

        stage = self.stages[self.stage_index]
        return {
            "scenario": stage.scenario,
            "target_motion": stage.target_motion,
            "enable_fatigue": stage.fatigue_enabled,
        }
