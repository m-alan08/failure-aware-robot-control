# Migrated from m-alan08/quantrl-fetch src/quantrl/disturbances/config.py
# (introduced in 7e7f67a, last changed in 3b3f495). See PROVENANCE.md.
"""Validated configuration for disturbance evaluation experiments.

An experiment file (for example ``configs/disturbances/pilot.json``) holds the shared settings
(checkpoints, episode seeds, perturbation seed, output directory) and lists condition files.
Each condition file holds condition entries; an entry with ``severities`` (and optionally
``onsets``) expands into one condition per combination. Unknown keys are errors.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DISTURBANCE_TYPES = (
    "action_scale",
    "action_noise",
    "observation_noise",
    "goal_shift",
    "action_delay",
)
# Fixed stream ids keep each perturbation's random numbers independent of which other
# perturbations share the condition. Never renumber these: it would change every draw.
STREAM_IDS = {name: index + 1 for index, name in enumerate(DISTURBANCE_TYPES)}
ACTION_TYPES = ("action_delay", "action_scale", "action_noise")  # application order
NOISY_OBSERVATION_FIELDS = ("observation", "desired_goal")
# "full": one raw-unit sigma on every entry of the selected fields (the original behaviour).
# "positions" and "velocities": physically grouped entries of the `observation` vector with
# field-appropriate units; see farc.envs.observation_layout.
OBSERVATION_MASKS = ("full", "positions", "velocities")
BASELINE_TYPE = "none"

# Episode seed ranges. Training-time evaluation uses 1_000_000+ and PTQ/QAT calibration uses
# 2_000_000+ (quantrl-fetch's EvalConfig). Disturbance pilots use [3_000_000, 3_500_000),
# validation runs (baseline characterization, design decisions) use [3_500_000, 4_000_000),
# and [4_000_000, 5_000_000) is reserved for the final held-out test.
SEED_RANGES = {
    "pilot": (3_000_000, 3_500_000),
    "validation": (3_500_000, 4_000_000),
    "heldout": (4_000_000, 5_000_000),
}


class DisturbanceConfigError(ValueError):
    pass


@dataclass(frozen=True)
class DisturbanceSpec:
    """One perturbation.

    ``onset`` is the 0-based index of the first action (decision step) the perturbation
    affects; ``duration`` is the number of affected decision steps, or ``None`` for the rest
    of the episode. Severity units depend on the type:

    - action_scale: fraction of the commanded action removed, executed = (1 - s) * action
    - action_noise: standard deviation of Gaussian noise added to the action (action units)
    - observation_noise: standard deviation of Gaussian noise added to the controller's copy
      of the observation. mask "full": raw units on every entry of ``fields``; mask
      "positions": metres on the gripper and object Cartesian positions; mask "velocities":
      metres per second on the linear velocities (converted to the observation's per-step
      units by multiplying by the control period)
    - goal_shift: minimum Euclidean distance (metres) between the old and the new goal
    - action_delay: number of decision steps of delay (integer)
    """

    type: str
    severity: float
    onset: int = 0
    duration: int | None = None
    fields: tuple[str, ...] = ("observation",)
    max_attempts: int = 1000
    mask: str = "full"

    def active(self, step: int) -> bool:
        if step < self.onset:
            return False
        return self.duration is None or step < self.onset + self.duration

    def to_dict(self) -> dict[str, Any]:
        values: dict[str, Any] = {
            "type": self.type,
            "severity": self.severity,
            "onset": self.onset,
            "duration": self.duration,
        }
        if self.type == "observation_noise":
            values["fields"] = list(self.fields)
            values["mask"] = self.mask
        if self.type == "goal_shift":
            values["max_attempts"] = self.max_attempts
        return values


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    disturbances: tuple[DisturbanceSpec, ...]
    env_ids: tuple[str, ...] | None = None  # None: every environment

    def applies_to(self, env_id: str) -> bool:
        return self.env_ids is None or env_id in self.env_ids

    @property
    def window_end(self) -> int | None:
        """Decision index after the last disturbed decision, or None if any disturbance lasts
        to the end of the episode (a permanent change, including every goal shift)."""
        if not self.disturbances or any(d.duration is None for d in self.disturbances):
            return None
        return max(d.onset + int(d.duration) for d in self.disturbances if d.duration)

    @property
    def regime(self) -> str:
        if self.is_baseline:
            return "none"
        return "transient" if self.window_end is not None else "permanent"

    @property
    def is_baseline(self) -> bool:
        return not self.disturbances

    @property
    def onset(self) -> int | None:
        return min((d.onset for d in self.disturbances), default=None)

    def to_dict(self) -> dict[str, Any]:
        values: dict[str, Any] = {
            "name": self.name,
            "disturbances": [d.to_dict() for d in self.disturbances],
        }
        if self.env_ids is not None:
            values["env_ids"] = list(self.env_ids)
        return values


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    stage: str
    episodes: int
    episode_seed_offset: int
    perturbation_seed: int
    checkpoints: tuple[str, ...]
    conditions: tuple[ConditionSpec, ...]
    output_dir: str
    save_trajectories: bool = True
    bootstrap_resamples: int = 10_000
    confidence: float = 0.95
    torch_threads: int = 1

    @property
    def episode_seeds(self) -> list[int]:
        return [self.episode_seed_offset + index for index in range(self.episodes)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stage": self.stage,
            "episodes": self.episodes,
            "episode_seed_offset": self.episode_seed_offset,
            "perturbation_seed": self.perturbation_seed,
            "checkpoints": list(self.checkpoints),
            "conditions": [condition.to_dict() for condition in self.conditions],
            "output_dir": self.output_dir,
            "save_trajectories": self.save_trajectories,
            "bootstrap_resamples": self.bootstrap_resamples,
            "confidence": self.confidence,
            "torch_threads": self.torch_threads,
        }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (_is_int(value) or isinstance(value, float)) and math.isfinite(value)


def _require_keys(where: str, values: dict[str, Any], allowed: set[str], required: set[str]):
    if not isinstance(values, dict):
        raise DisturbanceConfigError(f"{where} must be a JSON object, got {values!r}")
    unknown = set(values) - allowed
    if unknown:
        raise DisturbanceConfigError(f"{where}: unknown keys {sorted(unknown)}")
    missing = required - set(values)
    if missing:
        raise DisturbanceConfigError(f"{where}: missing keys {sorted(missing)}")


def validate_disturbance(spec: DisturbanceSpec, horizon: int | None = None) -> DisturbanceSpec:
    problems = []
    if spec.type not in DISTURBANCE_TYPES:
        problems.append(f"type must be one of {DISTURBANCE_TYPES}, got {spec.type!r}")
    if not _is_int(spec.onset) or spec.onset < 0:
        problems.append(f"onset must be a non-negative integer, got {spec.onset!r}")
    elif horizon is not None and spec.onset >= horizon:
        problems.append(f"onset {spec.onset} is not inside the {horizon}-step episode")
    if spec.duration is not None and (not _is_int(spec.duration) or spec.duration < 1):
        problems.append(f"duration must be null or a positive integer, got {spec.duration!r}")
    severity = spec.severity
    if not _is_number(severity):
        problems.append(f"severity must be a finite number, got {severity!r}")
    elif spec.type == "action_scale" and not 0 < severity <= 1:
        problems.append(f"action_scale severity must be in (0, 1], got {severity}")
    elif spec.type in ("action_noise", "observation_noise", "goal_shift") and severity <= 0:
        problems.append(f"{spec.type} severity must be positive, got {severity}")
    elif spec.type == "action_delay" and (not _is_int(severity) or severity < 1):
        problems.append(f"action_delay severity must be an integer of at least 1, got {severity}")
    if spec.type == "observation_noise" and (
        not spec.fields or any(f not in NOISY_OBSERVATION_FIELDS for f in spec.fields)
    ):
        problems.append(
            f"observation_noise fields must be a non-empty subset of "
            f"{NOISY_OBSERVATION_FIELDS}, got {spec.fields!r}"
        )
    if spec.type == "observation_noise":
        if spec.mask not in OBSERVATION_MASKS:
            problems.append(f"mask must be one of {OBSERVATION_MASKS}, got {spec.mask!r}")
        elif spec.mask != "full" and tuple(spec.fields) != ("observation",):
            problems.append(f"mask {spec.mask!r} only applies to fields ['observation']")
    if spec.type == "goal_shift":
        if spec.duration is not None:
            problems.append("goal_shift is a persistent change, so duration must be null")
        if not _is_int(spec.max_attempts) or spec.max_attempts < 1:
            problems.append(f"max_attempts must be a positive integer, got {spec.max_attempts!r}")
    if problems:
        raise DisturbanceConfigError(
            f"Invalid disturbance {spec}:\n" + "\n".join(f"  - {p}" for p in problems)
        )
    return spec


def condition_name(spec: DisturbanceSpec) -> str:
    duration = "inf" if spec.duration is None else str(spec.duration)
    kind = spec.type
    if spec.type == "observation_noise" and spec.mask != "full":
        kind = f"{spec.type}_{spec.mask}"
    return f"{kind}_sev{spec.severity:g}_on{spec.onset}_dur{duration}"


ENTRY_KEYS = {
    "type",
    "severities",
    "severity",
    "onsets",
    "onset",
    "duration",
    "fields",
    "max_attempts",
    "name",
    "mask",
    "env_ids",
}


def expand_condition_entry(entry: dict[str, Any], where: str) -> list[ConditionSpec]:
    _require_keys(where, entry, ENTRY_KEYS, {"type"})
    env_ids = entry.get("env_ids")
    if env_ids is not None:
        if (
            not isinstance(env_ids, list)
            or not env_ids
            or not all(isinstance(e, str) for e in env_ids)
        ):
            raise DisturbanceConfigError(f"{where}: env_ids must be a non-empty list of ids")
        env_ids = tuple(env_ids)
    if entry["type"] == BASELINE_TYPE:
        _require_keys(where, entry, {"type", "name", "env_ids"}, {"type"})
        return [ConditionSpec(entry.get("name", "undisturbed"), (), env_ids)]
    if ("severities" in entry) == ("severity" in entry):
        raise DisturbanceConfigError(f"{where}: give exactly one of severity or severities")
    if "onsets" in entry and "onset" in entry:
        raise DisturbanceConfigError(f"{where}: give at most one of onset or onsets")
    severities = entry.get("severities", [entry.get("severity")])
    onsets = entry.get("onsets", [entry.get("onset", 0)])
    if not isinstance(severities, list) or not severities:
        raise DisturbanceConfigError(f"{where}: severities must be a non-empty list")
    if not isinstance(onsets, list) or not onsets:
        raise DisturbanceConfigError(f"{where}: onsets must be a non-empty list")
    if "name" in entry and len(severities) * len(onsets) > 1:
        raise DisturbanceConfigError(f"{where}: name is only allowed for a single condition")
    conditions = []
    for severity, onset in itertools.product(severities, onsets):
        spec = DisturbanceSpec(
            type=entry["type"],
            severity=severity,
            onset=onset,
            duration=entry.get("duration"),
            fields=tuple(entry.get("fields", ("observation",))),
            max_attempts=entry.get("max_attempts", 1000),
            mask=entry.get("mask", "full"),
        )
        validate_disturbance(spec)
        name = entry.get("name", condition_name(spec))
        conditions.append(ConditionSpec(name, (spec,), env_ids))
    return conditions


EXPERIMENT_KEYS = {
    "name",
    "stage",
    "episodes",
    "episode_seed_offset",
    "perturbation_seed",
    "checkpoints",
    "condition_files",
    "conditions",
    "output_dir",
    "save_trajectories",
    "bootstrap_resamples",
    "confidence",
    "torch_threads",
    "description",
}


def validate_experiment(config: ExperimentConfig) -> ExperimentConfig:
    problems = []
    if not isinstance(config.name, str) or not config.name or "/" in config.name:
        problems.append(f"name must be a non-empty string without '/', got {config.name!r}")
    if config.stage not in SEED_RANGES:
        problems.append(f"stage must be one of {tuple(SEED_RANGES)}, got {config.stage!r}")
    if not _is_int(config.episodes) or config.episodes < 1:
        problems.append(f"episodes must be a positive integer, got {config.episodes!r}")
    if not _is_int(config.episode_seed_offset):
        problems.append(
            f"episode_seed_offset must be an integer, got {config.episode_seed_offset!r}"
        )
    elif config.stage in SEED_RANGES and _is_int(config.episodes):
        low, high = SEED_RANGES[config.stage]
        end = config.episode_seed_offset + config.episodes
        if config.episode_seed_offset < low or end > high:
            problems.append(
                f"{config.stage} episode seeds must lie in [{low}, {high}), "
                f"got [{config.episode_seed_offset}, {end})"
            )
    if not _is_int(config.perturbation_seed) or config.perturbation_seed < 0:
        problems.append(
            f"perturbation_seed must be a non-negative integer, got {config.perturbation_seed!r}"
        )
    if not config.checkpoints:
        problems.append("checkpoints must list at least one checkpoint")
    if not config.conditions:
        problems.append("at least one condition is needed")
    duplicates = sorted(
        {
            first.name
            for index, first in enumerate(config.conditions)
            for second in config.conditions[index + 1 :]
            if first.name == second.name
            and (
                first.env_ids is None
                or second.env_ids is None
                or set(first.env_ids) & set(second.env_ids)
            )
        }
    )
    if duplicates:
        problems.append(f"duplicate condition names {duplicates}")
    if not isinstance(config.save_trajectories, bool):
        problems.append("save_trajectories must be true or false")
    if not _is_int(config.bootstrap_resamples) or config.bootstrap_resamples < 1:
        problems.append("bootstrap_resamples must be a positive integer")
    if not _is_number(config.confidence) or not 0 < config.confidence < 1:
        problems.append("confidence must be in (0, 1)")
    if not _is_int(config.torch_threads) or config.torch_threads < 1:
        problems.append("torch_threads must be a positive integer")
    if problems:
        raise DisturbanceConfigError(
            "Invalid disturbance experiment:\n" + "\n".join(f"  - {p}" for p in problems)
        )
    return config


def experiment_from_dict(values: dict[str, Any], base_dir: Path | None = None) -> ExperimentConfig:
    _require_keys(
        "experiment",
        values,
        EXPERIMENT_KEYS,
        {
            "name",
            "stage",
            "episodes",
            "episode_seed_offset",
            "perturbation_seed",
            "checkpoints",
            "output_dir",
        },
    )
    base_dir = base_dir or Path(".")
    conditions: list[ConditionSpec] = []
    for file_name in values.get("condition_files", []):
        path = base_dir / file_name
        file_values = json.loads(path.read_text())
        _require_keys(str(path), file_values, {"conditions", "description"}, {"conditions"})
        for index, entry in enumerate(file_values["conditions"]):
            conditions.extend(expand_condition_entry(entry, f"{path} conditions[{index}]"))
    for index, entry in enumerate(values.get("conditions", [])):
        conditions.extend(expand_condition_entry(entry, f"experiment conditions[{index}]"))
    checkpoints = values["checkpoints"]
    if not isinstance(checkpoints, list) or not all(isinstance(c, str) for c in checkpoints):
        raise DisturbanceConfigError("checkpoints must be a list of paths")
    return validate_experiment(
        ExperimentConfig(
            name=values["name"],
            stage=values["stage"],
            episodes=values["episodes"],
            episode_seed_offset=values["episode_seed_offset"],
            perturbation_seed=values["perturbation_seed"],
            checkpoints=tuple(checkpoints),
            conditions=tuple(conditions),
            output_dir=values["output_dir"],
            save_trajectories=values.get("save_trajectories", True),
            bootstrap_resamples=values.get("bootstrap_resamples", 10_000),
            confidence=values.get("confidence", 0.95),
            torch_threads=values.get("torch_threads", 1),
        )
    )


def load_experiment(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    path = Path(path)
    values = json.loads(path.read_text())
    values.update(overrides or {})
    return experiment_from_dict(values, path.parent)
