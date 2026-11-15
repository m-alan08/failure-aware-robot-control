# Migrated from m-alan08/quantrl-fetch scripts/run_disturbance_eval.py
# (introduced in 0b6b61d, last changed in fad4381). See PROVENANCE.md.
from __future__ import annotations

import argparse

from farc.disturbances.config import load_experiment
from farc.disturbances.experiment import run_experiment
from farc.paths import resolve


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen policies under disturbances on fixed episode seeds."
    )
    parser.add_argument("--config", default="configs/disturbances/pilot.json")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Overrides the config's output_dir; results go to <output_dir>/<name>",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Override, e.g. for a smoke run")
    parser.add_argument("--name", default=None, help="Override the experiment name")
    parser.add_argument("--no-trajectories", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run in the same output directory, skipping finished "
        "conditions (the configuration must be identical)",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    overrides: dict = {}
    if arguments.episodes is not None:
        overrides["episodes"] = arguments.episodes
    if arguments.name is not None:
        overrides["name"] = arguments.name
    if arguments.output_dir is not None:
        overrides["output_dir"] = arguments.output_dir
    if arguments.no_trajectories:
        overrides["save_trajectories"] = False
    config = load_experiment(arguments.config, overrides)
    output_dir = resolve(config.output_dir) / config.name
    if output_dir.exists() and any(output_dir.iterdir()) and not arguments.resume:
        raise SystemExit(
            f"{output_dir} already exists and is not empty; pass --resume to continue it "
            "or choose another --name"
        )
    paths = run_experiment(config, output_dir, verbose=not arguments.quiet, resume=arguments.resume)
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
