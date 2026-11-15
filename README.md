# Failure-Aware Hierarchical Control

**Selective residual reinforcement learning for robust robotic manipulation.**

A well-trained goal-conditioned policy (here SAC + HER on the Fetch arm) works until something it was not trained for happens: weaker actuators, delayed or noisy commands, noisy sensing, or a goal that moves mid-task. This project asks whether such a controller can be made robust *without retraining it*, by keeping the base policy frozen and adding three layers on top:

1. a **failure detector** that recognizes, early enough to act, that an episode is heading for failure;
2. a **residual RL policy** that adds a correction to the base action only when the detector asks for it (selective activation);
3. a **recovery orchestrator** that chooses between continuing, residual correction, and higher-level recovery skills.

## Hypothesis

Gating a residual correction on a failure detector should recover more disturbed episodes than either the frozen policy alone or an always-on residual, with less interference on episodes the base policy already solves. That benefit depends on the detector warning early enough, at a low enough false-alarm rate, for a correction to finish before the deadline.

## Status

| Stage | State |
| --- | --- |
| Frozen fp32 SAC+HER base policies (FetchReach, FetchPush; 3 seeds each) | done, in quantrl-fetch; reused read-only here |
| Phase 1: disturbance framework and baseline characterization | done (validation seeds) |
| Reactive detectors: progress-deficit, phase-aware, matched-FPR threshold sweep | done (exploratory, validation seeds) |
| Learned failure predictor | not started (developed separately) |
| Selective residual RL | not started |
| Hierarchical recovery orchestration | not started |
| Held-out evaluation (seeds 4,000,000+) | not started; the range has never been used |

## Findings so far

All of the following are on validation-range episodes that also informed design choices: treat them as exploratory, not as final evidence. Details, intervals and caveats are in `docs/`.

**The frozen policies are robust to mild disturbances and fail in characteristic ways under strong ones** ([DISTURBANCES.md](docs/DISTURBANCES.md)). 3 training seeds × 100 episodes per condition; both tasks reach 100% success undisturbed.
- **FetchPush fails gradually under actuator weakness:** action scale 0.5 / 0.65 / 0.75 / 0.85 gives 89 / 70 / 50 / 20% success.
- **Late goal shifts are time-limited:** a shift of at least 10 cm at step 40 gives 60% success, at step 25 97%.
- **Command delay is the most damaging disturbance:** 3 steps leaves 8% (Reach) and 13% (Push) success.
- **Position noise up to 1 cm is tolerated (97 to 99%).** Noise on near-constant observation entries (the earlier full-vector model) is an artefact and is no longer used by default.
- **FetchReach failures are mostly holding failures:** the arm reaches the goal and then drifts out of the 5 cm radius.

**A distance-only progress-deficit detector** ([PROGRESS_DEFICIT_DETECTOR.md](docs/PROGRESS_DEFICIT_DETECTOR.md)) flags episodes whose distance to the goal is shrinking too slowly for the 50-step deadline. On FetchPush its raw detection rate is high, but mostly because it warns during the approach phase, when the gripper is still moving to the block.

**A phase-aware detector** ([PHASE_AWARE_DETECTOR.md](docs/PHASE_AWARE_DETECTOR.md)) separates approach, manipulation, and goal maintenance using ground-truth gripper and object positions. At a fixed threshold it trades detection for far fewer false alarms during approach.

**Matched-FPR threshold sweep** ([THRESHOLD_SWEEP.md](docs/THRESHOLD_SWEEP.md)). At the same false-positive rate, phase awareness improves *actionable* detection in the low-false-alarm regime but not above it:

| False-positive rate | Warnings with more than 5 steps left | Warnings with more than 10 steps left |
| --- | --- | --- |
| 2–10% | +6 to +11 points | +10 to +14 points |
| 20–30% | no advantage (−1 point) | no advantage (−1 point) |

- The low-FPR advantage holds for all three trained policies.
- It comes from persistent slow-progress disturbances (action scaling, long delays) and reverses for noise-driven and late failures, so the pooled number depends on the disturbance mix.
- No detector gives useful warning for a goal shift at step 40.

**Not yet shown:** that any warning is early enough for a correction to succeed (that needs the residual controller), or that any of this holds on held-out seeds.

## Architecture

```text
src/farc/
├── base_policy/     frozen SAC+HER policies: hash-verified read-only checkpoint loading,
│                    fp32 actor, goal normalizer, the original quantrl-fetch evaluator
├── envs/            Fetch environment factory, validated observation layout
├── disturbances/    perturbation specs and configs, seeded perturbations, frozen-policy
│                    rollouts with ground-truth logging, recovery metrics, resumable runner
├── detection/       progress-deficit and phase-aware detectors, matched evaluations,
│                    exact threshold sweep
├── statistics.py    episode and hierarchical (seed, episode) bootstrap intervals
└── paths.py         $FARC_DATA_DIR resolution
configs/             JSON configs for disturbance experiments and detector evaluations
scripts/             command-line entry points (see below)
results/             compact reference tables used by the reproduction check
docs/                research notes, data management, figures
tests/               unit and integration tests
```

Future layers (`residual/`, `orchestration/`, and a learned predictor developed separately) will consume the same rollout and disturbance interfaces. The base policy stays frozen throughout.

## Setup

Python 3.12 via [uv](https://docs.astral.sh/uv/); dependency versions are locked to those that produced the results.

```bash
uv sync
```

Checkpoints and outputs live outside the repository in `$FARC_DATA_DIR` (default `~/farc-data`). Import the frozen policies from a quantrl-fetch clone and verify them ([docs/DATA.md](docs/DATA.md)):

```bash
uv run python scripts/import_checkpoints.py --source-runs /path/to/quantrl-fetch/runs
```
```bash
uv run python scripts/verify_checkpoints.py
```

## Reproducing Phase 1

```bash
./scripts/reproduce_phase1.sh
```

This regenerates the extended disturbance evaluation (12,600 episodes) and all three detector evaluations from the checkpoints, then compares 17 result tables with `results/`. It takes about 15 minutes on one CPU core of an Apple M5 and writes about 100 MB to `$FARC_DATA_DIR/experiments`. At migration (commits `cd8ccf5` and `110591d`, clean trees) all 17 tables were bit-identical to the quantrl-fetch results. Individual steps:

| Script | Purpose |
| --- | --- |
| `scripts/run_disturbance_eval.py --config configs/disturbances/<file>.json [--resume]` | disturbance experiments |
| `scripts/evaluate_detector.py` | progress-deficit detector grid |
| `scripts/evaluate_phase_detector.py` | phase-aware vs distance-only, matched episodes |
| `scripts/compare_detectors.py` | episode-paired comparison of two saved detectors |
| `scripts/sweep_detector_thresholds.py` | exact matched-FPR threshold sweep |
| `scripts/check_reproduction.py` | compare regenerated outputs with `results/` |

## Development

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

Tests that need checkpoints or regenerated outputs skip when they are absent.

## Roadmap

1. **Learned failure predictor** (separate track): trained on validation-range rollouts, compared with the reactive detectors at matched FPR and lead time.
2. **Selective residual RL** (FetchPush first): a residual policy trained only on detector-triggered segments, compared with no residual and with an always-on residual. Its measured time-to-recover defines what "actionable lead time" means.
3. **Hierarchical recovery:** an orchestrator over continue, residual, and recovery skills (re-approach, re-grasp, re-target after a goal shift).
4. **Held-out evaluation:** freeze every choice, then run once on seeds 4,000,000+ with per-condition and condition-balanced reporting.
5. **Broader disturbances:** external forces, more tasks (FetchPickAndPlace, FetchSlide).

## Provenance

Split out of [m-alan08/quantrl-fetch](https://github.com/m-alan08/quantrl-fetch), which trained the base policies and hosted Phase 1 on its `phase1-disturbance-eval` branch. [PROVENANCE.md](PROVENANCE.md) maps every migrated file to its source path and commits.

## License

MIT
