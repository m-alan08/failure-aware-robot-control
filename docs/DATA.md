# Data, checkpoints and artifacts

Nothing large is stored in git. Checkpoints, trajectories and experiment outputs live in a data directory outside the repository:

```text
$FARC_DATA_DIR            (default ~/farc-data; keep it outside iCloud/Dropbox-synced folders)
├── checkpoints/<id>/checkpoint.pt      frozen base policies, read-only, hash-verified
└── experiments/<name>/                 outputs of scripts/run_disturbance_eval.py
    ├── parts/<env>_seed<k>/<condition>/  per-condition episodes, trajectories.csv.gz, done.json
    ├── episodes.csv, summary*.csv, distance_curves.csv, metadata.json, *.png
    └── detectors/<name>/               outputs of the detector scripts
```

- **Paths in configs.** Config files refer to this directory as `$FARC_DATA_DIR`, so they are portable; `farc.paths.resolve` expands it.
- **Committed results.** Only compact result tables (`results/`, under 2 MB) and a few figures (`docs/figures/`) are committed. They are the reference that `scripts/check_reproduction.py` compares regenerated outputs with.

## Frozen base policies

| Id | Task | Training seed | Steps | SHA-256 (first 12) |
| --- | --- | --- | --- | --- |
| `fetch_reach_fp32_seed0` | FetchReach-v4 | 0 | 30,000 | aba3ddaeb84f |
| `fetch_reach_fp32_seed1` | FetchReach-v4 | 1 | 30,000 | c96098003621 |
| `fetch_reach_fp32_seed2` | FetchReach-v4 | 2 | 30,000 | 79c2d8b3c1ac |
| `fetch_push_fp32_seed0` | FetchPush-v4 | 0 | 1,000,000 | 07f6244471f8 |
| `fetch_push_fp32_seed1` | FetchPush-v4 | 1 | 1,000,000 | 1f569ab405ed |
| `fetch_push_fp32_seed2` | FetchPush-v4 | 2 | 1,000,000 | 901ab0c01a2a |

Full hashes, sizes, and training commits are in `src/farc/base_policy/checkpoints.json`. Every load checks the file against that manifest and refuses anything else; nothing in this repository writes to a checkpoint.

### Obtaining them

**From a quantrl-fetch clone that has them** (`runs/<id>/checkpoint.pt`, the usual case):

```bash
uv run python scripts/import_checkpoints.py --source-runs /path/to/quantrl-fetch/runs
```
```bash
uv run python scripts/verify_checkpoints.py
```

The import copies each file to `$FARC_DATA_DIR/checkpoints/<id>/checkpoint.pt`, checks the SHA-256 of source and copy, and marks the copy read-only.

**By retraining in quantrl-fetch.** The FetchPush policies are recorded as trained at `fdb6266` (on GitHub) with `configs/fetch_push.json`, seeds 0 to 2, on CPU.

> **Correction (2026-11-17):**
> - `configs/fetch_push.json` does not exist at `fdb6266`. It was first committed two commits
>   later, in `95e9611` (verified with the GitHub API; see
>   `docs/PHASE_H_RETRAINED_BASELINE_PROPOSAL.md` §2).
> - To retrain, check out `95e9611`, which contains the config.
> - Whether that reproduces the recorded hashes is untested. For example, in quantrl-fetch:

```bash
uv run python scripts/train.py --config configs/fetch_push.json --seed 0 1 2 --device cpu
```

Training is deterministic on CPU in quantrl-fetch's tests, but a retrained file is only accepted here if its hash matches. The FetchReach policies were trained from an uncommitted tree on top of `6355bf7` (see [PROVENANCE.md](../PROVENANCE.md)), so they are not guaranteed to be regenerable bit for bit.

**Archival copy.** The GitHub release [`artifacts-v1`](https://github.com/m-alan08/failure-aware-robot-control/releases/tag/artifacts-v1) (2026-11-17, private like the repository) holds every artifact outside git:
- the six checkpoints;
- the F1 contact model and the Phase B failure-predictor models;
- every raw experiment directory, including both held-out confirmations: 29 tar archives, 764 files, about 1.1 GB.

`MANIFEST.json` in the release lists the SHA-256 of every archive and of every file inside it. Before upload, every archive was extracted and checked against the manifest, `checkpoints.json` and the raw-file hashes in `results/*/analysis.json`. After upload, every asset's size and GitHub SHA-256 digest matched the local copy.

To restore, extract every archive into `$FARC_DATA_DIR`:

```bash
for f in *.tar; do tar -xf "$f" -C "${FARC_DATA_DIR:-$HOME/farc-data}"; done
```
```bash
uv run python scripts/verify_checkpoints.py
```

The archives are not committed to git.

**Verifying a restore.** `scripts/verify_reproduction.py` performs every step of a restore and
checks it:
- verifies the release against `MANIFEST.json`;
- extracts it safely into an empty directory;
- checks the checkpoints, the F1 model and the raw-file hashes;
- recomputes the supported analyses and compares them with `results/`.

On 2026-11-17 it passed in a clean clone at the tagged commit (`docs/REPRODUCIBILITY_REPORT.md`).
Running it on another machine is still outstanding.

## Seed ranges

| Range | Use |
| --- | --- |
| 1,000,000+ | quantrl-fetch training-time and final evaluation |
| 2,000,000+ | quantrl-fetch PTQ/QAT calibration |
| 3,000,000 to 3,499,999 | disturbance pilots (`stage: "pilot"`) |
| 3,500,000 to 3,999,999 | validation: baseline characterization, detector design (`stage: "validation"`) |
| 4,000,000 to 4,999,999 | **reserved final held-out test** (`stage: "heldout"`). Used once each: 4,000,000–299 (E/F1 confirmation) and 4,000,300–599 (G4 confirmation). Nothing at or above 4,000,600 has been used. |

The config validator enforces these ranges.
