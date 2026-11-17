# Reproducibility report: archived artifacts → committed analyses

**Question:** can the committed analysis outputs of Phases E–G4 be recomputed from the
repository at the commit tagged `artifacts-v1`, plus the files in the `artifacts-v1` release,
with nothing else?

**Answer:**
- Yes, for the 10 analyses this check covers: every derived table and every summary is
  byte-identical.
- The check ran in a clean, isolated environment **on the original development machine**, so
  it is **not an independent reproduction**.
- No simulation was rerun. This verifies the *analyses* of the archived raw data, not the
  generation of that data.

Run on 2026-11-17 by Claude, the same agent that designed and ran Phases G1–G4.

## 1. Exact repository commit

| item | value |
|---|---|
| Release | `artifacts-v1` (private GitHub release, 30 assets) |
| Tag → commit | `e7aedddf0666b766f6fe13e9d0726298d1bce693`, which is also the release's recorded target |
| Checkout | a fresh `git clone` into a new temporary directory, outside this worktree and outside any data directory. The commit was checked out detached, and the tree stayed clean throughout. |
| `uv.lock` SHA-256 | `c6e63ad9e1045118…` (unchanged from the tagged commit) |
| Verifier | `scripts/verify_reproduction.py`, from this PR's branch. It was run from *outside* the pristine clone with `--repo`, so the clone was never modified. It imports only the clone's own analysis code. |

## 2. Operating system and dependencies

| item | value |
|---|---|
| OS | macOS 26.5, arm64 (Apple silicon) |
| Python | 3.12.14 (uv-managed; `.python-version` pins 3.12) |
| uv | 0.13.0 |
| Install | `uv sync --frozen` with a **new, empty** `UV_CACHE_DIR`, so every package was downloaded fresh and checked against the lockfile's hashes |
| Key packages | torch 2.14.1, numpy 2.5.4, pandas 3.0.6, mujoco 3.11.0, gymnasium 1.4.0. All match `uv.lock`; the verifier checks this. |

**What was not isolated:**
- the same physical machine and OS;
- the uv-managed Python interpreter installation, which is shared on the machine;
- the GitHub credentials used to download the private release.

## 3. Artifact verification

| check | result |
|---|---|
| Release assets equal the manifest's list: 29 archives + `MANIFEST.json`, nothing missing or extra | passed |
| SHA-256 and size of every archive against `MANIFEST.json` | **29 / 29** |
| Archive members: no absolute paths, no `..`, no links or special files, and exactly the manifest's file list | passed for all 29 |
| Extraction into a new empty data directory (`tarfile` "data" filter) | passed. The verifier refuses a non-empty directory. |
| SHA-256 of every extracted file against `MANIFEST.json` | **764 / 764** |
| Files present outside the manifest | none |
| Frozen SAC checkpoints against `src/farc/base_policy/checkpoints.json` | **6 / 6** |
| F1 contact model against its pinned hash (`8a960329…`) | passed |
| Raw files against the `raw_sha256` recorded in each committed `results/*/analysis.json` | 10 / 10 analyses: every recorded raw file matched, including raw `metadata.json` |

Before any analysis code runs, `FARC_DATA_DIR` is set to the restored directory, and every raw
directory must resolve inside it. So the original `~/farc-data` cannot be used as a silent
fallback.

## 4. Analysis reproduction

Each analysis was recomputed **in memory** from the restored raw files, using the clone's own
pre-registered derive function with its committed config. That config verifies the plan
SHA-256 where the phase pinned one.

- **Tables:** each derived table was serialised exactly as the original runner does
  (`DataFrame.to_csv(index=False)`) and compared byte for byte with `results/`.
- **Summaries:** each summary was serialised with the original JSON writer and compared with
  `analysis.json`.
- **Fallback:** a non-identical table would have been compared numerically (relative 1e-9,
  NaN-aware). None needed it.

| analysis | seeds | derived tables | summary | paired design complete |
|---|---|---|---|---|
| Phase E pilot | dev | 5 / 5 byte-identical | byte-identical | yes |
| E2 (pre-registered) | dev | 6 / 6 | byte-identical | yes |
| F1 evaluation (pre-registered) | dev | 6 / 6 | byte-identical | yes |
| **E/F1 held-out confirmation** | **HO 4,000,000–299** | 6 / 6 | byte-identical | yes |
| G1 | dev | 11 / 11 | byte-identical | yes |
| G2 Stage 1 (incl. branches) | dev | 7 / 7 | byte-identical | yes |
| **G2 Stage 3** | dev | 8 / 8 | byte-identical | yes |
| G2 sham check (exploratory) | dev | 4 / 4 | byte-identical | yes |
| **G3 mechanism** | dev | 11 / 11 | byte-identical | yes |
| **G4 held-out confirmation** | **HO 4,000,300–599** | 8 / 8 | byte-identical | yes |
| **Total** | | **72 / 72** | **10 / 10** | **10 / 10** |

Because every table and summary is byte-identical, all of the following reproduce exactly:
- success rates, paired effect sizes and their CIs;
- one-sided and two-sided p-values and Holm-adjusted p-values;
- hypothesis decisions and labels, safety gates;
- secondary, descriptive and step-level tables (latency, rates, switching);
- G3's divergence and sensitivity tables, and the recorded raw hashes.

**Exact vs statistical reproduction.** Reproduction here is *exact*, not merely within
bootstrap noise. Each bootstrap draws its indices from `numpy.random.default_rng(seed)` with
the seed pinned in the config, and the same NumPy version (2.5.4) was installed.

Exactness therefore depends on the locked versions. NumPy does not promise that `Generator`
streams stay identical across releases, and pandas' CSV float formatting can change. Under
other versions, expect *statistical* agreement:
- CI endpoints and p-values within Monte Carlo error. At 10,000 resamples, a p-value near
  0.025 has a standard error of about 0.0016.
- Point estimates, success rates and rescued/harmed counts still identical, since they do not
  depend on the bootstrap.

**Final run of the finished command:** it exited 0, with 170 checks and 0 failures. The
analyses took about 97 s in total, of which G3 took 33 s. The machine-readable report is
committed as `results/reproduction/verification_report.json`. Its paths are replaced by
placeholders, and it was checked for usernames, absolute paths and credentials (none).

### Not covered (unverified by this check)

| phase | reason |
|---|---|
| Phase 1 baseline and detectors | Their checker, `scripts/reproduce_phase1.sh`, regenerates episodes in the simulator, which this audit does not do. The Phase 1 raw directory is archived and hash-verified, but its derived tables were not recomputed here. |
| Phase B (failure predictors) | needs the failure-dataset build pipeline; not wired into the verifier |
| Phase C, Phase D | Their analyses rebuild predictor datasets and use a different raw layout (the committed Phase D metadata predates the analysis split); not wired in. Phase C and D raw files are archived and hash-verified. |
| F1 model training and selection; F1 exploratory ablations | Training is out of scope (no model may be trained). The ablation analysis lives inside its script. |
| G1 and G2 calibration | The calibration choices are recorded, but the verifier did not recompute them from the calibration raw files. |
| **Data generation itself** | No episode was re-simulated, so it is unverified that the raw files would be regenerated bit for bit from the checkpoints. Earlier runs checked within-machine determinism (replay checks, bitwise runner equivalence on debugging seeds), but not across machines. |

## 5. Statistical audit (independent inspection of the code)

Reproducing a script's output does not validate its method, so the analysis code was read
separately. The modules read were:
- `farc.adaptation.delay_eval` and `farc.adaptation.experiment`;
- `farc.contact.evaluation` and `farc.contact.heldout`;
- `farc.supervision.evaluation`;
- `farc.delaychange.stage3_analysis`, `mechanism_analysis` and `heldout_analysis`.

| item | finding |
|---|---|
| **Pairing keys** | (policy seed, condition, episode seed) in every phase. G4's condition names are unique across its two arms, which a test checks. The new completeness check confirmed one row per configuration per pair, with identical pair sets across configurations within each condition, in all 10 raw datasets. |
| **Clustering** | Episode seed is the resampling unit. Each episode seed fixes the initial state in every condition, and from G3 on also the delay schedule, so it is the right cluster. Policies and conditions are not resampled. |
| **Bootstrap** | Every phase uses the same construction: per-cluster sums and counts, cluster indices drawn with replacement (10,000 resamples; 2,000 in the E pilot), and the ratio of resampled sums. This is the standard cluster bootstrap for a mean of paired differences. |
| **CIs** | Percentile 95% intervals. With 100–1,000 clusters this is adequate; the E pilot's 30 clusters are on the small side for percentile intervals, and that phase is labelled a pilot. |
| **p-values** | One-sided: the share of draws ≤ bound. Two-sided: `2 · min(P(draw ≤ 0), P(draw ≥ 0))`. These are percentile-inversion p-values, consistent with the CIs, not tests under a null-centred resampling distribution. "0.0" means < 10⁻⁴. |
| **Fixed-sequence gatekeeping** | E/F1 held-out: H1 → H2 → H3, each "supported" only if its CI low exceeds the bound *and* the gate is still open. G3: G must pass before M1–M4 are confirmatory. G4: Step 2 (P3) only if Step 1 passes. All correct. |
| **Intersection-union** | G2 (H1 ∧ H2 per candidate) and G4 Step 1 (P1 ∧ P2) use p = max(p₁, p₂). This is the valid IUT p-value. |
| **Holm** | Step-down with monotone adjustment (`delay_eval.holm`), correctly implemented. It is used across G1's four hypotheses, G2's candidates, G2's 12 secondary tests, G3's M1–M4 and its 6 secondary tests, and G4's S1–S7. |
| **Non-inferiority** | Implemented as P(draw ≤ −margin) at one-sided α = 0.025. The margins in code match the plans: −1 point vs contact-aware (G2 H2, G4 P2), −2 vs frozen SAC (G2 S4, G4 S1/S2), −2 for robustness (G2 S5, G4 S7). |
| **Practical thresholds** | G2 and G4: +2 points, applied to the *point estimate* as pre-registered. In G4 the CI's lower end (+2.4) also exceeds it. G3 and G4 specificity minimum: +1 point, used in the labels exactly as specified. |

**Methodological concerns (none changes a reported decision):**

1. **Fixed policy effects.** Only 3 SAC policies, not resampled. Inference is conditional on
   them (see audit S1).
2. **Percentile-inversion p-values.** These are approximate; a null-centred or studentised
   bootstrap would be more conventional. The decisive results have z ≈ 4–20.
3. **Monte Carlo error.** A borderline p-value has Monte Carlo error of about ±0.0016 at 10,000
   resamples. The only recorded decisions near 0.025 are G3's `yoked_timing` (Holm p = 0.050)
   and RS-M1 (0.066), both "not supported".
4. **Silent data-shape assumptions.**
   - Every analysis pivots with `aggfunc="first"`, which would hide duplicates.
   - `farc.contact.heldout._outcomes` and `farc.supervision.evaluation._outcomes` cast pivot
     cells with `astype(bool)`, so a *missing* pair would count as a success.
   - The complete design makes both harmless for every committed result, as the completeness
     check confirmed. But the code would not detect a future incomplete run; the verifier now
     does.
5. **Degenerate gate CIs.** With no discordant pairs the CI is [0, 0], so "CI low > −2" is
   vacuous and the harmed-fraction criterion decides (audit S5).
6. **No cross-phase error control.** Many development hypotheses were run across phases. Only
   the two held-out confirmations are confirmatory (audit S3).
7. **The G3 replay interaction is sensitive to alarm divergence.** This is a design limit, not
   a computational one (audit B3).

## 6. Failures and deviations

**Verification failures:** none in the final run. The final run of the committed command
(fresh empty data directory, `--extract`, all 10 analyses) exited 0 with 170 checks and 0
failures.

**Deviations during the audit, all in the verifier, none in the data or results:**
1. On the first quick run, the verifier rejected the environment. It had kept only the last
   `torch` entry of `uv.lock`, which locks four builds (`2.14.1`, `+cpu`, `+cu126`, `+cu130`).
   The parser now accepts any locked build. A test covers this.
2. The paired-design completeness check was added after the first full run, which passed
   without it. The final run includes it.

**Inputs:**
- No committed result, raw file, checkpoint, model or release asset was modified.
- No simulation episode was run, and no seed was used.
- The original `~/farc-data` was not read by the verification.

## 7. Instructions for an independent researcher

Prerequisites: access to the private repository and its `artifacts-v1` release, `gh`
authenticated, and `uv` installed. All paths below are your own choice.

```bash
git clone https://github.com/m-alan08/failure-aware-robot-control.git
```
```bash
cd failure-aware-robot-control && git checkout --detach artifacts-v1
```
```bash
uv sync --frozen
```
```bash
gh release download artifacts-v1 --dir ../release
```

Get the verifier from this PR's branch, and keep it *outside* the checkout so the tree stays
clean:

```bash
git show origin/claude/reproducibility-audit:scripts/verify_reproduction.py > ../verify_reproduction.py
```
```bash
uv run --frozen python ../verify_reproduction.py --repo . --release-dir ../release --data-dir ../farc-data-restored --extract --report ../reproduction_report.json
```

**What to expect:**
- Exit status 0, and a JSON report listing every check. The report shows paths as
  `<REPO>`, `<RELEASE_DIR>`, `<DATA_DIR>` and `<HOME>`.
- About 3 minutes on an Apple-silicon laptop, plus the download.
- `--data-dir` must be empty, or not exist yet. To re-check an already restored directory,
  omit `--extract`.

**On a different OS or CPU:**
- Expect the artifact checks and point estimates to pass unchanged.
- A table may then come out "numerically equal" rather than "byte-identical", which the report
  accepts.
- If a table shows a MISMATCH beyond 1e-9, compare the installed NumPy and pandas versions with
  `uv.lock` first: bootstrap streams and CSV formatting are version-dependent.

**Once this PR is merged,** the verifier is part of the repository. A later release tag would
then include it and the separate `git show` step goes away.

## 8. Remaining limitations

1. **Not independent.**
   - The same machine, the same person or agent that ran G1–G4, and the same GitHub account.
   - A second machine, ideally another OS or CPU, run by someone else, is needed before
     claiming independent reproduction.
2. **Simulation not re-run.** That the raw files themselves can be regenerated from the
   checkpoints is unverified here.
3. **Private access.**
   - Both the repository and the release are private, so an outsider cannot run any of this
     without being granted access.
   - A public release would need a DOI-backed archive.
   - Before publishing, review the four archived metadata files that contain absolute local
     paths (see `docs/DATA.md`).
4. **Partial coverage.** Phase 1, B, C, D, the F1 training and ablations, and the calibration
   recomputations are not covered (§4).
5. **Version-bound exactness.** Byte-identity holds under the locked environment; elsewhere,
   agreement is statistical.
6. **The verifier is new code.** It has unit tests and a clean end-to-end run, but it has not
   been reviewed by anyone else.
