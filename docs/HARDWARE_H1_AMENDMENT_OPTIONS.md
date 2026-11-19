# H1 pre-registration: amendment options for P1–P4 (draft, for decision)

**Status:**
- **Decision (2026-11-18): the recommended amendments H1-A1–A4 were adopted.** They are recorded
  in `docs/HARDWARE_H1_PREREGISTRATION.md` §10 and implemented in `h1_analysis`.
- The text below is the options analysis as presented for the decision.
- H1 execution remains unapproved, and no data of any kind exist yet.
- Whichever option you choose becomes a numbered amendment, committed before any hardware
  trial. Because no H1 data exist, the change cannot be informed by outcomes.

**Evidence:**
- SYNTHETIC only: `scripts/h1_p1_amendment_check.py` → `results/hardware_h1_amendments/p1_check.json`.
- This is a deterministic synthetic kinematic plant (fixture seed 20264200), not the simulator
  and not a measurement.
- In it, the gripper velocity is the zero-order-hold command delayed by a transport lag L, and
  positions reach the controller S ms after they are measured.
- The **unchanged** Phase E estimator runs on what the controller would receive.
- The grid covers L = 5–145 ms in 10 ms steps × S ∈ {0, 15, 30} ms, with 20 episodes per point.

---

## P1 (with a related finding, P5): the timestamp latency does not measure what the estimator sees

### What is wrong

| | finding | effect |
|---|---|---|
| **P1** | §4.2 resamples zero-order-hold commands by **linear interpolation**. Each command is placed at its tick, not over its 40 ms hold, so L_ep reads about **+20 ms** high. | d₀_ts shifts by one step for about half of all latencies |
| **P5** (found while drafting) | §4.2 stamps velocity samples at their **measurement** time and says sensor latency "is not subtracted… the estimator sees it too". But a measurement-stamped L_ep *excludes* sensor latency S, while the estimator, which receives each position S ms late, *includes* it. | d₀_ts and the estimator's d₀ refer to different delays, L vs L + S |

**In the synthetic plant, the estimator's discrete lag equals round((L + S) / 40 ms)** at every
point where it is reliable (37 / 37). That is the quantity d₀_ts must reproduce for criterion
5.1 (d) to be a meaningful check.

### Options compared (SYNTHETIC)

| option | definition of L_ep | d₀_ts = estimator d₀ (at the 37 reliable points) | estimator-unreliable points flagged ambiguous (8) | reliable points flagged ambiguous |
|---|---|---|---|---|
| as written | linear interpolation, measurement stamps | 27 / 37 (agreement partly accidental: S offsets part of the +20 ms) | 5 / 8 | 16 |
| **A** | zero-order-hold resampling of commands, measurement stamps | 23 / 37 | 3 / 8 | 21 |
| **B** | commands stamped at the hold-interval centre, linear | 23 / 37 (identical to A) | 3 / 8 | 21 |
| **C** | as written, then −20 ms | 23 / 37 (identical to A) | 3 / 8 | 21 |
| **A + S (recommended)** | zero-order hold; each velocity sample stamped at the time the **controller receives it** | **37 / 37** | **8 / 8** | 11 |

**Reading:**
- **Fixing the interpolation alone does not help.** A, B and C all remove the +20 ms but still
  ignore S, and do worse than the uncorrected method, whose error happened to cancel against S
  at some points.
- **Only A + S measures the delay the estimator experiences.** It agrees at every reliable
  point, and its half-step band flags every point where the estimator itself was unreliable.

### Recommended amendment text (A + S)

> **Amendment H1-A1 (§4.2).** Step 1 is replaced by:
>
> "Let c(t) be the issued command as a zero-order hold: c_k holds from its driver-send time t_k
> until t_{k+1}, sampled on a common 1 ms grid. Let v(t) be the measured gripper velocity, with
> each sample stamped at the monotonic time the controller **receives** it (the logged receipt
> time, which includes sensor and transport latency), linearly interpolated onto the same grid."
>
> Step 3 is replaced by: "Because samples are stamped at receipt, L_ep includes sensor
> latency, as the estimator does. The sensor's own latency (`proprio_latency_s`) is reported
> separately for information."

**What it needs from software and hardware:**
- the logged receipt time per proprioception sample, a field to add to the driver (B4);
- the analysis change in `episode_latency`, made only after the amendment is approved;
- the P1 test updated from "pins the +20 ms" to "recovers L + S".

**Effect on other sections:** none on §5.3 (additivity, a difference of medians), the
thresholds or the criteria.

### Related observation (no change proposed): the ±8 ms half-step band is wide

- Under A + S the band flags all 8 estimator-unreliable points, as intended.
- It also flags **11 of 37 reliable points.** The band covers 16 of every 40 ms (40%), so
  roughly a third of plausible natural latencies would leave Q5 "unresolved" even when the
  estimator is consistent.
- This is conservative, it was approved, and it is **not proposed for change.** It is listed
  so the consequence is known: Q5 may stop H1 for latencies near a step boundary.
- If you want it revisited, that would be a separate amendment, decided now, before any data.

---

## P2: "pooled over x and y" (§4.2)

| option | definition |
|---|---|
| **P2-a (implemented, recommended)** | the lag maximising the mean of the two per-axis normalised cross-correlations |
| P2-b | the lag maximising one correlation computed on both axes stacked |
| P2-c | the mean of the two per-axis lags, rounded to 1 ms |

**Recommendation: P2-a.** It weights axes equally whatever their excitation scale, and reduces to
the per-axis answer when they agree.

> **Amendment H1-A2 (§4.2).** "Pooled L_ep is the lag in [0, 300] ms that maximises the mean of
> the x and y normalised cross-correlations."

## P3: aborted episodes in the denominator (§5.2)

| option | treatment |
|---|---|
| **P3-a (implemented, recommended)** | kept in the 20-episode denominator, counted as **not correct** and **not** a wrong latch. Consistent with "safety stop = failure" (proposal §4.5) and with the no-selective-re-run rule. |
| P3-b | excluded from the denominator (the criterion becomes ≥ 18 of the valid episodes) |

**Recommendation: P3-a.** P3-b would let safety stops raise the apparent success rate, which
conflicts with treating a stop as a failure.

> **Amendment H1-A3 (§5.2).** "Aborted episodes and episodes with an invalid log remain in the
> 20-episode denominator, count as not correct, and do not count as wrong latches. They are
> reported separately in the safety accounting."

## P4: which sequences each delay uses (§3.1, §3.3, §5.3)

**The problem:**
- §3.1 selects 60 admissible sequences.
- §3.3 runs Q5 on sequences 1–20 and Q1 as 5 delays × 20 sequences in 3 blocks, but does not say
  *which* 20 sequences each delay uses.
- The §5.3 paired bootstrap pairs Q5 and Q1 episodes by sequence id.

| option | assignment |
|---|---|
| **P4-a (recommended)** | **every** Q1 delay uses sequences 1–20, the same as Q5, so each sequence is paired across Q5 and all five delays. The 100 Q1 episodes (5 delays × 20 sequences) are split into 3 blocks of 34 / 33 / 33 in (sequence, delay) order randomised by seed 20264100. Sequences 21–60 are reserves, used only where §3.1's fallback applies. |
| P4-b | each delay uses its own 20 sequences. Pairing by id is then impossible, and the bootstrap would have to be unpaired. That would also be a methods change. |

**Recommendation: P4-a.** It matches the paired design in §5.3, keeps excitation identical
across delays, so delay is the only difference, and needs no change to the statistics.

> **Amendment H1-A4 (§3.3).** "Q5 and every Q1 delay use excitation sequences 1–20. The 100 Q1
> (sequence, delay) episodes are randomised by seed 20264100 and split in that order into
> three blocks of 34, 33 and 33. Sequences 21–60 are reserves for the §3.1 fallback only."

---

## Summary of recommendations

| | recommendation | changes thresholds or endpoints? |
|---|---|---|
| P1 + P5 | **A + S**: zero-order-hold command resampling, and velocity stamped at controller receipt | no (the measurement definition only) |
| P2 | P2-a (as implemented) | no |
| P3 | P3-a (as implemented) | no |
| P4 | P4-a: sequences 1–20 for Q5 and every delay | no |

**If you approve, next:**
- record H1-A1–A4 as numbered amendments in the pre-registration;
- update `episode_latency` and its tests for A1;
- add the receipt-time field to the driver requirements (B4).

All of this is software and documentation only. H1 execution stays unapproved.
