# Stage H1 readiness (after H0)

**H1 is not approved, and nothing here starts it.** H1 is free-space system identification
with robot motion and no object contact. This document lists what H0 established, what it
could not, and what must exist before H1 can be proposed for approval.

## 1. Verified in H0 (offline, simulation data)

| item | evidence |
|---|---|
| The hardware software path computes exactly what the confirmed simulation controllers compute (F, K, CA, C) | bitwise on 405,000 closed-loop-consistent steps and 10,800 open-loop episodes (`HARDWARE_H0_RESULTS.md` §3–4) |
| Delay injection reproduces the simulator's replay/drop semantics, including delay changes and resets | §6 and tests |
| The observation adapter's ordering, units, frames and operation order match FetchPush's `_get_obs` | tests; round trip ≤ 2 × 10⁻¹⁶ apart from a replay-only Euler artefact |
| Invalid, missing, stale, skewed and repeated observations stop the trial | tests |
| Structured, hash-chained, write-once logs with deterministic replay | tests |
| Compute fits easily on the **local** machine: p99 0.10–0.27 ms against a 10 ms budget | `HARDWARE_H0_RESULTS.md` §7 |

## 2. Not verified (needs hardware or the target computer)

- **T1:** timing on the target lab computer, including middleware, sensor acquisition and OS
  jitter. A real-time-priority test is needed.
- **The plant model:** whether the real arm's Cartesian tracking matches the estimator's
  assumptions. The nominal per-step gain K0 = 0.62 × 0.05 m, the linear lag model and the
  noise floor are all from simulation (Q1).
- **Natural latency and jitter** of the real command path (Q5).
- **Sensing:** noise and temporal correlation of real gripper and object positions. J-3 showed
  the estimator abstains above about 0.75 cm of independent noise on the gripper position.
- **F1 on real data:** whether the contact model predicts real object motion (Q3, needs H2
  logs).
- **The frozen policy on hardware:** whether it is competent at all (gate G-HW, H2).
- **The robot command interface:** action → Cartesian target. Also how the arm's controller
  realises a 5 cm/step maximum displacement within 40 ms, and how it saturates.

## 3. Required hardware interfaces (none implemented)

| interface | requirement |
|---|---|
| Cartesian command | 25 Hz end-effector position (or delta-position) targets in the base frame. Franka: libfranka / franka_ros2 Cartesian impedance or pose control. Fetch: arm Cartesian controller via ROS. The FIFO output maps through `Rᵀ` and is scaled by 0.05 m. |
| Proprioception | grip-site position and linear velocity in the base frame, with a monotonic timestamp at ≥ 25 Hz |
| Object sensing | 3D position and orientation quaternion of the puck at ≥ 25 Hz, timestamped on the same clock: motion capture (preferred) or a calibrated camera with fiducials |
| Gripper | fingers held closed; positions read or fixed |
| Safety layer | workspace box, velocity and force limits and e-stop **below** the policy, in the robot controller (proposal §4.5) |
| Clock | one monotonic time base across robot, sensors and controller; synchronisation method documented |
| Compute | a single process at 25 Hz on the lab computer, CPU only |

## 4. Required calibration (all placeholders in `configs/hardware/h0_template.json`)

1. `world_from_base_rotation` / `_translation`: the base-to-FetchPush-world convention. This
   includes the table-height alignment, so that the table top sits at the policy's
   z = 0.4 m.
2. Camera-to-base extrinsics (or the motion-capture frame), feeding the object position.
3. `object_marker_offset_m`: marker to object centre.
4. `camera_latency_s`, `proprio_latency_s`: measured.
5. `max_observation_age_s`, `max_sensor_skew_s`: set from the measured latencies **before**
   H1 trials, not after.
6. `workspace_min_m` / `workspace_max_m`.
7. `finger_position_m` if the fingers are not measured.
8. Sensor noise characterisation: per-axis std and lag-1 autocorrelation, on a static object
   and on a robot holding still.

## 5. Safety requirements before any motion

- The host lab's safety approval, an operator trained on the specific robot, and an e-stop
  within reach. No unattended operation.
- Robot-side workspace box, velocity, acceleration and force limits enforced independently of
  this software. They must be tested at injected delay d = 4 with no object, at low speed,
  before any closed-loop policy run.
- Software stops: the H0 safety-stop rule (any invalid observation aborts), plus a watchdog
  for a missed 40 ms cycle. **The watchdog is not yet implemented** (blocker B4).
- A written risk assessment for overshoot under injected delay.

## 6. Lab access requirements

- Access to a Fetch (preferred; exact simulator model) or a Franka FR3/Panda, through a host lab
  or collaborator. **No purchase is proposed.**
- A table area matching FetchPush's workspace, and a puck approximating the 5 cm block.
- Motion capture or a calibrated camera.
- Lab time: about 1 day for H1 (free-space system identification, about 100 episodes), plus
  setup and calibration time.
- Permission to run the controller software on the lab's computer, or an approved
  workstation.

## 7. Blockers to H1

| # | blocker | owner |
|---|---|---|
| B1 | **No hardware access is arranged** | user / host lab |
| B2 | **No safety approval** | host lab |
| B3 | Platform not chosen (Fetch vs Franka), so the robot-specific command and state interfaces cannot be written | user |
| B4 | Robot-side driver, watchdog and middleware bridge not implemented. They need the chosen platform's SDK, and testing them requires the robot. | after B3 |
| B5 | Calibration values unknown (all placeholders) | H1 setup |
| B6 | The H1 pre-registration (estimator accuracy per d, natural latency, gain, noise) is not yet written | next proposal step |

## 8. Is H1 ready for approval?

**No.** The offline software is ready, and its equivalence to the confirmed controllers is
established. H1 cannot be approved until hardware access (B1), safety approval (B2) and a
platform choice (B3) exist. The H1 pre-registration (B6) can be drafted without hardware, and
only on your instruction.
