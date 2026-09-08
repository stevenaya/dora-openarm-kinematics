# dora-openarm-kinematics

Dora nodes for forward and inverse kinematics on the OpenArm bimanual robot, backed by MuJoCo and [mink](https://github.com/kevinzakka/mink). FK/IK logic and CLI defaults come from the [`openarm_control`](https://github.com/enactic/openarm_control) package.


## Install

```bash
uv sync
```

## Dora Nodes


### `dora-openarm-fk` — Forward Kinematics

Converts per-arm joint angles to end-effector poses via `mj_forward`.


| | |
|---|---|
| **Inputs** | `position_right`, `position_left` `[{"qpos": float32[8]}]` — joints 1–7 + gripper (flat `float32[8]` also accepted) |
| **Outputs** | `pose_right`, `pose_left` `[{"pose": float32[8]}]` — `[px, py, pz, qw, qx, qy, qz, gripper_value]` |

See [Model and coordinate frames](#model-and-coordinate-frames) for the shared FK/IK options. The node also emits `status: ["ready"]` at startup.

---

### `dora-openarm-ik` — Differential IK

Solves joint angles from EE pose targets using mink's QP-based differential IK. Both arms share one `mink.Configuration`; each internal iteration solves one QP for both arms. Non-arm DOFs (finger joints, lifter) are frozen within the QP, and gripper commands come from the target payloads.

| | |
|---|---|
| **Inputs** | `target_right`, `target_left` `[{"pose": float32[8]}]` — `[px, py, pz, qw, qx, qy, qz, gripper_value]`; `position` `[{"qpos": float32[16]}]` — optional joint-state sync, right[8] followed by left[8] (flat arrays also accepted) |
| **Outputs** | `position_right`, `position_left` `[{"qpos": float32[8]}]` |

The node emits `status: ["ready"]` at startup. In bimanual mode, a solve waits for a new target from each arm. No `tick` input is needed. Limits use the internal IK configuration; the optional `position` input synchronizes that configuration.

## Configuration

The descriptions below follow `openarm-control` 0.4.0. Options and defaults are supplied by the installed control version; query the complete list instead of copying a fixed set of tuning values into every dataflow:

```bash
uv run dora-openarm-fk --help
uv run dora-openarm-ik --help
```

### Model and coordinate frames

These options are shared by FK and IK. Set them in the dataflow when selecting a different robot model or pose convention.

| Option | Purpose |
|---|---|
| `--mode` | Select `right`, `left`, or `bimanual`. The bundled examples use the default bimanual mode. |
| `--xml` | Path to the MuJoCo MJCF scene used for kinematics and joint limits. |
| `--keyframe` | Initial model configuration; also supplies the home reference for posture regularization. |
| `--frame-right`, `--frame-left` | End-effector frame names. Select their object types with `--frame-type-right` / `--frame-type-left` (`body`, `site`, or `geom`). |
| `--origin-frame`, `--origin-frame-type` | Reference frame for FK outputs and IK targets. `--origin-frame world` uses world coordinates. |

Connected FK and IK nodes must agree on the model, end-effector frames, and pose origin. An upstream pose producer must publish the same coordinate convention: translation in meters and quaternion in `qw, qx, qy, qz` order.

### Control rate and joint limits

Keep the nominal rate and explicit limit selection visible in the dataflow. For the dummy example's 4 ms source timer:

```yaml
args: "--tick-hz 250 --limit-velocity"
```

`--tick-hz` describes the nominal rate of complete target updates. It does not generate ticks, throttle events, or infer the rate from wiring. When changing the source timer or target producer rate, update this value too.

| Option | Purpose |
|---|---|
| `--tick-hz` | Nominal control rate in Hz; sets the outer control period to `1 / tick_hz` when `--dt` is omitted. |
| `--dt` | Explicit outer control period in seconds; overrides `--tick-hz`. This is not the per-iteration integration timestep. |
| `--max-iters` | Internal IK iterations per solve. Each iteration integrates `dt / max_iters`, so increasing this value does not increase the total control period. |
| `--limit-velocity` | Enable recoverable joint position and velocity constraints for the seven arm joints. Without it, joint position limits still apply, but these velocity caps are not enabled. |
| `--config` | YAML file overriding the built-in velocity caps, used only with `--limit-velocity`. Its `arm_velocity_limits` key must contain seven positive finite values in rad/s, ordered joint1 through joint7 and applied to both arms. |

For custom caps, append `--config velocity-limits.yaml` to the IK arguments. This option reads the velocity-limit list; it is not a general IK tuning configuration file. Gripper commands are separate from the seven-joint velocity limits.

### Advanced tuning

Start with control defaults and add individual overrides when needed. Keep rates and limit settings fixed while comparing solver weights.

| Option | Effect |
|---|---|
| `--pos-cost`, `--ori-cost` | Relative weights of end-effector position and orientation tracking. |
| `--damping` | Global Tikhonov regularization; penalizes large joint increments. |
| `--lm-damping` | Error-dependent Levenberg-Marquardt damping for frame tasks. |
| `--posture-cost` | Full-joint posture attraction toward the initial configuration; `0` disables this task. |
| `--nullspace-cost`, `--nullspace-return-rate` | Weight and return gain (1/s) for the 7-DoF arm's structural-nullspace home attraction. `--nullspace-cost 0` disables this task. |
| `--kinetic-energy-cost` | Inertia-weighted joint-motion penalty; `0` disables it. |
| `--frame-position-error-limit`, `--frame-orientation-error-limit` | Position (m) and orientation (rad) error-request budgets per outer solve, divided across iterations. `0` disables the corresponding bound. |
| `--joint-braking`, `--no-joint-braking` | Enable or disable preventive braking near joint position bounds; requires `--limit-velocity`. |
| `--joint-braking-distance` | Joint-space distance from a position bound at which preventive braking starts, in rad for arm joints. |
| `--singularity-max-approach-rate` | Maximum first-order decrease per second of the normalized Jacobian's `sigma_min / sigma_max` ratio, reduced near singularities. `0` disables this constraint. |
| `--solver` | QP backend name; the selected backend must be installed. |

Full posture attraction and nullspace attraction are separate tasks: `--posture-cost 0` does not disable nullspace regularization. Frame error-request limits bound what the task asks the solver to correct, not the final tracking error or physical end-effector speed. In control 0.4.0, position bounding is scheduled from target linear speed and accumulated lag; orientation bounding applies whenever its limit is positive.

## Quick Start

### FK — visualise leader arm poses

Reads joint angles from a dummy leader and publishes end-effector poses. No physical hardware is required. To use a real leader, replace the dummy node as indicated in the example dataflow.

```bash
uv run dora build example/dataflow-dummy-fk.yaml --uv
uv run dora run example/dataflow-dummy-fk.yaml --uv
```

---

### FK → IK roundtrip

Pipes a dummy leader's FK output into IK to inspect end-effector tracking. No physical hardware is required. A redundant arm may reach the same end-effector pose with different joint angles.

```bash
uv run dora build example/dataflow-dummy-ik.yaml --uv
uv run dora run example/dataflow-dummy-ik.yaml --uv
```

**Dataflow:** `leader` → `fk` (joints → poses) → `ik` (poses → joints) → `viewer`

The IK arguments keep the source rate and velocity-limit switch explicit; other settings use control defaults:

```yaml
args: "--tick-hz 250 --limit-velocity"
```

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
