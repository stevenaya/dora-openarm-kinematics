# dora-openarm-kinematics

Dora nodes for forward and inverse kinematics on the OpenArm bimanual robot, backed by MuJoCo and [mink](https://github.com/kevinzakka/mink). FK/IK logic lives in the [`openarm_control`](https://github.com/enactic/openarm_control) package, imported as `control`.


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

```
--mode           right | left | bimanual  (default: bimanual)
--frame-right    MuJoCo site/body/geom name for right EE  (default: right_ee_control_point)
--frame-left     MuJoCo site/body/geom name for left EE   (default: left_ee_control_point)
--frame-type-*   body | site | geom  (default: site)
--origin-frame   pose reference frame, or world  (default: arm_origin)
--origin-frame-type
                 body | site | geom  (default: site)
--keyframe       initial keyframe name  (default: home)
--xml            MJCF scene file
```

---

### `dora-openarm-ik` — Differential IK

Solves joint angles from EE pose targets using mink's QP-based differential IK. Both arms share one `mink.Configuration` and are solved in a single QP per step. DOFs not driven by IK outputs are frozen (finger joints, lifter).

| | |
|---|---|
| **Inputs** | `target_right`, `target_left` `[{"pose": float32[8]}]` — EE pose target plus gripper; `state_right`, `state_left` normalized driver state structs with `qpos[8]` and `qvel[8]` (measured qpos drives limits and sync); `syncstate` `bool[1]` — enable measured-configuration synchronization |
| **Outputs** | `position_right`, `position_left` `[{"qpos": float32[8]}]` |

Complete, fresh active-arm states update state-aware limits. While `syncstate`
is true they also synchronize the IK configuration; changing it to false
performs one final fresh-state sync before targets resume. After
`--measured-state-timeout`, limits fall back to command state. Single-arm modes
require and publish only their active side. If `syncstate` is not connected,
target-only dataflows keep their keyframe-based behavior.
Targets are ignored while synchronization is active or waiting for fresh state.
After either `syncstate` transition, every active side must provide a new target
before the next solve, so targets from opposite sides of a clutch cannot mix.

FK outputs and IK targets share `--origin-frame` (default: `arm_origin`).
Endpoint and origin objects may be bodies, sites, or geoms. VR translation
deltas use the `arm_origin` axes, so a rotated origin requires upstream
conversion.

```
--mode           right | left | bimanual  (default: bimanual)
--max-iters      IK iterations per event  (default: 5)
--dt             outer control period  (default: 1 / --tick-hz)
--tick-hz        nominal control rate  (default: 250)
--damping        global Tikhonov regularization  (default: 0.1)
--lm-damping     per-task LM damping  (default: 0.01)
--posture-cost   posture task weight, 0 = disabled  (default: 0.0)
--pos-cost       position task cost  (default: 12.0)
--ori-cost       orientation task cost  (default: 1.5)
--measured-state-timeout
                 measured-state lifetime for safety limits  (default: 0.1 s)
--solver         QP backend  (default: daqp)
--frame-right    site/body/geom name for right EE  (default: right_ee_control_point)
--frame-left     site/body/geom name for left EE   (default: left_ee_control_point)
--frame-type-*   body | site | geom  (default: site)
--origin-frame   pose reference frame, or world  (default: arm_origin)
--origin-frame-type
                 body | site | geom  (default: site)
--keyframe       initial keyframe  (default: home)
--xml            MJCF scene file
```

## Quick Start

### FK — visualise leader arm poses

Reads joint angles from a physical leader arm and publishes end-effector poses. Requires a connected leader device.

```bash
uv run dora build example/dataflow-dummy-fk.yaml --uv
uv run dora run example/dataflow-dummy-fk.yaml --uv
```

---

### FK → IK roundtrip

Pipes FK output directly back into IK to verify the solver round-trips correctly. No physical hardware needed beyond the leader.

```bash
uv run dora build example/dataflow-dummy-ik.yaml --uv
uv run dora run example/dataflow-dummy-ik.yaml --uv
```

**Dataflow:** `leader` → `fk` (joints → poses) → `ik` (poses → joints) → `viewer`

Tune the ik solver parameters in:

```yaml
args: "--mode bimanual --max-iters 5 --dt 0.1 --damping 0.25 --posture-cost 0.01 --lm-damping 0.01"
```

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
