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
--keyframe       initial keyframe name  (default: home)
--xml            MJCF scene file
```

---

### `dora-openarm-ik` — Differential IK

Solves joint angles from EE pose targets using mink's QP-based differential IK. Both arms share one `mink.Configuration` and are solved in a single QP per step. DOFs not driven by IK outputs are frozen (finger joints, lifter).

| | |
|---|---|
| **Inputs** | `target_right`, `target_left` `[{"pose": float32[8]}]` — EE pose plus gripper; `state_right`, `state_left` — normalized `qpos[8]` / `qvel[8]` state; `command_right`, `command_left` — driver-accepted `qpos[8]` commands; `syncstate` `bool[1]` — reference synchronization interval; optional `sync_mode` `string[1]` — `state` (default), `command`, or `reference`; optional `reset`, `active` `bool[1]` |
| **Outputs** | `position_right`, `position_left` `[{"qpos": float32[8]}]` |

`--target-mode absolute` preserves direct target handling. In `relative` mode,
targets received during `syncstate=true` are captured as source references when
synchronization is released, then their translation and rotation deltas are
applied to end-effector poses computed from the IK configuration. `state`
initializes that configuration from fresh measured joints, `command` uses the
latest complete driver-accepted command pair, and `reference` preserves the
existing configuration. A missing command pair falls back atomically to
`state`. Omitting `sync_mode` selects `state`.
The next complete active-arm target pair starts solving. If `active` is not
connected, output is enabled for compatibility with existing dataflows. Source
and absolute poses use the configured IK origin frame (normally `arm_origin`).
See [IK target modes runtime proposal](docs/ik-target-modes-proposal.md) for the
complete state machine, coordinate convention, and integration sequence.

```
--mode           right | left | bimanual  (default: bimanual)
--max-iters      IK iterations per event  (default: 5)
--dt             integration timestep per iteration  (default: 0.5)
--damping        global Tikhonov regularization  (default: 1e-3)
--lm-damping     per-task LM damping  (default: 1e-4)
--posture-cost   posture task weight, 0 = disabled  (default: 0.0)
--pos-cost       position task cost  (default: 1.0)
--ori-cost       orientation task cost  (default: 1.0)
--target-mode    absolute | relative  (default: absolute)
--measured-state-timeout
                 measured-state lifetime  (default: 0.1 s)
--relative-input-timeout
                 source-pose lifetime when capturing references  (default: 0.1 s)
--solver         QP backend  (default: daqp)
--frame-right    site/body name for right EE  (default: right_ee_control_point)
--frame-left     site/body name for left EE   (default: left_ee_control_point)
--keyframe       initial keyframe  (default: home)
--xml            MJCF scene file
```

---

### `dora-openarm-teleop` — Relative Teleoperation State

Owns the synchronization trigger state without forwarding pose, joint, or
action data. With no `command` input it starts in standalone teleoperation
mode. When connected to an evaluation UI, existing `start`, `intervene`,
`stop`, and `quit` commands disable, arm, or reset teleoperation.

| | |
|---|---|
| **Inputs** | `grip_left`, `grip_right` `float32[1]`; optional `force_state_sync` `bool[1]`; optional `command` `string[1]` |
| **Outputs** | `active`, `syncstate`, `reset` `bool[1]`; `sync_mode`, `status` `string[1]` |

The first synchronization after `intervene` uses `command`. Later clutches use
`reference`; pressing `force_state_sync` during a clutch upgrades that
synchronization to `state` until the clutch is released. Episode start and
terminal commands emit a one-shot `reset` event; `intervene` preserves the
latest command cache.

```
--sync-trigger            left | right | both  (default: left)
--grip-engage-threshold   press threshold  (default: 0.7)
--grip-release-threshold  release threshold  (default: 0.5)
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
