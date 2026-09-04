# Relative Teleoperation and Intervention Architecture

## 1. Scope

This document describes the current end-to-end relative teleoperation design,
including how an evaluation rollout changes into operator intervention. It
covers node ownership, Dora inputs and outputs, signal routing, synchronization
states, action arbitration, arm-session metadata, and recorder integration.

No single node owns the whole workflow:

- the evaluation UI owns the episode phase and requests intervention;
- `dora-openarm-teleop` owns clutch and synchronization state;
- `dora-openarm-ik` owns robot configuration, relative references, and IK
  validation;
- the action mux owns model-versus-VR actuator arbitration;
- `dora-openarm` owns hardware execution and the last accepted command;
- the recorder owns dataset persistence.

The teleop node is a control-plane state machine. It never receives or
forwards poses, joint states, or actions.

## 2. Runtime Graph

The control plane is:

```text
evaluation UI -- teleop_enable --> teleop -- active --------> action mux
                                      |  \-- syncstate -----> IK
                                      |  \-- sync_mode -----> IK
                                      |  \-- reset ----------> IK
                                      |
                                      +<-- IK status ---------+

teleop -- status --> evaluation UI
evaluation UI -- arm_command --> action mux, model path, and arm nodes
```

The pose and action plane is:

```text
                                  model_right/left
observer -> policy -> executor ---------+
                                         v
VR poses -> relative IK -> vr_right/left -> action mux -> arm-right/left
                ^                                         |
                |                                         |
                +-------- state + latest_command ----------+
```

Recording is a side branch, not part of actuator arbitration:

```text
evaluation UI/recorder_command ---------------------------> recorder
arm/state ------------------------------------------------> recorder observations
arm/latest_command ---------------------------------------> recorder actions
cameras --------------------------------------------------> recorder
policy/actions -------------------------------------------> recorder policy chunks
```

The recorder uses `arm/latest_command`, rather than raw model or IK output, so
the stored arm action is the final command accepted by the driver after its
checks and clamping.

## 3. Node Responsibilities and Contracts

### 3.1 Evaluation UI

The UI owns high-level phases such as rollout, intervention, and stopped. It
does not decide how a clutch synchronizes IK.

| Direction | Endpoint | Meaning |
|---|---|---|
| Input | `arm_status_right`, `arm_status_left` | Arm lifecycle status |
| Input | `teleop_status` | Operator-visible teleop state |
| Input | `recorder_result` | Recorder command acknowledgement |
| Input | `button_a`, `button_b`, `button_x` | Intervention success, failure, and cancel controls |
| Output | `arm_command` | Broadcast `start`, `intervene`, `stop`, or `quit` |
| Output | `teleop_enable` | Whether intervention teleoperation is allowed |
| Output | `recorder_command` | Episode persistence command |
| Output | `task_prompt` | Language prompt and episode identity |

During model rollout, `teleop_enable=false`. On intervention, the UI
broadcasts `intervene` and sends `teleop_enable=true`. Leaving intervention
sends `teleop_enable=false` before stopping or quitting the arm path.

`arm_command` is a broadcast protocol. Each receiving node consumes only the
commands relevant to it. In particular, `intervene` changes the mux and model
pipeline state while the already-started arms remain enabled.

### 3.2 VR Receiver

The VR receiver samples the latest controller packet on the arm tick and
publishes independent pose, grip, trigger, joystick, and button events.

| Output | Schema | Consumer in this architecture |
|---|---|---|
| `pose_right`, `pose_left` | `struct<pose: float32[8]>` | IK target/source poses |
| `grip_right`, `grip_left` | `float32[1]` | Teleop synchronization trigger |
| `button_y` | `bool[1]` | Force measured-state synchronization |
| `button_x` | `bool[1]` | UI cancel |
| `button_a`, `button_b` | `bool[1]` | UI success/failure during intervention |

Pose layout is:

```text
[px, py, pz, qw, qx, qy, qz, gripper]
```

The trigger controls the gripper value in the pose. The grip is the clutch.
The evaluation dataflow currently uses `--sync-trigger left`, but the teleop
node also supports `right` and `both`.

### 3.3 Teleop State Node

`dora-openarm-teleop` receives only state-machine inputs:

| Input | Schema | Required | Meaning |
|---|---|---|---|
| `grip_left` | floating point `[1]` | For `left` or `both` | Left analog clutch input |
| `grip_right` | floating point `[1]` | For `right` or `both` | Right analog clutch input |
| `enable` | `bool[1]` | No | External teleop gate; absent means enabled |
| `force_state_sync` | `bool[1]` | No | Upgrade the current clutch to `state` mode |
| `ik_status` | `string[1]` | No | IK feedback requesting recalibration |

It emits:

| Output | Schema | Meaning |
|---|---|---|
| `active` | `bool[1]` | IK output may control the arm |
| `syncstate` | `bool[1]` | A clutch synchronization interval is open |
| `sync_mode` | `string[1]` | `command`, `state`, or `reference` |
| `reset` | `bool[1]` | One-shot IK runtime reset |
| `status` | `string[1]` | `inactive`, `require_sync`, `syncing`, or `tracking` |

The optional `enable` input has edge semantics:

- the first received value resets IK and becomes authoritative;
- a later `false -> true` or `true -> false` edge resets IK;
- repeated equal values do nothing;
- with no connection, standalone data collection starts enabled.

After teleop initialization, an enable edge, or recalibration, the configured
clutch must be observed released before a new press is accepted. This prevents
a stale held button from opening synchronization accidentally.

| Current status | Event | Next status | Gate result |
|---|---|---|---|
| `inactive` | `enable=true` | `require_sync` | `active=false`, `syncstate=false`, reset IK |
| `require_sync` | Trigger first observed released | `require_sync` | Gates remain closed |
| `require_sync` | Trigger pressed after release | `syncing` | `active=false`, `syncstate=true` |
| `syncing` | Trigger released | `tracking` | `syncstate=false`, then `active=true` |
| `tracking` | Trigger pressed | `syncing` | `active=false`, `syncstate=true` |
| Any enabled state | IK requests recalibration | `require_sync` | Both gates closed |
| Any state | `enable=false` | `inactive` | Both gates closed, reset IK |

Gate output ordering is intentional:

- entering synchronization sends `active=false` before `syncstate=true`;
- leaving synchronization sends `syncstate=false` before `active=true`.

This keeps the mux in HOLD while IK captures or commits references.

### 3.4 Relative IK Node

`dora-openarm-ik --target-mode relative` owns all pose and robot-state data.

| Input | Schema | Meaning |
|---|---|---|
| `target_right`, `target_left` | `struct<pose: float32[8]>` | Current VR/source poses |
| `state_right`, `state_left` | normalized state containing `qpos[8]`, `qvel[8]` | Measured arm state |
| `command_right`, `command_left` | `struct<qpos: float32[8]>` | Latest driver-accepted command |
| `active` | `bool[1]` | Output gate; absent means active |
| `syncstate` | `bool[1]` | Synchronization interval gate |
| `sync_mode` | `string[1]` | Configuration source for this synchronization |
| `reset` | `bool[1]` | Clear all runtime configuration and references |

| Output | Schema | Meaning |
|---|---|---|
| `position_right`, `position_left` | `struct<qpos: float32[8]>` | Solved arm command |
| `status` | `string[1]` | `ready` or `relative_recalibration_required` |

IK solves only when all of the following are true:

```text
active
and not syncstate
and no synchronization is pending
and configuration is initialized
and relative calibration is valid
and one new complete target set is available
```

In bimanual mode, a complete target set contains one new right pose and one new
left pose. Cached targets are cleared across gate and calibration boundaries,
so samples from different phases cannot be paired.

### 3.5 Action Mux

The mux is the only node that selects which command source reaches the arms.
It validates canonical `struct<qpos>` values and forwards selected event
metadata unchanged.

| Mux event | Resulting mode |
|---|---|
| Process start | `HOLD` |
| `arm_command=start` | `MODEL` |
| `arm_command=intervene` | Intervention armed, mode `HOLD` |
| `teleop/active=true` while intervention is armed | `VR` |
| `teleop/active=false` while intervention is armed | `HOLD` |
| `arm_command=stop` or `quit` | `HOLD` |

`active` cannot switch the mux to VR before `intervene` has armed intervention.
Model actions are ignored in HOLD and VR modes; VR actions are ignored in HOLD
and MODEL modes.

### 3.6 Arm Nodes

Each `dora-openarm` node:

- publishes normalized measured `state`;
- publishes `latest_command` only for a command actually accepted by the
  driver;
- increments its process-local `start_epoch` after each successful `start()`;
- stamps arm outputs with that epoch;
- accepts a legacy command without an epoch, but rejects a supplied malformed
  or non-current epoch.

The right and left nodes maintain counters independently. The current bimanual
dataflow starts them together, and IK requires their state/command streams to
belong to one shared epoch.

## 4. Synchronization Modes and Relative Mapping

The teleop state machine chooses one mode for each clutch interval.

| Mode | IK configuration source | Use |
|---|---|---|
| `command` | Fresh right and left `latest_command` pair | First clutch after enable or recalibration |
| `state` | Fresh right and left measured-state pair | Explicit recovery or force-sync with Y |
| `reference` | Existing internal IK configuration | Normal re-clutch without pulling configuration to measured state |

### 4.1 Command Validation and Fallback

A command seed is accepted only when:

- both arm commands have arrived recently;
- both measured states have arrived recently;
- all values are finite;
- every arm-joint command/state difference is at most `0.2 rad`;
- the `start_epoch` protocol, once observed, is consistent.

The gripper is excluded from the `0.2 rad` comparison. If command validation
fails, the whole bimanual seed falls back to measured state. The implementation
never mixes command on one side with state on the other.

### 4.2 Reference Capture

While the clutch is held, IK does not solve. It caches fresh source poses and
prepares the selected configuration source. On clutch release it uses one
configuration snapshot for both final synchronization and FK, then atomically
anchors:

```text
VR pose at release <-> robot FK pose from the selected configuration
```

The source poses used for anchoring are not reused as movement commands. IK
waits for the next complete post-release target pair before solving.

### 4.3 Coordinates and Mapping

The VR source poses and robot FK poses are expressed in the configured IK
origin frame, normally:

```text
--origin-frame arm_origin
--origin-frame-type site
```

At clutch release, save source pose `(p_S0, q_S0)` and end-effector FK pose
`(p_E0, q_E0)`. For a later source pose `(p_S, q_S)`, calculate:

$$
\Delta p = p_S - p_{S0},
\qquad
q_\Delta = q_{S0}^{-1} \otimes q_S.
$$

Apply that motion to the saved end-effector pose:

$$
p_E^* = p_{E0} + \Delta p,
\qquad
q_E^* = q_{E0} \otimes q_\Delta.
$$

Translation follows the fixed `arm_origin` axes. The latest gripper value
passes through directly. Quaternions use `wxyz`, are normalized, and `q` and
`-q` are treated as the same rotation.

## 5. Intervention Lifecycle

### 5.1 Model Rollout

```text
UI: phase=rollout, teleop_enable=false
mux: MODEL
teleop: inactive, active=false
IK: reset and blocked from the actuator path
arm: executes model commands selected by mux
recorder: stores arm/latest_command as executed action
```

VR poses may continue to arrive, but they cannot reach the arm because teleop
is inactive and the mux is in MODEL mode.

### 5.2 Enter Intervention

The UI performs the high-level transition:

```text
1. finish the rollout recorder segment as failed
2. broadcast arm_command=intervene
3. send teleop_enable=true
4. open a linked intervention recorder segment
5. display teleop status=require_sync
```

The mux handles `intervene` immediately by leaving MODEL and entering HOLD.
The enable rising edge resets IK and puts teleop in `require_sync`. The UI does
not send `syncstate` or choose a sync mode.

### 5.3 First Clutch

Assuming the configured trigger is the left grip:

```text
left grip released at least once
left grip crosses engage threshold
  -> teleop emits sync_mode=command
  -> teleop emits active=false
  -> teleop emits syncstate=true
  -> mux remains HOLD
  -> IK follows a valid command seed, or atomically falls back to state
```

When the grip crosses the release threshold:

```text
teleop emits syncstate=false
  -> IK validates fresh VR poses and the selected configuration
  -> IK computes FK and commits relative references
teleop emits active=true
  -> mux enters VR
teleop status becomes tracking
```

IK still waits for a new right/left VR pose pair before sending the first arm
command.

### 5.4 Tracking

```text
new right pose ----+
                   +-> relative mapping -> one bimanual IK solve
new left pose -----+                         |
                                             v
                                      position_right/left
                                             |
                                             v
                                      mux VR inputs
                                             |
                                             v
                                           arms
```

Measured state remains available to state-aware IK safety limits. It is not
used to overwrite the internal configuration on every tracking step.

### 5.5 Normal Re-clutch

A later grip press selects `reference` mode:

```text
active=false -> mux HOLD
syncstate=true, sync_mode=reference
```

IK preserves its internal configuration and creates a new controller-to-robot
anchor on release. This allows the operator to reposition the controller
without snapping IK back to measured joints.

### 5.6 Forced State Synchronization

Pressing Y before or during a clutch changes that clutch to `state` mode. IK
then synchronizes its configuration from fresh measured arm state before
capturing the new relative reference.

This is an explicit recovery operation. It is not required for ordinary
re-clutching.

### 5.7 Leave Intervention

Success, failure, cancel, stop, or quit disables teleoperation. On the enable
falling edge, teleop sends gates closed and a one-shot IK reset. The mux handles
stop or quit by returning to HOLD.

## 6. Arm Session Generation (`start_epoch`)

`start_epoch` protects the control path from commands and references belonging
to an earlier arm start:

```text
arm state/latest_command metadata
        |
        v
IK caches and compares one shared epoch
        |
        v
IK position metadata
        |
        v
action mux forwards metadata unchanged
        |
        v
arm validates the supplied epoch
```

IK behavior is:

- before any epoch is observed, inputs without one remain compatible;
- the first valid epoch is cached;
- after caching, missing or malformed epochs invalidate relative calibration;
- an older epoch is dropped;
- a newer epoch clears input generations and requires a new clutch cycle;
- solved outputs carry the cached epoch.

On invalidation, IK emits `relative_recalibration_required`. Teleop consumes
that status, closes `active` and `syncstate`, returns to `require_sync`, and
requires a release followed by a new clutch press.

The counter is process-local. A complete arm-node process restart resets it;
the current protocol primarily distinguishes repeated `start()` calls within
one process.

## 7. Recording Identity (`episode_attempt_id`)

`episode_attempt_id` is separate from `start_epoch`. The UI creates a UUID for
each recorder episode attempt and passes it through the observer and policy
path. The recorder uses it to reject delayed policy chunks from another
attempt and stores it in dataset metadata.

Kinematics does not read or modify `episode_attempt_id`. Conversely, the
recorder does not use `start_epoch` as an episode identity.

A rollout and its intervention segment normally have different attempt IDs but
the same arm epoch because intervention does not restart the arms.

## 8. Freshness, Validation, and Safety

The control path uses several independent checks. Freshness prevents an old
sample from being used for synchronization; generation checks prevent samples
from different arm starts from being combined; the IK solver constrains the
motion it generates; and the mux plus arm driver decide whether that command
can reach hardware. No single check replaces the others.

### 8.1 Freshness Semantics

IK freshness is based on local arrival time from `time.monotonic()`. It does
not use Dora metadata `timestamp` or `executed_timestamp` to decide whether an
input is fresh. This avoids comparing clocks from different processes and
machines.

| Input | Freshness requirement | Where it is enforced | Expired behavior |
|---|---|---|---|
| Right and left measured state | Both received within `--measured-state-timeout` | `state` synchronization and command-seed validation | Synchronization waits or fails; command mode falls back to the complete state pair only if that pair is fresh |
| Right and left `latest_command` | Both received within `--measured-state-timeout` | `command` synchronization | Reject the complete command pair and attempt state synchronization |
| Right and left source pose | Both received after the current clutch began and within `--relative-input-timeout` | Relative-reference commit on clutch release | Invalidate calibration, emit `relative_recalibration_required`, and produce no IK output |
| `start_epoch` | Exact generation match after the first epoch is cached | Every state and command input | Drop an older event; invalidate calibration on a missing, malformed, or newer epoch |

Both timeout defaults are `0.1 s`, and the evaluation dataflow sets them
explicitly to that value. The checks run when the relevant Dora event is
handled; there is no background timeout task that changes status merely
because wall-clock time has passed.

`latest_command` may describe a HOLD command whose `executed_timestamp` is
older than the timeout. It remains eligible while the arm node is alive and
continues publishing that same accepted command snapshot in response to fresh
state requests. Freshness therefore means current confirmation from the arm
node, not recent physical motion.

During tracking, fresh measured state updates the state-aware joint and
singularity limits. If the pair becomes stale, IK clears those measured-state
references but currently continues target-only solving. Also,
`--relative-input-timeout` is checked when committing a relative reference,
not as a continuous tracking watchdog or as a maximum skew check for every
bimanual target pair.

### 8.2 Input and Synchronization Validation

Before synchronization or solving, the path applies these checks:

- grip, gate, mode, state, command, and pose inputs must have their expected
  Arrow schema;
- numeric state, command, and pose values must be finite;
- pose quaternions must be non-zero and are normalized before use;
- command synchronization requires a complete fresh command pair and a
  complete fresh measured-state pair;
- command and measured arm joints must differ by at most `0.2 rad` on each of
  the seven arm joints; the gripper is excluded;
- a failed command check falls back atomically to both measured arms, never to
  a command/state mixture;
- source poses used at clutch release must belong to that clutch interval;
- after a reset or recalibration request, the clutch must first be observed
  released, and its `0.7` engage / `0.5` release hysteresis rejects threshold
  chatter;
- bimanual IK waits for one new target from each side before each solve and
  clears pending samples across synchronization and calibration boundaries.

Malformed or non-finite events are skipped. A relative-reference error also
invalidates calibration, closes the teleop gates through status feedback, and
requires a new clutch cycle.

### 8.3 Motion and Actuator Safety Layers

The current evaluation invocation enables the following solver-side layers:

- bounded frame tasks limit the requested pose-error correction per outer IK
  solve (defaults: `0.02 m` translation and `0.25 rad` orientation);
- `--limit-velocity` enables per-joint position and velocity constraints plus
  preventive braking near joint limits;
- singularity approach limiting slows or stops motion toward poorly
  conditioned configurations;
- nullspace posture regularization, kinetic-energy regularization, task
  damping, and global QP damping bias redundant solutions away from abrupt or
  unstable motion;
- if the constrained QP has no solution, the solver restores its pre-solve
  configuration and emits no command for that target pair.

Outside the solver:

- teleop keeps `active=false` throughout synchronization;
- the mux stays in HOLD unless intervention is armed and teleop is active;
- the mux validates canonical `struct<qpos>` actions and preserves metadata;
- the arm node ignores commands while stopped and rejects a supplied
  malformed or non-current `start_epoch`;
- the driver may reject or clamp a requested position, and only its accepted,
  final `latest_command` is exposed to IK and the recorder.

These mechanisms provide transition and trajectory safeguards, but they are
not an end-to-end communication deadman. A hardware deployment still relies
on the arm driver's own communication and motor protections.

## 9. Failure and Recovery

| Condition | IK behavior | Teleop/mux behavior | Recovery |
|---|---|---|---|
| Command missing, stale, or more than `0.2 rad` from state | Fall back to the complete measured-state pair | Synchronization continues in HOLD | Release clutch normally if state and poses are valid |
| Source pose missing, stale, or predating clutch at reference commit | Emit `relative_recalibration_required`; no output | Teleop closes gates; mux HOLD | Release and press clutch again |
| `start_epoch` advances or becomes invalid | Clear configuration/reference generations | Teleop reports `require_sync`; mux HOLD | New clutch cycle |
| Only one bimanual target arrives | Do not solve | Existing mode unchanged | Wait for the other side |
| `enable=false` | Runtime reset | Teleop inactive; mux cannot enter VR | New enable edge and clutch cycle |

During tracking, stale measured state currently removes the measured-state
safety reference but does not by itself stop target-only solving. This is
existing compatibility behavior.

## 10. Core Evaluation Wiring

The relevant part of the current dataflow is:

```yaml
- id: teleop
  path: dora-openarm-teleop
  args: "--sync-trigger left"
  inputs:
    grip_right: vr-receiver/grip_right
    grip_left: vr-receiver/grip_left
    force_state_sync: vr-receiver/button_y
    enable: evaluation-ui/teleop_enable
    ik_status: vr-ik/status
  outputs: [active, syncstate, sync_mode, reset, status]

- id: vr-ik
  path: dora-openarm-ik
  args: >-
    --tick-hz 250 --mode bimanual
    --origin-frame arm_origin --origin-frame-type site
    --target-mode relative --relative-input-timeout 0.1
    --max-iters 5 --limit-velocity --measured-state-timeout 0.1
  inputs:
    state_right: arm-right/state
    state_left: arm-left/state
    command_right: arm-right/latest_command
    command_left: arm-left/latest_command
    active: teleop/active
    syncstate: teleop/syncstate
    sync_mode: teleop/sync_mode
    reset: teleop/reset
    target_right: vr-receiver/pose_right
    target_left: vr-receiver/pose_left
  outputs: [position_right, position_left, status]

- id: action-mux
  path: dora-openarm-action-mux
  inputs:
    command: evaluation-ui/arm_command
    active: teleop/active
    model_right: actions-executor/move_position_right
    model_left: actions-executor/move_position_left
    vr_right: vr-ik/position_right
    vr_left: vr-ik/position_left
  outputs: [move_position_right, move_position_left]
```

The arm nodes connect `action-mux/move_position_right` and
`action-mux/move_position_left` to their respective `move_position` inputs.
Their `state` and `latest_command` outputs return to IK and also branch to the
recorder.

## 11. Standalone Data Collection

The same teleop and IK nodes can be used without inference intervention:

- omit the teleop `enable` input, so teleoperation starts enabled;
- keep the clutch, `ik_status`, state, command, and pose wiring unchanged;
- connect IK output directly to the arms or through a mux fixed to the VR path;
- record `arm/latest_command` as action and `arm/state` as observation.

Episode changes do not have to reset teleop calibration. Stopping the arm or
ending the dataflow terminates control. If a running arm is started again, its
new epoch causes IK to require synchronization automatically.

## 12. Current Limitations

- `start_epoch` is process-local and is not globally unique across arm-node
  process restarts.
- Right and left arm nodes count starts independently; the current bimanual
  design expects them to be started together.
- The VR receiver republishes its latest packet on tick. IK freshness measures
  Dora target arrival, so `relative_input_timeout` alone cannot detect a
  stalled network source while the receiver continues replaying the last
  packet.
- Teleop status is advisory for the UI. Safety does not depend on the UI
  displaying or acknowledging it.
- The mux performs source arbitration, not trajectory blending. Transitioning
  through HOLD and synchronizing relative references provides continuity.
