# IK Absolute and Relative Target Modes

## 1. What This Change Does

The IK node accepts two target modes:

- `absolute`: the input pose is used directly as the end-effector target.
- `relative`: clutch release connects the current VR pose to the robot's current FK pose. Later VR motion moves the end effector by the same amount.

This PR changes only `dora-openarm-ik`. It does not change intervention, action mux, VR filtering, rate limiting, or standalone FK. VR lag checking is deferred.

## 2. Coordinates and Inputs

The current MuJoCo model contains `arm_origin`. By default, `openarm_control` expresses both FK and IK poses in that frame:

```text
--origin-frame arm_origin
--origin-frame-type site
```

The VR pose must use the same `arm_origin` axes. No fixed world offset is needed.

| Input | Meaning |
|---|---|
| `target_right/left` | `[px, py, pz, qw, qx, qy, qz, gripper]`; EE target in absolute mode, VR/source pose in relative mode |
| `state_right/left` | Measured `qpos[8]` and `qvel[8]`; qpos is used for synchronization and safety limits |
| `syncstate` | `true` while clutch is held and IK should follow measured joints instead of solving |
| `active` | Optional output gate; defaults to `true` when not connected |

New arguments:

```text
--target-mode absolute|relative   # default: absolute
--measured-state-timeout 0.1
--relative-input-timeout 0.1
```

`--mode right|left|bimanual` determines which arms are required. Single-arm mode never waits for the other arm.

## 3. Common Output Rules

IK solves only when:

$$
\text{active}
\land \neg\text{syncing}
\land \text{reference valid}
\land \text{one new complete target set available}.
$$

For bimanual operation, right and left poses arrive as separate events. A solve uses one new pose from each arm:

```text
right arrives -> wait for left
left arrives  -> solve once
after solve   -> wait for a new right + left pair
```

The pair is cleared whenever `active` or `syncstate` changes. This prevents a pre-clutch pose from one arm being combined with a post-clutch pose from the other.

## 4. Absolute Mode

Without `syncstate` or measured state, behavior is unchanged from mainline:

```text
complete target pair -> IK solve -> joint commands
```

When synchronization is used:

```text
syncstate=true
  - stop solving
  - cache targets
  - repeatedly kin.sync(latest measured qpos)

syncstate: true -> false
  - perform one final kin.sync(latest measured qpos)
  - resume solving
```

If measured state is not fresh at release, absolute mode waits for the next state packet before resuming.

## 5. Relative Mode

Relative mode has three phases:

```text
1. CLUTCH HELD
   cache latest VR poses and measured joints
   keep the IK configuration synchronized
   do not solve

2. CLUTCH RELEASE
   validate fresh VR poses and measured state
   copy one q_ref snapshot
   final sync(q_ref)
   FK(q_ref)
   atomically save VR and FK references

3. TRACKING
   wait for the next complete VR-pose set
   map VR motion from the saved references
   solve IK
```

The release transaction uses one joint snapshot:

```python
q_ref = state_qpos.copy()
kin.sync(q_ref)
fk_right, fk_left = kin.fk_bimanual(q_ref[:8], q_ref[8:16])
mapper.anchor(vr_right, vr_left, fk_right, fk_left)
```

Using the same `q_ref` for final sync and FK prevents the internal IK state and the reference pose from describing different robot instants.

The VR poses used to establish the reference are not also used as motion commands. The first solve waits for the next post-release pose set.

## 6. Relative Mapping

Let:

- `A` be the shared `arm_origin` frame;
- `S` be the VR/source frame;
- `E` be the robot end-effector frame;
- subscript `0` mean "at clutch release."

At clutch release, save:

$$
{}^{A}\mathbf p_{S0},\quad {}^{A}\mathbf R_{S0},\quad
{}^{A}\mathbf p_{E0},\quad {}^{A}\mathbf R_{E0}.
$$

For the current VR pose $({}^{A}\mathbf p_S,{}^{A}\mathbf R_S)$, calculate its motion from the reference:

$$
{}^{A}\Delta\mathbf p_S
= {}^{A}\mathbf p_S-{}^{A}\mathbf p_{S0},
$$

$$
\Delta\mathbf R_S
= ({}^{A}\mathbf R_{S0})^{\mathsf{T}}{}^{A}\mathbf R_S.
$$

The first value is controller displacement in `arm_origin` axes. The second is controller rotation since clutch release; for a rotation matrix, $\mathbf R^{-1}=\mathbf R^{\mathsf{T}}$.

Apply that motion to the FK reference:

$$
{}^{A}\mathbf p_E^*
= {}^{A}\mathbf p_{E0}+{}^{A}\Delta\mathbf p_S,
$$

$$
{}^{A}\mathbf R_E^*
= {}^{A}\mathbf R_{E0}\Delta\mathbf R_S.
$$

Quaternion form (`wxyz`) is:

$$
q_{\Delta}=q_{S0}^{-1}\otimes q_S,
\qquad
q_E^*=q_{E0}\otimes q_{\Delta}.
$$

The latest gripper value passes through directly:

$$
g_E^*=g_S.
$$

### Simple example

At clutch release, the VR pose is linked to the current robot pose. If the user then:

- moves the controller `+2 cm` along `arm_origin` X, the EE target moves `+2 cm` along the same X axis;
- rolls the controller `+10 degrees`, the EE target rolls `+10 degrees` from its saved orientation.

Translation intentionally follows the fixed `arm_origin` axes. This is not full rigid-transform composition:

$$
{}^{A}\mathbf T_E^*
\ne
{}^{A}\mathbf T_{E0}
({}^{A}\mathbf T_{S0})^{-1}
{}^{A}\mathbf T_S.
$$

The full SE(3) expression would rotate the translation delta as well, changing the existing teleoperation behavior. The mapper also normalizes quaternions and treats `q` and `-q` as the same rotation.

## 7. Failure and Recovery

Relative reference capture fails when any required VR pose or measured state is missing, stale, or invalid:

```text
reference invalid
status = relative_recalibration_required
no IK output
```

Late packets do not silently unlock the node. A new complete `syncstate=true -> false` clutch cycle is required.

During normal tracking, stale measured state clears the measured-state safety reference but does not stop solving. This preserves existing target-only behavior and can be tightened in a later safety change.

## 8. Planned Wiring

```text
vr-receiver/pose_right  -> vr-ik/target_right
vr-receiver/pose_left   -> vr-ik/target_left
intervention/active     -> vr-ik/active
intervention/syncstate  -> vr-ik/syncstate
state_right/left        -> vr-ik/state_right/left
```

IK runs with:

```text
--target-mode relative
--origin-frame arm_origin
```

## 9. Future Work

### 9.1 VR stability and lag validation

Before phase 2 commits a relative reference, the IK node should consume fresh per-arm VR health or lag inputs and verify that:

- the VR samples are complete and recent;
- the source stream is stable enough to define a reference;
- position and orientation lag are below configurable limits.

The check belongs before `sync(q_ref)` and `FK(q_ref)`. A failure should keep the reference invalid, report `relative_recalibration_required`, and require another clutch cycle. It should not change the mapping equations.

### 9.2 Runtime target-mode switching

A future optional Dora input should switch the node between `absolute` and `relative` without restarting it:

```text
target_mode: "absolute" | "relative"
```

The CLI argument remains the startup default when this input is not connected. Every mode change must be a hard target-generation boundary:

- clear pending target pairs so cached poses cannot cross modes;
- entering `relative` invalidates the old reference and requires a new clutch calibration cycle;
- entering `absolute` discards the relative reference and waits for a fresh absolute target set;
- never reinterpret a target cached under one mode as a target in the other mode.

## 10. Minimum Tests

1. Absolute target-only operation remains compatible with mainline.
2. Absolute sync blocks solving and performs final sync.
3. Relative translation, rotation, quaternion sign, and gripper mapping are correct.
4. Bimanual final sync and FK use the same `q_ref`.
5. Missing or stale reference inputs block output until the next clutch cycle.
6. The first relative solve requires a new post-release target set.
7. Single-arm modes do not wait for the inactive arm.
