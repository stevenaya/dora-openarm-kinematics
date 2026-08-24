# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dora node: mink-based differential IK solver for OpenArm.

Accepts end-effector pose targets and solves joint angles via mink's QP-based
differential IK. Both arms share one mink.Configuration and one QP solve per
step.

Pose convention:  float32[8] = [px, py, pz, qw, qx, qy, qz, gripper_angle]
Inputs:
  target_right – [{"pose": float32[8]}]  right EE target pose + gripper angle
  target_left  – [{"pose": float32[8]}]  left  EE target pose + gripper angle
  state_right/left – normalized state with qpos[8] and qvel[8]; measured qpos
                     is used by state-aware limits and configuration sync
  command_right/left – latest driver-accepted qpos[8] command
  reset – optional bool[1] one-shot event that clears all runtime state
  syncstate – bool[1], true while reference synchronization is active
  sync_mode – optional string[1], "state", "command", or "reference";
              defaults to "state"
  active – optional bool[1] output gate; absent means active
  Flat float32 arrays are also accepted for target inputs.

Outputs:
  position_right – [{"qpos": float32[8]}] solved right arm joint angles
  position_left  – [{"qpos": float32[8]}] solved left arm joint angles
  status         – "ready" or "relative_recalibration_required"
"""

from __future__ import annotations

import argparse
import numbers
import time

import dora
import numpy as np
import pyarrow as pa

from openarm_control import (
    Kinematics,
    register_common_args,
    register_ik_args,
    ik_params_from_args,
    setup_from_args,
)

from dora_openarm_kinematics.relative_target import RelativeTargetMapper


_QPOS_STRUCT_TYPE = pa.struct({"qpos": pa.list_(pa.float32())})
_SYNC_MODES = {"state", "command", "reference"}
_DEFAULT_SYNC_MODE = "state"
_COMMAND_STATE_MAX_DELTA = 0.2


def build_qpos_output(qpos: np.ndarray) -> pa.Array:
    """Wrap joint angles as a length-1 StructArray: [{"qpos": [...]}]."""
    return pa.array([{"qpos": qpos}], type=_QPOS_STRUCT_TYPE)


def extract_values(value: pa.Array, key: str) -> np.ndarray:
    """Read `key` from a length-1 StructArray, or a flat array as-is."""
    if pa.types.is_struct(value.type):
        value = value.field(key)[0].values
    return np.array(value, dtype=np.float32)


def extract_bool(value: pa.Array, name: str) -> bool:
    """Read a non-null length-one boolean array."""
    if len(value) != 1 or not pa.types.is_boolean(value.type) or not value[0].is_valid:
        raise ValueError(f"expected {name} bool[1]")
    return bool(value[0].as_py())


def extract_sync_mode(value: pa.Array) -> str:
    """Read a supported synchronization mode from a length-one string array."""
    if len(value) != 1 or not pa.types.is_string(value.type) or not value[0].is_valid:
        raise ValueError("expected sync_mode string[1]")
    mode = str(value[0].as_py())
    if mode not in _SYNC_MODES:
        raise ValueError(f"unsupported sync_mode {mode!r}")
    return mode


def _has_fresh_state(
    received_at: dict[str, float],
    sides: list[str],
    now: float,
    timeout: float,
) -> bool:
    return all(now - received_at.get(side, -np.inf) <= timeout for side in sides)


def _metadata_start_epoch(metadata: dict) -> tuple[bool, int | None]:
    """Return whether an optional start_epoch is well formed and its value."""
    if "start_epoch" not in metadata:
        return True, None
    value = metadata["start_epoch"]
    if (
        not isinstance(value, numbers.Integral)
        or isinstance(value, bool)
        or int(value) < 0
    ):
        return False, None
    return True, int(value)


def _run(args: argparse.Namespace) -> None:
    if (
        not np.isfinite(args.measured_state_timeout)
        or args.measured_state_timeout <= 0.0
    ):
        raise ValueError("--measured-state-timeout must be finite and positive.")
    if (
        not np.isfinite(args.relative_input_timeout)
        or args.relative_input_timeout <= 0.0
    ):
        raise ValueError("--relative-input-timeout must be finite and positive.")

    kin = Kinematics(setup_from_args(args), ik_params_from_args(args))

    node = dora.Node()
    node.send_output("status", pa.array(["ready"]))
    status = "ready"
    sides = kin.setup.sides
    initial_configuration_qpos = np.concatenate(
        [
            np.append(*kin.setup.joint_resolver.get_driver(kin.setup.data.qpos, side))
            for side in ("right", "left")
        ]
    ).astype(np.float32)
    state_qpos = initial_configuration_qpos.copy()
    command_qpos = initial_configuration_qpos.copy()
    configuration_qpos = state_qpos.copy()
    state_received_at: dict[str, float] = {}
    command_received_at: dict[str, float] = {}
    targets: dict[str, np.ndarray] = {}
    target_received_at: dict[str, float] = {}
    pending_targets = set(sides)
    mapper = RelativeTargetMapper()
    active = True
    sync_enabled = False
    sync_pending = False
    sync_started_at: float | None = None
    requested_sync_mode = _DEFAULT_SYNC_MODE
    active_sync_mode: str | None = None
    sync_reference_qpos: np.ndarray | None = None
    configuration_initialized = True
    calibration_valid = args.target_mode == "absolute"
    cached_start_epoch: int | None = None

    def set_status(value: str) -> None:
        nonlocal status
        if value != status:
            status = value
            node.send_output("status", pa.array([value]))

    def sync_from_cached_state(now: float, *, final: bool = False) -> bool:
        nonlocal configuration_qpos, configuration_initialized
        nonlocal sync_reference_qpos
        if not _has_fresh_state(
            state_received_at,
            sides,
            now,
            args.measured_state_timeout,
        ):
            return False
        q_snapshot = state_qpos.copy()
        kin.sync(q_snapshot)
        configuration_qpos = q_snapshot
        configuration_initialized = True
        if active_sync_mode == "state":
            sync_reference_qpos = configuration_qpos.copy()
        if final:
            print("[ik] Synchronized configuration from measured state.", flush=True)
        return True

    def command_seed_rejection_reasons(now: float) -> list[str]:
        reasons = []
        for source, received_at in (
            ("executed command", command_received_at),
            ("measured state", state_received_at),
        ):
            for side in sides:
                timestamp = received_at.get(side)
                if timestamp is None:
                    reasons.append(f"missing {side} {source}")
                elif now - timestamp > args.measured_state_timeout:
                    reasons.append(f"{side} {source} is {now - timestamp:.3f}s old")
        if reasons:
            return reasons
        for side in sides:
            arm_slice = slice(0, 8) if side == "right" else slice(8, 16)
            delta = np.abs(command_qpos[arm_slice][:7] - state_qpos[arm_slice][:7])
            if np.any(delta > _COMMAND_STATE_MAX_DELTA):
                reasons.append(
                    f"{side} command/state delta {float(np.max(delta)):.3f}rad "
                    f"exceeds {_COMMAND_STATE_MAX_DELTA:.3f}rad"
                )
        return reasons

    def sync_from_cached_command(now: float, *, final: bool = False) -> bool:
        nonlocal configuration_qpos, configuration_initialized
        nonlocal sync_reference_qpos
        if command_seed_rejection_reasons(now):
            return False
        q_snapshot = command_qpos.copy()
        kin.sync(q_snapshot)
        configuration_qpos = q_snapshot
        configuration_initialized = True
        if active_sync_mode == "command":
            sync_reference_qpos = configuration_qpos.copy()
        if final:
            print("[ik] Synchronized configuration from executed command.", flush=True)
        return True

    def sync_from_active_source(now: float, *, final: bool = False) -> bool:
        nonlocal active_sync_mode, sync_reference_qpos
        if active_sync_mode == "command":
            rejection_reasons = command_seed_rejection_reasons(now)
            if not rejection_reasons and sync_from_cached_command(now, final=final):
                return True
            active_sync_mode = "state"
            print(
                "[ik] Executed command seed rejected; falling back to state sync: "
                + "; ".join(rejection_reasons),
                flush=True,
            )
        if active_sync_mode == "state":
            return sync_from_cached_state(now, final=final)
        if active_sync_mode == "reference":
            if configuration_initialized:
                sync_reference_qpos = configuration_qpos.copy()
                return True
            active_sync_mode = "state"
            print(
                "[ik] Configuration is not initialized; falling back to state sync.",
                flush=True,
            )
            return sync_from_cached_state(now, final=final)
        return False

    def reset_runtime_state() -> None:
        nonlocal state_qpos, command_qpos, configuration_qpos
        nonlocal pending_targets, active, sync_enabled, sync_pending
        nonlocal sync_started_at, requested_sync_mode, active_sync_mode
        nonlocal sync_reference_qpos, configuration_initialized
        nonlocal calibration_valid, cached_start_epoch
        active = False
        sync_enabled = False
        sync_pending = False
        sync_started_at = None
        requested_sync_mode = _DEFAULT_SYNC_MODE
        active_sync_mode = None
        sync_reference_qpos = None
        configuration_initialized = False
        calibration_valid = args.target_mode == "absolute"
        state_qpos = initial_configuration_qpos.copy()
        command_qpos = initial_configuration_qpos.copy()
        configuration_qpos = initial_configuration_qpos.copy()
        state_received_at.clear()
        command_received_at.clear()
        targets.clear()
        target_received_at.clear()
        pending_targets = set(sides)
        mapper.reset()
        kin.sync(initial_configuration_qpos)
        kin.clear_measured_state()
        cached_start_epoch = None
        set_status("ready")
        print("[ik] Runtime state reset.", flush=True)

    def fail_relative_calibration(reasons: list[str]) -> None:
        nonlocal calibration_valid, pending_targets, sync_enabled, sync_pending
        nonlocal sync_started_at, active_sync_mode, sync_reference_qpos
        nonlocal configuration_initialized
        calibration_valid = False
        sync_enabled = False
        sync_pending = False
        sync_started_at = None
        active_sync_mode = None
        sync_reference_qpos = None
        configuration_initialized = False
        state_received_at.clear()
        command_received_at.clear()
        targets.clear()
        target_received_at.clear()
        pending_targets = set(sides)
        mapper.reset()
        kin.clear_measured_state()
        print(
            "[ik] Relative calibration failed: " + "; ".join(reasons),
            flush=True,
        )
        set_status("relative_recalibration_required")

    def invalidate_input_generation(reason: str) -> None:
        """Discard samples that may belong to a different arm start."""
        nonlocal pending_targets
        if args.target_mode == "relative":
            fail_relative_calibration([reason])
            return
        state_received_at.clear()
        command_received_at.clear()
        targets.clear()
        target_received_at.clear()
        pending_targets = set(sides)
        kin.clear_measured_state()

    def accept_input_epoch(metadata: dict, event_id: str) -> bool:
        """Apply the shared epoch protocol to state and command inputs."""
        nonlocal cached_start_epoch
        valid, incoming_epoch = _metadata_start_epoch(metadata)
        if not valid:
            invalidate_input_generation(f"malformed start_epoch on {event_id}")
            return False
        if cached_start_epoch is None:
            if incoming_epoch is not None:
                cached_start_epoch = incoming_epoch
            return True
        if incoming_epoch is None:
            invalidate_input_generation(f"missing start_epoch on {event_id}")
            return False
        if incoming_epoch == cached_start_epoch:
            return True
        if incoming_epoch < cached_start_epoch:
            print(
                f"[ik] Dropping stale {event_id} start_epoch={incoming_epoch}; "
                f"current={cached_start_epoch}.",
                flush=True,
            )
            return False
        previous_epoch = cached_start_epoch
        cached_start_epoch = incoming_epoch
        invalidate_input_generation(
            f"start_epoch advanced from {previous_epoch} to {incoming_epoch} "
            f"on {event_id}"
        )
        return True

    def calibrate_relative(now: float) -> bool:
        nonlocal calibration_valid, pending_targets, sync_started_at
        nonlocal active_sync_mode, sync_reference_qpos
        mode = active_sync_mode or _DEFAULT_SYNC_MODE
        if mode == "command":
            rejection_reasons = command_seed_rejection_reasons(now)
            if rejection_reasons:
                mode = "state"
                active_sync_mode = "state"
                print(
                    "[ik] Executed command seed rejected at sync release; "
                    "falling back to state sync: " + "; ".join(rejection_reasons),
                    flush=True,
                )
        reasons: list[str] = []
        if sync_started_at is None:
            reasons.append("no preceding sync interval")
        else:
            for side in sides:
                received_at = target_received_at.get(side)
                if received_at is None:
                    reasons.append(f"missing {side} source pose")
                elif received_at < sync_started_at:
                    reasons.append(f"{side} source pose predates this sync interval")
                elif now - received_at > args.relative_input_timeout:
                    reasons.append(
                        f"{side} source pose is {now - received_at:.3f}s old"
                    )
        if mode == "state":
            for side in sides:
                received_at = state_received_at.get(side)
                if received_at is None:
                    reasons.append(f"missing {side} measured state")
                elif now - received_at > args.measured_state_timeout:
                    reasons.append(
                        f"{side} measured state is {now - received_at:.3f}s old"
                    )
        elif mode == "command":
            reasons.extend(command_seed_rejection_reasons(now))
        if sync_reference_qpos is None:
            reasons.append("missing IK configuration reference")
        if reasons:
            fail_relative_calibration(reasons)
            sync_started_at = None
            active_sync_mode = None
            sync_reference_qpos = None
            return False

        if mode == "state" and not sync_from_cached_state(now, final=True):
            fail_relative_calibration(["measured state became stale"])
            sync_started_at = None
            active_sync_mode = None
            sync_reference_qpos = None
            return False
        if mode == "command" and not sync_from_cached_command(now, final=True):
            fail_relative_calibration(["executed command pair became unavailable"])
            sync_started_at = None
            active_sync_mode = None
            sync_reference_qpos = None
            return False

        q_snapshot = sync_reference_qpos.copy()
        try:
            fk_right: np.ndarray | None = None
            fk_left: np.ndarray | None = None
            if sides == ["right", "left"]:
                fk_right, fk_left = kin.fk_bimanual(q_snapshot[:8], q_snapshot[8:16])
            elif sides == ["right"]:
                fk_right = kin.fk("right", q_snapshot[:8])
            else:
                fk_left = kin.fk("left", q_snapshot[8:16])
            mapper.anchor(
                targets.get("right"),
                targets.get("left"),
                fk_right,
                fk_left,
            )
        except (RuntimeError, ValueError) as exc:
            fail_relative_calibration([str(exc)])
            sync_started_at = None
            active_sync_mode = None
            sync_reference_qpos = None
            return False

        calibration_valid = True
        pending_targets = set(sides)
        sync_started_at = None
        active_sync_mode = None
        sync_reference_qpos = None
        set_status("ready")
        print(f"[ik] Relative target references calibrated ({mode}).", flush=True)
        return True

    def try_solve(now: float) -> None:
        nonlocal configuration_qpos, pending_targets
        if (
            not active
            or sync_enabled
            or sync_pending
            or not configuration_initialized
            or not calibration_valid
            or pending_targets
        ):
            return

        try:
            mapped_targets = {
                side: (
                    mapper.map(side, targets[side])
                    if args.target_mode == "relative"
                    else targets[side]
                )
                for side in sides
            }
        except (KeyError, RuntimeError, ValueError) as exc:
            pending_targets = set(sides)
            if args.target_mode == "relative":
                fail_relative_calibration([str(exc)])
            else:
                print(f"Warning: invalid IK target: {exc}. Skipping.")
            return

        pending_targets = set(sides)
        for side, values in mapped_targets.items():
            kin.set_target(side, values[:7])
            kin.set_gripper(side, values[7])
        if not kin.ready():
            return

        if _has_fresh_state(
            state_received_at,
            sides,
            now,
            args.measured_state_timeout,
        ):
            kin.update_measured_state(state_qpos)
        elif state_received_at:
            kin.clear_measured_state()
        result = kin.solve()
        if result is None:
            return
        configuration_qpos = result.copy()

        ts = {"timestamp": time.time_ns()}
        if cached_start_epoch is not None:
            ts["start_epoch"] = cached_start_epoch
        if "right" in sides:
            node.send_output("position_right", build_qpos_output(result[:8]), ts)
        if "left" in sides:
            node.send_output("position_left", build_qpos_output(result[8:16]), ts)

    def finish_absolute_sync(now: float) -> None:
        nonlocal active_sync_mode, sync_pending, sync_reference_qpos
        nonlocal sync_started_at, pending_targets
        if not sync_from_cached_state(now, final=True):
            return
        sync_pending = False
        if active and sync_started_at is not None:
            pending_targets = {
                side
                for side in sides
                if target_received_at.get(side, -np.inf) < sync_started_at
            }
        else:
            pending_targets = set(sides)
        active_sync_mode = None
        sync_reference_qpos = None
        sync_started_at = None
        try_solve(now)

    for event in node:
        if event["type"] != "INPUT":
            continue

        eid = event["id"]
        now = time.monotonic()

        if eid == "reset":
            try:
                requested = extract_bool(event["value"], "reset")
            except ValueError as exc:
                print(f"Warning: {exc}. Skipping.")
                continue
            if requested:
                reset_runtime_state()
            continue

        if eid == "sync_mode":
            try:
                mode = extract_sync_mode(event["value"])
            except ValueError as exc:
                print(f"Warning: {exc}. Skipping.", flush=True)
                continue
            requested_sync_mode = mode
            if sync_enabled and mode == "state" and active_sync_mode != "state":
                active_sync_mode = "state"
                sync_from_cached_state(now)
                print("[ik] Promoted active synchronization to state.", flush=True)
            continue

        if eid == "active":
            try:
                requested = extract_bool(event["value"], "active")
            except ValueError as exc:
                print(f"Warning: {exc}. Skipping.")
                continue
            if requested != active:
                active = requested
                pending_targets = set(sides)
            continue

        if eid == "syncstate":
            try:
                requested = extract_bool(event["value"], "syncstate")
            except ValueError as exc:
                print(f"Warning: {exc}. Skipping.")
                continue
            if requested == sync_enabled:
                if requested:
                    sync_from_active_source(now)
                continue

            sync_enabled = requested
            pending_targets = set(sides)
            if requested:
                sync_pending = False
                sync_started_at = now
                active_sync_mode = requested_sync_mode
                if args.target_mode == "absolute":
                    active_sync_mode = "state"
                sync_reference_qpos = configuration_qpos.copy()
                if args.target_mode == "relative":
                    calibration_valid = False
                    mapper.reset()
                sync_from_active_source(now)
            elif args.target_mode == "relative":
                sync_pending = False
                calibrate_relative(now)
            else:
                sync_pending = True
                finish_absolute_sync(now)
            continue

        if eid in {"state_right", "state_left"}:
            if not accept_input_epoch(event["metadata"], eid):
                continue
            if not pa.types.is_struct(event["value"].type):
                print(f"Warning: expected normalized struct for {eid}. Skipping.")
                continue
            qpos = extract_values(event["value"], "qpos")
            qvel = extract_values(event["value"], "qvel")
            if (
                qpos.shape != (8,)
                or qvel.shape != (8,)
                or not np.all(np.isfinite(qpos))
                or not np.all(np.isfinite(qvel))
            ):
                print(
                    f"Warning: expected finite {eid} qpos[8]/qvel[8], "
                    f"got {qpos.shape}/{qvel.shape}. Skipping."
                )
                continue
            side = eid.removeprefix("state_")
            arm_slice = slice(0, 8) if side == "right" else slice(8, 16)
            state_qpos[arm_slice] = qpos
            state_received_at[side] = now
            if sync_enabled and active_sync_mode == "state":
                sync_from_cached_state(now)
            elif sync_pending:
                finish_absolute_sync(now)
            continue

        if eid in {"command_right", "command_left"}:
            if not accept_input_epoch(event["metadata"], eid):
                continue
            side = eid.removeprefix("command_")
            if side not in sides:
                continue
            qpos = extract_values(event["value"], "qpos")
            if qpos.shape != (8,) or not np.all(np.isfinite(qpos)):
                print(f"Warning: expected finite {eid} qpos[8]. Skipping.")
                continue
            arm_slice = slice(0, 8) if side == "right" else slice(8, 16)
            command_qpos[arm_slice] = qpos
            command_received_at[side] = now
            if sync_enabled and active_sync_mode == "command":
                sync_from_cached_command(now)
            continue

        if eid in {"target_right", "target_left"}:
            side = eid.removeprefix("target_")
            if side not in sides:
                continue
            values = extract_values(event["value"], "pose")
            if values.shape != (8,):
                print(f"Warning: expected {eid}[8], got {values.shape}. Skipping.")
                continue
            targets[side] = values.copy()
            target_received_at[side] = now
            if active and not sync_enabled and not sync_pending and calibration_valid:
                pending_targets.discard(side)
                try_solve(now)


def main() -> None:
    """Inverse kinematics for OpenArm."""
    parser = argparse.ArgumentParser(
        description="Mink IK dora node – OpenArm end-effector pose → joint angles"
    )
    register_common_args(parser)
    register_ik_args(parser)
    parser.add_argument(
        "--target-mode",
        choices=("absolute", "relative"),
        default="absolute",
        help="Interpret targets directly or relative to clutch-time source poses.",
    )
    parser.add_argument("--measured-state-timeout", type=float, default=0.1)
    parser.add_argument("--relative-input-timeout", type=float, default=0.1)
    args = parser.parse_args()
    _run(args)


if __name__ == "__main__":
    main()
