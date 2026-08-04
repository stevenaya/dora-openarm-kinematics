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
  syncstate – bool[1], true while measured q should overwrite IK configuration
  active – optional bool[1] output gate; absent means active
  Flat float32 arrays are also accepted for target inputs.

Outputs:
  position_right – [{"qpos": float32[8]}] solved right arm joint angles
  position_left  – [{"qpos": float32[8]}] solved left arm joint angles
  status         – "ready" or "relative_recalibration_required"
"""

from __future__ import annotations

import argparse
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


def _has_fresh_state(
    received_at: dict[str, float],
    sides: list[str],
    now: float,
    timeout: float,
) -> bool:
    return all(now - received_at.get(side, -np.inf) <= timeout for side in sides)


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
    state_qpos = np.concatenate(
        [
            np.append(*kin.setup.joint_resolver.get_driver(kin.setup.data.qpos, side))
            for side in ("right", "left")
        ]
    ).astype(np.float32)
    state_received_at: dict[str, float] = {}
    targets: dict[str, np.ndarray] = {}
    target_received_at: dict[str, float] = {}
    pending_targets = set(sides)
    mapper = RelativeTargetMapper()
    active = True
    sync_enabled = False
    sync_pending = False
    sync_started_at: float | None = None
    calibration_valid = args.target_mode == "absolute"

    def set_status(value: str) -> None:
        nonlocal status
        if value != status:
            status = value
            node.send_output("status", pa.array([value]))

    def sync_from_cached_state(now: float, *, final: bool = False) -> bool:
        if not _has_fresh_state(
            state_received_at,
            sides,
            now,
            args.measured_state_timeout,
        ):
            return False
        kin.sync(state_qpos.copy())
        if final:
            print("[ik] Synchronized configuration from measured state.", flush=True)
        return True

    def fail_relative_calibration(reasons: list[str]) -> None:
        nonlocal calibration_valid, pending_targets
        calibration_valid = False
        pending_targets = set(sides)
        mapper.reset()
        print(
            "[ik] Relative calibration failed: " + "; ".join(reasons),
            flush=True,
        )
        set_status("relative_recalibration_required")

    def calibrate_relative(now: float) -> bool:
        nonlocal calibration_valid, pending_targets, sync_started_at
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
        for side in sides:
            received_at = state_received_at.get(side)
            if received_at is None:
                reasons.append(f"missing {side} measured state")
            elif now - received_at > args.measured_state_timeout:
                reasons.append(f"{side} measured state is {now - received_at:.3f}s old")
        if reasons:
            fail_relative_calibration(reasons)
            sync_started_at = None
            return False

        q_snapshot = state_qpos.copy()
        try:
            kin.sync(q_snapshot)
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
            return False

        calibration_valid = True
        pending_targets = set(sides)
        sync_started_at = None
        set_status("ready")
        print("[ik] Relative target references calibrated.", flush=True)
        return True

    def try_solve(now: float) -> None:
        nonlocal pending_targets
        if (
            not active
            or sync_enabled
            or sync_pending
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

        ts = {"timestamp": time.time_ns()}
        if "right" in sides:
            node.send_output("position_right", build_qpos_output(result[:8]), ts)
        if "left" in sides:
            node.send_output("position_left", build_qpos_output(result[8:16]), ts)

    def finish_absolute_sync(now: float) -> None:
        nonlocal sync_pending, sync_started_at, pending_targets
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
        sync_started_at = None
        try_solve(now)

    for event in node:
        if event["type"] != "INPUT":
            continue

        eid = event["id"]
        now = time.monotonic()

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
                    sync_from_cached_state(now)
                continue

            sync_enabled = requested
            pending_targets = set(sides)
            if requested:
                sync_pending = False
                sync_started_at = now
                if args.target_mode == "relative":
                    calibration_valid = False
                    mapper.reset()
                sync_from_cached_state(now)
            elif args.target_mode == "relative":
                sync_pending = False
                calibrate_relative(now)
            else:
                sync_pending = True
                finish_absolute_sync(now)
            continue

        if eid in {"state_right", "state_left"}:
            if not pa.types.is_struct(event["value"].type):
                print(f"Warning: expected normalized struct for {eid}. Skipping.")
                continue
            qpos = extract_values(event["value"], "qpos")
            qvel = extract_values(event["value"], "qvel")
            if qpos.shape != (8,) or qvel.shape != (8,):
                print(
                    f"Warning: expected {eid} qpos[8]/qvel[8], "
                    f"got {qpos.shape}/{qvel.shape}. Skipping."
                )
                continue
            side = eid.removeprefix("state_")
            arm_slice = slice(0, 8) if side == "right" else slice(8, 16)
            state_qpos[arm_slice] = qpos
            state_received_at[side] = now
            if sync_enabled:
                sync_from_cached_state(now)
            elif sync_pending:
                finish_absolute_sync(now)
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
