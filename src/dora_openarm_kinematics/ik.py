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
  Flat float32 arrays are also accepted for target inputs.

Outputs:
  position_right – [{"qpos": float32[8]}] solved right arm joint angles
  position_left  – [{"qpos": float32[8]}] solved left arm joint angles
  status         – ["ready"] on startup
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


_QPOS_STRUCT_TYPE = pa.struct({"qpos": pa.list_(pa.float32())})


def build_qpos_output(qpos: np.ndarray) -> pa.Array:
    """Wrap joint angles as a length-1 StructArray: [{"qpos": [...]}]."""
    return pa.array([{"qpos": qpos}], type=_QPOS_STRUCT_TYPE)


def extract_values(value: pa.Array, key: str) -> np.ndarray:
    """Read `key` from a length-1 StructArray, or a flat array as-is."""
    if pa.types.is_struct(value.type):
        value = value.field(key)[0].values
    return np.array(value, dtype=np.float32)


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

    kin = Kinematics(setup_from_args(args), ik_params_from_args(args))

    node = dora.Node()
    node.send_output("status", pa.array(["ready"]))
    state_qpos = np.hstack(
        [
            np.append(*kin.setup.joint_resolver.get_driver(kin.setup.data.qpos, side))
            for side in ("right", "left")
        ]
    ).astype(np.float32)
    state_received_at: dict[str, float] = {}
    sync_enabled = False
    sync_pending = False
    pending_targets = set(kin.setup.sides)

    def sync_from_cached_state(now: float) -> None:
        nonlocal sync_pending
        if not (sync_enabled or sync_pending) or not _has_fresh_state(
            state_received_at,
            kin.setup.sides,
            now,
            args.measured_state_timeout,
        ):
            return
        kin.sync(state_qpos)
        if sync_pending:
            print("[ik] Synchronized configuration from measured state.", flush=True)
        sync_pending = False

    for event in node:
        if event["type"] != "INPUT":
            continue

        eid = event["id"]
        now = time.monotonic()

        if eid == "syncstate":
            value = event["value"]
            if (
                len(value) != 1
                or not pa.types.is_boolean(value.type)
                or not value[0].is_valid
            ):
                print("Warning: expected syncstate bool[1]. Skipping.")
                continue
            requested = bool(value[0].as_py())
            if requested != sync_enabled:
                sync_enabled = requested
                sync_pending = True
                pending_targets = set(kin.setup.sides)
            sync_from_cached_state(now)
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
            sync_from_cached_state(now)
            continue

        if eid in {"target_right", "target_left"} and (sync_enabled or sync_pending):
            continue

        if eid == "target_right" and "right" in kin.setup.sides:
            values = extract_values(event["value"], "pose")
            if values.shape != (8,):
                print(
                    f"Warning: expected target_right[8], got {values.shape}. Skipping."
                )
                continue
            pose = values[:7]
            gripper_angle = values[7]
            kin.set_target("right", pose)
            kin.set_gripper("right", gripper_angle)
            pending_targets.discard("right")

        elif eid == "target_left" and "left" in kin.setup.sides:
            values = extract_values(event["value"], "pose")
            if values.shape != (8,):
                print(
                    f"Warning: expected target_left[8], got {values.shape}. Skipping."
                )
                continue
            pose = values[:7]
            gripper_angle = values[7]
            kin.set_target("left", pose)
            kin.set_gripper("left", gripper_angle)
            pending_targets.discard("left")

        else:
            continue

        if pending_targets:
            continue
        pending_targets = set(kin.setup.sides)
        if not kin.ready():
            continue

        if _has_fresh_state(
            state_received_at,
            kin.setup.sides,
            now,
            args.measured_state_timeout,
        ):
            kin.update_measured_state(state_qpos)
        elif state_received_at:
            kin.clear_measured_state()
        result = kin.solve()
        if result is None:
            continue

        ts = {"timestamp": time.time_ns()}
        if "right" in kin.setup.sides:
            node.send_output("position_right", build_qpos_output(result[:8]), ts)
        if "left" in kin.setup.sides:
            node.send_output("position_left", build_qpos_output(result[8:16]), ts)


def main() -> None:
    """Inverse kinematics for OpenArm."""
    parser = argparse.ArgumentParser(
        description="Mink IK dora node – OpenArm end-effector pose → joint angles"
    )
    register_common_args(parser)
    register_ik_args(parser)
    parser.add_argument("--measured-state-timeout", type=float, default=0.1)
    args = parser.parse_args()
    _run(args)


if __name__ == "__main__":
    main()
