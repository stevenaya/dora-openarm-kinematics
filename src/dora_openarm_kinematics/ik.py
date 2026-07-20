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
  position_right – [{"qpos": float32[8]}] current right joint state
  position_left  – [{"qpos": float32[8]}] current left joint state
                   (paired internally for optional sync)
  active       – bool[1]  true while intervention drives the arm
  command      – string[1]  episode/intervention lifecycle command
  Flat float32 arrays are also accepted for all inputs.

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


def _sync_before_solve(
    kin: Kinematics,
    measured_qpos: np.ndarray | None,
    enabled: bool,
) -> None:
    if enabled and measured_qpos is not None:
        kin.sync(measured_qpos)


class _BimanualPositionBuffer:
    """Emit right+left qpos after receiving a fresh sample from each arm."""

    def __init__(self) -> None:
        self._positions: dict[str, np.ndarray] = {}
        self._updated: set[str] = set()

    def update(self, side: str, qpos: np.ndarray) -> np.ndarray | None:
        if side not in {"right", "left"}:
            raise ValueError(f"Unknown arm side: {side}")
        if qpos.shape != (8,):
            raise ValueError(f"Per-arm qpos must contain 8 values, got {qpos.shape}")

        self._positions[side] = qpos.copy()
        self._updated.add(side)
        if self._updated != {"right", "left"}:
            return None

        self._updated.clear()
        return np.concatenate(
            [self._positions["right"], self._positions["left"]]
        ).astype(np.float32)


def _run(args: argparse.Namespace) -> None:
    kin = Kinematics(setup_from_args(args), ik_params_from_args(args))

    node = dora.Node()
    node.send_output("status", pa.array(["ready"]))
    sync_enabled = True
    intervention_armed = False
    measured_qpos: np.ndarray | None = None
    position_buffer = _BimanualPositionBuffer()

    for event in node:
        if event["type"] != "INPUT":
            continue

        eid = event["id"]

        if eid == "command":
            command = event["value"][0].as_py()
            if command == "intervene":
                intervention_armed = True
            elif command in {"start", "stop", "cancel", "success", "fail", "quit"}:
                intervention_armed = False
            sync_enabled = True
            continue

        if eid == "active":
            if intervention_armed:
                sync_enabled = not bool(event["value"][0].as_py())
            continue

        if eid in {"position_right", "position_left"}:
            values = extract_values(event["value"], "qpos")
            if values.shape != (8,):
                print(f"Warning: expected {eid}[8], got {values.shape}. Skipping.")
                continue
            side = eid.removeprefix("position_")
            paired_qpos = position_buffer.update(side, values)
            if paired_qpos is not None:
                measured_qpos = paired_qpos
                if sync_enabled:
                    kin.sync(paired_qpos)
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

        else:
            continue

        if not kin.ready():
            continue

        _sync_before_solve(kin, measured_qpos, args.sync_during_active)
        result = kin.solve()
        if result is None:
            continue

        ts = {"timestamp": time.time_ns()}
        node.send_output("position_right", build_qpos_output(result[:8]), ts)
        node.send_output("position_left", build_qpos_output(result[8:16]), ts)
        sync_enabled = False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mink IK dora node – OpenArm end-effector pose → joint angles"
    )
    register_common_args(parser)
    register_ik_args(parser)
    parser.add_argument(
        "--sync-during-active",
        action="store_true",
        help="Sync the latest measured arm qpos once before each active IK solve.",
    )
    return parser


def main() -> None:
    """Inverse kinematics for OpenArm."""
    _run(_build_parser().parse_args())


if __name__ == "__main__":
    main()
