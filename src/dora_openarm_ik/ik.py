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

Pose convention (inputs and outputs):  float32[7] = [px, py, pz, qw, qx, qy, qz]
Inputs:
  target_right – float32[7]  right EE target pose
  target_left  – float32[7]  left  EE target pose
  position     – float32[16] current joint state right[8]+left[8] (optional sync)

Outputs:
  position_right – float32[8] solved right arm joint angles
  position_left  – float32[8] solved left arm joint angles
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


def _map_range(x: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """Map a value from one range to another, with clipping."""
    if in_max == in_min:
        return out_min
    x = max(min(x, in_max), in_min) if in_min < in_max else max(min(x, in_min), in_max)
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def _map_trigger_to_gripper(trigger: float, side: str) -> float:
    """trigger 0.0~1.0 → gripper angle (radians)"""
    if side == "right":
        return np.deg2rad(_map_range(trigger, 0.0, 1.0, -45.0, 8.0))
    else:
        return np.deg2rad(_map_range(trigger, 0.0, 1.0, 45.0, -8.0))


def _run(args: argparse.Namespace) -> None:
    kin = Kinematics(setup_from_args(args), ik_params_from_args(args))

    node = dora.Node()
    node.send_output("status", pa.array(["ready"]))

    sync_enabled = True

    for event in node:
        if event["type"] != "INPUT":
            continue

        eid = event["id"]
        if eid == "command":
            command = event["value"][0].as_py()
            if command in ("start", "intervene"):
                kin.reset_posture_target()
                sync_enabled = True
            elif command in ("stop", "cancel", "success", "fail", "quit"):
                sync_enabled = True
            continue

        if eid == "active":
            active = bool(event["value"][0].as_py())
            sync_enabled = not active
            continue

        values = np.array(event["value"], dtype=np.float32)

        if eid == "position":
            if sync_enabled and values.shape == (16,):
                kin.sync(values)
            continue

        if eid == "target_right" and "right" in kin.setup.sides:
            if values.shape != (7,):
                print(f"Warning: expected target_right[7], got {values.shape}. Skipping.")
                continue
            kin.set_target("right", values)

        elif eid == "target_left" and "left" in kin.setup.sides:
            if values.shape != (7,):
                print(f"Warning: expected target_left[7], got {values.shape}. Skipping.")
                continue
            kin.set_target("left", values)

        elif eid == "trigger_right":
            kin.set_gripper("right", _map_trigger_to_gripper(float(values[0]), "right"))
            continue

        elif eid == "trigger_left":
            kin.set_gripper("left", _map_trigger_to_gripper(float(values[0]), "left"))
            continue

        else:
            continue

        if not kin.ready():
            continue

        result = kin.solve()
        if result is None:
            continue

        ts = {"timestamp": time.time_ns()}
        node.send_output("position_right", pa.array(result[:8],  type=pa.float32()), ts)
        node.send_output("position_left",  pa.array(result[8:16], type=pa.float32()), ts)
        sync_enabled = False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mink IK dora node – OpenArm end-effector pose → joint angles"
    )
    register_common_args(parser)
    register_ik_args(parser)
    args = parser.parse_args()
    _run(args)


if __name__ == "__main__":
    main()
