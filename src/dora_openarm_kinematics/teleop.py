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

"""Dora node: teleoperation synchronization state machine.

The node owns only teleoperation state. It does not receive or forward poses,
joint states, or actions.

Inputs:
  grip_left/right - float[1] analog synchronization triggers
  force_full_sync - optional bool[1] button; upgrades the active sync to full
  command - optional string[1] evaluation command (start/intervene/stop/quit)

Outputs:
  active - bool[1], true while relative teleoperation may drive the arm
  syncstate - bool[1], true while IK should synchronize references
  sync_mode - "full" for measured configuration + references, or "reference"
  status - inactive, waiting_trigger, syncing, or tracking
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import math

import dora
import pyarrow as pa


STOP_COMMANDS = {"stop", "cancel", "success", "fail", "quit"}
STATUSES = {"inactive", "waiting_trigger", "syncing", "tracking"}


@dataclass
class _GripState:
    pressed: bool = False

    def update(self, value: float, engage: float, release: float) -> None:
        if self.pressed:
            if value <= release:
                self.pressed = False
        elif value >= engage:
            self.pressed = True


@dataclass
class _TeleopState:
    sync_trigger: str
    enabled: bool = True
    status: str = "waiting_trigger"
    active: bool = False
    syncstate: bool = False
    sync_mode: str = "full"
    first_sync: bool = True
    force_full_pressed: bool = False
    left: _GripState = field(default_factory=_GripState)
    right: _GripState = field(default_factory=_GripState)

    @property
    def triggered(self) -> bool:
        if self.sync_trigger == "left":
            return self.left.pressed
        if self.sync_trigger == "right":
            return self.right.pressed
        return self.left.pressed and self.right.pressed

    def reset_grips(self) -> None:
        self.left.pressed = False
        self.right.pressed = False


def _extract_grip(value: pa.Array, name: str) -> float:
    if len(value) != 1 or not pa.types.is_floating(value.type) or not value[0].is_valid:
        raise ValueError(f"expected {name} float[1]")
    result = float(value[0].as_py())
    if not math.isfinite(result):
        raise ValueError(f"expected finite {name}")
    return result


def _extract_command(value: pa.Array) -> str:
    if len(value) != 1 or not pa.types.is_string(value.type) or not value[0].is_valid:
        raise ValueError("expected command string[1]")
    return str(value[0].as_py())


def _extract_bool(value: pa.Array, name: str) -> bool:
    if len(value) != 1 or not pa.types.is_boolean(value.type) or not value[0].is_valid:
        raise ValueError(f"expected {name} bool[1]")
    return bool(value[0].as_py())


def _run(args: argparse.Namespace) -> None:
    if not 0.0 <= args.grip_release_threshold < args.grip_engage_threshold <= 1.0:
        raise ValueError("grip thresholds must satisfy 0 <= release < engage <= 1")

    node = dora.Node()
    state = _TeleopState(sync_trigger=args.sync_trigger)

    def send_bool(output_id: str, value: bool, metadata: dict) -> None:
        node.send_output(output_id, pa.array([value], type=pa.bool_()), metadata)

    def set_gates(
        *, active: bool, syncstate: bool, metadata: dict, force: bool = False
    ) -> None:
        active_changed = force or active != state.active
        sync_changed = force or syncstate != state.syncstate

        if active_changed and not active:
            state.active = False
            send_bool("active", False, metadata)
        if sync_changed:
            state.syncstate = syncstate
            send_bool("syncstate", syncstate, metadata)
        if active_changed and active:
            state.active = True
            send_bool("active", True, metadata)

    def set_status(value: str, metadata: dict, *, force: bool = False) -> None:
        if value not in STATUSES:
            raise ValueError(f"unknown teleop status: {value}")
        if force or value != state.status:
            state.status = value
            node.send_output("status", pa.array([value]), metadata)

    def set_sync_mode(value: str, metadata: dict) -> None:
        if value != state.sync_mode:
            state.sync_mode = value
            node.send_output("sync_mode", pa.array([value]), metadata)

    def reset(*, enabled: bool, metadata: dict) -> None:
        state.enabled = enabled
        state.first_sync = True
        state.force_full_pressed = False
        state.reset_grips()
        set_gates(active=False, syncstate=False, metadata=metadata, force=True)
        set_sync_mode("full", metadata)
        set_status(
            "waiting_trigger" if enabled else "inactive",
            metadata,
            force=True,
        )

    def update_trigger(metadata: dict) -> None:
        if not state.enabled:
            return
        if state.triggered:
            if state.status != "syncing":
                mode = (
                    "full"
                    if state.first_sync or state.force_full_pressed
                    else "reference"
                )
                set_sync_mode(mode, metadata)
                set_gates(active=False, syncstate=True, metadata=metadata)
                set_status("syncing", metadata)
                state.first_sync = False
        elif state.status == "syncing":
            set_gates(active=True, syncstate=False, metadata=metadata)
            set_status("tracking", metadata)

    reset(enabled=True, metadata={})

    for event in node:
        if event["type"] != "INPUT":
            continue

        event_id = event["id"]
        metadata = event["metadata"]

        if event_id == "command":
            try:
                command = _extract_command(event["value"])
            except ValueError as exc:
                print(f"Warning: {exc}. Skipping.")
                continue
            if command == "intervene":
                reset(enabled=True, metadata=metadata)
            elif command == "start" or command in STOP_COMMANDS:
                reset(enabled=False, metadata=metadata)
            continue

        if event_id == "force_full_sync":
            try:
                pressed = _extract_bool(event["value"], event_id)
            except ValueError as exc:
                print(f"Warning: {exc}. Skipping.")
                continue
            rising_edge = pressed and not state.force_full_pressed
            state.force_full_pressed = pressed
            if (
                rising_edge
                and state.enabled
                and state.triggered
                and state.status == "syncing"
                and state.sync_mode == "reference"
            ):
                set_sync_mode("full", metadata)
            continue

        if event_id not in {"grip_left", "grip_right"}:
            continue
        try:
            value = _extract_grip(event["value"], event_id)
        except ValueError as exc:
            print(f"Warning: {exc}. Skipping.")
            continue

        grip = state.left if event_id == "grip_left" else state.right
        grip.update(
            value,
            args.grip_engage_threshold,
            args.grip_release_threshold,
        )
        update_trigger(metadata)


def main() -> None:
    """Run the teleoperation synchronization state machine."""
    parser = argparse.ArgumentParser(
        description="OpenArm relative teleoperation synchronization state machine"
    )
    parser.add_argument(
        "--sync-trigger",
        choices=("left", "right", "both"),
        default="left",
        help="Grip input(s) used to enter synchronization (default: left).",
    )
    parser.add_argument("--grip-engage-threshold", type=float, default=0.7)
    parser.add_argument("--grip-release-threshold", type=float, default=0.5)
    _run(parser.parse_args())


if __name__ == "__main__":
    main()
