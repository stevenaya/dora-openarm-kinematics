"""Event-level tests for absolute and relative IK target handling."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pyarrow as pa

from dora_openarm_kinematics import ik


_POSE_TYPE = pa.struct({"pose": pa.list_(pa.float32())})
_STATE_TYPE = pa.struct(
    {"qpos": pa.list_(pa.float32()), "qvel": pa.list_(pa.float32())}
)


def _pose(values: list[float]) -> pa.Array:
    return pa.array([{"pose": values}], type=_POSE_TYPE)


def _state(offset: float) -> pa.Array:
    return pa.array(
        [
            {
                "qpos": np.arange(8, dtype=np.float32) + offset,
                "qvel": np.zeros(8, dtype=np.float32),
            }
        ],
        type=_STATE_TYPE,
    )


def _event(event_id: str, value: pa.Array) -> dict:
    return {"type": "INPUT", "id": event_id, "value": value, "metadata": {}}


@dataclass
class _Output:
    output_id: str
    value: pa.Array
    metadata: dict


class _FakeNode:
    def __init__(self, events: list[dict]) -> None:
        self.events = events
        self.outputs: list[_Output] = []

    def __iter__(self):
        return iter(self.events)

    def send_output(self, output_id, value, metadata=None) -> None:
        self.outputs.append(_Output(output_id, value, metadata or {}))


class _JointResolver:
    def get_driver(self, _qpos, _side):
        return np.zeros(7), 0.0


class _FakeKinematics:
    def __init__(self, sides: list[str]) -> None:
        self.setup = SimpleNamespace(
            sides=sides,
            joint_resolver=_JointResolver(),
            data=SimpleNamespace(qpos=np.zeros(20)),
        )
        self.targets: dict[str, np.ndarray] = {}
        self.sync_history: list[np.ndarray] = []
        self.solve_history: list[dict[str, np.ndarray]] = []
        self.measured_history: list[np.ndarray] = []
        self.clear_count = 0
        self.fk_bimanual_history: list[tuple[np.ndarray, np.ndarray]] = []
        self.fk_history: list[tuple[str, np.ndarray]] = []

    def sync(self, qpos: np.ndarray) -> None:
        self.sync_history.append(qpos.copy())

    def fk_bimanual(
        self, right: np.ndarray, left: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        self.fk_bimanual_history.append((right.copy(), left.copy()))
        return (
            np.array([10, 20, 30, 1, 0, 0, 0], dtype=np.float32),
            np.array([-10, 20, 30, 1, 0, 0, 0], dtype=np.float32),
        )

    def fk(self, side: str, qpos: np.ndarray) -> np.ndarray:
        self.fk_history.append((side, qpos.copy()))
        x = 10.0 if side == "right" else -10.0
        return np.array([x, 20, 30, 1, 0, 0, 0], dtype=np.float32)

    def set_target(self, side: str, pose: np.ndarray) -> None:
        self.targets[side] = np.asarray(pose).copy()

    def set_gripper(self, side: str, value: float) -> None:
        self.targets[f"{side}_gripper"] = np.array(value)

    def ready(self) -> bool:
        return all(side in self.targets for side in self.setup.sides)

    def update_measured_state(self, qpos: np.ndarray) -> None:
        self.measured_history.append(qpos.copy())

    def clear_measured_state(self) -> None:
        self.clear_count += 1

    def solve(self) -> np.ndarray:
        self.solve_history.append(
            {side: self.targets[side].copy() for side in self.setup.sides}
        )
        return np.arange(16, dtype=np.float32)


def _run(
    events: list[dict],
    *,
    target_mode: str = "absolute",
    sides: list[str] | None = None,
    times: list[float] | None = None,
) -> tuple[_FakeKinematics, _FakeNode]:
    selected_sides = sides or ["right", "left"]
    kinematics = _FakeKinematics(selected_sides)
    node = _FakeNode(events)
    args = argparse.Namespace(
        measured_state_timeout=0.1,
        relative_input_timeout=0.1,
        target_mode=target_mode,
    )
    clock = times or [index * 0.01 for index in range(len(events))]
    with (
        mock.patch.object(ik, "Kinematics", return_value=kinematics),
        mock.patch.object(ik, "setup_from_args", return_value=object()),
        mock.patch.object(ik, "ik_params_from_args", return_value=object()),
        mock.patch.object(ik.dora, "Node", return_value=node),
        mock.patch.object(ik.time, "monotonic", side_effect=clock),
        mock.patch.object(ik.time, "time_ns", return_value=123),
    ):
        ik._run(args)
    return kinematics, node


_RIGHT_REF = [1, 2, 3, 1, 0, 0, 0, 0.1]
_LEFT_REF = [-1, 2, 3, 1, 0, 0, 0, -0.1]


class IKNodeTest(unittest.TestCase):
    """Verify synchronization, gating, and relative calibration transitions."""

    def test_default_absolute_mode_needs_a_fresh_pair(self) -> None:
        """An unconnected active input preserves target-only mainline behavior."""
        kin, node = _run(
            [
                _event("target_right", _pose(_RIGHT_REF)),
                _event("target_right", _pose(_RIGHT_REF)),
                _event("target_left", _pose(_LEFT_REF)),
            ]
        )
        self.assertEqual(len(kin.solve_history), 1)
        self.assertEqual(
            [output.output_id for output in node.outputs],
            ["status", "position_right", "position_left"],
        )

    def test_absolute_mode_syncs_and_uses_cached_targets_on_release(self) -> None:
        """Sync mode blocks solving and performs one final measured-state sync."""
        kin, _ = _run(
            [
                _event("syncstate", pa.array([True])),
                _event("target_right", _pose(_RIGHT_REF)),
                _event("target_left", _pose(_LEFT_REF)),
                _event("state_right", _state(1.0)),
                _event("state_left", _state(2.0)),
                _event("syncstate", pa.array([False])),
            ]
        )
        self.assertEqual(len(kin.sync_history), 2)
        self.assertEqual(len(kin.solve_history), 1)
        np.testing.assert_allclose(kin.solve_history[0]["right"], _RIGHT_REF[:7])

    def test_active_false_blocks_until_a_new_pair_arrives(self) -> None:
        """Re-enabling output cannot complete a pair started while inactive."""
        kin, _ = _run(
            [
                _event("active", pa.array([False])),
                _event("target_right", _pose(_RIGHT_REF)),
                _event("target_left", _pose(_LEFT_REF)),
                _event("active", pa.array([True])),
                _event("target_right", _pose(_RIGHT_REF)),
                _event("target_left", _pose(_LEFT_REF)),
            ]
        )
        self.assertEqual(len(kin.solve_history), 1)

    def test_relative_mode_anchors_then_maps_the_next_pair(self) -> None:
        """A complete sync cycle anchors both arms to one measured snapshot."""
        right_next = [1.1, 1.8, 3.3, 1, 0, 0, 0, 0.4]
        left_next = [-0.9, 2.1, 2.8, 1, 0, 0, 0, -0.4]
        kin, _ = _run(
            [
                _event("active", pa.array([False])),
                _event("syncstate", pa.array([True])),
                _event("state_right", _state(1.0)),
                _event("state_left", _state(2.0)),
                _event("target_right", _pose(_RIGHT_REF)),
                _event("target_left", _pose(_LEFT_REF)),
                _event("active", pa.array([True])),
                _event("syncstate", pa.array([False])),
                _event("target_right", _pose(right_next)),
                _event("target_left", _pose(left_next)),
            ],
            target_mode="relative",
        )
        self.assertEqual(len(kin.fk_bimanual_history), 1)
        right_qpos, left_qpos = kin.fk_bimanual_history[0]
        np.testing.assert_allclose(right_qpos, np.arange(8) + 1.0)
        np.testing.assert_allclose(left_qpos, np.arange(8) + 2.0)
        self.assertEqual(len(kin.solve_history), 1)
        np.testing.assert_allclose(
            kin.solve_history[0]["right"][:3], [10.1, 19.8, 30.3], atol=1e-6
        )
        np.testing.assert_allclose(
            kin.solve_history[0]["left"][:3], [-9.9, 20.1, 29.8], atol=1e-6
        )

    def test_failed_relative_calibration_requires_another_sync_cycle(self) -> None:
        """Late inputs cannot silently unlock a failed clutch release."""
        events = [
            _event("syncstate", pa.array([True])),
            _event("state_right", _state(1.0)),
            _event("state_left", _state(2.0)),
            _event("target_right", _pose(_RIGHT_REF)),
            _event("target_left", _pose(_LEFT_REF)),
            _event("syncstate", pa.array([False])),
            _event("target_right", _pose(_RIGHT_REF)),
            _event("target_left", _pose(_LEFT_REF)),
            _event("syncstate", pa.array([True])),
            _event("state_right", _state(3.0)),
            _event("state_left", _state(4.0)),
            _event("target_right", _pose(_RIGHT_REF)),
            _event("target_left", _pose(_LEFT_REF)),
            _event("syncstate", pa.array([False])),
            _event("target_right", _pose(_RIGHT_REF)),
            _event("target_left", _pose(_LEFT_REF)),
        ]
        times = [0.0, 0.01, 0.02, 0.03, 0.04, 0.20, 0.21, 0.22]
        times += [0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30]

        kin, node = _run(events, target_mode="relative", times=times)

        self.assertEqual(len(kin.solve_history), 1)
        statuses = [
            output.value[0].as_py()
            for output in node.outputs
            if output.output_id == "status"
        ]
        self.assertEqual(
            statuses, ["ready", "relative_recalibration_required", "ready"]
        )

    def test_relative_single_arm_mode_needs_only_its_selected_side(self) -> None:
        """Single-arm calibration and output do not wait for the inactive arm."""
        kin, node = _run(
            [
                _event("syncstate", pa.array([True])),
                _event("state_right", _state(1.0)),
                _event("target_right", _pose(_RIGHT_REF)),
                _event("syncstate", pa.array([False])),
                _event("target_right", _pose(_RIGHT_REF)),
            ],
            target_mode="relative",
            sides=["right"],
        )
        self.assertEqual(len(kin.fk_history), 1)
        self.assertEqual(len(kin.solve_history), 1)
        self.assertNotIn("position_left", [output.output_id for output in node.outputs])


if __name__ == "__main__":
    unittest.main()
