"""Tests for optional measured-state synchronization in the IK node."""

from __future__ import annotations

import sys
import types
import unittest

import numpy as np


if "dora" not in sys.modules:
    sys.modules["dora"] = types.ModuleType("dora")

if "pyarrow" not in sys.modules:
    pyarrow = types.ModuleType("pyarrow")
    pyarrow.Array = object
    pyarrow.float32 = lambda: object()
    pyarrow.list_ = lambda _value: object()
    pyarrow.struct = lambda _fields: object()
    sys.modules["pyarrow"] = pyarrow

from dora_openarm_kinematics.ik import (
    _BimanualPositionBuffer,
    _BimanualStateBuffer,
    _MeasuredState,
    _apply_state_feedback,
    _build_parser,
    _sync_before_solve,
)


class _FakeKinematics:
    def __init__(self) -> None:
        self.synced: list[np.ndarray] = []
        self.blended: list[tuple[np.ndarray, np.ndarray, float, float]] = []

    def sync(self, values: np.ndarray) -> None:
        self.synced.append(values.copy())

    def blend_state(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        blend: float,
        prediction_dt: float,
    ) -> None:
        self.blended.append(
            (qpos.copy(), qvel.copy(), blend, prediction_dt)
        )


class MeasuredSyncTest(unittest.TestCase):
    """Verify the active synchronization switch and helper."""

    def test_flag_is_disabled_by_default(self) -> None:
        """The command-line switch is opt-in."""
        parser = _build_parser()
        defaults = parser.parse_args([])
        self.assertFalse(defaults.sync_during_active)
        self.assertEqual(defaults.state_feedback_blend, 0.0)
        self.assertEqual(defaults.state_feedback_max_prediction_dt, 0.004)
        self.assertTrue(parser.parse_args(["--sync-during-active"]).sync_during_active)

    def test_disabled_sync_preserves_current_behavior(self) -> None:
        """Disabled mode does not add a pre-solve sync."""
        kin = _FakeKinematics()
        _sync_before_solve(kin, np.arange(16, dtype=np.float32), enabled=False)
        self.assertEqual(kin.synced, [])

    def test_enabled_sync_applies_latest_measurement_once(self) -> None:
        """Enabled mode applies one cached measurement per helper call."""
        kin = _FakeKinematics()
        measured = np.arange(16, dtype=np.float32)
        _sync_before_solve(kin, measured, enabled=True)

        self.assertEqual(len(kin.synced), 1)
        np.testing.assert_array_equal(kin.synced[0], measured)

    def test_enabled_sync_without_measurement_is_a_noop(self) -> None:
        """Enabled mode waits until a measurement has arrived."""
        kin = _FakeKinematics()
        _sync_before_solve(kin, None, enabled=True)
        self.assertEqual(kin.synced, [])


class BimanualPositionBufferTest(unittest.TestCase):
    """Verify measured arm positions are paired without an adapter node."""

    def test_waits_for_both_sides_and_preserves_order(self) -> None:
        """A pair is emitted in canonical right-then-left order."""
        buffer = _BimanualPositionBuffer()
        right = np.arange(8, dtype=np.float32)
        left = np.arange(10, 18, dtype=np.float32)

        self.assertIsNone(buffer.update("left", left))
        paired = buffer.update("right", right)

        np.testing.assert_array_equal(paired, np.concatenate([right, left]))

    def test_requires_a_fresh_sample_from_each_side(self) -> None:
        """Each emitted pair consumes one fresh sample from both arms."""
        buffer = _BimanualPositionBuffer()
        right = np.arange(8, dtype=np.float32)
        left = np.arange(10, 18, dtype=np.float32)

        self.assertIsNone(buffer.update("right", right))
        self.assertIsNotNone(buffer.update("left", left))
        self.assertIsNone(buffer.update("right", right + 1))
        self.assertIsNone(buffer.update("right", right + 2))
        paired = buffer.update("left", left + 1)

        np.testing.assert_array_equal(
            paired,
            np.concatenate([right + 2, left + 1]),
        )

    def test_rejects_invalid_per_arm_shape(self) -> None:
        """Per-arm measurements must contain all eight joint values."""
        buffer = _BimanualPositionBuffer()
        with self.assertRaisesRegex(ValueError, "8 values"):
            buffer.update("right", np.arange(7, dtype=np.float32))


class MeasuredStateFeedbackTest(unittest.TestCase):
    """Verify fresh q/dq pairing and bounded state prediction."""

    def test_state_buffer_pairs_position_velocity_and_latest_time(self) -> None:
        """A pair preserves each field and uses the newer receive time."""
        buffer = _BimanualStateBuffer()
        right_qpos = np.arange(8, dtype=np.float32)
        right_qvel = np.arange(10, 18, dtype=np.float32)
        left_qpos = np.arange(20, 28, dtype=np.float32)
        left_qvel = np.arange(30, 38, dtype=np.float32)

        self.assertIsNone(
            buffer.update("right", right_qpos, right_qvel, received_at=10.0)
        )
        paired = buffer.update(
            "left", left_qpos, left_qvel, received_at=10.002
        )

        assert paired is not None
        np.testing.assert_array_equal(
            paired.qpos, np.concatenate([right_qpos, left_qpos])
        )
        np.testing.assert_array_equal(
            paired.qvel, np.concatenate([right_qvel, left_qvel])
        )
        self.assertEqual(paired.received_at, 10.002)

    def test_feedback_blends_once_with_capped_prediction_age(self) -> None:
        """Feedback caps qvel extrapolation at the configured horizon."""
        kin = _FakeKinematics()
        state = _MeasuredState(
            qpos=np.arange(16, dtype=np.float32),
            qvel=np.arange(20, 36, dtype=np.float32),
            received_at=10.0,
        )

        consumed = _apply_state_feedback(
            kin,
            state,
            blend=0.5,
            max_prediction_dt=0.004,
            now=10.02,
        )

        self.assertTrue(consumed)
        self.assertEqual(len(kin.blended), 1)
        qpos, qvel, blend, prediction_dt = kin.blended[0]
        np.testing.assert_array_equal(qpos, state.qpos)
        np.testing.assert_array_equal(qvel, state.qvel)
        self.assertEqual(blend, 0.5)
        self.assertEqual(prediction_dt, 0.004)

    def test_disabled_feedback_is_exact_feedforward_noop(self) -> None:
        """A zero blend leaves the feedforward path untouched."""
        kin = _FakeKinematics()
        state = _MeasuredState(
            qpos=np.zeros(16, dtype=np.float32),
            qvel=np.zeros(16, dtype=np.float32),
            received_at=10.0,
        )

        consumed = _apply_state_feedback(
            kin,
            state,
            blend=0.0,
            max_prediction_dt=0.004,
            now=10.001,
        )

        self.assertFalse(consumed)
        self.assertEqual(kin.blended, [])


if __name__ == "__main__":
    unittest.main()
