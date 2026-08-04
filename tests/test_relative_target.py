"""Tests for clutch-relative workspace target mapping."""

from __future__ import annotations

import unittest

import numpy as np

from dora_openarm_kinematics.relative_target import RelativeTargetMapper


class RelativeTargetMapperTest(unittest.TestCase):
    """Verify reference capture and SE(3) delta composition."""

    def setUp(self) -> None:
        """Create a mapper and representative right-arm references."""
        self.mapper = RelativeTargetMapper()
        self.source_ref = np.array([1, 2, 3, 1, 0, 0, 0, -0.5], dtype=float)
        sqrt_half = np.sqrt(0.5)
        self.fk_ref = np.array([10, 20, 30, sqrt_half, sqrt_half, 0, 0], dtype=float)
        self.mapper.anchor(self.source_ref, None, self.fk_ref, None)

    def test_map_applies_delta_and_latest_gripper(self) -> None:
        """Translation is additive and rotation is composed after the FK anchor."""
        sqrt_half = np.sqrt(0.5)
        source = np.array(
            [1.1, 1.8, 3.3, sqrt_half, 0, 0, sqrt_half, 0.25], dtype=float
        )

        target = self.mapper.map("right", source)

        np.testing.assert_allclose(target[:3], [10.1, 19.8, 30.3], atol=1e-6)
        np.testing.assert_allclose(target[3:7], [0.5, 0.5, -0.5, 0.5], atol=1e-6)
        self.assertAlmostEqual(float(target[7]), 0.25)

    def test_quaternion_sign_does_not_change_target(self) -> None:
        """Equivalent source quaternion signs produce the same mapped pose."""
        source = self.source_ref.copy()
        flipped = source.copy()
        flipped[3:7] *= -1.0
        np.testing.assert_allclose(
            self.mapper.map("right", source),
            self.mapper.map("right", flipped),
            atol=1e-7,
        )

    def test_failed_anchor_keeps_previous_reference(self) -> None:
        """Reference replacement is atomic when validation fails."""
        expected = self.mapper.map("right", self.source_ref)
        invalid = self.source_ref.copy()
        invalid[3:7] = 0.0

        with self.assertRaises(ValueError):
            self.mapper.anchor(invalid, None, self.fk_ref, None)

        np.testing.assert_allclose(
            self.mapper.map("right", self.source_ref), expected, atol=1e-7
        )

    def test_reset_requires_a_new_anchor(self) -> None:
        """Mapping is unavailable after reset."""
        self.mapper.reset()
        with self.assertRaises(RuntimeError):
            self.mapper.map("right", self.source_ref)


if __name__ == "__main__":
    unittest.main()
