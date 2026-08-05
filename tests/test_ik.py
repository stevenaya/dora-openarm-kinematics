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

"""Tests for measured-state freshness in the IK node."""

from __future__ import annotations

import unittest

from dora_openarm_kinematics.ik import _has_fresh_state


class FreshStateTest(unittest.TestCase):
    """Verify active-arm state completeness and freshness."""

    def test_active_sides_must_all_be_fresh(self) -> None:
        """Require fresh state for every arm selected by the node mode."""
        cases = [
            ({"right": 1.0}, ["right", "left"], 1.0, False),
            ({"right": 1.0, "left": 1.02}, ["right", "left"], 1.03, True),
            ({"right": 2.0}, ["right"], 2.01, True),
            ({"right": 3.0, "left": 3.2}, ["right", "left"], 3.2, False),
        ]
        for received_at, sides, now, expected in cases:
            with self.subTest(sides=sides, now=now):
                self.assertEqual(
                    _has_fresh_state(received_at, sides, now, 0.1),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
