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

"""Clutch-relative workspace target mapping."""

from __future__ import annotations

import numpy as np


def _normalize_pose(values: np.ndarray, size: int, label: str) -> np.ndarray:
    pose = np.asarray(values, dtype=np.float64)
    if pose.shape != (size,) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{label} must be a finite shape-({size},) array.")
    norm = float(np.linalg.norm(pose[3:7]))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError(f"{label} quaternion must be non-zero.")
    pose = pose.copy()
    pose[3:7] /= norm
    return pose


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    result = np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )
    return result / np.linalg.norm(result)


def _quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
    return quaternion * np.array([1.0, -1.0, -1.0, -1.0])


class RelativeTargetMapper:
    """Map source-pose deltas onto end-effector poses captured at clutch release."""

    def __init__(self) -> None:
        """Initialize without references."""
        self._references: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def reset(self) -> None:
        """Discard every source and end-effector reference."""
        self._references.clear()

    def anchor(
        self,
        source_right: np.ndarray | None,
        source_left: np.ndarray | None,
        fk_right: np.ndarray | None,
        fk_left: np.ndarray | None,
    ) -> None:
        """Atomically capture references for each supplied arm."""
        references: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for side, source, fk_pose in (
            ("right", source_right, fk_right),
            ("left", source_left, fk_left),
        ):
            if source is None and fk_pose is None:
                continue
            if source is None or fk_pose is None:
                raise ValueError(f"{side} source and FK references must be paired.")
            references[side] = (
                _normalize_pose(source, 8, f"{side} source reference"),
                _normalize_pose(fk_pose, 7, f"{side} FK reference"),
            )
        if not references:
            raise ValueError("At least one arm reference is required.")
        self._references = references

    def map(self, side: str, source_pose: np.ndarray) -> np.ndarray:
        """Apply one source delta and pass through its latest gripper value."""
        if side not in self._references:
            raise RuntimeError(f"No relative target reference for {side} arm.")

        source = _normalize_pose(source_pose, 8, f"{side} source pose")
        source_ref, fk_ref = self._references[side]
        if np.dot(source_ref[3:7], source[3:7]) < 0.0:
            source[3:7] *= -1.0

        delta_rotation = _quat_multiply(_quat_conjugate(source_ref[3:7]), source[3:7])
        target_rotation = _quat_multiply(fk_ref[3:7], delta_rotation)
        if np.dot(fk_ref[3:7], target_rotation) < 0.0:
            target_rotation *= -1.0

        return np.concatenate(
            [
                fk_ref[:3] + source[:3] - source_ref[:3],
                target_rotation,
                source[7:8],
            ]
        ).astype(np.float32)
