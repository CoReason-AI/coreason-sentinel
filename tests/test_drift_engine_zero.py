# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_sentinel

import unittest

import numpy as np
import pytest

from coreason_sentinel.drift_engine import DriftEngine


class TestDriftEngineZeroVectors(unittest.TestCase):
    def test_compute_cosine_similarity_both_zero(self) -> None:
        """
        Verify that similarity of two zero vectors is handled definedly.
        Current implementation returns 1.0 (Identical).
        """
        v1 = [0.0, 0.0]
        v2 = [0.0, 0.0]

        sim = DriftEngine.compute_cosine_similarity(v1, v2)
        self.assertEqual(sim, 1.0)

    def test_compute_cosine_similarity_one_zero(self) -> None:
        """
        Verify that similarity of one zero vector and one non-zero vector is 0.0.
        """
        v1 = [0.0, 0.0]
        v2 = [1.0, 0.0]

        sim = DriftEngine.compute_cosine_similarity(v1, v2)
        self.assertEqual(sim, 0.0)

    def test_detect_vector_drift_zero_baseline(self) -> None:
        """
        Verify behavior when baseline batch centroid is zero vector.
        """
        baseline = [[0.0, 0.0], [0.0, 0.0]]
        live = [[1.0, 0.0]]

        # Centroid of baseline is [0, 0].
        # Centroid of live is [1, 0].
        # Sim([0,0], [1,0]) -> 0.0.
        # Drift = 1.0 - 0.0 = 1.0.

        drift = DriftEngine.detect_vector_drift(baseline, live)
        self.assertEqual(drift, 1.0)

    def test_detect_vector_drift_both_zero(self) -> None:
        """
        Verify behavior when both batches are zero vectors.
        """
        baseline = [[0.0, 0.0]]
        live = [[0.0, 0.0]]

        # Sim([0,0], [0,0]) -> 1.0.
        # Drift = 1.0 - 1.0 = 0.0.

        drift = DriftEngine.detect_vector_drift(baseline, live)
        self.assertEqual(drift, 0.0)
