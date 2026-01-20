# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_sentinel

import pytest

from coreason_sentinel.drift_engine import DriftEngine


class TestDriftEngineRelevance:
    def test_compute_relevance_drift_identical(self) -> None:
        """
        Test that identical vectors have 0.0 relevance drift.
        """
        v1 = [1.0, 0.0]
        v2 = [1.0, 0.0]
        drift = DriftEngine.compute_relevance_drift(v1, v2)
        assert drift == 0.0

    def test_compute_relevance_drift_orthogonal(self) -> None:
        """
        Test that orthogonal vectors have 1.0 relevance drift.
        """
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        drift = DriftEngine.compute_relevance_drift(v1, v2)
        assert drift == 1.0

    def test_compute_relevance_drift_opposite(self) -> None:
        """
        Test that opposite vectors have 2.0 relevance drift.
        (Cosine Similarity -1 => Distance 1 - (-1) = 2)
        """
        v1 = [1.0, 0.0]
        v2 = [-1.0, 0.0]
        drift = DriftEngine.compute_relevance_drift(v1, v2)
        assert drift == 2.0

    def test_compute_relevance_drift_dimension_mismatch(self) -> None:
        """
        Test that mismatched dimensions raise ValueError.
        """
        v1 = [1.0, 0.0]
        v2 = [1.0, 0.0, 1.0]
        with pytest.raises(ValueError, match="must have same dimension"):
            DriftEngine.compute_relevance_drift(v1, v2)

    def test_compute_relevance_drift_zero_vectors(self) -> None:
        """
        Test behavior with zero vectors.
        """
        v1 = [0.0, 0.0]
        v2 = [1.0, 0.0]
        # Implementation details: compute_cosine_similarity returns 0.0 similarity if one is zero
        # So drift = 1 - 0 = 1.0
        drift = DriftEngine.compute_relevance_drift(v1, v2)
        assert drift == 1.0

        # Both zero -> similarity 1.0 -> drift 0.0
        drift_both_zero = DriftEngine.compute_relevance_drift(v1, v1)
        assert drift_both_zero == 0.0
