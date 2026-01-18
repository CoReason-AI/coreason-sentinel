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

from coreason_sentinel.drift_engine import DriftEngine


class TestDriftEngine(unittest.TestCase):
    def test_compute_cosine_similarity_identical(self) -> None:
        """Test similarity for identical vectors is 1.0."""
        v1 = [1.0, 2.0, 3.0]
        v2 = [1.0, 2.0, 3.0]
        sim = DriftEngine.compute_cosine_similarity(v1, v2)
        self.assertAlmostEqual(sim, 1.0)

    def test_compute_cosine_similarity_orthogonal(self) -> None:
        """Test similarity for orthogonal vectors is 0.0."""
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        sim = DriftEngine.compute_cosine_similarity(v1, v2)
        self.assertAlmostEqual(sim, 0.0)

    def test_compute_cosine_similarity_opposite(self) -> None:
        """Test similarity for opposite vectors is -1.0."""
        v1 = [1.0, 1.0]
        v2 = [-1.0, -1.0]
        sim = DriftEngine.compute_cosine_similarity(v1, v2)
        self.assertAlmostEqual(sim, -1.0)

    def test_compute_cosine_similarity_zero_vector(self) -> None:
        """Test handling of zero vectors."""
        v1 = [0.0, 0.0]
        v2 = [1.0, 1.0]
        # Should be 0.0 (no similarity)
        sim = DriftEngine.compute_cosine_similarity(v1, v2)
        self.assertEqual(sim, 0.0)

        # Both zero
        sim = DriftEngine.compute_cosine_similarity([0.0], [0.0])
        self.assertEqual(sim, 1.0)  # Convention for both zero

    def test_kl_divergence_identical(self) -> None:
        """Test KL Divergence for identical distributions is 0.0."""
        p = [0.1, 0.9]
        q = [0.1, 0.9]
        div = DriftEngine.compute_kl_divergence(p, q)
        self.assertAlmostEqual(div, 0.0)

    def test_kl_divergence_different(self) -> None:
        """Test KL Divergence for different distributions."""
        p = [0.9, 0.1]
        q = [0.1, 0.9]
        div = DriftEngine.compute_kl_divergence(p, q)
        # Manually: 0.9*log(9) + 0.1*log(1/9)
        # = 0.9*2.197 - 0.1*2.197 = 0.8*2.197 approx 1.75
        self.assertTrue(div > 1.0)

    def test_kl_divergence_smoothing(self) -> None:
        """Test that zero probabilities don't crash (epsilon smoothing)."""
        p = [1.0, 0.0]
        q = [0.5, 0.5]
        # p has a zero, q doesn't.
        # But if q had a zero where p is non-zero, KL explodes.
        # Our implementation adds epsilon to both.
        div = DriftEngine.compute_kl_divergence(p, q)
        self.assertTrue(div >= 0.0)

    def test_detect_vector_drift(self) -> None:
        """Test batch drift detection."""
        baseline = [[1.0, 0.0], [1.0, 0.0]]  # Centroid [1.0, 0.0]
        live = [[0.0, 1.0], [0.0, 1.0]]  # Centroid [0.0, 1.0]
        # Orthogonal centroids -> Sim 0.0 -> Drift 1.0
        drift = DriftEngine.detect_vector_drift(baseline, live)
        self.assertAlmostEqual(drift, 1.0)

    def test_detect_vector_drift_no_drift(self) -> None:
        """Test batch drift detection with no drift."""
        baseline = [[1.0, 1.0], [2.0, 2.0]]
        live = [[1.0, 1.0], [2.0, 2.0]]
        drift = DriftEngine.detect_vector_drift(baseline, live)
        self.assertAlmostEqual(drift, 0.0)

    def test_detect_vector_drift_empty(self) -> None:
        """Test empty batch raises error."""
        with self.assertRaises(ValueError):
            DriftEngine.detect_vector_drift([], [[1.0]])

    def test_detect_vector_drift_mismatch_dim(self) -> None:
        """Test mismatched dimensions raises error."""
        baseline = [[1.0, 0.0]]
        live = [[1.0, 0.0, 0.0]]
        with self.assertRaises(ValueError):
            DriftEngine.detect_vector_drift(baseline, live)

    def test_detect_vector_drift_ragged_batch(self) -> None:
        """Test ragged batch (inconsistent dimensions inside a batch) raises error."""
        # numpy.array will fail or create object array if dimensions differ
        baseline = [[1.0, 0.0], [1.0]]
        live = [[1.0, 0.0]]
        # The drift engine tries to convert to float array, which should fail for ragged lists
        with self.assertRaises(ValueError):
            DriftEngine.detect_vector_drift(baseline, live)

    def test_kl_divergence_input_validation(self) -> None:
        """Test validation for KL divergence inputs."""
        # Negative probabilities
        with self.assertRaises(ValueError):
            DriftEngine.compute_kl_divergence([-0.1, 1.1], [0.5, 0.5])

        # Mismatched dimensions
        with self.assertRaises(ValueError):
            DriftEngine.compute_kl_divergence([0.5, 0.5], [0.5, 0.5, 0.0])

    def test_detect_vector_drift_array_conversion_error(self) -> None:
        """Test error when converting to numpy array fails."""
        # This one covers the "except (ValueError, TypeError) as e" block
        # Passing something that isn't a list of lists of numbers
        with self.assertRaises(ValueError):
            DriftEngine.detect_vector_drift("not_a_list", [[1.0]])  # type: ignore

    def test_detect_vector_drift_high_dim(self) -> None:
        """Test 3D array input (which should fail validation)."""
        # baseline is 3D: batch of 2D matrices instead of vectors
        baseline = [[[1.0]], [[1.0]]]
        live = [[[1.0]], [[1.0]]]
        with self.assertRaises(ValueError):
            DriftEngine.detect_vector_drift(baseline, live)  # type: ignore

    def test_distribution_samples_outside_bins(self) -> None:
        """Test when samples are all outside bin edges (counts sum to 0)."""
        samples = [10.0, 20.0]
        bin_edges = [0.0, 5.0]  # Samples outside range
        # bins: [0, 5]. 10 and 20 are ignored? np.histogram ignores outliers?
        # np.histogram behavior: values outside range are not counted.

        dist = DriftEngine.compute_distribution_from_samples(samples, bin_edges)
        # Should return list of 0.0s (len = len(edges)-1 = 1)
        self.assertEqual(dist, [0.0])
