import numpy as np
import pytest

from coreason_sentinel.drift_engine import DriftEngine


class TestDriftEngine:
    def test_cosine_similarity_identical(self) -> None:
        """Test that identical vectors have similarity 1.0"""
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        sim = DriftEngine.compute_cosine_similarity(v1, v2)
        assert sim == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self) -> None:
        """Test that orthogonal vectors have similarity 0.0"""
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        sim = DriftEngine.compute_cosine_similarity(v1, v2)
        assert sim == pytest.approx(0.0)

    def test_cosine_similarity_opposite(self) -> None:
        """Test that opposite vectors have similarity -1.0"""
        v1 = [1.0, 0.0, 0.0]
        v2 = [-1.0, 0.0, 0.0]
        sim = DriftEngine.compute_cosine_similarity(v1, v2)
        assert sim == pytest.approx(-1.0)

    def test_cosine_similarity_dimension_mismatch(self) -> None:
        v1 = [1.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        with pytest.raises(ValueError, match="must have same dimension"):
            DriftEngine.compute_cosine_similarity(v1, v2)

    def test_cosine_similarity_zero_vectors(self) -> None:
        v1 = [0.0, 0.0]
        v2 = [1.0, 1.0]
        # Similarity involving zero vector is 0.0 usually
        assert DriftEngine.compute_cosine_similarity(v1, v2) == 0.0
        assert DriftEngine.compute_cosine_similarity(v1, v1) == 1.0

    def test_kl_divergence_identical(self) -> None:
        """KL Divergence of identical distributions should be 0.0"""
        p = [0.5, 0.5]
        q = [0.5, 0.5]
        kl = DriftEngine.compute_kl_divergence(p, q)
        assert kl == pytest.approx(0.0, abs=1e-9)

    def test_kl_divergence_different(self) -> None:
        """KL Divergence of different distributions should be > 0.0"""
        p = [0.9, 0.1]
        q = [0.1, 0.9]
        kl = DriftEngine.compute_kl_divergence(p, q)
        assert kl > 0.0
        # Manual Calc:
        # P=[0.9, 0.1], Q=[0.1, 0.9]
        # 0.9 * log(0.9/0.1) + 0.1 * log(0.1/0.9)
        # 0.9 * log(9) + 0.1 * log(1/9)
        # 0.9 * 2.197 - 0.1 * 2.197
        # 1.977 - 0.219 = 1.758
        assert kl == pytest.approx(1.7577, rel=1e-3)

    def test_kl_divergence_normalization(self) -> None:
        """Test that input distributions are normalized automatically"""
        p = [9, 1]  # Sums to 10
        q = [1, 9]  # Sums to 10
        # Should behave same as [0.9, 0.1] and [0.1, 0.9]
        kl = DriftEngine.compute_kl_divergence(p, q)
        assert kl == pytest.approx(1.7577, rel=1e-3)

    def test_kl_divergence_zero_handling(self) -> None:
        """Test that zeros don't cause nan/inf due to smoothing"""
        p = [1.0, 0.0]
        q = [0.5, 0.5]
        # P(x)=0 case: 0 * log(0/q) -> 0
        # P(x)=1 case: 1 * log(1/0.5) -> log(2) -> 0.693
        kl = DriftEngine.compute_kl_divergence(p, q)
        assert kl == pytest.approx(0.6931, rel=1e-3)

    def test_kl_divergence_mismatch(self) -> None:
        """Test mismatched dimensions"""
        p = [1.0, 0.0]
        q = [0.5, 0.5, 0.0]
        with pytest.raises(ValueError, match="Distributions must have same dimension"):
            DriftEngine.compute_kl_divergence(p, q)

    def test_detect_vector_drift_no_drift(self) -> None:
        # Centroids are identical
        baseline = [[1.0, 0.0], [0.0, 1.0]]  # Mean: [0.5, 0.5]
        live = [[0.5, 0.5], [0.5, 0.5]]  # Mean: [0.5, 0.5]
        drift = DriftEngine.detect_vector_drift(baseline, live)
        assert drift == pytest.approx(0.0)

    def test_detect_vector_drift_high_drift(self) -> None:
        # Centroids are orthogonal
        baseline = [[1.0, 0.0], [1.0, 0.0]]  # Mean: [1.0, 0.0]
        live = [[0.0, 1.0], [0.0, 1.0]]  # Mean: [0.0, 1.0]
        drift = DriftEngine.detect_vector_drift(baseline, live)
        # Cosine distance should be 1.0
        assert drift == pytest.approx(1.0)

    def test_detect_vector_drift_empty(self) -> None:
        with pytest.raises(ValueError, match="Batches cannot be empty"):
            DriftEngine.detect_vector_drift([], [[1.0]])
        with pytest.raises(ValueError, match="Batches cannot be empty"):
            DriftEngine.detect_vector_drift([[1.0]], [])

    # --- Edge Cases & Complex Scenarios ---

    def test_detect_vector_drift_different_batch_sizes(self) -> None:
        """Test comparing a large baseline to a small live batch."""
        # Baseline: 100 vectors clustering around [1.0, 0.0]
        # Live: 10 vectors clustering around [1.0, 0.0]
        # Should have near zero drift.
        baseline = [[1.0, 0.0]] * 100
        live = [[1.0, 0.0]] * 10
        drift = DriftEngine.detect_vector_drift(baseline, live)
        assert drift == pytest.approx(0.0)

        # Baseline: 100 vectors at [1,0], Live 10 vectors at [0,1]
        baseline = [[1.0, 0.0]] * 100
        live = [[0.0, 1.0]] * 10
        drift = DriftEngine.detect_vector_drift(baseline, live)
        assert drift == pytest.approx(1.0)

    def test_detect_vector_drift_high_dimensionality(self) -> None:
        """Test with high-dimensional vectors (e.g., 1536 for embeddings)."""
        dim = 1536
        # Generate random vectors
        np.random.seed(42)
        v1 = np.random.rand(dim).tolist()
        v2 = np.random.rand(dim).tolist()  # Likely orthogonalish, but let's just check it runs

        # Exact duplicate batch
        baseline = [v1]
        live = [v1]
        drift = DriftEngine.detect_vector_drift(baseline, live)
        assert drift == pytest.approx(0.0)

        # Different vectors
        baseline = [v1]
        live = [v2]
        drift = DriftEngine.detect_vector_drift(baseline, live)
        # Should be a float between 0 and 2
        assert 0.0 <= drift <= 2.0

    def test_cosine_similarity_magnitude_independence(self) -> None:
        """Test that scaling a vector does not change its cosine similarity."""
        v1 = [1.0, 2.0, 3.0]
        v2 = [100.0, 200.0, 300.0]  # Same direction, scaled
        sim = DriftEngine.compute_cosine_similarity(v1, v2)
        assert sim == pytest.approx(1.0)

        v3 = [-1.0, -2.0, -3.0]  # Opposite direction
        sim = DriftEngine.compute_cosine_similarity(v1, v3)
        assert sim == pytest.approx(-1.0)

    def test_kl_divergence_extreme_values(self) -> None:
        """Test KL Divergence with extremely small probabilities."""
        p = [1e-20, 1.0]
        q = [1.0, 1e-20]
        # Should not raise math error and return a high divergence
        kl = DriftEngine.compute_kl_divergence(p, q)
        assert kl > 10.0  # Expect large divergence

    def test_nan_inputs(self) -> None:
        """Test behavior with NaN inputs. Should likely propagate NaN or raise error."""
        # Current implementation relies on scipy/numpy.
        # Generally, it's safer if it raises an error or returns NaN, but we should know which.
        v1 = [np.nan, 1.0]
        v2 = [1.0, 1.0]
        # Cosine of NaN usually results in NaN
        sim = DriftEngine.compute_cosine_similarity(v1, v2)
        assert np.isnan(sim)
