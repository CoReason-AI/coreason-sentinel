# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_sentinel

from typing import Any, List, Optional, Union, cast

import numpy as np
from coreason_identity.models import UserContext
from numpy.typing import NDArray
from scipy.spatial.distance import cosine
from scipy.special import rel_entr


class DriftEngine:
    """
    The Statistician: Responsible for detecting drift in model inputs and outputs.
    """

    @staticmethod
    def compute_cosine_similarity(
        baseline: Union[List[float], NDArray[np.float64]], live: Union[List[float], NDArray[np.float64]]
    ) -> float:
        """
        Computes the Cosine Similarity between two vectors.

        Result range: [-1.0, 1.0]
        1.0: Identical direction
        0.0: Orthogonal
        -1.0: Opposite direction

        Note: scipy.spatial.distance.cosine returns Cosine DISTANCE (1 - similarity).
        So Similarity = 1 - Distance.

        Args:
            baseline: The baseline vector.
            live: The live/observed vector.

        Returns:
            float: The cosine similarity between the two vectors.

        Raises:
            ValueError: If vectors have different dimensions.
        """
        # Ensure inputs are numpy arrays
        u = np.asarray(baseline, dtype=np.float64)
        v = np.asarray(live, dtype=np.float64)

        if u.shape != v.shape:
            raise ValueError(f"Vectors must have same dimension. Got {u.shape} and {v.shape}")

        # Check for zero vectors to avoid division by zero in internal calculation
        if np.all(u == 0) or np.all(v == 0):
            # Similarity is undefined or 0 for zero vectors depending on definition.
            # Usually we return 0.0 if one is zero and other is not, or 1.0 if both are zero?
            # Let's assume 0.0 for safety if undefined.
            if np.all(u == 0) and np.all(v == 0):
                return 1.0  # Both are "nothing", so identical?
            return 0.0

        distance = cosine(u, v)
        return float(1.0 - distance)

    @staticmethod
    def compute_kl_divergence(
        baseline: Union[List[float], NDArray[np.float64]],
        live: Union[List[float], NDArray[np.float64]],
        epsilon: float = 1e-10,
    ) -> float:
        """
        Computes the Kullback-Leibler (KL) Divergence between two probability distributions.
        KL(P || Q) = sum(P(x) * log(P(x) / Q(x)))

        Args:
            baseline (P): The reference distribution (ground truth / baseline).
            live (Q): The observed distribution (approximation / live).
            epsilon: Small smoothing factor to avoid division by zero.

        Returns:
            float: The divergence score (>= 0.0). 0.0 indicates identical distributions.

        Raises:
            ValueError: If distributions have different dimensions or contain negative values.
        """
        p = np.asarray(baseline, dtype=np.float64)
        q = np.asarray(live, dtype=np.float64)

        if p.shape != q.shape:
            raise ValueError(f"Distributions must have same dimension. Got {p.shape} and {q.shape}")

        if np.any(p < 0) or np.any(q < 0):
            raise ValueError("Probabilities cannot be negative")

        # Add epsilon to avoid zero probabilities and re-normalize
        p = p + epsilon
        q = q + epsilon

        p = p / np.sum(p)
        q = q / np.sum(q)

        # scipy.special.rel_entr computes element-wise P * log(P/Q)
        # Summing it gives KL Divergence
        return float(np.sum(rel_entr(p, q)))

    @classmethod
    def detect_vector_drift(cls, baseline_batch: List[List[float]], live_batch: List[List[float]]) -> float:
        """
        Detects drift between a batch of baseline vectors and a batch of live vectors.

        This implementation computes the Cosine Similarity between the
        CENTROID (mean) of the baseline batch and the CENTROID of the live batch.

        Args:
            baseline_batch: List of baseline vectors (embeddings).
            live_batch: List of live vectors (embeddings).

        Returns:
            float: Drift magnitude (1.0 - similarity).
                   0.0 means no drift (centroids match).
                   1.0 means max drift (centroids orthogonal).

        Raises:
            ValueError: If batches are empty or dimensions mismatch.
        """
        if not baseline_batch or not live_batch:
            raise ValueError("Batches cannot be empty")

        try:
            baseline_arr = np.array(baseline_batch, dtype=np.float64)
            live_arr = np.array(live_batch, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Failed to create array from batches: {e}. Ensure all vectors have consistent dimensions."
            ) from e

        if baseline_arr.ndim != 2 or live_arr.ndim != 2:
            raise ValueError("Input batches must be 2D arrays (List of Lists).")

        # Calculate Centroids
        baseline_centroid = np.mean(baseline_arr, axis=0)
        live_centroid = np.mean(live_arr, axis=0)

        similarity = cls.compute_cosine_similarity(baseline_centroid, live_centroid)

        # Convert similarity to a drift metric (Distance)
        # If similarity is 1.0, drift is 0.0
        # If similarity is 0.0, drift is 1.0
        # If similarity is -1.0, drift is 2.0 (but usually we care about distance 0-1 range for cosine distance)
        # Scipy cosine returns 0 to 2.

        # Re-using scipy cosine distance logic:
        # Distance = 1 - Similarity
        return 1.0 - similarity

    @classmethod
    def compute_relevance_drift(cls, query_embedding: List[float], response_embedding: List[float]) -> float:
        """
        Computes the Relevance Drift between a Query and a Response using Cosine Distance.
        Relevance Drift = 1.0 - Cosine Similarity.

        Args:
            query_embedding: The embedding vector of the user query.
            response_embedding: The embedding vector of the model response.

        Returns:
            float: Drift score. 0.0 means perfectly relevant (identical direction).
                   1.0 means orthogonal. > 1.0 means opposite.

        Raises:
            ValueError: If vectors have different dimensions.
        """
        # reuse the static method
        similarity = cls.compute_cosine_similarity(query_embedding, response_embedding)
        return 1.0 - similarity

    @staticmethod
    def compute_distribution_from_samples(samples: List[float], bin_edges: List[float]) -> List[float]:
        """
        Converts a list of raw samples into a probability distribution (PMF)
        based on the provided bin edges.

        Args:
            samples: List of raw values (e.g., output lengths).
            bin_edges: List of float values defining the bin edges.
                       Must be monotonically increasing.
                       Length must be len(output_distribution) + 1.

        Returns:
            List[float]: Probability of samples falling into each bin.
        """
        if not samples:
            # If no samples, return a uniform distribution or zeros?
            # Returning zeros might cause KL issues, but compute_kl_divergence handles epsilon.
            # However, standard practice is uniform or raising error.
            # Let's return zeros, relying on KL smoothing.
            return [0.0] * (len(bin_edges) - 1)

        counts, _ = np.histogram(samples, bins=bin_edges)
        total = np.sum(counts)
        if total == 0:
            return [0.0] * (len(bin_edges) - 1)

        probabilities = counts / total
        return cast(List[float], probabilities.tolist())

    @classmethod
    def detect_drift(cls, content: str, user_context: Optional[UserContext] = None, **kwargs: Any) -> float:
        """
        High-level interface for detecting drift with user context awareness.

        This method is a placeholder for the orchestration logic currently handled
        by TelemetryIngestor.process_drift, which uses:
        1. BaselineProvider to fetch cohort-specific baselines (via user_context.groups).
        2. DriftEngine.detect_vector_drift for the mathematical computation.

        Args:
            content: The raw input content.
            user_context: The user context for cohort selection.
            **kwargs: Additional arguments (e.g., embeddings, baselines).

        Returns:
            float: The drift score.

        Raises:
            NotImplementedError: As the logic is currently in Ingestor.
        """
        raise NotImplementedError(
            "Drift detection orchestration is currently handled in TelemetryIngestor.process_drift"
        )
