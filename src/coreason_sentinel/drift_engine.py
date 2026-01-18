# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_sentinel

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cosine
from scipy.stats import entropy


class DriftEngine:
    """
    The Statistician: Responsible for detecting drift in model inputs and outputs.
    """

    @staticmethod
    def compute_cosine_similarity(
        baseline: list[float] | NDArray[np.float64], live: list[float] | NDArray[np.float64]
    ) -> float:
        """
        Computes the Cosine Similarity between two vectors.
        Result range: [-1.0, 1.0]
        1.0: Identical direction
        0.0: Orthogonal
        -1.0: Opposite direction

        Note: scipy.spatial.distance.cosine returns Cosine DISTANCE (1 - similarity).
        So Similarity = 1 - Distance.
        """
        # Ensure inputs are numpy arrays
        u = np.asarray(baseline, dtype=np.float64)
        v = np.asarray(live, dtype=np.float64)

        if u.shape != v.shape:
            raise ValueError(f"Vectors must have same dimension. Got {u.shape} and {v.shape}")

        # Check for zero vectors to avoid division by zero in internal calculation
        u_is_zero = np.all(u == 0)
        v_is_zero = np.all(v == 0)

        if u_is_zero or v_is_zero:
            # If both are zero, they are identical (similarity 1.0)
            if u_is_zero and v_is_zero:
                return 1.0
            # If one is zero and the other is not, they are orthogonal (similarity 0.0)
            return 0.0

        # scipy.spatial.distance.cosine returns 1 - (u . v) / (||u|| ||v||)
        distance = cosine(u, v)
        return float(1.0 - distance)

    @staticmethod
    def compute_kl_divergence(
        baseline: list[float] | NDArray[np.float64],
        live: list[float] | NDArray[np.float64],
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

        # Use scipy.stats.entropy for efficient calculation
        # entropy(pk, qk) calculates S = sum(pk * log(pk / qk))
        return float(entropy(p, q))

    @classmethod
    def detect_vector_drift(cls, baseline_batch: list[list[float]], live_batch: list[list[float]]) -> float:
        """
        Detects drift between a batch of baseline vectors and a batch of live vectors.
        Computes the Cosine Similarity between the CENTROID (mean) of the baseline batch
        and the CENTROID of the live batch.

        Returns:
            drift_magnitude: 1.0 - similarity.
            0.0 means no drift (centroids match).
            1.0 means max drift (centroids orthogonal).
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

        # Distance = 1 - Similarity
        return 1.0 - similarity

    @classmethod
    def compute_relevance_drift(cls, query_embedding: list[float], response_embedding: list[float]) -> float:
        """
        Computes the Relevance Drift between a Query and a Response using Cosine Distance.
        Relevance Drift = 1.0 - Cosine Similarity.

        Args:
            query_embedding: The embedding vector of the user query.
            response_embedding: The embedding vector of the model response.

        Returns:
            float: Drift score. 0.0 means perfectly relevant (identical direction).
                   1.0 means orthogonal. > 1.0 means opposite.
        """
        # reuse the static method
        similarity = cls.compute_cosine_similarity(query_embedding, response_embedding)
        return 1.0 - similarity

    @staticmethod
    def compute_distribution_from_samples(samples: list[float], bin_edges: list[float]) -> list[float]:
        """
        Converts a list of raw samples into a probability distribution (PMF)
        based on the provided bin edges.

        Args:
            samples: List of raw values (e.g., output lengths).
            bin_edges: List of float values defining the bin edges.
                       Must be monotonically increasing.
                       Length must be len(output_distribution) + 1.

        Returns:
            list[float]: Probability of samples falling into each bin.
        """
        if not samples:
            # Return zeros, relying on KL smoothing (epsilon) later.
            return [0.0] * (len(bin_edges) - 1)

        # Use histogram to count samples in bins
        counts, _ = np.histogram(samples, bins=bin_edges)
        total = np.sum(counts)

        if total == 0:
            return [0.0] * (len(bin_edges) - 1)

        probabilities = counts / total
        return cast(list[float], probabilities.tolist())
