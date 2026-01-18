# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_sentinel

from typing import cast

from coreason_sentinel.drift_engine import DriftEngine
from coreason_sentinel.interfaces import BaselineProviderProtocol, VeritasEvent
from coreason_sentinel.metric_store import MetricStore
from coreason_sentinel.models import SentinelConfig
from coreason_sentinel.utils.logger import logger


class DriftMonitor:
    """
    Orchestrates Drift Detection.
    Connects Events -> DriftEngine -> MetricStore.
    """

    def __init__(
        self,
        config: SentinelConfig,
        baseline_provider: BaselineProviderProtocol,
        metric_store: MetricStore,
    ):
        self.config = config
        self.baseline_provider = baseline_provider
        self.metric_store = metric_store

    def process_event(self, event: VeritasEvent) -> None:
        """
        Runs all drift detection checks for a single event.
        Records metrics if drift is calculated.
        """
        logger.info(f"Processing drift for event {event.event_id}")

        self._check_vector_drift(event)
        self._check_output_drift(event)
        self._check_relevance_drift(event)

    def _check_vector_drift(self, event: VeritasEvent) -> None:
        """
        Checks for embedding drift (Cosine Similarity of centroids).
        """
        embedding = event.metadata.get("embedding")
        if not embedding or not isinstance(embedding, list):
            return

        try:
            baselines = self.baseline_provider.get_baseline_vectors(event.agent_id)
            if baselines:
                # Live batch is just the current event embedding wrapped in a list
                # Cast to list[list[float]] assuming valid input
                live_batch = [cast(list[float], embedding)]
                drift_score = DriftEngine.detect_vector_drift(baselines, live_batch)

                self._record(event.agent_id, "vector_drift", drift_score)
        except Exception as e:
            logger.error(f"Failed to process vector drift detection: {e}")

    def _check_output_drift(self, event: VeritasEvent) -> None:
        """
        Checks for output length drift (KL Divergence).
        """
        # 1. Determine Output Length
        output_length = 0.0
        if "completion_tokens" in event.metrics:
            output_length = float(event.metrics["completion_tokens"])
        elif "token_count" in event.metrics:
            output_length = float(event.metrics["token_count"])
        else:
            # Fallback
            output_length = float(len(event.output_text.split()))

        # 2. Record this metric so it's in the sliding window
        self._record(event.agent_id, "output_length", output_length)

        try:
            # 3. Retrieve Baseline Distribution
            result = self.baseline_provider.get_baseline_output_length_distribution(event.agent_id)
            if not result:
                return
            baseline_dist, bin_edges = result

            if not baseline_dist or not bin_edges:
                return

            # 4. Construct Live Distribution
            recent_samples = self.metric_store.get_recent_values(
                event.agent_id, "output_length", limit=self.config.drift_sample_window
            )
            if not recent_samples:
                return

            live_dist = DriftEngine.compute_distribution_from_samples(recent_samples, bin_edges)

            # 5. Compute KL Divergence
            kl_divergence = DriftEngine.compute_kl_divergence(baseline_dist, live_dist)
            self._record(event.agent_id, "output_drift_kl", kl_divergence)

        except (AttributeError, NotImplementedError) as e:
            logger.debug(f"Baseline provider does not support output drift: {e}")
        except Exception as e:
            logger.error(f"Failed to process output drift detection: {e}")

    def _check_relevance_drift(self, event: VeritasEvent) -> None:
        """
        Checks for relevance drift (Query vs Response).
        """
        query_embedding = event.metadata.get("query_embedding")
        response_embedding = event.metadata.get("response_embedding")

        if (
            query_embedding
            and isinstance(query_embedding, list)
            and response_embedding
            and isinstance(response_embedding, list)
        ):
            try:
                # Casts assuming valid input from metadata
                q_emb = cast(list[float], query_embedding)
                r_emb = cast(list[float], response_embedding)

                relevance_drift = DriftEngine.compute_relevance_drift(q_emb, r_emb)
                self._record(event.agent_id, "relevance_drift", relevance_drift)
            except Exception as e:
                logger.error(f"Failed to process relevance drift detection: {e}")

    def _record(self, agent_id: str, metric_name: str, value: float) -> None:
        """
        Helper to record metric to store.
        Uses a default window of 1 hour (3600s) unless triggers override it.
        """
        max_window = self.config.get_max_window_for_metric(metric_name)
        self.metric_store.record_metric(agent_id, metric_name, value, retention_window=max_window)
