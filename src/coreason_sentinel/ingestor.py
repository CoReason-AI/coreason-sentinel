# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_sentinel


import re
from typing import Dict

from coreason_sentinel.circuit_breaker import CircuitBreaker
from coreason_sentinel.drift_engine import DriftEngine
from coreason_sentinel.interfaces import BaselineProviderProtocol, VeritasEvent
from coreason_sentinel.models import SentinelConfig
from coreason_sentinel.spot_checker import SpotChecker
from coreason_sentinel.utils.logger import logger


class TelemetryIngestor:
    """
    The Listener: Orchestrates the monitoring pipeline.
    Routes events to Circuit Breaker, Spot Checker, and Drift Engine.
    """

    def __init__(
        self,
        config: SentinelConfig,
        circuit_breaker: CircuitBreaker,
        spot_checker: SpotChecker,
        baseline_provider: BaselineProviderProtocol,
    ):
        self.config = config
        self.circuit_breaker = circuit_breaker
        self.spot_checker = spot_checker
        self.baseline_provider = baseline_provider

    def process_event(self, event: VeritasEvent) -> None:
        """
        Processes a single telemetry event from Veritas.
        """
        logger.info(f"Processing event {event.event_id} for agent {event.agent_id}")

        # 1. Record Metrics for Circuit Breaker
        # Extract metrics from event.metrics
        for metric_name, value in event.metrics.items():
            if isinstance(value, (int, float)):
                self.circuit_breaker.record_metric(metric_name, float(value))

        # 1.5 Extract Custom Metrics (Refusal & Sentiment)
        custom_metrics = self._extract_custom_metrics(event)
        for metric_name, value in custom_metrics.items():
            self.circuit_breaker.record_metric(metric_name, value)

        # 2. Spot Check (Audit)
        conversation = {
            "input": event.input_text,
            "output": event.output_text,
            "metadata": event.metadata,
        }
        if self.spot_checker.should_sample(event.metadata):
            grade = self.spot_checker.check_sample(conversation)
            if grade:
                # If grade is low, we might want to trigger something.
                # For now, we just log (or maybe record a "low_quality" metric?)
                # PRD says: "If estimated quality drops... Sentinel effectively pulls the plug"
                # So we should record the score as a metric.
                self.circuit_breaker.record_metric("faithfulness_score", grade.faithfulness_score)
                self.circuit_breaker.record_metric("safety_score", grade.safety_score)

        # 3. Drift Detection (Vector)
        embedding = event.metadata.get("embedding")
        if embedding and isinstance(embedding, list):
            try:
                # Retrieve baselines
                baselines = self.baseline_provider.get_baseline_vectors(event.agent_id)
                if baselines:
                    # Calculate drift
                    # Live batch is just the current event embedding wrapped in a list
                    drift_score = DriftEngine.detect_vector_drift(baselines, [embedding])
                    self.circuit_breaker.record_metric("vector_drift", drift_score)
            except Exception as e:
                logger.error(f"Failed to process vector drift detection: {e}")

        # 4. Drift Detection (Output Length)
        try:
            self._process_output_drift(event)
        except Exception as e:
            logger.error(f"Failed to process output drift detection: {e}")

        # Final trigger check after all metrics
        self.circuit_breaker.check_triggers()

    def _extract_custom_metrics(self, event: VeritasEvent) -> Dict[str, float]:
        """
        Extracts custom metrics based on metadata flags and regex patterns.
        """
        metrics: Dict[str, float] = {}

        # 1. Refusal Detection
        if event.metadata.get("is_refusal"):
            metrics["refusal_count"] = 1.0

        # 2. Sentiment Detection (Regex)
        # We check the input_text for user frustration signals
        for pattern in self.config.sentiment_regex_patterns:
            try:
                if re.search(pattern, event.input_text, re.IGNORECASE):
                    metrics["sentiment_frustration_count"] = 1.0
                    break  # Count once per event
            except re.error as e:
                logger.error(f"Invalid regex pattern '{pattern}' in configuration: {e}")
                continue

        return metrics

    def _process_output_drift(self, event: VeritasEvent) -> None:
        """
        Detects drift in output length (token count) using KL Divergence.
        """
        # 1. Determine Output Length
        output_length = 0.0
        if "completion_tokens" in event.metrics:
            output_length = float(event.metrics["completion_tokens"])
        elif "token_count" in event.metrics:
            output_length = float(event.metrics["token_count"])
        else:
            # Fallback to crude approximation
            output_length = float(len(event.output_text.split()))

        # 2. Record this metric so it's in the sliding window
        self.circuit_breaker.record_metric("output_length", output_length)

        # 3. Retrieve Baseline Distribution
        try:
            baseline_dist, bin_edges = self.baseline_provider.get_baseline_output_length_distribution(event.agent_id)
        except (AttributeError, NotImplementedError):
            # Provider might not support it yet
            return

        if not baseline_dist or not bin_edges:
            return

        # 4. Construct Live Distribution
        # Fetch recent samples
        recent_samples = self.circuit_breaker.get_recent_values("output_length", self.config.drift_sample_window)
        if not recent_samples:
            return

        live_dist = DriftEngine.compute_distribution_from_samples(recent_samples, bin_edges)

        # 5. Compute KL Divergence
        try:
            kl_divergence = DriftEngine.compute_kl_divergence(baseline_dist, live_dist)
            self.circuit_breaker.record_metric("output_drift_kl", kl_divergence)
        except ValueError as e:
            logger.warning(f"Skipping KL calculation due to validation error: {e}")
