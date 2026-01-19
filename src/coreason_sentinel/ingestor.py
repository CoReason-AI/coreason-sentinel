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
from datetime import datetime
from typing import Any, Dict

from coreason_sentinel.circuit_breaker import CircuitBreaker
from coreason_sentinel.drift_engine import DriftEngine
from coreason_sentinel.interfaces import BaselineProviderProtocol, OTELSpan, VeritasClientProtocol, VeritasEvent
from coreason_sentinel.models import SentinelConfig
from coreason_sentinel.spot_checker import SpotChecker
from coreason_sentinel.utils.logger import logger


class TelemetryIngestor:
    """
    The Listener: Orchestrates the monitoring pipeline.

    The Omni-Ingestor ingests both OTEL Spans (real-time traces) and Veritas Logs (long-term data).
    It routes events to the Circuit Breaker for metric tracking and trigger evaluation,
    to the Spot Checker for auditing, and to the Drift Engine for statistical analysis.
    """

    def __init__(
        self,
        config: SentinelConfig,
        circuit_breaker: CircuitBreaker,
        spot_checker: SpotChecker,
        baseline_provider: BaselineProviderProtocol,
        veritas_client: VeritasClientProtocol,
    ):
        """
        Initializes the TelemetryIngestor.

        Args:
            config: Configuration for ingestion rules.
            circuit_breaker: Instance of CircuitBreaker to record metrics and check triggers.
            spot_checker: Instance of SpotChecker to audit samples.
            baseline_provider: Provider for baseline vectors and distributions for drift detection.
            veritas_client: Client to fetch historical/batched logs.
        """
        self.config = config
        self.circuit_breaker = circuit_breaker
        self.spot_checker = spot_checker
        self.baseline_provider = baseline_provider
        self.veritas_client = veritas_client

    def process_otel_span(self, span: OTELSpan) -> None:
        """
        Processes a single OpenTelemetry Span.

        Extracts Latency, Tokens, Cost, Sentiment, and Refusal metrics for Circuit Breaker monitoring.
        This is the real-time ingestion path.

        Args:
            span: The OpenTelemetry span object to process.
        """
        logger.info(f"Processing OTEL Span {span.span_id} - {span.name}")

        # 1. Calculate Latency (seconds)
        if span.end_time_unix_nano > span.start_time_unix_nano:
            latency_sec = (span.end_time_unix_nano - span.start_time_unix_nano) / 1e9
            self.circuit_breaker.record_metric("latency", latency_sec)

        attributes = span.attributes or {}

        # 2. Extract Token Counts
        # Try standard semantic conventions
        token_count = 0.0

        try:
            # "llm.token_count.total" (OpenLLMetry / common convention)
            if "llm.token_count.total" in attributes:
                token_count = float(attributes["llm.token_count.total"])
            # "gen_ai.usage.total_tokens" (OTEL Semantic Conventions for GenAI)
            elif "gen_ai.usage.total_tokens" in attributes:
                token_count = float(attributes["gen_ai.usage.total_tokens"])
            # "llm.usage.total_tokens" (Another variant)
            elif "llm.usage.total_tokens" in attributes:
                token_count = float(attributes["llm.usage.total_tokens"])
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse token count from span attributes: {e}")
            token_count = 0.0

        if token_count > 0:
            self.circuit_breaker.record_metric("token_count", token_count)

            # 3. Calculate Cost
            # Cost = (Tokens / 1000) * CostPer1k
            if self.config.cost_per_1k_tokens > 0:
                cost = (token_count / 1000.0) * self.config.cost_per_1k_tokens
                self.circuit_breaker.record_metric("cost", cost)

        # 4. Extract Custom Metrics (Refusal & Sentiment)
        # Extract input text from standard semantic conventions
        input_text = ""
        if "gen_ai.prompt" in attributes:
            input_text = str(attributes["gen_ai.prompt"])
        elif "llm.input_messages" in attributes:
            input_text = str(attributes["llm.input_messages"])
        # Add more fallbacks if needed, but gen_ai.prompt is the target.

        custom_metrics = self._extract_custom_metrics(input_text, attributes)
        for metric_name, value in custom_metrics.items():
            self.circuit_breaker.record_metric(metric_name, value)

        # 5. Check Triggers
        self.circuit_breaker.check_triggers()

    def ingest_from_veritas_since(self, since: datetime) -> int:
        """
        Polls Veritas for logs since the given timestamp and processes them.

        This is the batch ingestion path.

        Args:
            since: The timestamp to fetch logs from.

        Returns:
            int: The number of events successfully processed.
        """
        try:
            events = self.veritas_client.fetch_logs(self.config.agent_id, since)
        except Exception as e:
            logger.error(f"Failed to fetch logs from Veritas: {e}")
            return 0

        if not events:
            return 0

        count = 0
        for event in events:
            try:
                self.process_event(event)

                # Run drift detection synchronously as per "Drift Execution Model" requirement
                # PRD Note: "Drift is inevitable" - we must check it.
                self.process_drift(event)

                count += 1
            except Exception as e:
                logger.error(f"Failed to process event {event.event_id}: {e}")
                # Continue processing other events
                continue

        return count

    def process_event(self, event: VeritasEvent) -> None:
        """
        Processes a single telemetry event from Veritas.

        This path is lightweight: Metrics extraction and Spot Checking only.
        Drift detection is offloaded to process_drift().

        Args:
            event: The VeritasEvent to process.
        """
        logger.info(f"Processing event {event.event_id} for agent {event.agent_id}")

        # 1. Record Metrics for Circuit Breaker
        # Extract metrics from event.metrics
        for metric_name, value in event.metrics.items():
            if isinstance(value, (int, float)):
                self.circuit_breaker.record_metric(metric_name, float(value))

        # 1.5 Extract Custom Metrics (Refusal & Sentiment)
        custom_metrics = self._extract_custom_metrics(event.input_text, event.metadata)
        for metric_name, value in custom_metrics.items():
            self.circuit_breaker.record_metric(metric_name, value)

        # 2. Spot Check (Audit)
        # Combine event metadata with derived metrics for spot checking rules
        combined_metadata = event.metadata.copy()
        combined_metadata.update(custom_metrics)

        conversation = {
            "input": event.input_text,
            "output": event.output_text,
            "metadata": combined_metadata,
        }
        if self.spot_checker.should_sample(combined_metadata):
            grade = self.spot_checker.check_sample(conversation)
            if grade:
                # If grade is low, we might want to trigger something.
                # For now, we just log (or maybe record a "low_quality" metric?)
                # PRD says: "If estimated quality drops... Sentinel effectively pulls the plug"
                # So we should record the score as a metric.
                self.circuit_breaker.record_metric("faithfulness_score", grade.faithfulness_score)
                self.circuit_breaker.record_metric("retrieval_precision_score", grade.retrieval_precision_score)
                self.circuit_breaker.record_metric("safety_score", grade.safety_score)

        # Final trigger check after all metrics
        self.circuit_breaker.check_triggers()

    def process_drift(self, event: VeritasEvent) -> None:
        """
        Processes Drift Detection for a single event.

        Intended to be run asynchronously/in background.
        Calculates Vector, Output, and Relevance Drift.

        Args:
            event: The VeritasEvent to analyze for drift.

        Returns:
            None: This method records metrics into the Circuit Breaker but returns nothing.
        """
        logger.info(f"Processing drift for event {event.event_id}")

        # 1. Drift Detection (Vector)
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

        # 2. Drift Detection (Output Length)
        try:
            self._process_output_drift(event)
        except Exception as e:
            logger.error(f"Failed to process output drift detection: {e}")

        # 3. Drift Detection (Relevance - Query vs Response)
        query_embedding = event.metadata.get("query_embedding")
        response_embedding = event.metadata.get("response_embedding")

        if (
            query_embedding
            and isinstance(query_embedding, list)
            and response_embedding
            and isinstance(response_embedding, list)
        ):
            try:
                relevance_drift = DriftEngine.compute_relevance_drift(query_embedding, response_embedding)
                self.circuit_breaker.record_metric("relevance_drift", relevance_drift)
            except Exception as e:
                logger.error(f"Failed to process relevance drift detection: {e}")

        # Trigger check? Maybe not strictly necessary if metrics are passive,
        # but if drift violates a trigger, we should trip.
        self.circuit_breaker.check_triggers()

    def _extract_custom_metrics(self, input_text: str, metadata: Dict[str, Any]) -> Dict[str, float]:
        """
        Extracts custom metrics based on metadata flags and regex patterns.

        Args:
            input_text: The user input text.
            metadata: Event/Span metadata.

        Returns:
            Dict[str, float]: Extracted metrics (e.g., {"refusal_count": 1.0}).
        """
        metrics: Dict[str, float] = {}

        # 1. Refusal Detection
        if metadata.get("is_refusal"):
            metrics["refusal_count"] = 1.0

        # 2. Sentiment Detection (Regex)
        # We check the input_text for user frustration signals
        for pattern in self.config.sentiment_regex_patterns:
            try:
                if re.search(pattern, input_text, re.IGNORECASE):
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
