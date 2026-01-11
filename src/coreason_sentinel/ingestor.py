# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_sentinel


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

        # Check triggers immediately after update
        self.circuit_breaker.check_triggers()

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
                self.circuit_breaker.check_triggers()

        # 3. Drift Detection
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
                    self.circuit_breaker.check_triggers()
            except Exception as e:
                logger.error(f"Failed to process drift detection: {e}")
