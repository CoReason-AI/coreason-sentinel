# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_sentinel

from datetime import datetime

from coreason_sentinel.circuit_breaker import CircuitBreaker
from coreason_sentinel.drift_monitor import DriftMonitor
from coreason_sentinel.interfaces import VeritasClientProtocol, VeritasEvent
from coreason_sentinel.metric_store import MetricStore
from coreason_sentinel.models import SentinelConfig
from coreason_sentinel.spot_checker import SpotChecker
from coreason_sentinel.utils.logger import logger
from coreason_sentinel.utils.metric_extractor import MetricExtractor


class VeritasIngestionHandler:
    """
    Handles ingestion of events from Veritas (Data Source).
    Orchestrates Metrics, Spot Checks, and Drift Detection.
    """

    def __init__(
        self,
        config: SentinelConfig,
        veritas_client: VeritasClientProtocol,
        metric_store: MetricStore,
        circuit_breaker: CircuitBreaker,
        spot_checker: SpotChecker,
        drift_monitor: DriftMonitor,
        metric_extractor: MetricExtractor,
    ):
        self.config = config
        self.veritas_client = veritas_client
        self.metric_store = metric_store
        self.circuit_breaker = circuit_breaker
        self.spot_checker = spot_checker
        self.drift_monitor = drift_monitor
        self.metric_extractor = metric_extractor

    def ingest_since(self, since: datetime) -> int:
        """
        Polls Veritas for logs since the given timestamp and processes them.
        Returns the number of events processed.
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
                count += 1
            except Exception as e:
                logger.error(f"Failed to process event {event.event_id}: {e}")
                # Continue processing other events
                continue

        return count

    def process_event(self, event: VeritasEvent) -> None:
        """
        Processes a single telemetry event from Veritas.
        1. Record Metrics
        2. Spot Check (Audit)
        3. Drift Detection
        4. Check Triggers
        """
        logger.info(f"Processing event {event.event_id} for agent {event.agent_id}")

        # 1. Record Metrics for Circuit Breaker
        # Extract metrics from event.metrics
        for metric_name, value in event.metrics.items():
            if isinstance(value, (int, float)):
                self._record(event.agent_id, metric_name, float(value))

        # 1.5 Extract Custom Metrics (Refusal & Sentiment)
        custom_metrics = self.metric_extractor.extract(event.input_text, event.metadata)
        for metric_name, value in custom_metrics.items():
            self._record(event.agent_id, metric_name, value)

        # 2. Spot Check (Audit)
        # Combine event metadata with derived metrics for spot checking rules
        combined_metadata = event.metadata.copy()
        combined_metadata.update(custom_metrics)

        if self.spot_checker.should_sample(combined_metadata):
            conversation = {
                "input": event.input_text,
                "output": event.output_text,
                "metadata": combined_metadata,
            }
            grade = self.spot_checker.check_sample(conversation)
            if grade:
                self._record(event.agent_id, "faithfulness_score", grade.faithfulness_score)
                self._record(event.agent_id, "retrieval_precision_score", grade.retrieval_precision_score)
                self._record(event.agent_id, "safety_score", grade.safety_score)

        # 3. Drift Detection (Synchronous as per requirements)
        self.drift_monitor.process_event(event)

        # 4. Final trigger check after all metrics
        self.circuit_breaker.check_triggers()

    def _record(self, agent_id: str, metric_name: str, value: float) -> None:
        """
        Helper to record metrics using the metric store.
        """
        max_window = self.config.get_max_window_for_metric(metric_name)
        self.metric_store.record_metric(agent_id, metric_name, value, retention_window=max_window)
