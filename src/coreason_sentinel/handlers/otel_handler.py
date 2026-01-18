# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_sentinel

from typing import Any

from coreason_sentinel.circuit_breaker import CircuitBreaker
from coreason_sentinel.interfaces import OTELSpan
from coreason_sentinel.metric_store import MetricStore
from coreason_sentinel.models import SentinelConfig
from coreason_sentinel.utils.logger import logger
from coreason_sentinel.utils.metric_extractor import MetricExtractor


class OtelIngestionHandler:
    """
    Handles ingestion of OpenTelemetry Spans.
    """

    def __init__(
        self,
        config: SentinelConfig,
        metric_store: MetricStore,
        circuit_breaker: CircuitBreaker,
        metric_extractor: MetricExtractor,
    ):
        self.config = config
        self.metric_store = metric_store
        self.circuit_breaker = circuit_breaker
        self.metric_extractor = metric_extractor

    def process_span(self, span: OTELSpan) -> None:
        """
        Processes a single OpenTelemetry Span.
        Extracts Latency, Tokens, Cost, Sentiment, and Refusal metrics.
        """
        logger.info(f"Processing OTEL Span {span.span_id} - {span.name}")
        # agent_id is not in span. Assuming it's in config for the single-tenant deployment model
        # or we might need to extract it from span attributes if multi-tenant.
        # Based on current CircuitBreaker design, agent_id is in config.
        agent_id = self.config.agent_id

        # 1. Calculate Latency (seconds)
        if span.end_time_unix_nano > span.start_time_unix_nano:
            latency_sec = (span.end_time_unix_nano - span.start_time_unix_nano) / 1e9
            self._record(agent_id, "latency", latency_sec)

        attributes = span.attributes or {}

        # 2. Extract Token Counts
        token_count = self._extract_token_count(attributes)

        if token_count > 0:
            self._record(agent_id, "token_count", token_count)

            # 3. Calculate Cost
            if self.config.cost_per_1k_tokens > 0:
                cost = (token_count / 1000.0) * self.config.cost_per_1k_tokens
                self._record(agent_id, "cost", cost)

        # 4. Extract Custom Metrics (Refusal & Sentiment)
        input_text = self._extract_input_text(attributes)
        custom_metrics = self.metric_extractor.extract(input_text, attributes)

        for metric_name, value in custom_metrics.items():
            self._record(agent_id, metric_name, value)

        # 5. Check Triggers
        # Triggers might depend on updated metrics.
        self.circuit_breaker.check_triggers()

    def _extract_token_count(self, attributes: dict[str, Any]) -> float:
        try:
            if "llm.token_count.total" in attributes:
                return float(attributes["llm.token_count.total"])
            if "gen_ai.usage.total_tokens" in attributes:
                return float(attributes["gen_ai.usage.total_tokens"])
            if "llm.usage.total_tokens" in attributes:
                return float(attributes["llm.usage.total_tokens"])
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse token count from span attributes: {e}")
        return 0.0

    def _extract_input_text(self, attributes: dict[str, Any]) -> str:
        if "gen_ai.prompt" in attributes:
            return str(attributes["gen_ai.prompt"])
        elif "llm.input_messages" in attributes:
            return str(attributes["llm.input_messages"])
        return ""

    def _record(self, agent_id: str, metric_name: str, value: float) -> None:
        """
        Helper to record metrics using the metric store.
        """
        max_window = self.config.get_max_window_for_metric(metric_name)
        self.metric_store.record_metric(agent_id, metric_name, value, retention_window=max_window)
