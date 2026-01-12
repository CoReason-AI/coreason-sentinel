# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_sentinel

from unittest.mock import MagicMock

import pytest

from coreason_sentinel.ingestor import TelemetryIngestor
from coreason_sentinel.interfaces import OTELSpan
from coreason_sentinel.models import SentinelConfig


class TestOTELSpanIngestion:
    @pytest.fixture
    def mock_components(self) -> tuple[TelemetryIngestor, MagicMock]:
        config = SentinelConfig(
            agent_id="test-agent-otel",
            cost_per_1k_tokens=0.01,  # $0.01 per 1k tokens
        )
        circuit_breaker = MagicMock()
        spot_checker = MagicMock()
        baseline_provider = MagicMock()

        ingestor = TelemetryIngestor(config, circuit_breaker, spot_checker, baseline_provider)
        return ingestor, circuit_breaker

    def test_process_span_latency_calculation(self, mock_components: tuple[TelemetryIngestor, MagicMock]) -> None:
        ingestor, circuit_breaker = mock_components

        # 1 second duration
        start = 1700000000_000_000_000  # ns
        end = 1700000001_000_000_000  # ns

        span = OTELSpan(
            trace_id="abc",
            span_id="123",
            name="test_span",
            start_time_unix_nano=start,
            end_time_unix_nano=end,
            attributes={},
        )

        ingestor.process_otel_span(span)

        # Check latency metric
        circuit_breaker.record_metric.assert_any_call("latency", 1.0)
        circuit_breaker.check_triggers.assert_called_once()

    def test_process_span_token_and_cost_extraction(self, mock_components: tuple[TelemetryIngestor, MagicMock]) -> None:
        ingestor, circuit_breaker = mock_components

        # 2000 tokens, cost rate 0.01 per 1k -> expected cost 0.02
        attributes = {"llm.token_count.total": 2000}

        span = OTELSpan(
            trace_id="abc",
            span_id="123",
            name="llm_call",
            start_time_unix_nano=100,
            end_time_unix_nano=200,
            attributes=attributes,
        )

        ingestor.process_otel_span(span)

        circuit_breaker.record_metric.assert_any_call("token_count", 2000.0)
        circuit_breaker.record_metric.assert_any_call("cost", 0.02)

    def test_process_span_alternative_token_attributes(
        self, mock_components: tuple[TelemetryIngestor, MagicMock]
    ) -> None:
        ingestor, circuit_breaker = mock_components

        # Check priority: llm.token_count.total > gen_ai.usage.total_tokens
        attributes = {"gen_ai.usage.total_tokens": 500}

        span = OTELSpan(
            trace_id="abc",
            span_id="456",
            name="gen_ai_call",
            start_time_unix_nano=100,
            end_time_unix_nano=200,
            attributes=attributes,
        )

        ingestor.process_otel_span(span)
        circuit_breaker.record_metric.assert_any_call("token_count", 500.0)

    def test_process_span_no_metrics(self, mock_components: tuple[TelemetryIngestor, MagicMock]) -> None:
        ingestor, circuit_breaker = mock_components

        # Zero duration, no attributes
        span = OTELSpan(
            trace_id="abc", span_id="789", name="empty", start_time_unix_nano=100, end_time_unix_nano=100, attributes={}
        )

        ingestor.process_otel_span(span)

        # Latency 0 might be recorded if > start, but here start==end
        # Logic: if end > start: record latency. So no latency recorded here.
        # No tokens, no cost.
        assert circuit_breaker.record_metric.call_count == 0
        circuit_breaker.check_triggers.assert_called_once()
