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
from coreason_sentinel.models import CircuitBreakerTrigger, SentinelConfig


class TestOTELSpanIngestion:
    @pytest.fixture
    def mock_components(self) -> tuple[TelemetryIngestor, MagicMock]:
        config = SentinelConfig(
            agent_id="test-agent-otel",
            owner_email="test@example.com",
            phoenix_endpoint="http://localhost:6006",
            cost_per_1k_tokens=0.01,  # $0.01 per 1k tokens
        )
        circuit_breaker = MagicMock()
        spot_checker = MagicMock()
        baseline_provider = MagicMock()
        veritas_client = MagicMock()

        ingestor = TelemetryIngestor(config, circuit_breaker, spot_checker, baseline_provider, veritas_client)
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

    def test_process_span_edge_cases(self, mock_components: tuple[TelemetryIngestor, MagicMock]) -> None:
        """Test edge cases: negative duration, malformed/negative tokens."""
        ingestor, circuit_breaker = mock_components

        # 1. Negative Duration (Clock Skew)
        span_skew = OTELSpan(
            trace_id="skew",
            span_id="1",
            name="skewed",
            start_time_unix_nano=200,
            end_time_unix_nano=100,  # End before start
            attributes={},
        )
        ingestor.process_otel_span(span_skew)
        # Should NOT record latency
        for call in circuit_breaker.record_metric.call_args_list:
            assert call[0][0] != "latency"

        # 2. Malformed Token Attribute
        span_malformed = OTELSpan(
            trace_id="mal",
            span_id="2",
            name="bad_tokens",
            start_time_unix_nano=100,
            end_time_unix_nano=200,
            attributes={"llm.token_count.total": "undefined_value"},
        )
        # Should gracefully handle ValueError and NOT record token_count
        ingestor.process_otel_span(span_malformed)

        # 3. Negative Token Count (Logic check)
        span_neg = OTELSpan(
            trace_id="neg",
            span_id="3",
            name="neg_tokens",
            start_time_unix_nano=100,
            end_time_unix_nano=200,
            attributes={"llm.token_count.total": -50},
        )
        ingestor.process_otel_span(span_neg)

        # Verify: circuit_breaker.record_metric should NOT have been called with "token_count" for these spans.
        # Note: Latency might be called for malformed/neg spans because time is valid (100 -> 200).
        # We only check for token_count calls.

        # Filter calls for token_count
        token_calls = [c for c in circuit_breaker.record_metric.call_args_list if c[0][0] == "token_count"]

        # Expecting 0 calls for token_count across all 3 ingestions above
        assert len(token_calls) == 0, f"Unexpected token_count calls: {token_calls}"

    def test_complex_scenario_cost_trigger(self) -> None:
        """
        Complex Scenario:
        1. Setup Circuit Breaker with Cost Trigger (e.g., > $0.05).
        2. Ingest normal spans (under threshold).
        3. Ingest expensive span (over threshold).
        4. Verify Circuit Breaker trip logic is invoked.
        """
        # We need a REAL Circuit Breaker logic simulation or detailed mock behavior.
        # Since CircuitBreaker.check_triggers() is called, we want to ensure data flows correctly.
        # If we use the real Circuit Breaker class with a Mock Redis, we can test integration.
        # But here we are unit testing Ingestor. So we verify that Ingestor calculates Cost correctly
        # and passes it to RecordMetric, and calls CheckTriggers.

        # Let's verify the integration flow with a specific setup.
        trigger = CircuitBreakerTrigger(
            metric="cost", threshold=0.05, window_seconds=60, operator=">", aggregation_method="SUM"
        )
        config = SentinelConfig(
            agent_id="complex-agent",
            owner_email="test@example.com",
            phoenix_endpoint="http://localhost:6006",
            cost_per_1k_tokens=0.02,  # $0.02 per 1k
            triggers=[trigger],
        )

        circuit_breaker = MagicMock()
        spot_checker = MagicMock()
        baseline_provider = MagicMock()
        veritas_client = MagicMock()

        ingestor = TelemetryIngestor(config, circuit_breaker, spot_checker, baseline_provider, veritas_client)

        # 1. Normal Span: 1000 tokens -> $0.02 cost.
        span_normal = OTELSpan(
            trace_id="t1",
            span_id="s1",
            name="normal",
            start_time_unix_nano=0,
            end_time_unix_nano=100,
            attributes={"llm.token_count.total": 1000},
        )
        ingestor.process_otel_span(span_normal)
        circuit_breaker.record_metric.assert_any_call("cost", 0.02)
        circuit_breaker.check_triggers.assert_called()

        circuit_breaker.reset_mock()

        # 2. Expensive Span: 3000 tokens -> $0.06 cost.
        span_expensive = OTELSpan(
            trace_id="t2",
            span_id="s2",
            name="expensive",
            start_time_unix_nano=0,
            end_time_unix_nano=100,
            attributes={"llm.token_count.total": 3000},
        )
        ingestor.process_otel_span(span_expensive)
        circuit_breaker.record_metric.assert_any_call("cost", 0.06)
        circuit_breaker.check_triggers.assert_called()

    def test_process_span_third_token_attribute_variant(
        self, mock_components: tuple[TelemetryIngestor, MagicMock]
    ) -> None:
        """Test the third variant of token attribute: llm.usage.total_tokens"""
        ingestor, circuit_breaker = mock_components

        attributes = {"llm.usage.total_tokens": 123}

        span = OTELSpan(
            trace_id="abc",
            span_id="789",
            name="variant_call",
            start_time_unix_nano=100,
            end_time_unix_nano=200,
            attributes=attributes,
        )

        ingestor.process_otel_span(span)
        circuit_breaker.record_metric.assert_any_call("token_count", 123.0)
