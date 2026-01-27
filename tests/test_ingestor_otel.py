# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_sentinel

import unittest
from unittest.mock import AsyncMock, MagicMock

import pytest

from coreason_sentinel.ingestor import TelemetryIngestorAsync
from coreason_sentinel.interfaces import OTELSpan
from coreason_sentinel.models import CircuitBreakerTrigger, SentinelConfig


@pytest.mark.asyncio
class TestOTELSpanIngestion(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.config = SentinelConfig(
            agent_id="test-agent-otel",
            owner_email="test@example.com",
            phoenix_endpoint="http://localhost:6006",
            cost_per_1k_tokens=0.01,  # $0.01 per 1k tokens
        )
        self.circuit_breaker = MagicMock()
        # Mock async methods
        self.circuit_breaker.record_metric = AsyncMock()
        self.circuit_breaker.check_triggers = AsyncMock()

        self.spot_checker = MagicMock()
        self.baseline_provider = MagicMock()
        self.veritas_client = MagicMock()

        self.ingestor = TelemetryIngestorAsync(
            self.config, self.circuit_breaker, self.spot_checker, self.baseline_provider, self.veritas_client
        )

    async def test_process_span_latency_calculation(self) -> None:
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

        await self.ingestor.process_otel_span(span)

        # Check latency metric
        self.circuit_breaker.record_metric.assert_any_call("latency", 1.0, None)
        self.circuit_breaker.check_triggers.assert_called_once()

    async def test_process_span_token_and_cost_extraction(self) -> None:
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

        await self.ingestor.process_otel_span(span)

        self.circuit_breaker.record_metric.assert_any_call("token_count", 2000.0, None)
        self.circuit_breaker.record_metric.assert_any_call("cost", 0.02, None)

    async def test_process_span_alternative_token_attributes(self) -> None:
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

        await self.ingestor.process_otel_span(span)
        self.circuit_breaker.record_metric.assert_any_call("token_count", 500.0, None)

    async def test_process_span_no_metrics(self) -> None:
        # Zero duration, no attributes
        span = OTELSpan(
            trace_id="abc", span_id="789", name="empty", start_time_unix_nano=100, end_time_unix_nano=100, attributes={}
        )

        await self.ingestor.process_otel_span(span)

        # Latency 0 might be recorded if > start, but here start==end
        # Logic: if end > start: record latency. So no latency recorded here.
        # No tokens, no cost.
        assert self.circuit_breaker.record_metric.call_count == 0
        self.circuit_breaker.check_triggers.assert_called_once()

    async def test_process_span_edge_cases(self) -> None:
        """Test edge cases: negative duration, malformed/negative tokens."""
        # 1. Negative Duration (Clock Skew)
        span_skew = OTELSpan(
            trace_id="skew",
            span_id="1",
            name="skewed",
            start_time_unix_nano=200,
            end_time_unix_nano=100,  # End before start
            attributes={},
        )
        await self.ingestor.process_otel_span(span_skew)
        # Should NOT record latency
        for call in self.circuit_breaker.record_metric.call_args_list:
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
        await self.ingestor.process_otel_span(span_malformed)

        # 3. Negative Token Count (Logic check)
        span_neg = OTELSpan(
            trace_id="neg",
            span_id="3",
            name="neg_tokens",
            start_time_unix_nano=100,
            end_time_unix_nano=200,
            attributes={"llm.token_count.total": -50},
        )
        await self.ingestor.process_otel_span(span_neg)

        # Verify: circuit_breaker.record_metric should NOT have been called with "token_count" for these spans.
        # Filter calls for token_count
        token_calls = [c for c in self.circuit_breaker.record_metric.call_args_list if c[0][0] == "token_count"]

        # Expecting 0 calls for token_count across all 3 ingestions above
        assert len(token_calls) == 0, f"Unexpected token_count calls: {token_calls}"

    async def test_complex_scenario_cost_trigger(self) -> None:
        """
        Complex Scenario:
        1. Setup Circuit Breaker with Cost Trigger (e.g., > $0.05).
        2. Ingest normal spans (under threshold).
        3. Ingest expensive span (over threshold).
        4. Verify Circuit Breaker trip logic is invoked.
        """
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
        circuit_breaker.record_metric = AsyncMock()
        circuit_breaker.check_triggers = AsyncMock()

        spot_checker = MagicMock()
        baseline_provider = MagicMock()
        veritas_client = MagicMock()

        ingestor = TelemetryIngestorAsync(config, circuit_breaker, spot_checker, baseline_provider, veritas_client)

        # 1. Normal Span: 1000 tokens -> $0.02 cost.
        span_normal = OTELSpan(
            trace_id="t1",
            span_id="s1",
            name="normal",
            start_time_unix_nano=0,
            end_time_unix_nano=100,
            attributes={"llm.token_count.total": 1000},
        )
        await ingestor.process_otel_span(span_normal)
        circuit_breaker.record_metric.assert_any_call("cost", 0.02, None)
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
        await ingestor.process_otel_span(span_expensive)
        circuit_breaker.record_metric.assert_any_call("cost", 0.06, None)
        circuit_breaker.check_triggers.assert_called()

    async def test_process_span_third_token_attribute_variant(self) -> None:
        """Test the third variant of token attribute: llm.usage.total_tokens"""
        attributes = {"llm.usage.total_tokens": 123}

        span = OTELSpan(
            trace_id="abc",
            span_id="789",
            name="variant_call",
            start_time_unix_nano=100,
            end_time_unix_nano=200,
            attributes=attributes,
        )

        await self.ingestor.process_otel_span(span)
        self.circuit_breaker.record_metric.assert_any_call("token_count", 123.0, None)

    async def test_process_span_sentiment_extraction(self) -> None:
        """Test extraction of sentiment metrics from gen_ai.prompt."""
        attributes = {"gen_ai.prompt": "This is a STOP request because it is WRONG"}

        span = OTELSpan(
            trace_id="s1",
            span_id="1",
            name="sentiment_check",
            start_time_unix_nano=100,
            end_time_unix_nano=200,
            attributes=attributes,
        )

        await self.ingestor.process_otel_span(span)

        # "STOP" and "WRONG" are default regex patterns
        # Only one metric per event is recorded for sentiment frustration
        self.circuit_breaker.record_metric.assert_any_call("sentiment_frustration_count", 1.0, None)

    async def test_process_span_sentiment_extraction_fallback(self) -> None:
        """Test extraction of sentiment metrics from llm.input_messages fallback."""
        attributes = {"llm.input_messages": "Please STOP now"}

        span = OTELSpan(
            trace_id="s2",
            span_id="2",
            name="sentiment_fallback",
            start_time_unix_nano=100,
            end_time_unix_nano=200,
            attributes=attributes,
        )

        await self.ingestor.process_otel_span(span)
        self.circuit_breaker.record_metric.assert_any_call("sentiment_frustration_count", 1.0, None)

    async def test_process_span_refusal_extraction(self) -> None:
        """Test extraction of refusal metrics from is_refusal attribute."""
        attributes = {"is_refusal": True}

        span = OTELSpan(
            trace_id="r1",
            span_id="1",
            name="refusal_check",
            start_time_unix_nano=100,
            end_time_unix_nano=200,
            attributes=attributes,
        )

        await self.ingestor.process_otel_span(span)
        self.circuit_breaker.record_metric.assert_any_call("refusal_count", 1.0, None)

    async def test_process_span_refusal_false(self) -> None:
        """Test extraction of refusal metrics when is_refusal is False."""
        attributes = {"is_refusal": False}

        span = OTELSpan(
            trace_id="r2",
            span_id="2",
            name="refusal_check_false",
            start_time_unix_nano=100,
            end_time_unix_nano=200,
            attributes=attributes,
        )

        await self.ingestor.process_otel_span(span)
        # Should NOT record refusal_count
        calls = [c[0][0] for c in self.circuit_breaker.record_metric.call_args_list]
        assert "refusal_count" not in calls

    async def test_process_span_no_custom_metrics(self) -> None:
        """Test processing a span with no sentiment or refusal signals."""
        attributes = {"gen_ai.prompt": "Hello world", "is_refusal": False}

        span = OTELSpan(
            trace_id="n1",
            span_id="1",
            name="normal_span",
            start_time_unix_nano=100,
            end_time_unix_nano=200,
            attributes=attributes,
        )

        await self.ingestor.process_otel_span(span)

        calls = [c[0][0] for c in self.circuit_breaker.record_metric.call_args_list]
        assert "sentiment_frustration_count" not in calls
        assert "refusal_count" not in calls
