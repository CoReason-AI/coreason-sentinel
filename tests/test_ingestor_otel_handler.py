import unittest
from typing import Any
from unittest.mock import MagicMock

from coreason_sentinel.circuit_breaker import CircuitBreaker
from coreason_sentinel.handlers.otel_handler import OtelIngestionHandler
from coreason_sentinel.interfaces import OTELSpan
from coreason_sentinel.metric_store import MetricStore
from coreason_sentinel.models import SentinelConfig
from coreason_sentinel.utils.metric_extractor import MetricExtractor


class TestOtelIngestionHandler(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SentinelConfig(
            agent_id="test_agent", owner_email="test@example.com", phoenix_endpoint="http://localhost", triggers=[]
        )
        self.mock_metric_store = MagicMock(spec=MetricStore)
        self.mock_circuit_breaker = MagicMock(spec=CircuitBreaker)
        self.mock_metric_extractor = MagicMock(spec=MetricExtractor)
        self.handler = OtelIngestionHandler(
            self.config, self.mock_metric_store, self.mock_circuit_breaker, self.mock_metric_extractor
        )

    def test_process_span_latency(self) -> None:
        span = OTELSpan(
            trace_id="t1",
            span_id="s1",
            name="op",
            start_time_unix_nano=1_000_000_000,
            end_time_unix_nano=1_500_000_000,  # 0.5s latency
            attributes={},
        )
        self.mock_metric_extractor.extract.return_value = {}

        self.handler.process_span(span)

        self.mock_metric_store.record_metric.assert_any_call("test_agent", "latency", 0.5, retention_window=3600)
        self.mock_circuit_breaker.check_triggers.assert_called_once()

    def test_process_span_tokens_and_cost(self) -> None:
        span = OTELSpan(
            trace_id="t1",
            span_id="s1",
            name="op",
            start_time_unix_nano=1,
            end_time_unix_nano=2,
            attributes={"llm.token_count.total": 1000},
        )
        self.mock_metric_extractor.extract.return_value = {}

        self.handler.process_span(span)

        # Check token count
        self.mock_metric_store.record_metric.assert_any_call("test_agent", "token_count", 1000.0, retention_window=3600)
        # Check cost: 1000 tokens * 0.002 per 1k = 0.002
        self.mock_metric_store.record_metric.assert_any_call("test_agent", "cost", 0.002, retention_window=3600)

    def test_process_span_custom_metrics(self) -> None:
        span = OTELSpan(
            trace_id="t1",
            span_id="s1",
            name="op",
            start_time_unix_nano=1,
            end_time_unix_nano=2,
            attributes={"gen_ai.prompt": "bad bot"},
        )
        self.mock_metric_extractor.extract.return_value = {"sentiment_frustration_count": 1.0}

        self.handler.process_span(span)

        self.mock_metric_extractor.extract.assert_called_with("bad bot", span.attributes)
        self.mock_metric_store.record_metric.assert_any_call(
            "test_agent", "sentiment_frustration_count", 1.0, retention_window=3600
        )

    def test_process_span_token_extraction_variants(self) -> None:
        # Test fallback keys
        attrs: dict[str, Any] = {"gen_ai.usage.total_tokens": 500}
        span = OTELSpan(
            trace_id="t1", span_id="s1", name="op", start_time_unix_nano=1, end_time_unix_nano=2, attributes=attrs
        )
        self.mock_metric_extractor.extract.return_value = {}

        self.handler.process_span(span)
        self.mock_metric_store.record_metric.assert_any_call("test_agent", "token_count", 500.0, retention_window=3600)

    def test_process_span_token_parse_error(self) -> None:
        attrs: dict[str, Any] = {"llm.token_count.total": "not_a_number"}
        span = OTELSpan(
            trace_id="t1", span_id="s1", name="op", start_time_unix_nano=1, end_time_unix_nano=2, attributes=attrs
        )
        self.mock_metric_extractor.extract.return_value = {}

        self.handler.process_span(span)
        # Should not record token_count
        # Assert record_metric not called with "token_count"
        for call in self.mock_metric_store.record_metric.call_args_list:
            self.assertNotEqual(call[0][1], "token_count")

    def test_extract_input_text_fallback(self) -> None:
        attrs: dict[str, Any] = {"llm.input_messages": "hello"}
        span = OTELSpan(
            trace_id="t1", span_id="s1", name="op", start_time_unix_nano=1, end_time_unix_nano=2, attributes=attrs
        )
        self.mock_metric_extractor.extract.return_value = {}

        self.handler.process_span(span)
        self.mock_metric_extractor.extract.assert_called_with("hello", attrs)

    def test_extract_input_text_none(self) -> None:
        attrs: dict[str, Any] = {}
        span = OTELSpan(
            trace_id="t1", span_id="s1", name="op", start_time_unix_nano=1, end_time_unix_nano=2, attributes=attrs
        )
        self.mock_metric_extractor.extract.return_value = {}

        self.handler.process_span(span)
        # Should call extract with empty string
        self.mock_metric_extractor.extract.assert_called_with("", attrs)
