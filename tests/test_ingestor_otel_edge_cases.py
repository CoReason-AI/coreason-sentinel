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


class TestOTELSpanIngestionEdgeCases:
    @pytest.fixture
    def mock_components(self) -> tuple[TelemetryIngestor, MagicMock]:
        config = SentinelConfig(
            agent_id="test-agent-otel-edge",
            owner_email="test@example.com",
            phoenix_endpoint="http://localhost:6006",
            cost_per_1k_tokens=0.01,
        )
        circuit_breaker = MagicMock()
        spot_checker = MagicMock()
        baseline_provider = MagicMock()
        veritas_client = MagicMock()

        ingestor = TelemetryIngestor(config, circuit_breaker, spot_checker, baseline_provider, veritas_client)
        return ingestor, circuit_breaker

    def test_non_string_prompt_list(self, mock_components: tuple[TelemetryIngestor, MagicMock]) -> None:
        """
        Test that a list-based prompt (e.g. list of messages) is converted to string
        and regex still finds the pattern.
        """
        ingestor, circuit_breaker = mock_components

        # "STOP" is inside the list strings
        attributes = {"gen_ai.prompt": ["User: Hello", "System: STOP that"]}

        span = OTELSpan(
            trace_id="edge1",
            span_id="1",
            name="list_prompt",
            start_time_unix_nano=100,
            end_time_unix_nano=200,
            attributes=attributes,
        )

        ingestor.process_otel_span(span)

        # str(["...", "STOP..."]) -> "['...', 'STOP...']"
        # Regex "STOP" should match.
        circuit_breaker.record_metric.assert_any_call("sentiment_frustration_count", 1.0)

    def test_attribute_precedence_masking(self, mock_components: tuple[TelemetryIngestor, MagicMock]) -> None:
        """
        Test that gen_ai.prompt takes precedence over llm.input_messages.
        If gen_ai.prompt is clean, but llm.input_messages has 'STOP', logic should use gen_ai.prompt
        and thus record NO sentiment.
        """
        ingestor, circuit_breaker = mock_components

        attributes = {
            "gen_ai.prompt": "Everything is fine here",
            "llm.input_messages": "Please STOP this madness",
        }

        span = OTELSpan(
            trace_id="edge2",
            span_id="2",
            name="precedence_check",
            start_time_unix_nano=100,
            end_time_unix_nano=200,
            attributes=attributes,
        )

        ingestor.process_otel_span(span)

        # Should NOT record sentiment because prioritized source is clean
        calls = [c[0][0] for c in circuit_breaker.record_metric.call_args_list]
        assert "sentiment_frustration_count" not in calls

    def test_refusal_truthiness_integer(self, mock_components: tuple[TelemetryIngestor, MagicMock]) -> None:
        """Test that integer 1 counts as True for refusal."""
        ingestor, circuit_breaker = mock_components

        attributes = {"is_refusal": 1}

        span = OTELSpan(
            trace_id="edge3",
            span_id="3",
            name="refusal_int",
            start_time_unix_nano=100,
            end_time_unix_nano=200,
            attributes=attributes,
        )

        ingestor.process_otel_span(span)
        circuit_breaker.record_metric.assert_any_call("refusal_count", 1.0)

    def test_refusal_truthiness_string_false(self, mock_components: tuple[TelemetryIngestor, MagicMock]) -> None:
        """
        Test behavior with string "False".
        Current implementation uses `if metadata.get("is_refusal"):`.
        "False" is truthy in Python.
        This test documents current behavior (it WILL record refusal).
        If this is undesirable, code must change.
        For now, we assert the CURRENT behavior to ensure we know if it changes.
        """
        ingestor, circuit_breaker = mock_components

        attributes = {"is_refusal": "False"}

        span = OTELSpan(
            trace_id="edge4",
            span_id="4",
            name="refusal_str_false",
            start_time_unix_nano=100,
            end_time_unix_nano=200,
            attributes=attributes,
        )

        ingestor.process_otel_span(span)

        # It SHOULD record refusal count because "False" is True in boolean context
        circuit_breaker.record_metric.assert_any_call("refusal_count", 1.0)

    def test_large_input_text(self, mock_components: tuple[TelemetryIngestor, MagicMock]) -> None:
        """Test processing with a large text payload to ensure no crashes."""
        ingestor, circuit_breaker = mock_components

        # Create 1MB string
        large_text = "word " * 200000 + "STOP"

        attributes = {"gen_ai.prompt": large_text}

        span = OTELSpan(
            trace_id="edge5",
            span_id="5",
            name="large_payload",
            start_time_unix_nano=100,
            end_time_unix_nano=200,
            attributes=attributes,
        )

        ingestor.process_otel_span(span)

        circuit_breaker.record_metric.assert_any_call("sentiment_frustration_count", 1.0)

    def test_missing_prompt_attributes(self, mock_components: tuple[TelemetryIngestor, MagicMock]) -> None:
        """Test span with NO prompt attributes handled gracefully."""
        ingestor, circuit_breaker = mock_components

        attributes = {"some.other.attr": "value"}

        span = OTELSpan(
            trace_id="edge6",
            span_id="6",
            name="no_prompt",
            start_time_unix_nano=100,
            end_time_unix_nano=200,
            attributes=attributes,
        )

        ingestor.process_otel_span(span)

        # No crash, no metrics (except latency)
        calls = [c[0][0] for c in circuit_breaker.record_metric.call_args_list]
        assert "sentiment_frustration_count" not in calls
