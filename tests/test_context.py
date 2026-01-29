import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from coreason_identity.models import UserContext

from coreason_sentinel.circuit_breaker import CircuitBreaker
from coreason_sentinel.ingestor import TelemetryIngestorAsync
from coreason_sentinel.interfaces import OTELSpan, VeritasEvent
from coreason_sentinel.models import SentinelConfig


@pytest.mark.asyncio
class TestContext(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.config = SentinelConfig(
            agent_id="test-agent",
            owner_email="test@example.com",
            phoenix_endpoint="http://localhost:6006",
            cost_per_1k_tokens=0.01,
        )
        self.circuit_breaker = MagicMock(spec=CircuitBreaker)
        self.circuit_breaker.record_metric = AsyncMock()
        self.circuit_breaker.check_triggers = AsyncMock()

        self.spot_checker = MagicMock()
        self.baseline_provider = MagicMock()
        self.veritas_client = MagicMock()

        self.ingestor = TelemetryIngestorAsync(
            self.config, self.circuit_breaker, self.spot_checker, self.baseline_provider, self.veritas_client
        )

    async def test_process_otel_span_with_user_context(self) -> None:
        """Test processing OTEL span with user context."""
        span = OTELSpan(
            trace_id="t1",
            span_id="s1",
            name="test",
            start_time_unix_nano=100,
            end_time_unix_nano=200,
            attributes={},
        )
        user_context = UserContext(user_id="user123", sub="user123", email="u@e.com", groups=["admin", "editor"])

        await self.ingestor.process_otel_span(span, user_context)

        # Verify attribute mapping
        assert span.attributes["enduser.id"] == "user123"
        assert span.attributes["enduser.role"] == "admin,editor"

        # Verify metric recording passes user_context
        # record_metric calls: latency, then check_triggers
        # latency call
        # 100ns = 1e-7 seconds
        self.circuit_breaker.record_metric.assert_any_call("latency", 1e-7, user_context)
        # triggers call
        self.circuit_breaker.check_triggers.assert_called_with(user_context)

    async def test_security_violation_trigger(self) -> None:
        """Test that security violation logs a critical alert."""
        span = OTELSpan(
            trace_id="t2",
            span_id="s2",
            name="test_sec",
            start_time_unix_nano=100,
            end_time_unix_nano=200,
            attributes={"security_violation": True},
        )
        user_context = UserContext(user_id="bad_actor", sub="bad_actor", email="bad@e.com")

        with patch("coreason_sentinel.ingestor.logger") as mock_logger:
            await self.ingestor.process_otel_span(span, user_context)
            mock_logger.critical.assert_called_with("SECURITY VIOLATION detected for User ID: bad_actor")

    async def test_process_event_with_user_context(self) -> None:
        """Test processing Veritas event with user context."""
        event = VeritasEvent(
            event_id="e1",
            timestamp="2024-01-01T00:00:00Z",
            agent_id="agent1",
            session_id="sess1",
            input_text="hi",
            output_text="bye",
            metrics={"latency": 0.5},
            metadata={},
        )
        user_context = UserContext(user_id="user456", sub="user456", email="u@e.com")

        await self.ingestor.process_event(event, user_context)

        self.circuit_breaker.record_metric.assert_any_call("latency", 0.5, user_context)
        self.circuit_breaker.check_triggers.assert_called_with(user_context)
