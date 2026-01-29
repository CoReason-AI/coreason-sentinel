import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from coreason_sentinel.circuit_breaker import CircuitBreaker
from coreason_sentinel.ingestor import TelemetryIngestorAsync
from coreason_sentinel.interfaces import (
    BaselineProviderProtocol,
    OTELSpan,
    VeritasClientProtocol,
    VeritasEvent,
)
from coreason_sentinel.models import SentinelConfig
from coreason_sentinel.spot_checker import SpotChecker


class TestIngestorUserContextEdgeCases(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.config = SentinelConfig(
            agent_id="test-agent",
            owner_email="test@example.com",
            phoenix_endpoint="http://localhost:6006",
            sampling_rate=1.0,
            triggers=[],
        )
        self.mock_cb = MagicMock(spec=CircuitBreaker)
        self.mock_cb.record_metric = AsyncMock()
        self.mock_cb.check_triggers = AsyncMock()

        self.mock_sc = MagicMock(spec=SpotChecker)
        self.mock_sc.should_sample.return_value = False
        self.mock_bp = MagicMock(spec=BaselineProviderProtocol)
        self.mock_veritas = MagicMock(spec=VeritasClientProtocol)

        self.ingestor = TelemetryIngestorAsync(self.config, self.mock_cb, self.mock_sc, self.mock_bp, self.mock_veritas)

    async def test_process_otel_span_none_context(self) -> None:
        span = OTELSpan(
            trace_id="t1", span_id="s1", name="test", start_time_unix_nano=100, end_time_unix_nano=200, attributes={}
        )
        await self.ingestor.process_otel_span(span, None)
        assert "enduser.id" not in span.attributes
        self.mock_cb.record_metric.assert_called()

    async def test_process_otel_span_empty_context(self) -> None:
        # Construct context with no ID/Groups (if allowed)
        # Using dummy values that are falsy if possible, or just checking attribute mapping logic
        # Since we use getattr with defaults, if we pass args that don't map, we get None.
        # But UserContext constructor requires args.
        # We can mock UserContext object.
        mock_context = MagicMock()
        # Mock getattr to return None
        # But getattr checks the object.
        # If we use a real UserContext but don't set the attributes (if possible in 0.4.x via some init?)
        # Let's just use a MagicMock that acts like a UserContext but has no attributes
        del mock_context.user_id
        del mock_context.sub
        del mock_context.groups
        del mock_context.permissions

        span = OTELSpan(
            trace_id="t1", span_id="s1", name="test", start_time_unix_nano=100, end_time_unix_nano=200, attributes={}
        )

        # We need to type ignore or cast because process_otel_span expects UserContext
        await self.ingestor.process_otel_span(span, mock_context)

        assert "enduser.id" not in span.attributes
        assert "enduser.role" not in span.attributes

    async def test_process_event_none_context(self) -> None:
        event = VeritasEvent(
            event_id="e1",
            timestamp="2024-01-01T00:00:00Z",
            agent_id="a1",
            session_id="s1",
            input_text="hi",
            output_text="bye",
            metrics={},
            metadata={},
        )
        # Should not crash
        await self.ingestor.process_event(event, None)
        self.mock_cb.check_triggers.assert_called_with(None)

    async def test_security_violation_none_context(self) -> None:
        span = OTELSpan(
            trace_id="t1",
            span_id="s1",
            name="test",
            start_time_unix_nano=100,
            end_time_unix_nano=200,
            attributes={"security_violation": True},
        )
        with patch("coreason_sentinel.ingestor.logger") as mock_logger:
            await self.ingestor.process_otel_span(span, None)
            mock_logger.critical.assert_called_with("SECURITY VIOLATION detected for User ID: unknown")

    async def test_security_violation_empty_context(self) -> None:
        span = OTELSpan(
            trace_id="t1", span_id="s1", name="test",
            start_time_unix_nano=100, end_time_unix_nano=200,
            attributes={"security_violation": True}
        )
        mock_ctx = MagicMock()
        del mock_ctx.user_id
        del mock_ctx.sub
        with patch("coreason_sentinel.ingestor.logger") as mock_logger:
            await self.ingestor.process_otel_span(span, mock_ctx) # type: ignore
            mock_logger.critical.assert_called_with("SECURITY VIOLATION detected for User ID: unknown")
