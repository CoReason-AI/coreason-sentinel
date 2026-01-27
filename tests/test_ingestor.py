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
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from coreason_sentinel.circuit_breaker import CircuitBreaker
from coreason_sentinel.ingestor import TelemetryIngestor, TelemetryIngestorAsync
from coreason_sentinel.interfaces import (
    BaselineProviderProtocol,
    VeritasClientProtocol,
    VeritasEvent,
)
from coreason_sentinel.models import SentinelConfig
from coreason_sentinel.spot_checker import SpotChecker


@pytest.mark.asyncio
class TestTelemetryIngestorAsync(unittest.IsolatedAsyncioTestCase):
    async def test_lifecycle(self) -> None:
        config = SentinelConfig(
            agent_id="test-agent",
            owner_email="test@example.com",
            phoenix_endpoint="http://localhost:6006",
            sampling_rate=1.0,
            triggers=[],
        )
        mock_cb = MagicMock(spec=CircuitBreaker)
        mock_sc = MagicMock(spec=SpotChecker)
        mock_bp = MagicMock(spec=BaselineProviderProtocol)
        mock_veritas = MagicMock(spec=VeritasClientProtocol)

        async with TelemetryIngestorAsync(config, mock_cb, mock_sc, mock_bp, mock_veritas) as ingestor:
            assert ingestor is not None
            # Internal client should be active
            assert not ingestor._client.is_closed

        # Internal client should be closed
        assert ingestor._client.is_closed

    async def test_lifecycle_with_external_client(self) -> None:
        """Test initializing with an external client."""
        config = SentinelConfig(
            agent_id="test-agent",
            owner_email="test@example.com",
            phoenix_endpoint="http://localhost:6006",
            triggers=[],
        )
        mock_cb = MagicMock(spec=CircuitBreaker)
        mock_sc = MagicMock(spec=SpotChecker)
        mock_bp = MagicMock(spec=BaselineProviderProtocol)
        mock_veritas = MagicMock(spec=VeritasClientProtocol)
        external_client = httpx.AsyncClient()

        async with TelemetryIngestorAsync(
            config, mock_cb, mock_sc, mock_bp, mock_veritas, client=external_client
        ) as ingestor:
            assert ingestor._client is external_client
            assert not ingestor._internal_client

        # External client should NOT be closed by __aexit__
        assert not external_client.is_closed
        await external_client.aclose()

    async def test_ingestor_async_client_resurrection(self) -> None:
        """Test that the client is recreated if closed upon re-entry."""
        config = SentinelConfig(
            agent_id="test",
            owner_email="test@example.com",
            phoenix_endpoint="http://localhost:6006",
            triggers=[],
        )
        mock_cb = MagicMock(spec=CircuitBreaker)
        mock_sc = MagicMock(spec=SpotChecker)
        mock_bp = MagicMock(spec=BaselineProviderProtocol)
        mock_veritas = MagicMock(spec=VeritasClientProtocol)

        ingestor = TelemetryIngestorAsync(config, mock_cb, mock_sc, mock_bp, mock_veritas)

        # Close the default client manually
        await ingestor._client.aclose()
        assert ingestor._client.is_closed

        # Re-enter context - should trigger resurrection logic
        async with ingestor:
            assert not ingestor._client.is_closed
            assert ingestor._client is not None

    async def test_ingest_from_veritas_triggers_drift_logic(self) -> None:
        config = SentinelConfig(
            agent_id="test-agent",
            owner_email="test@example.com",
            phoenix_endpoint="http://localhost:6006",
            sampling_rate=1.0,
            triggers=[],
        )
        mock_cb = MagicMock(spec=CircuitBreaker)
        mock_cb.record_metric = AsyncMock()
        mock_cb.check_triggers = AsyncMock()

        mock_sc = MagicMock(spec=SpotChecker)
        mock_sc.should_sample.return_value = False
        mock_bp = MagicMock(spec=BaselineProviderProtocol)
        mock_veritas = MagicMock(spec=VeritasClientProtocol)

        ingestor = TelemetryIngestorAsync(config, mock_cb, mock_sc, mock_bp, mock_veritas)

        event = VeritasEvent(
            event_id="evt_drift_check",
            timestamp=datetime.now(timezone.utc),
            agent_id="test-agent",
            session_id="sess_1",
            input_text="Hello",
            output_text="World",
            metrics={"latency": 0.1},
            metadata={
                "query_embedding": [0.1, 0.2],
                "response_embedding": [0.1, 0.25],
                "embedding": [0.1, 0.2],
            },
        )

        with patch("anyio.to_thread.run_sync", side_effect=lambda func, *args: func(*args)):
            mock_veritas.fetch_logs.return_value = [event]
            mock_bp.get_baseline_vectors.return_value = [[0.1, 0.2]]

            count = await ingestor.ingest_from_veritas_since(datetime.now(timezone.utc))

        assert count == 1
        mock_cb.record_metric.assert_any_call("latency", 0.1, None)

        calls = [call.args[0] for call in mock_cb.record_metric.call_args_list]
        assert "relevance_drift" in calls
        assert "vector_drift" in calls

    async def test_process_event_records_metrics(self) -> None:
        config = SentinelConfig(
            agent_id="test-agent",
            owner_email="test@example.com",
            phoenix_endpoint="http://localhost:6006",
            sampling_rate=1.0,
            triggers=[],
        )
        mock_cb = MagicMock(spec=CircuitBreaker)
        mock_cb.record_metric = AsyncMock()
        mock_cb.check_triggers = AsyncMock()

        mock_sc = MagicMock(spec=SpotChecker)
        mock_sc.should_sample.return_value = False
        mock_bp = MagicMock(spec=BaselineProviderProtocol)
        mock_veritas = MagicMock(spec=VeritasClientProtocol)

        ingestor = TelemetryIngestorAsync(config, mock_cb, mock_sc, mock_bp, mock_veritas)

        event = VeritasEvent(
            event_id="evt-1",
            timestamp=datetime.now(timezone.utc),
            agent_id="test-agent",
            session_id="sess-1",
            input_text="hello",
            output_text="world",
            metrics={"latency": 100, "token_count": 50},
            metadata={"user_tier": "free"},
        )

        await ingestor.process_event(event)

        mock_cb.record_metric.assert_any_call("latency", 100.0, None)
        mock_cb.record_metric.assert_any_call("token_count", 50.0, None)
        mock_cb.check_triggers.assert_called_once()

    async def test_process_drift_vector(self) -> None:
        config = SentinelConfig(
            agent_id="test-agent",
            owner_email="test@example.com",
            phoenix_endpoint="http://localhost:6006",
            sampling_rate=1.0,
            triggers=[],
        )
        mock_cb = MagicMock(spec=CircuitBreaker)
        mock_cb.record_metric = AsyncMock()
        mock_cb.check_triggers = AsyncMock()

        mock_sc = MagicMock(spec=SpotChecker)
        mock_bp = MagicMock(spec=BaselineProviderProtocol)
        mock_veritas = MagicMock(spec=VeritasClientProtocol)

        ingestor = TelemetryIngestorAsync(config, mock_cb, mock_sc, mock_bp, mock_veritas)

        event = VeritasEvent(
            event_id="evt-1",
            timestamp=datetime.now(timezone.utc),
            agent_id="test-agent",
            session_id="sess-1",
            input_text="hello",
            output_text="world",
            metrics={},
            metadata={"embedding": [0.1, 0.2, 0.3]},
        )

        mock_bp.get_baseline_vectors.return_value = [[0.1, 0.2, 0.3]]

        with patch("coreason_sentinel.ingestor.DriftEngine.detect_vector_drift", return_value=0.5) as mock_detect_drift:
            with patch("anyio.to_thread.run_sync", side_effect=lambda func, *args: func(*args)):
                await ingestor.process_drift(event)
                mock_detect_drift.assert_called()
                mock_cb.record_metric.assert_any_call("vector_drift", 0.5, None)


class TestTelemetryIngestorSync:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
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
        self.ingestor = TelemetryIngestor(self.config, self.mock_cb, self.mock_sc, self.mock_bp, self.mock_veritas)

        self.event = VeritasEvent(
            event_id="evt-1",
            timestamp=datetime.now(timezone.utc),
            agent_id="test-agent",
            session_id="sess-1",
            input_text="hello",
            output_text="world",
            metrics={"latency": 100, "token_count": 50},
            metadata={"user_tier": "free"},
        )

    def test_context_manager_lifecycle(self) -> None:
        """Test that the sync context manager cleans up resources using the portal."""
        assert self.ingestor is not None

        with self.ingestor as svc:
            assert svc is self.ingestor
            assert self.ingestor._portal is not None

        assert self.ingestor._portal is None

    def test_process_event_sync_with_portal(self) -> None:
        """Test that process_event uses the portal if available."""
        with patch.object(self.ingestor, "_async") as mock_async:
            mock_async.process_event = AsyncMock()
            mock_async.__aenter__ = AsyncMock()
            mock_async.__aexit__ = AsyncMock()

            with self.ingestor as svc:
                svc.process_event(self.event)

            assert mock_async.process_event.call_count == 1

    def test_process_event_sync_without_portal(self) -> None:
        """Test fallback to anyio.run if not in context manager."""
        # Using mock method on real async object to avoid complex patch issues
        with patch.object(self.ingestor._async, "process_event", new_callable=AsyncMock):
            # Should raise error now
            with pytest.raises(RuntimeError):
                self.ingestor.process_event(self.event)

    def test_process_otel_span_sync_without_portal(self) -> None:
        """Test fallback to anyio.run for process_otel_span."""
        from coreason_sentinel.interfaces import OTELSpan

        span = OTELSpan(trace_id="t1", span_id="s1", name="test", start_time_unix_nano=0, end_time_unix_nano=1)
        with patch.object(self.ingestor._async, "process_otel_span", new_callable=AsyncMock):
            with pytest.raises(RuntimeError):
                self.ingestor.process_otel_span(span)

    def test_ingest_from_veritas_since_sync_without_portal(self) -> None:
        """Test fallback to anyio.run for ingest_from_veritas_since."""
        with patch.object(self.ingestor._async, "ingest_from_veritas_since", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = 5
            with pytest.raises(RuntimeError):
                self.ingestor.ingest_from_veritas_since(datetime.now(timezone.utc))

    def test_process_drift_sync_without_portal(self) -> None:
        """Test fallback to anyio.run for process_drift."""
        with patch.object(self.ingestor._async, "process_drift", new_callable=AsyncMock):
            with pytest.raises(RuntimeError):
                self.ingestor.process_drift(self.event)
