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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coreason_sentinel.circuit_breaker import CircuitBreaker
from coreason_sentinel.ingestor import TelemetryIngestor
from coreason_sentinel.interfaces import VeritasEvent
from coreason_sentinel.models import SentinelConfig


class TestIngestorRelevanceDrift:
    @pytest.fixture
    def mock_components(self) -> tuple[TelemetryIngestor, MagicMock]:
        config = SentinelConfig(
            agent_id="test_agent",
            owner_email="test@example.com",
            phoenix_endpoint="http://localhost:6006",
        )
        breaker = MagicMock(spec=CircuitBreaker)
        breaker.record_metric = AsyncMock()
        breaker.check_triggers = AsyncMock()

        spot_checker = MagicMock()
        baseline_provider = MagicMock()
        veritas_client = MagicMock()

        ingestor = TelemetryIngestor(config, breaker, spot_checker, baseline_provider, veritas_client)
        return ingestor, breaker

    def test_process_event_records_relevance_drift(self, mock_components: tuple[TelemetryIngestor, MagicMock]) -> None:
        ingestor, breaker = mock_components

        event = VeritasEvent(
            event_id="evt_123",
            timestamp=datetime.now(),
            agent_id="test_agent",
            session_id="sess_1",
            input_text="query",
            output_text="response",
            metrics={},
            metadata={"query_embedding": [1.0, 0.0], "response_embedding": [0.0, 1.0]},
        )

        with patch("anyio.to_thread.run_sync", side_effect=lambda func, *args: func(*args)):
            with ingestor:
                ingestor.process_drift(event)

        # Verify relevance_drift was recorded
        # Orthogonal vectors -> Drift 1.0
        # Wait, if using `with ingestor`, it runs in portal thread.
        # MagicMock isn't thread-safe for assertions if race condition, but usually ok for simple tests.
        # But wait, `ingestor` (sync) calls `process_drift` via portal.
        # `process_drift` (async) calls `await breaker.record_metric`.
        # `breaker.record_metric` is `AsyncMock`.

        # We need to wait for completion? Portal.call blocks. So it is done.

        breaker.record_metric.assert_any_call("relevance_drift", 1.0)

    def test_process_event_skips_missing_embeddings(self, mock_components: tuple[TelemetryIngestor, MagicMock]) -> None:
        ingestor, breaker = mock_components

        event = VeritasEvent(
            event_id="evt_123",
            timestamp=datetime.now(),
            agent_id="test_agent",
            session_id="sess_1",
            input_text="query",
            output_text="response",
            metrics={},
            metadata={
                "query_embedding": [1.0, 0.0]
                # response_embedding missing
            },
        )

        with patch("anyio.to_thread.run_sync", side_effect=lambda func, *args: func(*args)):
            with ingestor:
                ingestor.process_drift(event)

        # Should not record relevance_drift
        calls = [call.args[0] for call in breaker.record_metric.call_args_list]
        assert "relevance_drift" not in calls

    def test_process_event_handles_calculation_error(
        self, mock_components: tuple[TelemetryIngestor, MagicMock]
    ) -> None:
        ingestor, breaker = mock_components

        event = VeritasEvent(
            event_id="evt_123",
            timestamp=datetime.now(),
            agent_id="test_agent",
            session_id="sess_1",
            input_text="query",
            output_text="response",
            metrics={},
            metadata={
                "query_embedding": [1.0, 0.0],
                "response_embedding": [1.0, 0.0, 0.0],  # Mismatched dimension
            },
        )

        # Should not raise exception
        with patch("anyio.to_thread.run_sync", side_effect=lambda func, *args: func(*args)):
            with ingestor:
                ingestor.process_drift(event)

        # Should not record relevance_drift
        calls = [call.args[0] for call in breaker.record_metric.call_args_list]
        assert "relevance_drift" not in calls
