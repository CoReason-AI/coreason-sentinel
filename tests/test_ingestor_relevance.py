from datetime import datetime
from unittest.mock import MagicMock

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

        ingestor.process_event(event)

        # Verify relevance_drift was recorded
        # Orthogonal vectors -> Drift 1.0
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

        ingestor.process_event(event)

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
        ingestor.process_event(event)

        # Should not record relevance_drift
        calls = [call.args[0] for call in breaker.record_metric.call_args_list]
        assert "relevance_drift" not in calls
