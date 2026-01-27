# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_sentinel

import time
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from coreason_sentinel.ingestor import TelemetryIngestorAsync
from coreason_sentinel.main import app, get_telemetry_ingestor
from coreason_sentinel.models import HealthReport, SentinelConfig

client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_get_telemetry_ingestor_default() -> None:
    """
    Test that the default dependency raises RuntimeError if state not set
    (or NotImplementedError if not using state).
    """
    # In main.py we changed it to check app.state.
    # But locally running this test, app.state might not be populated if lifespan didn't run.
    # TestClient runs lifespan if using `with TestClient(app) as client:`.
    # But here `client` is global.
    # The default dependency raises RuntimeError now.
    with pytest.raises(RuntimeError, match="TelemetryIngestor not initialized in app.state"):
        await get_telemetry_ingestor()


def test_ingest_otel_span_success() -> None:
    # Mock the ingestor
    mock_ingestor = MagicMock(spec=TelemetryIngestorAsync)
    mock_ingestor.process_otel_span = AsyncMock()

    # Override the dependency
    app.dependency_overrides[get_telemetry_ingestor] = lambda: mock_ingestor

    try:
        span_data = {
            "trace_id": uuid.uuid4().hex,
            "span_id": uuid.uuid4().hex[:16],
            "name": "test_span",
            "start_time_unix_nano": int(time.time() * 1e9),
            "end_time_unix_nano": int((time.time() + 1) * 1e9),
            "attributes": {"llm.token_count.total": 100},
            "status_code": "OK",
        }

        response = client.post("/ingest/otel/span", json=span_data)

        assert response.status_code == 202
        assert response.json() == {"status": "accepted", "span_id": span_data["span_id"]}

        # Verify background task was likely scheduled/executed
        mock_ingestor.process_otel_span.assert_called_once()

        called_arg = mock_ingestor.process_otel_span.call_args[0][0]
        assert called_arg.span_id == span_data["span_id"]
        assert called_arg.name == span_data["name"]

    finally:
        app.dependency_overrides = {}


def test_ingest_otel_span_invalid_data() -> None:
    mock_ingestor = MagicMock(spec=TelemetryIngestorAsync)
    mock_ingestor.process_otel_span = AsyncMock()
    app.dependency_overrides[get_telemetry_ingestor] = lambda: mock_ingestor

    try:
        # Missing required fields
        span_data = {"name": "incomplete_span"}

        response = client.post("/ingest/otel/span", json=span_data)
        assert response.status_code == 422
    finally:
        app.dependency_overrides = {}


def test_ingest_otel_span_bad_type() -> None:
    """Edge Case: Sending a string where an int is expected."""
    mock_ingestor = MagicMock(spec=TelemetryIngestorAsync)
    mock_ingestor.process_otel_span = AsyncMock()
    app.dependency_overrides[get_telemetry_ingestor] = lambda: mock_ingestor

    try:
        span_data = {
            "trace_id": uuid.uuid4().hex,
            "span_id": uuid.uuid4().hex[:16],
            "name": "bad_type_span",
            "start_time_unix_nano": "not_an_int",  # Bad type
            "end_time_unix_nano": int((time.time() + 1) * 1e9),
        }

        response = client.post("/ingest/otel/span", json=span_data)
        assert response.status_code == 422
        # Verify validation error details roughly
        assert "start_time_unix_nano" in str(response.json())
    finally:
        app.dependency_overrides = {}


def test_ingest_otel_span_malformed_attributes() -> None:
    """Edge Case: Sending a string instead of a dictionary for attributes."""
    mock_ingestor = MagicMock(spec=TelemetryIngestorAsync)
    mock_ingestor.process_otel_span = AsyncMock()
    app.dependency_overrides[get_telemetry_ingestor] = lambda: mock_ingestor

    try:
        span_data = {
            "trace_id": uuid.uuid4().hex,
            "span_id": uuid.uuid4().hex[:16],
            "name": "malformed_attr_span",
            "start_time_unix_nano": int(time.time() * 1e9),
            "end_time_unix_nano": int((time.time() + 1) * 1e9),
            "attributes": "this_should_be_a_dict",  # Bad type
        }

        response = client.post("/ingest/otel/span", json=span_data)
        assert response.status_code == 422
        assert "attributes" in str(response.json())
    finally:
        app.dependency_overrides = {}


def test_ingest_otel_span_background_exception() -> None:
    """Edge Case: Ensure API returns 202 even if background task fails (mocked)."""
    mock_ingestor = MagicMock(spec=TelemetryIngestorAsync)
    # Simulate an error inside the ingestor
    mock_ingestor.process_otel_span = AsyncMock(side_effect=ValueError("Something exploded!"))

    app.dependency_overrides[get_telemetry_ingestor] = lambda: mock_ingestor

    try:
        span_data = {
            "trace_id": uuid.uuid4().hex,
            "span_id": uuid.uuid4().hex[:16],
            "name": "exploding_span",
            "start_time_unix_nano": int(time.time() * 1e9),
            "end_time_unix_nano": int((time.time() + 1) * 1e9),
        }

        # TestClient executes background tasks *synchronously* after the request is handled
        # but before returning control.
        # Actually, TestClient catches exceptions in background tasks and re-raises them in the test.
        # So we expect the client.post to raise ValueError.

        try:
            client.post("/ingest/otel/span", json=span_data)
        except ValueError as e:
            assert str(e) == "Something exploded!"

    finally:
        app.dependency_overrides = {}


def test_complex_batch_ingestion() -> None:
    """Complex Scenario: Simulate a batch of 10 requests."""
    mock_ingestor = MagicMock(spec=TelemetryIngestorAsync)
    mock_ingestor.process_otel_span = AsyncMock()
    app.dependency_overrides[get_telemetry_ingestor] = lambda: mock_ingestor

    try:
        batch_size = 10
        for i in range(batch_size):
            span_data = {
                "trace_id": uuid.uuid4().hex,
                "span_id": f"span_{i}",
                "name": f"batch_span_{i}",
                "start_time_unix_nano": int(time.time() * 1e9),
                "end_time_unix_nano": int((time.time() + 1) * 1e9),
                "attributes": {"batch_index": i},
            }
            response = client.post("/ingest/otel/span", json=span_data)
            assert response.status_code == 202
            assert response.json()["span_id"] == f"span_{i}"

        # Verify called 10 times
        assert mock_ingestor.process_otel_span.call_count == batch_size
    finally:
        app.dependency_overrides = {}


def test_agent_health_check_success() -> None:
    """Test health check for valid agent."""
    # We don't use spec=TelemetryIngestorAsync because it doesn't see instance attributes set in __init__
    mock_ingestor = MagicMock()
    mock_ingestor.config = SentinelConfig(
        agent_id="agent-001", owner_email="ops@coreason.ai", phoenix_endpoint="http://localhost:6006"
    )
    # Mock the circuit breaker attribute
    mock_cb = MagicMock()
    mock_cb.get_health_report = AsyncMock(
        return_value=HealthReport(timestamp=datetime.now(), breaker_state="CLOSED", metrics={"avg_latency": 0.5})
    )
    mock_ingestor.circuit_breaker = mock_cb

    app.dependency_overrides[get_telemetry_ingestor] = lambda: mock_ingestor

    try:
        response = client.get("/health/agent-001")
        assert response.status_code == 200
        data = response.json()
        assert data["breaker_state"] == "CLOSED"
        assert data["metrics"]["avg_latency"] == 0.5
    finally:
        app.dependency_overrides = {}


def test_agent_health_check_not_found() -> None:
    """Test health check for invalid agent ID."""
    mock_ingestor = MagicMock()
    mock_ingestor.config = SentinelConfig(
        agent_id="agent-001", owner_email="ops@coreason.ai", phoenix_endpoint="http://localhost:6006"
    )

    app.dependency_overrides[get_telemetry_ingestor] = lambda: mock_ingestor

    try:
        response = client.get("/health/wrong-agent")
        assert response.status_code == 404
    finally:
        app.dependency_overrides = {}


def test_agent_status_check() -> None:
    """Test status check."""
    mock_ingestor = MagicMock()
    mock_ingestor.config = SentinelConfig(
        agent_id="agent-001", owner_email="ops@coreason.ai", phoenix_endpoint="http://localhost:6006"
    )
    mock_cb = MagicMock()
    mock_cb.allow_request = AsyncMock(return_value=True)
    mock_ingestor.circuit_breaker = mock_cb

    app.dependency_overrides[get_telemetry_ingestor] = lambda: mock_ingestor

    try:
        response = client.get("/status/agent-001")
        assert response.status_code == 200
        assert response.json() is True
    finally:
        app.dependency_overrides = {}


def test_ingest_veritas_event_success() -> None:
    """Test Veritas event ingestion."""
    mock_ingestor = MagicMock(spec=TelemetryIngestorAsync)
    mock_ingestor.process_event = AsyncMock()

    app.dependency_overrides[get_telemetry_ingestor] = lambda: mock_ingestor

    try:
        event_data = {
            "event_id": "evt_123",
            "timestamp": datetime.now().isoformat(),
            "agent_id": "agent-001",
            "session_id": "sess_1",
            "input_text": "Hello",
            "output_text": "World",
            "metrics": {"latency": 0.1},
            "metadata": {},
        }

        response = client.post("/ingest/veritas", json=event_data)
        assert response.status_code == 202
        assert response.json() == {"status": "accepted", "event_id": "evt_123"}

        mock_ingestor.process_event.assert_called_once()
    finally:
        app.dependency_overrides = {}
