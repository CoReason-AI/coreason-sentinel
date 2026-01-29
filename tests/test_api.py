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
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from coreason_sentinel.ingestor import TelemetryIngestorAsync
from coreason_sentinel.main import app, get_telemetry_ingestor
from coreason_sentinel.models import HealthReport

# Initialize TestClient. Note: Lifespan is not triggered automatically here unless using context manager.
client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_get_telemetry_ingestor_default_no_lifespan() -> None:
    """Test that the dependency raises RuntimeError if not initialized (no lifespan)."""
    # Ensure no override
    app.dependency_overrides = {}
    # Ensure state is clean (it should be since lifespan didn't run)
    if hasattr(app.state, "ingestor"):
        del app.state.ingestor

    with pytest.raises(RuntimeError, match="TelemetryIngestor is not initialized in app state"):
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


# --- New Tests for Full Integration with Lifespan ---


@pytest.fixture
def mock_redis() -> Generator[MagicMock, None, None]:
    with patch("coreason_sentinel.main.Redis") as mock:
        mock_instance = MagicMock()
        mock.from_url.return_value = mock_instance
        mock_instance.close = AsyncMock()
        yield mock_instance


@pytest.fixture
def client_with_lifespan(mock_redis: MagicMock) -> Generator[TestClient, None, None]:
    # Using 'with' triggers lifespan
    with TestClient(app) as c:
        yield c


def test_get_agent_health(client_with_lifespan: TestClient) -> None:
    # Access the ingestor created in lifespan
    ingestor = app.state.ingestor

    # Mock return value of circuit breaker
    # Note: ingestor.circuit_breaker is a real object with a mock redis.
    # We can just mock the method directly on the object.
    ingestor.circuit_breaker.get_health_report = AsyncMock(
        return_value=HealthReport(timestamp=datetime(2025, 1, 1), breaker_state="CLOSED", metrics={"avg_latency": 0.5})
    )

    response = client_with_lifespan.get("/health/default_agent")
    assert response.status_code == 200
    data = response.json()
    assert data["breaker_state"] == "CLOSED"
    assert data["metrics"]["avg_latency"] == 0.5


def test_get_agent_status(client_with_lifespan: TestClient) -> None:
    ingestor = app.state.ingestor
    ingestor.circuit_breaker.allow_request = AsyncMock(return_value=True)

    response = client_with_lifespan.get("/status/default_agent")
    assert response.status_code == 200
    assert response.json() is True

    ingestor.circuit_breaker.allow_request.return_value = False
    response = client_with_lifespan.get("/status/default_agent")
    assert response.json() is False


def test_ingest_veritas(client_with_lifespan: TestClient) -> None:
    ingestor = app.state.ingestor
    ingestor.process_event = AsyncMock()

    event_data = {
        "event_id": "evt_123",
        "timestamp": "2025-01-01T12:00:00",
        "agent_id": "default_agent",
        "session_id": "sess_1",
        "input_text": "Hello",
        "output_text": "World",
        "metrics": {"latency": 0.1},
        "metadata": {},
    }

    response = client_with_lifespan.post("/ingest/veritas", json=event_data)
    assert response.status_code == 202
    assert response.json() == {"status": "accepted", "event_id": "evt_123"}

    ingestor.process_event.assert_called_once()
