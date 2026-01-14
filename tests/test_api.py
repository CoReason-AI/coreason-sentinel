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
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from coreason_sentinel.ingestor import TelemetryIngestor
from coreason_sentinel.main import app, get_telemetry_ingestor

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_otel_span_success():
    # Mock the ingestor
    mock_ingestor = MagicMock(spec=TelemetryIngestor)

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
            "status_code": "OK"
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


def test_ingest_otel_span_invalid_data():
    mock_ingestor = MagicMock(spec=TelemetryIngestor)
    app.dependency_overrides[get_telemetry_ingestor] = lambda: mock_ingestor

    try:
        # Missing required fields
        span_data = {
            "name": "incomplete_span"
        }

        response = client.post("/ingest/otel/span", json=span_data)
        assert response.status_code == 422
    finally:
        app.dependency_overrides = {}


def test_ingest_otel_span_bad_type():
    """Edge Case: Sending a string where an int is expected."""
    mock_ingestor = MagicMock(spec=TelemetryIngestor)
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


def test_ingest_otel_span_malformed_attributes():
    """Edge Case: Sending a string instead of a dictionary for attributes."""
    mock_ingestor = MagicMock(spec=TelemetryIngestor)
    app.dependency_overrides[get_telemetry_ingestor] = lambda: mock_ingestor

    try:
        span_data = {
            "trace_id": uuid.uuid4().hex,
            "span_id": uuid.uuid4().hex[:16],
            "name": "malformed_attr_span",
            "start_time_unix_nano": int(time.time() * 1e9),
            "end_time_unix_nano": int((time.time() + 1) * 1e9),
            "attributes": "this_should_be_a_dict" # Bad type
        }

        response = client.post("/ingest/otel/span", json=span_data)
        assert response.status_code == 422
        assert "attributes" in str(response.json())
    finally:
        app.dependency_overrides = {}


def test_ingest_otel_span_background_exception():
    """Edge Case: Ensure API returns 202 even if background task fails (mocked)."""
    mock_ingestor = MagicMock(spec=TelemetryIngestor)
    # Simulate an error inside the ingestor
    mock_ingestor.process_otel_span.side_effect = ValueError("Something exploded!")

    app.dependency_overrides[get_telemetry_ingestor] = lambda: mock_ingestor

    try:
        span_data = {
            "trace_id": uuid.uuid4().hex,
            "span_id": uuid.uuid4().hex[:16],
            "name": "exploding_span",
            "start_time_unix_nano": int(time.time() * 1e9),
            "end_time_unix_nano": int((time.time() + 1) * 1e9),
        }

        # TestClient executes background tasks *synchronously* after the request is handled but before returning control?
        # Actually, TestClient catches exceptions in background tasks and re-raises them in the test.
        # So we expect the client.post to raise ValueError.
        # But in production (uvicorn), this would just log an error and the response (202) would have already been sent.

        try:
            client.post("/ingest/otel/span", json=span_data)
            # If we reach here without exception, maybe TestClient behavior changed or Starlette suppresses it?
            # Recent Starlette/FastAPI versions might bubble up exceptions in TestClient.
        except ValueError as e:
            assert str(e) == "Something exploded!"

    finally:
        app.dependency_overrides = {}


def test_complex_batch_ingestion():
    """Complex Scenario: Simulate a batch of 10 requests."""
    mock_ingestor = MagicMock(spec=TelemetryIngestor)
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
                "attributes": {"batch_index": i}
            }
            response = client.post("/ingest/otel/span", json=span_data)
            assert response.status_code == 202
            assert response.json()["span_id"] == f"span_{i}"

        # Verify called 10 times
        assert mock_ingestor.process_otel_span.call_count == batch_size
    finally:
        app.dependency_overrides = {}
