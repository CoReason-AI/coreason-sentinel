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
        # Note: TestClient runs background tasks synchronously after the response
        mock_ingestor.process_otel_span.assert_called_once()

        # Verify argument passed matches (roughly)
        called_arg = mock_ingestor.process_otel_span.call_args[0][0]
        assert called_arg.span_id == span_data["span_id"]
        assert called_arg.name == span_data["name"]

    finally:
        # Clear override
        app.dependency_overrides = {}


def test_ingest_otel_span_invalid_data():
    # Mock the ingestor even for invalid data test to avoid Dependency error
    # Although validation happens before dependency injection usually,
    # depending on FastAPI version/order, it might trigger dependency if in body.
    # Actually, in this case, dependency is injected for the function.
    # If validation fails, it might not reach the function, but TestClient
    # might still try to resolve dependencies if they are top level.
    # However, let's override to be safe and avoid the NotImplementedError.

    mock_ingestor = MagicMock(spec=TelemetryIngestor)
    app.dependency_overrides[get_telemetry_ingestor] = lambda: mock_ingestor

    try:
        # Missing required fields
        span_data = {
            "name": "incomplete_span"
        }

        response = client.post("/ingest/otel/span", json=span_data)
        assert response.status_code == 422  # Validation Error
    finally:
        app.dependency_overrides = {}
