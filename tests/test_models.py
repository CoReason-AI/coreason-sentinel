import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from coreason_sentinel.models import (
    CircuitBreakerTrigger,
    HealthReport,
    SentinelConfig,
)


def test_trigger_creation() -> None:
    trigger = CircuitBreakerTrigger(metric="error_rate", threshold=0.05, window_seconds=60)
    assert trigger.metric == "error_rate"
    assert trigger.threshold == 0.05
    assert trigger.window_seconds == 60
    assert trigger.operator == ">"


def test_trigger_validation() -> None:
    with pytest.raises(ValidationError):
        CircuitBreakerTrigger(metric="error_rate", threshold="invalid", window_seconds=60)


def test_trigger_edge_cases() -> None:
    # Window seconds must be positive
    with pytest.raises(ValidationError):
        CircuitBreakerTrigger(metric="error_rate", threshold=0.05, window_seconds=0)

    with pytest.raises(ValidationError):
        CircuitBreakerTrigger(metric="error_rate", threshold=0.05, window_seconds=-10)

    # Threshold can be negative (valid mathematical comparison)
    t = CircuitBreakerTrigger(metric="score", threshold=-5.0, window_seconds=10)
    assert t.threshold == -5.0


def test_sentinel_config_defaults() -> None:
    config = SentinelConfig(
        agent_id="test-agent", owner_email="test@example.com", phoenix_endpoint="http://localhost:6006"
    )
    assert config.agent_id == "test-agent"
    assert config.owner_email == "test@example.com"
    assert config.phoenix_endpoint == "http://localhost:6006"
    assert config.sampling_rate == 0.01
    assert config.drift_threshold_kl == 0.5
    assert config.triggers == []


def test_sentinel_config_full() -> None:
    trigger = CircuitBreakerTrigger(metric="cost", threshold=100, window_seconds=3600)
    config = SentinelConfig(
        agent_id="test-agent",
        owner_email="test@example.com",
        phoenix_endpoint="http://localhost:6006",
        sampling_rate=0.1,
        drift_threshold_kl=0.8,
        triggers=[trigger],
    )
    assert config.sampling_rate == 0.1
    assert len(config.triggers) == 1
    assert config.triggers[0].metric == "cost"


def test_sentinel_config_edge_cases() -> None:
    # Sample rate limits
    with pytest.raises(ValidationError):
        SentinelConfig(agent_id="test", owner_email="a", phoenix_endpoint="b", sampling_rate=-0.01)

    with pytest.raises(ValidationError):
        SentinelConfig(agent_id="test", owner_email="a", phoenix_endpoint="b", sampling_rate=1.01)

    # Valid boundary sample rates
    c1 = SentinelConfig(agent_id="test", owner_email="a", phoenix_endpoint="b", sampling_rate=0.0)
    assert c1.sampling_rate == 0.0
    c2 = SentinelConfig(agent_id="test", owner_email="a", phoenix_endpoint="b", sampling_rate=1.0)
    assert c2.sampling_rate == 1.0

    # Drift threshold limits
    with pytest.raises(ValidationError):
        SentinelConfig(agent_id="test", owner_email="a", phoenix_endpoint="b", drift_threshold_kl=-0.1)

    c3 = SentinelConfig(agent_id="test", owner_email="a", phoenix_endpoint="b", drift_threshold_kl=0.0)
    assert c3.drift_threshold_kl == 0.0


def test_sentinel_config_complex_serialization() -> None:
    trigger1 = CircuitBreakerTrigger(metric="latency", threshold=500, window_seconds=60)
    trigger2 = CircuitBreakerTrigger(metric="cost", threshold=100, window_seconds=3600, operator=">")

    config = SentinelConfig(
        agent_id="complex-agent",
        owner_email="test@example.com",
        phoenix_endpoint="http://localhost:6006",
        sampling_rate=0.5,
        drift_threshold_kl=0.75,
        triggers=[trigger1, trigger2],
    )

    # Round trip JSON
    json_str = config.model_dump_json()
    restored = SentinelConfig.model_validate_json(json_str)

    assert restored == config
    assert len(restored.triggers) == 2
    assert restored.triggers[0].metric == "latency"
    assert restored.triggers[1].window_seconds == 3600


def test_health_report_creation() -> None:
    now = datetime.now(timezone.utc)
    report = HealthReport(
        timestamp=now,
        breaker_state="CLOSED",
        metrics={"avg_latency": "200ms"},
    )
    assert report.timestamp == now
    assert report.breaker_state == "CLOSED"
    assert report.metrics["avg_latency"] == "200ms"


def test_health_report_complex_metrics() -> None:
    now = datetime.now(timezone.utc)
    complex_metrics = {
        "latency_p99": 120.5,
        "latency_p95": 100.0,
        "token_usage": {"prompt": 500, "completion": 200, "total": 700},
        "tags": ["prod", "v2"],
        "is_active": True,
    }

    report = HealthReport(timestamp=now, breaker_state="CLOSED", metrics=complex_metrics)

    # Verify nested access
    assert report.metrics["token_usage"]["total"] == 700
    assert report.metrics["tags"][1] == "v2"

    # JSON Round trip
    json_str = report.model_dump_json()
    # Note: JSON serialization converts datetime to string, so exact equality comparison
    # requires parsing it back.
    data = json.loads(json_str)
    assert data["metrics"]["token_usage"]["prompt"] == 500
    assert data["breaker_state"] == "CLOSED"


def test_health_report_status_validation() -> None:
    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError):
        HealthReport(timestamp=now, breaker_state="INVALID_STATUS")


# Edge Cases & Complex Scenarios


def test_breaker_state_case_sensitivity() -> None:
    now = datetime.now(timezone.utc)
    # Pydantic Literals are strict strings and case-sensitive
    with pytest.raises(ValidationError):
        HealthReport(timestamp=now, breaker_state="closed")


def test_breaker_state_whitespace() -> None:
    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError):
        HealthReport(timestamp=now, breaker_state=" CLOSED")
    with pytest.raises(ValidationError):
        HealthReport(timestamp=now, breaker_state="CLOSED ")


def test_breaker_state_empty_string() -> None:
    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError):
        HealthReport(timestamp=now, breaker_state="")


def test_health_report_history_bulk_serialization() -> None:
    """
    Simulates a scenario where we have a history of health reports (e.g. 50 snapshots).
    Ensures that bulk serialization/deserialization works correctly and performantly.
    """
    history = []
    base_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    for i in range(50):
        state = "CLOSED"
        if i % 10 == 0:
            state = "OPEN"
        elif i % 5 == 0:
            state = "HALF_OPEN"

        report = HealthReport(
            timestamp=base_time,  # In real scenario timestamp would increment
            breaker_state=state,
            metrics={"tick": i},
        )
        history.append(report)

    # Serialize list
    json_output = json.dumps([h.model_dump(mode="json") for h in history])

    # Deserialize list
    loaded_data = json.loads(json_output)
    restored_history = [HealthReport.model_validate(d) for d in loaded_data]

    assert len(restored_history) == 50
    assert restored_history[0].breaker_state == "OPEN"  # 0 % 10 == 0
    assert restored_history[5].breaker_state == "HALF_OPEN"  # 5 % 5 == 0
    assert restored_history[1].breaker_state == "CLOSED"
    assert restored_history[49].metrics["tick"] == 49


def test_health_report_timestamp_robustness() -> None:
    """
    Tests that HealthReport can handle slight variations in ISO timestamp strings
    that Pydantic usually supports.
    """
    # 1. Standard ISO with Z
    json_z = '{"timestamp": "2025-01-01T12:00:00Z", "breaker_state": "CLOSED", "metrics": {}}'
    report_z = HealthReport.model_validate_json(json_z)
    assert report_z.timestamp.year == 2025
    assert report_z.timestamp.tzinfo == timezone.utc

    # 2. ISO with offset
    json_offset = '{"timestamp": "2025-01-01T12:00:00+00:00", "breaker_state": "CLOSED", "metrics": {}}'
    report_offset = HealthReport.model_validate_json(json_offset)
    assert report_offset.timestamp == report_z.timestamp

    # 3. ISO without explicit timezone (Pydantic might interpret as naive, but we should check behavior)
    # The field is just 'datetime', so it accepts naive. Best practice is to ensure UTC though.
    json_naive = '{"timestamp": "2025-01-01T12:00:00", "breaker_state": "CLOSED", "metrics": {}}'
    report_naive = HealthReport.model_validate_json(json_naive)
    assert report_naive.timestamp.year == 2025
    assert report_naive.timestamp.tzinfo is None
