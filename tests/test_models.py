import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from coreason_sentinel.models import (
    AgentStatus,
    Alert,
    AlertSeverity,
    HealthReport,
    SentinelConfig,
    Trigger,
)


def test_trigger_creation() -> None:
    trigger = Trigger(metric_name="error_rate", threshold=0.05, window_seconds=60)
    assert trigger.metric_name == "error_rate"
    assert trigger.threshold == 0.05
    assert trigger.window_seconds == 60
    assert trigger.operator == ">"


def test_trigger_validation() -> None:
    with pytest.raises(ValidationError):
        Trigger(metric_name="error_rate", threshold="invalid", window_seconds=60)


def test_trigger_edge_cases() -> None:
    # Window seconds must be positive
    with pytest.raises(ValidationError):
        Trigger(metric_name="error_rate", threshold=0.05, window_seconds=0)

    with pytest.raises(ValidationError):
        Trigger(metric_name="error_rate", threshold=0.05, window_seconds=-10)

    # Threshold can be negative (valid mathematical comparison)
    t = Trigger(metric_name="score", threshold=-5.0, window_seconds=10)
    assert t.threshold == -5.0


def test_sentinel_config_defaults() -> None:
    config = SentinelConfig(agent_id="test-agent")
    assert config.agent_id == "test-agent"
    assert config.sample_rate == 0.01
    assert config.drift_threshold_kl == 0.5
    assert config.circuit_breaker_triggers == []
    assert config.notification_channels == []


def test_sentinel_config_full() -> None:
    trigger = Trigger(metric_name="cost", threshold=100, window_seconds=3600)
    config = SentinelConfig(
        agent_id="test-agent",
        sample_rate=0.1,
        drift_threshold_kl=0.8,
        circuit_breaker_triggers=[trigger],
        notification_channels=["admin@example.com"],
    )
    assert config.sample_rate == 0.1
    assert len(config.circuit_breaker_triggers) == 1
    assert config.circuit_breaker_triggers[0].metric_name == "cost"


def test_sentinel_config_edge_cases() -> None:
    # Sample rate limits
    with pytest.raises(ValidationError):
        SentinelConfig(agent_id="test", sample_rate=-0.01)

    with pytest.raises(ValidationError):
        SentinelConfig(agent_id="test", sample_rate=1.01)

    # Valid boundary sample rates
    c1 = SentinelConfig(agent_id="test", sample_rate=0.0)
    assert c1.sample_rate == 0.0
    c2 = SentinelConfig(agent_id="test", sample_rate=1.0)
    assert c2.sample_rate == 1.0

    # Drift threshold limits
    with pytest.raises(ValidationError):
        SentinelConfig(agent_id="test", drift_threshold_kl=-0.1)

    c3 = SentinelConfig(agent_id="test", drift_threshold_kl=0.0)
    assert c3.drift_threshold_kl == 0.0


def test_sentinel_config_complex_serialization() -> None:
    trigger1 = Trigger(metric_name="latency", threshold=500, window_seconds=60)
    trigger2 = Trigger(metric_name="cost", threshold=100, window_seconds=3600, operator=">")

    config = SentinelConfig(
        agent_id="complex-agent",
        sample_rate=0.5,
        drift_threshold_kl=0.75,
        circuit_breaker_triggers=[trigger1, trigger2],
        notification_channels=["alert@example.com", "ops@example.com"],
    )

    # Round trip JSON
    json_str = config.model_dump_json()
    restored = SentinelConfig.model_validate_json(json_str)

    assert restored == config
    assert len(restored.circuit_breaker_triggers) == 2
    assert restored.circuit_breaker_triggers[0].metric_name == "latency"
    assert restored.circuit_breaker_triggers[1].window_seconds == 3600


def test_alert_creation() -> None:
    alert = Alert(message="Something went wrong", severity=AlertSeverity.CRITICAL)
    assert alert.message == "Something went wrong"
    assert alert.severity == AlertSeverity.CRITICAL
    assert isinstance(alert.timestamp, datetime)


def test_health_report_creation() -> None:
    now = datetime.now(timezone.utc)
    report = HealthReport(
        timestamp=now,
        agent_status=AgentStatus.HEALTHY,
        metrics={"avg_latency": "200ms"},
    )
    assert report.timestamp == now
    assert report.agent_status == AgentStatus.HEALTHY
    assert report.metrics["avg_latency"] == "200ms"
    assert report.active_alerts == []


def test_health_report_complex_metrics() -> None:
    now = datetime.now(timezone.utc)
    complex_metrics = {
        "latency_p99": 120.5,
        "latency_p95": 100.0,
        "token_usage": {"prompt": 500, "completion": 200, "total": 700},
        "tags": ["prod", "v2"],
        "is_active": True,
    }

    report = HealthReport(timestamp=now, agent_status=AgentStatus.DEGRADED, metrics=complex_metrics)

    # Verify nested access
    assert report.metrics["token_usage"]["total"] == 700
    assert report.metrics["tags"][1] == "v2"

    # JSON Round trip
    json_str = report.model_dump_json()
    # Note: JSON serialization converts datetime to string, so exact equality comparison
    # requires parsing it back.
    data = json.loads(json_str)
    assert data["metrics"]["token_usage"]["prompt"] == 500
    assert data["agent_status"] == "DEGRADED"


def test_health_report_with_alerts() -> None:
    now = datetime.now(timezone.utc)
    alert = Alert(message="High latency", severity=AlertSeverity.WARNING)
    report = HealthReport(
        timestamp=now,
        agent_status=AgentStatus.DEGRADED,
        active_alerts=[alert],
    )
    assert len(report.active_alerts) == 1
    assert report.active_alerts[0].message == "High latency"


def test_enums() -> None:
    assert AgentStatus.HEALTHY == "HEALTHY"
    assert AlertSeverity.INFO == "INFO"
