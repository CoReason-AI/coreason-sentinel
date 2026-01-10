from datetime import datetime

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


def test_alert_creation() -> None:
    alert = Alert(message="Something went wrong", severity=AlertSeverity.CRITICAL)
    assert alert.message == "Something went wrong"
    assert alert.severity == AlertSeverity.CRITICAL
    assert isinstance(alert.timestamp, datetime)


def test_health_report_creation() -> None:
    now = datetime.utcnow()
    report = HealthReport(
        timestamp=now,
        agent_status=AgentStatus.HEALTHY,
        metrics={"avg_latency": "200ms"},
    )
    assert report.timestamp == now
    assert report.agent_status == AgentStatus.HEALTHY
    assert report.metrics["avg_latency"] == "200ms"
    assert report.active_alerts == []


def test_health_report_with_alerts() -> None:
    now = datetime.utcnow()
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
