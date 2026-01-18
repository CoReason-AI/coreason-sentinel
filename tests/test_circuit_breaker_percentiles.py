# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_sentinel

from typing import Any
from unittest.mock import MagicMock

import pytest

from coreason_sentinel.circuit_breaker import CircuitBreaker, CircuitBreakerState
from coreason_sentinel.models import CircuitBreakerTrigger, SentinelConfig


class MockRedis(MagicMock):
    """
    A functional mock of Redis for Circuit Breaker tests.
    Stores data in a simple dictionary structure to allow stateful testing.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._data: dict[str, Any] = {}
        self._sorted_sets: dict[str, list[tuple[float, bytes]]] = {}

    def get(self, key: str) -> Any:
        return self._data.get(key)

    def set(self, key: str, value: Any) -> bool:
        if isinstance(value, str):
            value = value.encode("utf-8")
        self._data[key] = value
        return True

    def getset(self, key: str, value: Any) -> Any:
        old_value = self._data.get(key)
        self.set(key, value)
        return old_value

    def exists(self, key: str) -> bool:
        return key in self._data

    def setex(self, key: str, time: int, value: Any) -> bool:
        self.set(key, value)
        return True

    def zadd(self, key: str, mapping: dict[str | bytes, float]) -> int:
        if key not in self._sorted_sets:
            self._sorted_sets[key] = []
        # Mapping is {member: score}
        for member, score in mapping.items():
            # member should be bytes
            if isinstance(member, str):
                member_bytes = member.encode("utf-8")
            else:
                member_bytes = member
            self._sorted_sets[key].append((score, member_bytes))
        # Sort by score
        self._sorted_sets[key].sort(key=lambda x: x[0])
        return 1

    def zrangebyscore(self, key: str, min_score: float | str, max_score: float | str) -> list[bytes]:
        if key not in self._sorted_sets:
            return []

        # Determine effective range
        # Simple implementation for testing: ignore +/- inf parsing for now
        # Assumes float inputs or uses a very simple check

        effective_min = float("-inf") if min_score == "-inf" else float(min_score)
        effective_max = float("inf") if max_score == "+inf" else float(max_score)

        result = []
        for score, member in self._sorted_sets[key]:
            if effective_min <= score <= effective_max:
                result.append(member)
        return result

    def zremrangebyscore(self, key: str, min_score: float | str, max_score: float | str) -> int:
        if key not in self._sorted_sets:
            return 0

        effective_min = float("-inf") if min_score == "-inf" else float(min_score)
        effective_max = float("inf") if max_score == "+inf" else float(max_score)

        new_list = []
        removed_count = 0
        for score, member in self._sorted_sets[key]:
            if not (effective_min <= score <= effective_max):
                new_list.append((score, member))
            else:
                removed_count += 1
        self._sorted_sets[key] = new_list
        return removed_count

    def expire(self, key: str, time: int) -> bool:
        return True


@pytest.fixture
def mock_redis() -> MockRedis:
    return MockRedis()


@pytest.fixture
def mock_notification_service() -> MagicMock:
    return MagicMock()


@pytest.fixture
def basic_config() -> SentinelConfig:
    return SentinelConfig(
        agent_id="test_agent", owner_email="admin@coreason.ai", phoenix_endpoint="http://localhost:6006", triggers=[]
    )


def test_percentile_calculation_p50(
    mock_redis: MockRedis, mock_notification_service: MagicMock, basic_config: SentinelConfig
) -> None:
    # Setup: 10 values from 1 to 10
    # P50 of [1..10] depends on interpolation, usually 5.5

    trigger = CircuitBreakerTrigger(
        metric="latency", threshold=5.0, window_seconds=60, aggregation_method="P50", operator=">"
    )
    basic_config.triggers = [trigger]

    cb = CircuitBreaker(mock_redis, basic_config, mock_notification_service)

    # Record metrics: 1.0, 2.0, ..., 10.0
    for i in range(1, 11):
        cb.record_metric("latency", float(i))

    # Check triggers
    # Median of 1..10 is 5.5. Threshold is 5.0. 5.5 > 5.0 -> Should trip.
    cb.check_triggers()

    assert cb.get_state() == CircuitBreakerState.OPEN
    mock_notification_service.send_critical_alert.assert_called_once()


def test_percentile_calculation_p90(
    mock_redis: MockRedis, mock_notification_service: MagicMock, basic_config: SentinelConfig
) -> None:
    # Setup: 100 values from 1 to 100
    # P90 should be roughly 90.

    trigger = CircuitBreakerTrigger(
        metric="latency", threshold=85.0, window_seconds=60, aggregation_method="P90", operator=">"
    )
    basic_config.triggers = [trigger]

    cb = CircuitBreaker(mock_redis, basic_config, mock_notification_service)

    for i in range(1, 101):
        cb.record_metric("latency", float(i))

    # P90 of 1..100 is approx 90.1 (depending on method). 90.1 > 85 -> Trip
    cb.check_triggers()
    assert cb.get_state() == CircuitBreakerState.OPEN


def test_percentile_calculation_p99(
    mock_redis: MockRedis, mock_notification_service: MagicMock, basic_config: SentinelConfig
) -> None:
    # Setup: 100 values. 99 values are 0.1, one value is 100.0 (outlier)

    trigger = CircuitBreakerTrigger(
        metric="latency", threshold=10.0, window_seconds=60, aggregation_method="P99", operator=">"
    )
    basic_config.triggers = [trigger]

    cb = CircuitBreaker(mock_redis, basic_config, mock_notification_service)

    for _ in range(99):
        cb.record_metric("latency", 0.1)
    cb.record_metric("latency", 100.0)

    # P99 of 99*0.1 and 1*100.0
    # with linear interpolation, P99 falls very close to the max value or is influenced by it.
    # In a set of 100 items, P99 is the 99th percentile.
    # Sorted: indices 0..98 are 0.1, index 99 is 100.0.
    # P99 corresponds to index ~98.01. So it should be close to 0.1 or interpolation between 0.1 and 100.0?
    # Wait, numpy percentile default is linear.
    # let's verify numpy behavior separately or assume standard behavior.
    # Actually, let's use a simpler distribution for deterministic testing.
    # 0, 10, 20, ... 100 (11 values).

    # Reset redis mock data for clean state or re-instantiate
    mock_redis._data = {}
    mock_redis._sorted_sets = {}

    # values: 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
    values = [float(x) for x in range(0, 101, 10)]
    for v in values:
        cb.record_metric("latency", v)

    # P99 of [0..100 step 10].
    # With 11 items, 100 is the 100th percentile. 90 is the 90th percentile.
    # P99 should be 99.0.
    # Threshold 10.0 -> Trip.

    cb.check_triggers()
    assert cb.get_state() == CircuitBreakerState.OPEN


def test_percentile_no_violation(
    mock_redis: MockRedis, mock_notification_service: MagicMock, basic_config: SentinelConfig
) -> None:
    trigger = CircuitBreakerTrigger(
        metric="latency",
        threshold=100.0,  # High threshold
        window_seconds=60,
        aggregation_method="P95",
        operator=">",
    )
    basic_config.triggers = [trigger]

    cb = CircuitBreaker(mock_redis, basic_config, mock_notification_service)

    for i in range(1, 11):
        cb.record_metric("latency", float(i))

    # Max is 10. P95 < 10. Threshold 100. No trip.
    cb.check_triggers()
    assert cb.get_state() == CircuitBreakerState.CLOSED
    mock_notification_service.send_critical_alert.assert_not_called()


def test_percentile_empty_data(
    mock_redis: MockRedis, mock_notification_service: MagicMock, basic_config: SentinelConfig
) -> None:
    trigger = CircuitBreakerTrigger(
        metric="latency", threshold=1.0, window_seconds=60, aggregation_method="P99", operator=">"
    )
    basic_config.triggers = [trigger]
    cb = CircuitBreaker(mock_redis, basic_config, mock_notification_service)

    # No metrics recorded
    cb.check_triggers()

    # Should not trip
    assert cb.get_state() == CircuitBreakerState.CLOSED


def test_percentile_invalid_format(
    mock_redis: MockRedis, mock_notification_service: MagicMock, basic_config: SentinelConfig
) -> None:
    # Use MagicMock to bypass Pydantic validation and simulate an invalid aggregation method
    # potentially coming from a raw dict or corrupted state
    trigger = MagicMock(spec=CircuitBreakerTrigger)
    trigger.metric = "latency"
    trigger.threshold = 10.0
    trigger.window_seconds = 60
    trigger.aggregation_method = "PXX"  # Invalid format
    trigger.operator = ">"

    basic_config.triggers = [trigger]

    cb = CircuitBreaker(mock_redis, basic_config, mock_notification_service)

    # Record some metrics so code reaches the parsing block
    cb.record_metric("latency", 5.0)

    # Should catch ValueError/IndexError and log error, returning False (no trip)
    cb.check_triggers()

    assert cb.get_state() == CircuitBreakerState.CLOSED


def test_percentile_numpy_error(
    mock_redis: MockRedis, mock_notification_service: MagicMock, basic_config: SentinelConfig
) -> None:
    # Trigger that causes np.percentile to fail (e.g., empty sequence, though we guard against empty)
    # or invalid quantile
    trigger = MagicMock(spec=CircuitBreakerTrigger)
    trigger.metric = "latency"
    trigger.threshold = 10.0
    trigger.window_seconds = 60
    trigger.aggregation_method = "P101" # Invalid percentile > 100
    trigger.operator = ">"

    basic_config.triggers = [trigger]
    cb = CircuitBreaker(mock_redis, basic_config, mock_notification_service)
    cb.record_metric("latency", 5.0)

    # Should catch ValueError inside _evaluate_trigger
    cb.check_triggers()
    assert cb.get_state() == CircuitBreakerState.CLOSED


def test_percentile_single_value(
    mock_redis: MockRedis, mock_notification_service: MagicMock, basic_config: SentinelConfig
) -> None:
    # Single value should be the same for any percentile
    trigger = CircuitBreakerTrigger(
        metric="latency", threshold=5.0, window_seconds=60, aggregation_method="P99", operator=">"
    )
    basic_config.triggers = [trigger]

    cb = CircuitBreaker(mock_redis, basic_config, mock_notification_service)

    cb.record_metric("latency", 4.0)
    cb.check_triggers()
    assert cb.get_state() == CircuitBreakerState.CLOSED

    cb.record_metric("latency", 6.0)  # Total metrics: 4.0, 6.0
    # P99 of [4.0, 6.0] should be close to 6.0 -> Trip
    cb.check_triggers()
    assert cb.get_state() == CircuitBreakerState.OPEN


def test_percentile_identical_values(
    mock_redis: MockRedis, mock_notification_service: MagicMock, basic_config: SentinelConfig
) -> None:
    # All values are 10.0
    trigger = CircuitBreakerTrigger(
        metric="latency", threshold=9.0, window_seconds=60, aggregation_method="P99", operator=">"
    )
    basic_config.triggers = [trigger]

    cb = CircuitBreaker(mock_redis, basic_config, mock_notification_service)

    for _ in range(50):
        cb.record_metric("latency", 10.0)

    cb.check_triggers()
    assert cb.get_state() == CircuitBreakerState.OPEN


def test_complex_spiky_traffic(
    mock_redis: MockRedis, mock_notification_service: MagicMock, basic_config: SentinelConfig
) -> None:
    # Scenario: Normal traffic is low (0.1 - 0.5s), but we have occasional spikes (5.0s, 10.0s)
    # P50 should remain low (no trip)
    # P99 should catch the spikes (trip)

    p50_trigger = CircuitBreakerTrigger(
        metric="latency", threshold=1.0, window_seconds=60, aggregation_method="P50", operator=">"
    )
    p99_trigger = CircuitBreakerTrigger(
        metric="latency", threshold=4.0, window_seconds=60, aggregation_method="P99", operator=">"
    )
    # We test them separately to verify behavior

    # 1. Test P50 stability
    basic_config.triggers = [p50_trigger]
    cb = CircuitBreaker(mock_redis, basic_config, mock_notification_service)

    # 90 requests at 0.2s, 10 requests at 10.0s
    for _ in range(90):
        cb.record_metric("latency", 0.2)
    for _ in range(10):
        cb.record_metric("latency", 10.0)

    # Median (P50) of 100 items (sorted: 90x0.2, 10x10.0) is 0.2.
    # Threshold is 1.0. Should NOT trip.
    cb.check_triggers()
    assert cb.get_state() == CircuitBreakerState.CLOSED

    # 2. Test P99 sensitivity
    basic_config.triggers = [p99_trigger]
    # Reset state (simulated by re-init or just assuming closed if we didn't trip)
    # But metrics are still in mock_redis.
    # P99 of 100 items (sorted: 0.2 ... 10.0). Index ~99 is 10.0.
    # Threshold is 4.0. Should trip.

    cb.check_triggers()
    assert cb.get_state() == CircuitBreakerState.OPEN
