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
import unittest
from unittest.mock import MagicMock, patch

from redis import Redis

from coreason_sentinel.circuit_breaker import CircuitBreaker, CircuitBreakerState
from coreason_sentinel.models import SentinelConfig, Trigger


class TestCircuitBreakerState(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_redis = MagicMock(spec=Redis)
        self.config = SentinelConfig(agent_id="test-agent", circuit_breaker_triggers=[], recovery_timeout=60)
        self.breaker = CircuitBreaker(self.mock_redis, self.config)

    def test_get_state_default(self) -> None:
        """Test that get_state returns CLOSED if key is missing."""
        self.mock_redis.get.return_value = None
        state = self.breaker.get_state()
        self.assertEqual(state, CircuitBreakerState.CLOSED)
        self.mock_redis.get.assert_called_with("sentinel:breaker:test-agent:state")

    def test_get_state_existing(self) -> None:
        """Test that get_state returns the stored state."""
        self.mock_redis.get.return_value = b"CLOSED"
        state = self.breaker.get_state()
        self.assertEqual(state, CircuitBreakerState.CLOSED)

    def test_get_state_redis_exception(self) -> None:
        """Test exception handling in get_state."""
        self.mock_redis.get.side_effect = Exception("Connection Error")
        state = self.breaker.get_state()
        self.assertEqual(state, CircuitBreakerState.CLOSED)

    def test_set_state(self) -> None:
        """Test transitioning state."""
        self.breaker.set_state(CircuitBreakerState.HALF_OPEN)
        self.mock_redis.set.assert_called_with("sentinel:breaker:test-agent:state", "HALF_OPEN")

    def test_set_state_open_sets_cooldown(self) -> None:
        """Test that setting OPEN sets the cooldown key."""
        self.breaker.set_state(CircuitBreakerState.OPEN)
        self.mock_redis.set.assert_called_with("sentinel:breaker:test-agent:state", "OPEN")
        self.mock_redis.setex.assert_called_with("sentinel:breaker:test-agent:cooldown", 60, "1")

    def test_get_state_auto_recovery_from_open(self) -> None:
        """Test OPEN -> HALF_OPEN transition when cooldown expires."""
        self.mock_redis.get.return_value = b"OPEN"
        # exists returns 0 (False) meaning key expired
        self.mock_redis.exists.return_value = 0

        state = self.breaker.get_state()

        # Should transition to HALF_OPEN
        self.assertEqual(state, CircuitBreakerState.HALF_OPEN)
        self.mock_redis.set.assert_called_with("sentinel:breaker:test-agent:state", "HALF_OPEN")

    def test_get_state_open_waiting_for_cooldown(self) -> None:
        """Test OPEN state persists if cooldown exists."""
        self.mock_redis.get.return_value = b"OPEN"
        # exists returns 1 (True)
        self.mock_redis.exists.return_value = 1

        state = self.breaker.get_state()
        self.assertEqual(state, CircuitBreakerState.OPEN)
        # Should NOT transition
        self.mock_redis.set.assert_not_called()

    def test_set_state_redis_exception(self) -> None:
        """Test exception handling in set_state."""
        self.mock_redis.set.side_effect = Exception("Connection Error")
        with self.assertRaises(Exception):  # noqa: B017
            self.breaker.set_state(CircuitBreakerState.OPEN)

    def test_redis_failure_handling(self) -> None:
        """Test that get_state defaults to CLOSED on Redis error."""
        self.mock_redis.get.side_effect = Exception("Redis connection failed")
        state = self.breaker.get_state()
        self.assertEqual(state, CircuitBreakerState.CLOSED)


class TestCircuitBreakerLogic(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_redis = MagicMock(spec=Redis)
        self.trigger = Trigger(
            metric_name="error_count",
            threshold=5,
            window_seconds=60,
            operator=">",
            aggregation_method="SUM",
        )
        self.config = SentinelConfig(agent_id="test-agent", circuit_breaker_triggers=[self.trigger])
        self.breaker = CircuitBreaker(self.mock_redis, self.config)

    def test_allow_request_closed(self) -> None:
        self.mock_redis.get.return_value = b"CLOSED"
        self.assertTrue(self.breaker.allow_request())

    def test_allow_request_open(self) -> None:
        # Assuming cooldown exists
        self.mock_redis.get.return_value = b"OPEN"
        self.mock_redis.exists.return_value = 1
        self.assertFalse(self.breaker.allow_request())

    def test_allow_request_half_open(self) -> None:
        self.mock_redis.get.return_value = b"HALF_OPEN"
        # Probabilistic, so we patch random
        with patch("random.random", return_value=0.04):  # < 0.05
            self.assertTrue(self.breaker.allow_request())
        with patch("random.random", return_value=0.06):  # > 0.05
            self.assertFalse(self.breaker.allow_request())

    def test_check_triggers_recovery_to_closed(self) -> None:
        """Test HALF_OPEN -> CLOSED if no violation."""
        self.mock_redis.get.return_value = b"HALF_OPEN"
        self.mock_redis.zrangebyscore.return_value = []  # No events

        self.breaker.check_triggers()

        # Should set CLOSED
        self.mock_redis.set.assert_called_with("sentinel:breaker:test-agent:state", "CLOSED")

    def test_check_triggers_half_open_failure(self) -> None:
        """Test HALF_OPEN -> OPEN if violation."""
        self.mock_redis.get.return_value = b"HALF_OPEN"
        # Events that violate threshold
        now = time.time()
        members = [f"{now}:10.0:id1".encode("utf-8")]
        self.mock_redis.zrangebyscore.return_value = members

        self.breaker.check_triggers()

        # Should set OPEN (and reset cooldown)
        self.mock_redis.set.assert_called_with("sentinel:breaker:test-agent:state", "OPEN")
        # Ensure cooldown set
        self.mock_redis.setex.assert_called()

    def test_record_metric(self) -> None:
        """Test recording a metric event."""
        self.breaker.record_metric("error_count", 1.0)
        self.mock_redis.zadd.assert_called_once()
        args, _ = self.mock_redis.zadd.call_args
        key = args[0]
        mapping = args[1]
        self.assertEqual(key, "sentinel:metrics:test-agent:error_count")
        self.assertEqual(len(mapping), 1)

    def test_record_metric_exception(self) -> None:
        """Test exception handling in record_metric."""
        self.mock_redis.zadd.side_effect = Exception("Redis Error")
        # Should catch and log error, not raise
        self.breaker.record_metric("error_count", 1.0)

    def test_record_metric_pruning(self) -> None:
        """Test that record_metric prunes old events."""
        self.breaker.record_metric("error_count", 1.0)
        self.mock_redis.zremrangebyscore.assert_called_once()

    def test_record_metric_nan(self) -> None:
        """Test that NaN values are ignored."""
        self.breaker.record_metric("error_count", float("nan"))
        self.mock_redis.zadd.assert_not_called()

    def test_record_metric_inf(self) -> None:
        """Test that Inf values are ignored."""
        self.breaker.record_metric("error_count", float("inf"))
        self.mock_redis.zadd.assert_not_called()

    def test_evaluate_trigger_trips(self) -> None:
        """Test that exceeding threshold trips the breaker."""
        now = time.time()
        members = [f"{now}:1.0:id{i}".encode("utf-8") for i in range(6)]
        self.mock_redis.zrangebyscore.return_value = members
        self.mock_redis.get.return_value = b"CLOSED"

        self.breaker.check_triggers()
        self.mock_redis.set.assert_called_with("sentinel:breaker:test-agent:state", "OPEN")

    def test_evaluate_trigger_already_open(self) -> None:
        """Test that checks are skipped if breaker is already OPEN."""
        self.mock_redis.get.return_value = b"OPEN"
        self.breaker.check_triggers()
        self.mock_redis.zrangebyscore.assert_not_called()

    def test_evaluate_trigger_no_trip(self) -> None:
        """Test that staying under threshold does NOT trip breaker."""
        now = time.time()
        members = [f"{now}:1.0:id{i}".encode("utf-8") for i in range(3)]
        self.mock_redis.zrangebyscore.return_value = members
        self.mock_redis.get.return_value = b"CLOSED"

        self.breaker.check_triggers()
        self.mock_redis.set.assert_not_called()

    def test_evaluate_trigger_exception(self) -> None:
        """Test exception handling in check_triggers."""
        self.mock_redis.zrangebyscore.side_effect = Exception("Redis Error")
        self.mock_redis.get.return_value = b"CLOSED"

        # Should not crash
        self.breaker.check_triggers()
        self.mock_redis.set.assert_not_called()

    def test_sum_metric_logic(self) -> None:
        """Test that values are summed correctly (e.g. Cost)."""
        cost_trigger = Trigger(
            metric_name="cost", threshold=100, window_seconds=60, operator=">", aggregation_method="SUM"
        )
        self.config.circuit_breaker_triggers = [cost_trigger]
        self.breaker = CircuitBreaker(self.mock_redis, self.config)

        now = time.time()
        members = [f"{now}:60.0:id1".encode("utf-8"), f"{now}:60.0:id2".encode("utf-8")]
        self.mock_redis.zrangebyscore.return_value = members
        self.mock_redis.get.return_value = b"CLOSED"

        self.breaker.check_triggers()
        self.mock_redis.set.assert_called_with("sentinel:breaker:test-agent:state", "OPEN")

    def test_bad_member_format(self) -> None:
        """Test resilience against malformed Redis data."""
        members = [b"malformed_string_without_colons"]
        self.mock_redis.zrangebyscore.return_value = members
        self.mock_redis.get.return_value = b"CLOSED"
        self.breaker.check_triggers()
        self.mock_redis.set.assert_not_called()

    def test_bad_member_format_non_float(self) -> None:
        """Test parsing error in member."""
        members = [b"timestamp:not_a_float:id"]
        self.mock_redis.zrangebyscore.return_value = members
        self.mock_redis.get.return_value = b"CLOSED"
        self.breaker.check_triggers()
        self.mock_redis.set.assert_not_called()

    def test_operator_less_than(self) -> None:
        """Test '<' operator."""
        trigger = Trigger(
            metric_name="quality", threshold=0.5, window_seconds=60, operator="<", aggregation_method="AVG"
        )
        self.config.circuit_breaker_triggers = [trigger]
        self.breaker = CircuitBreaker(self.mock_redis, self.config)

        now = time.time()
        members = [f"{now}:0.4:id1".encode("utf-8")]
        self.mock_redis.zrangebyscore.return_value = members
        self.mock_redis.get.return_value = b"CLOSED"

        self.breaker.check_triggers()
        self.mock_redis.set.assert_called_with("sentinel:breaker:test-agent:state", "OPEN")

    def test_evaluate_empty_events(self) -> None:
        """Test trigger evaluation when no events exist in window."""
        self.mock_redis.zrangebyscore.return_value = []
        self.mock_redis.get.return_value = b"CLOSED"

        trigger = Trigger(
            metric_name="quality", threshold=0.5, window_seconds=60, operator="<", aggregation_method="AVG"
        )
        self.config.circuit_breaker_triggers = [trigger]
        self.breaker = CircuitBreaker(self.mock_redis, self.config)

        self.breaker.check_triggers()
        self.mock_redis.set.assert_not_called()

    def test_aggregation_count(self) -> None:
        """Test COUNT aggregation."""
        trigger = Trigger(
            metric_name="errors", threshold=2, window_seconds=60, operator=">", aggregation_method="COUNT"
        )
        self.config.circuit_breaker_triggers = [trigger]
        self.breaker = CircuitBreaker(self.mock_redis, self.config)

        now = time.time()
        members = [f"{now}:1.0:id{i}".encode("utf-8") for i in range(3)]
        self.mock_redis.zrangebyscore.return_value = members
        self.mock_redis.get.return_value = b"CLOSED"

        self.breaker.check_triggers()
        self.mock_redis.set.assert_called_with("sentinel:breaker:test-agent:state", "OPEN")

    def test_aggregation_min_max(self) -> None:
        """Test MIN and MAX aggregation."""
        now = time.time()
        members = [f"{now}:1.0:id1".encode("utf-8"), f"{now}:5.0:id2".encode("utf-8")]
        self.mock_redis.zrangebyscore.return_value = members
        self.mock_redis.get.return_value = b"CLOSED"

        # MAX > 4 -> Trip
        trigger = Trigger(metric_name="test", threshold=4, window_seconds=60, operator=">", aggregation_method="MAX")
        self.config.circuit_breaker_triggers = [trigger]
        self.breaker = CircuitBreaker(self.mock_redis, self.config)
        self.breaker.check_triggers()
        self.mock_redis.set.assert_called()
        self.mock_redis.set.reset_mock()

        # MIN < 2 -> Trip
        trigger = Trigger(metric_name="test", threshold=2, window_seconds=60, operator="<", aggregation_method="MIN")
        self.config.circuit_breaker_triggers = [trigger]
        self.breaker = CircuitBreaker(self.mock_redis, self.config)
        self.breaker.check_triggers()
        self.mock_redis.set.assert_called()

    def test_return_false_on_unknown_operator(self) -> None:
        """Test invalid operator."""
        mock_trigger = MagicMock(spec=Trigger)
        mock_trigger.metric_name = "test"
        mock_trigger.threshold = 10
        mock_trigger.window_seconds = 60
        mock_trigger.operator = "=="
        mock_trigger.aggregation_method = "SUM"

        self.config.circuit_breaker_triggers = [mock_trigger]
        self.breaker = CircuitBreaker(self.mock_redis, self.config)

        now = time.time()
        members = [f"{now}:100.0:id1".encode("utf-8")]
        self.mock_redis.zrangebyscore.return_value = members
        self.mock_redis.get.return_value = b"CLOSED"

        self.breaker.check_triggers()
        self.mock_redis.set.assert_not_called()

    def test_get_recent_values_exception(self) -> None:
        """Test error handling when fetching recent values fails."""
        self.mock_redis.zrevrange.side_effect = Exception("Redis Down")
        values = self.breaker.get_recent_values("test_metric")
        self.assertEqual(values, [])

    def test_get_recent_values_empty(self) -> None:
        """Test fetching recent values returns empty list if none found."""
        self.mock_redis.zrevrange.return_value = []
        values = self.breaker.get_recent_values("test_metric")
        self.assertEqual(values, [])

    def test_get_recent_values_success(self) -> None:
        """Test fetching recent values with valid data."""
        now = time.time()
        members = [f"{now}:10.0:id1".encode("utf-8"), f"{now}:20.0:id2".encode("utf-8")]
        self.mock_redis.zrevrange.return_value = members
        values = self.breaker.get_recent_values("test_metric")
        self.assertEqual(values, [10.0, 20.0])

    def test_allow_request_unknown_state(self) -> None:
        """Test allow_request fallback for unknown state."""
        self.mock_redis.get.return_value = b"UNKNOWN"
        # get_state returns "UNKNOWN" if we force it, but get_state logic casts to Enum or defaults to CLOSED.
        # However, get_state code: `return CircuitBreakerState(state_bytes.decode("utf-8"))`.
        # If redis returns "UNKNOWN", CircuitBreakerState constructor will raise ValueError!
        # And get_state exception handler catches it and returns CLOSED.
        # So it's hard to make get_state return "UNKNOWN".
        # We must mock get_state directly on the breaker instance.
        with patch.object(self.breaker, "get_state", return_value="UNKNOWN"):
            self.assertTrue(self.breaker.allow_request())
