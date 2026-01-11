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
from unittest.mock import MagicMock

from redis import Redis

from coreason_sentinel.circuit_breaker import CircuitBreaker, CircuitBreakerState
from coreason_sentinel.models import SentinelConfig, Trigger


class TestCircuitBreakerState(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_redis = MagicMock(spec=Redis)
        self.config = SentinelConfig(agent_id="test-agent", circuit_breaker_triggers=[])
        self.breaker = CircuitBreaker(self.mock_redis, self.config)

    def test_get_state_default(self) -> None:
        """Test that get_state returns CLOSED if key is missing."""
        self.mock_redis.get.return_value = None
        state = self.breaker.get_state()
        self.assertEqual(state, CircuitBreakerState.CLOSED)
        self.mock_redis.get.assert_called_with("sentinel:breaker:test-agent:state")

    def test_get_state_existing(self) -> None:
        """Test that get_state returns the stored state."""
        self.mock_redis.get.return_value = b"OPEN"
        state = self.breaker.get_state()
        self.assertEqual(state, CircuitBreakerState.OPEN)

    def test_get_state_redis_exception(self) -> None:
        """Test exception handling in get_state."""
        self.mock_redis.get.side_effect = Exception("Connection Error")
        state = self.breaker.get_state()
        self.assertEqual(state, CircuitBreakerState.CLOSED)

    def test_set_state(self) -> None:
        """Test transitioning state."""
        self.breaker.set_state(CircuitBreakerState.HALF_OPEN)
        self.mock_redis.set.assert_called_with("sentinel:breaker:test-agent:state", "HALF_OPEN")

    def test_set_state_redis_exception(self) -> None:
        """Test exception handling in set_state."""
        self.mock_redis.set.side_effect = Exception("Connection Error")
        with self.assertRaises(Exception):  # noqa: B017
            self.breaker.set_state(CircuitBreakerState.OPEN)

    def test_redis_failure_handling(self) -> None:
        """Test that get_state defaults to CLOSED on Redis error."""
        self.mock_redis.get.side_effect = Exception("Redis connection failed")
        state = self.breaker.get_state()
        # Should default to CLOSED (Fail Safe)
        self.assertEqual(state, CircuitBreakerState.CLOSED)


class TestCircuitBreakerLogic(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_redis = MagicMock(spec=Redis)
        self.trigger = Trigger(metric_name="error_count", threshold=5, window_seconds=60, operator=">")
        self.config = SentinelConfig(agent_id="test-agent", circuit_breaker_triggers=[self.trigger])
        self.breaker = CircuitBreaker(self.mock_redis, self.config)

    def test_record_metric(self) -> None:
        """Test recording a metric event."""
        self.breaker.record_metric("error_count", 1.0)
        self.mock_redis.zadd.assert_called_once()
        args, _ = self.mock_redis.zadd.call_args
        key = args[0]
        mapping = args[1]
        self.assertEqual(key, "sentinel:metrics:test-agent:error_count")
        # Ensure mapping has 1 item
        self.assertEqual(len(mapping), 1)

    def test_record_metric_exception(self) -> None:
        """Test exception handling in record_metric."""
        self.mock_redis.zadd.side_effect = Exception("Redis Error")
        # Should catch and log error, not raise
        self.breaker.record_metric("error_count", 1.0)
        # Verify log happened? Logger is global, tricky to verify without patch.
        # But we verify it doesn't crash.

    def test_record_metric_pruning(self) -> None:
        """Test that record_metric prunes old events."""
        self.breaker.record_metric("error_count", 1.0)

        # Verify zremrangebyscore was called
        self.mock_redis.zremrangebyscore.assert_called_once()
        args, _ = self.mock_redis.zremrangebyscore.call_args
        self.assertEqual(args[0], "sentinel:metrics:test-agent:error_count")
        self.assertEqual(args[1], "-inf")
        # Ensure the 3rd arg is a float timestamp
        self.assertIsInstance(args[2], float)

    def test_evaluate_trigger_trips(self) -> None:
        """Test that exceeding threshold trips the breaker."""
        # Mock Redis returning 6 events (threshold is 5)
        # Member format: "{timestamp}:{value}:{uuid}"
        now = time.time()
        members = [f"{now}:1.0:id{i}".encode("utf-8") for i in range(6)]
        self.mock_redis.zrangebyscore.return_value = members

        # Mock current state as CLOSED
        self.mock_redis.get.return_value = b"CLOSED"

        self.breaker.check_triggers()

        # Should transition to OPEN
        self.mock_redis.set.assert_called_with("sentinel:breaker:test-agent:state", "OPEN")

    def test_evaluate_trigger_already_open(self) -> None:
        """Test that checks are skipped if breaker is already OPEN."""
        self.mock_redis.get.return_value = b"OPEN"
        self.breaker.check_triggers()
        # Should return early
        self.mock_redis.zrangebyscore.assert_not_called()

    def test_evaluate_trigger_no_trip(self) -> None:
        """Test that staying under threshold does NOT trip breaker."""
        # Mock Redis returning 3 events (threshold is 5)
        now = time.time()
        members = [f"{now}:1.0:id{i}".encode("utf-8") for i in range(3)]
        self.mock_redis.zrangebyscore.return_value = members

        self.mock_redis.get.return_value = b"CLOSED"

        self.breaker.check_triggers()

        # Should NOT transition to OPEN
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
        # Trigger: Cost > 100
        cost_trigger = Trigger(metric_name="cost", threshold=100, window_seconds=60, operator=">")
        self.config.circuit_breaker_triggers = [cost_trigger]
        self.breaker = CircuitBreaker(self.mock_redis, self.config)

        # Mock Redis returning 2 events of value 60 (Sum = 120 > 100)
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

        # Should default to value 1.0. If threshold is 5, sum is 1. No trip.
        self.breaker.check_triggers()
        self.mock_redis.set.assert_not_called()

    def test_bad_member_format_non_float(self) -> None:
        """Test parsing error in member."""
        members = [b"timestamp:not_a_float:id"]
        self.mock_redis.zrangebyscore.return_value = members
        self.mock_redis.get.return_value = b"CLOSED"

        # Should default to 1.0. 1.0 < 5. No trip.
        self.breaker.check_triggers()
        self.mock_redis.set.assert_not_called()

    def test_operator_less_than(self) -> None:
        """Test '<' operator."""
        # Trigger: quality < 0.5
        trigger = Trigger(metric_name="quality", threshold=0.5, window_seconds=60, operator="<")
        self.config.circuit_breaker_triggers = [trigger]
        self.breaker = CircuitBreaker(self.mock_redis, self.config)

        # Redis has value 0.4
        now = time.time()
        members = [f"{now}:0.4:id1".encode("utf-8")]
        self.mock_redis.zrangebyscore.return_value = members
        self.mock_redis.get.return_value = b"CLOSED"

        self.breaker.check_triggers()

        # Should trip
        self.mock_redis.set.assert_called_with("sentinel:breaker:test-agent:state", "OPEN")

    def test_evaluate_empty_events(self) -> None:
        """Test trigger evaluation when no events exist in window."""
        self.mock_redis.zrangebyscore.return_value = []
        self.mock_redis.get.return_value = b"CLOSED"

        # Total value = 0.0. 0.0 !> 5. No trip.
        self.breaker.check_triggers()
        self.mock_redis.set.assert_not_called()

        # If trigger is < 0.1, it should trip (since 0.0 < 0.1)
        trigger_low = Trigger(metric_name="quality", threshold=0.1, window_seconds=60, operator="<")
        self.config.circuit_breaker_triggers = [trigger_low]
        self.breaker = CircuitBreaker(self.mock_redis, self.config)
        self.breaker.check_triggers()
        self.mock_redis.set.assert_called_with("sentinel:breaker:test-agent:state", "OPEN")

    def test_return_false_on_unknown_operator(self) -> None:
        """Test invalid operator."""
        # Force an invalid operator (bypass strict type check in constructor via object assignment or mock)
        # Or just use logic check if code handles else.
        # Code: if > .. elif < .. return False.

        # We need to construct a trigger with invalid operator.
        # Since pydantic validates, we mock it.
        mock_trigger = MagicMock(spec=Trigger)
        mock_trigger.metric_name = "test"
        mock_trigger.threshold = 10
        mock_trigger.window_seconds = 60
        mock_trigger.operator = "=="  # Invalid

        self.config.circuit_breaker_triggers = [mock_trigger]
        self.breaker = CircuitBreaker(self.mock_redis, self.config)

        now = time.time()
        members = [f"{now}:100.0:id1".encode("utf-8")]
        self.mock_redis.zrangebyscore.return_value = members
        self.mock_redis.get.return_value = b"CLOSED"

        self.breaker.check_triggers()

        # Should not trip because "==" is not handled -> returns False
        self.mock_redis.set.assert_not_called()
