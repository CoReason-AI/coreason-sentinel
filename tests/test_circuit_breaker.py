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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.asyncio import Redis

from coreason_sentinel.circuit_breaker import CircuitBreaker, CircuitBreakerState
from coreason_sentinel.interfaces import NotificationServiceProtocol
from coreason_sentinel.models import CircuitBreakerTrigger, SentinelConfig


@pytest.mark.asyncio
class TestCircuitBreakerState(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.mock_redis = MagicMock(spec=Redis)
        # Mock async methods on Redis
        self.mock_redis.get = AsyncMock()
        self.mock_redis.getset = AsyncMock(return_value=b"CLOSED")
        self.mock_redis.setex = AsyncMock()
        self.mock_redis.exists = AsyncMock()
        self.mock_redis.zadd = AsyncMock()
        self.mock_redis.zremrangebyscore = AsyncMock()
        self.mock_redis.expire = AsyncMock()
        self.mock_redis.zrangebyscore = AsyncMock()
        self.mock_redis.zrevrange = AsyncMock()

        self.mock_notification_service = MagicMock(spec=NotificationServiceProtocol)
        self.config = SentinelConfig(
            agent_id="test-agent",
            owner_email="test@example.com",
            phoenix_endpoint="http://localhost:6006",
            triggers=[],
            recovery_timeout=60,
        )
        self.breaker = CircuitBreaker(self.mock_redis, self.config, self.mock_notification_service)

    async def test_get_state_default(self) -> None:
        """Test that get_state returns CLOSED if key is missing."""
        self.mock_redis.get.return_value = None
        state = await self.breaker.get_state()
        self.assertEqual(state, CircuitBreakerState.CLOSED)
        self.mock_redis.get.assert_called_with("sentinel:breaker:test-agent:state")

    async def test_get_state_existing(self) -> None:
        """Test that get_state returns the stored state."""
        self.mock_redis.get.return_value = b"CLOSED"
        state = await self.breaker.get_state()
        self.assertEqual(state, CircuitBreakerState.CLOSED)

    async def test_get_state_redis_exception(self) -> None:
        """Test exception handling in get_state."""
        self.mock_redis.get.side_effect = Exception("Connection Error")
        state = await self.breaker.get_state()
        self.assertEqual(state, CircuitBreakerState.CLOSED)

    async def test_set_state(self) -> None:
        """Test transitioning state."""
        await self.breaker.set_state(CircuitBreakerState.HALF_OPEN)
        self.mock_redis.getset.assert_called_with("sentinel:breaker:test-agent:state", "HALF_OPEN")

    async def test_set_state_open_sets_cooldown(self) -> None:
        """Test that setting OPEN sets the cooldown key."""
        # Mock old state as CLOSED (so we trip)
        self.mock_redis.getset.return_value = b"CLOSED"

        await self.breaker.set_state(CircuitBreakerState.OPEN)
        # Check that getset was used
        self.mock_redis.getset.assert_called_with("sentinel:breaker:test-agent:state", "OPEN")
        self.mock_redis.setex.assert_called_with("sentinel:breaker:test-agent:cooldown", 60, "1")

    async def test_set_state_open_sends_alert(self) -> None:
        """Test that setting OPEN sends a critical alert."""
        # Mock old state as CLOSED
        self.mock_redis.getset.return_value = b"CLOSED"
        reason = "Manual Trip"
        await self.breaker.set_state(CircuitBreakerState.OPEN, reason=reason)
        self.mock_notification_service.send_critical_alert.assert_called_once_with(
            email="test@example.com", agent_id="test-agent", reason=reason
        )

    async def test_idempotent_open_alert(self) -> None:
        """
        Test that setting OPEN when ALREADY OPEN does NOT send another alert.
        """
        # Mock old state as OPEN
        self.mock_redis.getset.return_value = b"OPEN"
        reason = "Manual Trip"
        await self.breaker.set_state(CircuitBreakerState.OPEN, reason=reason)

        # Should NOT alert
        self.mock_notification_service.send_critical_alert.assert_not_called()
        # Should NOT reset cooldown
        self.mock_redis.setex.assert_not_called()

    async def test_set_state_open_alert_failure(self) -> None:
        """Test that alert failure doesn't crash the breaker."""
        self.mock_redis.getset.return_value = b"CLOSED"
        self.mock_notification_service.send_critical_alert.side_effect = Exception("Email Down")
        await self.breaker.set_state(CircuitBreakerState.OPEN)
        # Should proceed to log but not crash
        self.mock_redis.getset.assert_called_with("sentinel:breaker:test-agent:state", "OPEN")

    async def test_missing_email_config(self) -> None:
        """Test that no alert is sent if owner_email is missing."""
        self.config.owner_email = ""
        self.mock_redis.getset.return_value = b"CLOSED"

        await self.breaker.set_state(CircuitBreakerState.OPEN)

        self.mock_notification_service.send_critical_alert.assert_not_called()

    async def test_flapping_cycle_alerts(self) -> None:
        """
        Test that flapping (Open -> Closed -> Open) sends 2 alerts.
        """
        # 1. First Trip
        self.mock_redis.getset.return_value = b"CLOSED"
        await self.breaker.set_state(CircuitBreakerState.OPEN)
        self.assertEqual(self.mock_notification_service.send_critical_alert.call_count, 1)

        # 2. Reset mock
        self.mock_notification_service.reset_mock()

        # 3. Recovery to CLOSED (getset returns OPEN)
        self.mock_redis.getset.return_value = b"OPEN"
        await self.breaker.set_state(CircuitBreakerState.CLOSED)
        self.mock_notification_service.send_critical_alert.assert_not_called()

        # 4. Second Trip (getset returns CLOSED)
        self.mock_redis.getset.return_value = b"CLOSED"
        await self.breaker.set_state(CircuitBreakerState.OPEN)
        self.assertEqual(self.mock_notification_service.send_critical_alert.call_count, 1)

    async def test_get_state_auto_recovery_from_open(self) -> None:
        """Test OPEN -> HALF_OPEN transition when cooldown expires."""
        self.mock_redis.get.return_value = b"OPEN"
        # exists returns 0 (False) meaning key expired
        self.mock_redis.exists.return_value = 0

        state = await self.breaker.get_state()

        # Should transition to HALF_OPEN
        self.assertEqual(state, CircuitBreakerState.HALF_OPEN)
        self.mock_redis.getset.assert_called_with("sentinel:breaker:test-agent:state", "HALF_OPEN")

    async def test_get_state_open_waiting_for_cooldown(self) -> None:
        """Test OPEN state persists if cooldown exists."""
        self.mock_redis.get.return_value = b"OPEN"
        # exists returns 1 (True)
        self.mock_redis.exists.return_value = 1

        state = await self.breaker.get_state()
        self.assertEqual(state, CircuitBreakerState.OPEN)
        # Should NOT transition
        self.mock_redis.set.assert_not_called()

    async def test_set_state_redis_exception(self) -> None:
        """Test exception handling in set_state."""
        self.mock_redis.getset.side_effect = Exception("Connection Error")
        with self.assertRaises(Exception):  # noqa: B017
            await self.breaker.set_state(CircuitBreakerState.OPEN)

    async def test_redis_failure_handling(self) -> None:
        """Test that get_state defaults to CLOSED on Redis error."""
        self.mock_redis.get.side_effect = Exception("Redis connection failed")
        state = await self.breaker.get_state()
        self.assertEqual(state, CircuitBreakerState.CLOSED)


@pytest.mark.asyncio
class TestCircuitBreakerLogic(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.mock_redis = MagicMock(spec=Redis)
        self.mock_redis.get = AsyncMock()
        self.mock_redis.getset = AsyncMock(return_value=b"CLOSED")
        self.mock_redis.setex = AsyncMock()
        self.mock_redis.exists = AsyncMock()
        self.mock_redis.zadd = AsyncMock()
        self.mock_redis.zremrangebyscore = AsyncMock()
        self.mock_redis.expire = AsyncMock()
        self.mock_redis.zrangebyscore = AsyncMock()
        self.mock_redis.zrevrange = AsyncMock()

        self.mock_notification_service = MagicMock(spec=NotificationServiceProtocol)
        self.trigger = CircuitBreakerTrigger(
            metric="error_count",
            threshold=5,
            window_seconds=60,
            operator=">",
            aggregation_method="SUM",
        )
        self.config = SentinelConfig(
            agent_id="test-agent",
            owner_email="test@example.com",
            phoenix_endpoint="http://localhost:6006",
            triggers=[self.trigger],
        )
        self.breaker = CircuitBreaker(self.mock_redis, self.config, self.mock_notification_service)

    async def test_allow_request_closed(self) -> None:
        self.mock_redis.get.return_value = b"CLOSED"
        self.assertTrue(await self.breaker.allow_request())

    async def test_allow_request_open(self) -> None:
        # Assuming cooldown exists
        self.mock_redis.get.return_value = b"OPEN"
        self.mock_redis.exists.return_value = 1
        self.assertFalse(await self.breaker.allow_request())

    async def test_allow_request_half_open(self) -> None:
        self.mock_redis.get.return_value = b"HALF_OPEN"
        # Probabilistic, so we patch random
        with patch("random.random", return_value=0.04):  # < 0.05
            self.assertTrue(await self.breaker.allow_request())
        with patch("random.random", return_value=0.06):  # > 0.05
            self.assertFalse(await self.breaker.allow_request())

    async def test_check_triggers_recovery_to_closed(self) -> None:
        """Test HALF_OPEN -> CLOSED if no violation."""
        self.mock_redis.get.return_value = b"HALF_OPEN"
        self.mock_redis.zrangebyscore.return_value = []  # No events

        await self.breaker.check_triggers()

        # Should set CLOSED
        self.mock_redis.getset.assert_called_with("sentinel:breaker:test-agent:state", "CLOSED")

    async def test_check_triggers_half_open_failure(self) -> None:
        """Test HALF_OPEN -> OPEN if violation."""
        self.mock_redis.get.return_value = b"HALF_OPEN"
        # Events that violate threshold
        now = time.time()
        members = [f"{now}:10.0:id1".encode("utf-8")]
        self.mock_redis.zrangebyscore.return_value = members

        await self.breaker.check_triggers()

        # Should set OPEN (and reset cooldown)
        self.mock_redis.getset.assert_called_with("sentinel:breaker:test-agent:state", "OPEN")
        # Ensure cooldown set
        self.mock_redis.setex.assert_called()

    async def test_record_metric(self) -> None:
        """Test recording a metric event."""
        await self.breaker.record_metric("error_count", 1.0)
        self.mock_redis.zadd.assert_called_once()
        args, _ = self.mock_redis.zadd.call_args
        key = args[0]
        mapping = args[1]
        self.assertEqual(key, "sentinel:metrics:test-agent:error_count")
        self.assertEqual(len(mapping), 1)

    async def test_record_metric_exception(self) -> None:
        """Test exception handling in record_metric."""
        self.mock_redis.zadd.side_effect = Exception("Redis Error")
        # Should catch and log error, not raise
        await self.breaker.record_metric("error_count", 1.0)

    async def test_record_metric_pruning(self) -> None:
        """Test that record_metric prunes old events."""
        await self.breaker.record_metric("error_count", 1.0)
        self.mock_redis.zremrangebyscore.assert_called_once()

    async def test_record_metric_nan(self) -> None:
        """Test that NaN values are ignored."""
        await self.breaker.record_metric("error_count", float("nan"))
        self.mock_redis.zadd.assert_not_called()

    async def test_record_metric_inf(self) -> None:
        """Test that Inf values are ignored."""
        await self.breaker.record_metric("error_count", float("inf"))
        self.mock_redis.zadd.assert_not_called()

    async def test_evaluate_trigger_trips(self) -> None:
        """Test that exceeding threshold trips the breaker."""
        now = time.time()
        members = [f"{now}:1.0:id{i}".encode("utf-8") for i in range(6)]
        self.mock_redis.zrangebyscore.return_value = members
        self.mock_redis.get.return_value = b"CLOSED"

        await self.breaker.check_triggers()
        self.mock_redis.getset.assert_called_with("sentinel:breaker:test-agent:state", "OPEN")

    async def test_evaluate_trigger_already_open(self) -> None:
        """Test that checks are skipped if breaker is already OPEN."""
        self.mock_redis.get.return_value = b"OPEN"
        await self.breaker.check_triggers()
        self.mock_redis.zrangebyscore.assert_not_called()

    async def test_evaluate_trigger_no_trip(self) -> None:
        """Test that staying under threshold does NOT trip breaker."""
        now = time.time()
        members = [f"{now}:1.0:id{i}".encode("utf-8") for i in range(3)]
        self.mock_redis.zrangebyscore.return_value = members
        self.mock_redis.get.return_value = b"CLOSED"

        await self.breaker.check_triggers()
        self.mock_redis.set.assert_not_called()

    async def test_evaluate_trigger_exception(self) -> None:
        """Test exception handling in check_triggers."""
        self.mock_redis.zrangebyscore.side_effect = Exception("Redis Error")
        self.mock_redis.get.return_value = b"CLOSED"

        # Should not crash
        await self.breaker.check_triggers()
        self.mock_redis.set.assert_not_called()

    async def test_sum_metric_logic(self) -> None:
        """Test that values are summed correctly (e.g. Cost)."""
        cost_trigger = CircuitBreakerTrigger(
            metric="cost", threshold=100, window_seconds=60, operator=">", aggregation_method="SUM"
        )
        self.config.triggers = [cost_trigger]
        self.breaker = CircuitBreaker(self.mock_redis, self.config, self.mock_notification_service)

        now = time.time()
        members = [f"{now}:60.0:id1".encode("utf-8"), f"{now}:60.0:id2".encode("utf-8")]
        self.mock_redis.zrangebyscore.return_value = members
        self.mock_redis.get.return_value = b"CLOSED"

        await self.breaker.check_triggers()
        self.mock_redis.getset.assert_called_with("sentinel:breaker:test-agent:state", "OPEN")

    async def test_bad_member_format(self) -> None:
        """Test resilience against malformed Redis data."""
        members = [b"malformed_string_without_colons"]
        self.mock_redis.zrangebyscore.return_value = members
        self.mock_redis.get.return_value = b"CLOSED"
        await self.breaker.check_triggers()
        self.mock_redis.set.assert_not_called()

    async def test_bad_member_format_non_float(self) -> None:
        """Test parsing error in member."""
        members = [b"timestamp:not_a_float:id"]
        self.mock_redis.zrangebyscore.return_value = members
        self.mock_redis.get.return_value = b"CLOSED"
        await self.breaker.check_triggers()
        self.mock_redis.set.assert_not_called()

    async def test_operator_less_than(self) -> None:
        """Test '<' operator."""
        trigger = CircuitBreakerTrigger(
            metric="quality", threshold=0.5, window_seconds=60, operator="<", aggregation_method="AVG"
        )
        self.config.triggers = [trigger]
        self.breaker = CircuitBreaker(self.mock_redis, self.config, self.mock_notification_service)

        now = time.time()
        members = [f"{now}:0.4:id1".encode("utf-8")]
        self.mock_redis.zrangebyscore.return_value = members
        self.mock_redis.get.return_value = b"CLOSED"

        await self.breaker.check_triggers()
        self.mock_redis.getset.assert_called_with("sentinel:breaker:test-agent:state", "OPEN")

    async def test_evaluate_empty_events(self) -> None:
        """Test trigger evaluation when no events exist in window."""
        self.mock_redis.zrangebyscore.return_value = []
        self.mock_redis.get.return_value = b"CLOSED"

        trigger = CircuitBreakerTrigger(
            metric="quality", threshold=0.5, window_seconds=60, operator="<", aggregation_method="AVG"
        )
        self.config.triggers = [trigger]
        self.breaker = CircuitBreaker(self.mock_redis, self.config, self.mock_notification_service)

        await self.breaker.check_triggers()
        self.mock_redis.set.assert_not_called()

    async def test_aggregation_count(self) -> None:
        """Test COUNT aggregation."""
        trigger = CircuitBreakerTrigger(
            metric="errors", threshold=2, window_seconds=60, operator=">", aggregation_method="COUNT"
        )
        self.config.triggers = [trigger]
        self.breaker = CircuitBreaker(self.mock_redis, self.config, self.mock_notification_service)

        now = time.time()
        members = [f"{now}:1.0:id{i}".encode("utf-8") for i in range(3)]
        self.mock_redis.zrangebyscore.return_value = members
        self.mock_redis.get.return_value = b"CLOSED"

        await self.breaker.check_triggers()
        self.mock_redis.getset.assert_called_with("sentinel:breaker:test-agent:state", "OPEN")

    async def test_aggregation_min_max(self) -> None:
        """Test MIN and MAX aggregation."""
        now = time.time()
        members = [f"{now}:1.0:id1".encode("utf-8"), f"{now}:5.0:id2".encode("utf-8")]
        self.mock_redis.zrangebyscore.return_value = members
        self.mock_redis.get.return_value = b"CLOSED"

        # MAX > 4 -> Trip
        trigger = CircuitBreakerTrigger(
            metric="test", threshold=4, window_seconds=60, operator=">", aggregation_method="MAX"
        )
        self.config.triggers = [trigger]
        self.breaker = CircuitBreaker(self.mock_redis, self.config, self.mock_notification_service)
        await self.breaker.check_triggers()
        self.mock_redis.getset.assert_called()
        self.mock_redis.getset.reset_mock()

        # MIN < 2 -> Trip
        trigger = CircuitBreakerTrigger(
            metric="test", threshold=2, window_seconds=60, operator="<", aggregation_method="MIN"
        )
        self.config.triggers = [trigger]
        self.breaker = CircuitBreaker(self.mock_redis, self.config, self.mock_notification_service)
        await self.breaker.check_triggers()
        self.mock_redis.getset.assert_called()

    async def test_return_false_on_unknown_operator(self) -> None:
        """Test invalid operator."""
        mock_trigger = MagicMock(spec=CircuitBreakerTrigger)
        mock_trigger.metric = "test"
        mock_trigger.threshold = 10
        mock_trigger.window_seconds = 60
        mock_trigger.operator = "=="
        mock_trigger.aggregation_method = "SUM"

        self.config.triggers = [mock_trigger]
        self.breaker = CircuitBreaker(self.mock_redis, self.config, self.mock_notification_service)

        now = time.time()
        members = [f"{now}:100.0:id1".encode("utf-8")]
        self.mock_redis.zrangebyscore.return_value = members
        self.mock_redis.get.return_value = b"CLOSED"

        await self.breaker.check_triggers()
        self.mock_redis.set.assert_not_called()

    async def test_get_recent_values_exception(self) -> None:
        """Test error handling when fetching recent values fails."""
        self.mock_redis.zrevrange.side_effect = Exception("Redis Down")
        values = await self.breaker.get_recent_values("test_metric")
        self.assertEqual(values, [])

    async def test_get_recent_values_empty(self) -> None:
        """Test fetching recent values returns empty list if none found."""
        self.mock_redis.zrevrange.return_value = []
        values = await self.breaker.get_recent_values("test_metric")
        self.assertEqual(values, [])

    async def test_get_recent_values_success(self) -> None:
        """Test fetching recent values with valid data."""
        now = time.time()
        members = [f"{now}:10.0:id1".encode("utf-8"), f"{now}:20.0:id2".encode("utf-8")]
        self.mock_redis.zrevrange.return_value = members
        values = await self.breaker.get_recent_values("test_metric")
        self.assertEqual(values, [10.0, 20.0])

    async def test_allow_request_unknown_state(self) -> None:
        """Test allow_request fallback for unknown state."""
        self.mock_redis.get.return_value = b"UNKNOWN"
        with patch.object(self.breaker, "get_state", return_value="UNKNOWN"):
            self.assertTrue(await self.breaker.allow_request())


@pytest.mark.asyncio
class TestCircuitBreakerComplexScenarios(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.mock_redis = MagicMock(spec=Redis)
        self.mock_redis.get = AsyncMock()
        self.mock_redis.getset = AsyncMock(return_value=b"CLOSED")
        self.mock_redis.setex = AsyncMock()
        self.mock_redis.exists = AsyncMock()
        self.mock_redis.zadd = AsyncMock()
        self.mock_redis.zremrangebyscore = AsyncMock()
        self.mock_redis.expire = AsyncMock()
        self.mock_redis.zrangebyscore = AsyncMock()
        self.mock_redis.zrevrange = AsyncMock()

        self.mock_notification_service = MagicMock(spec=NotificationServiceProtocol)
        self.trigger_error = CircuitBreakerTrigger(
            metric="errors",
            threshold=5,
            window_seconds=10,
            operator=">",
            aggregation_method="SUM",
        )
        self.trigger_latency = CircuitBreakerTrigger(
            metric="latency",
            threshold=1000,
            window_seconds=10,
            operator=">",
            aggregation_method="AVG",
        )
        self.config = SentinelConfig(
            agent_id="complex-agent",
            owner_email="test@example.com",
            phoenix_endpoint="http://localhost:6006",
            triggers=[self.trigger_error, self.trigger_latency],
            recovery_timeout=60,
        )
        self.breaker = CircuitBreaker(self.mock_redis, self.config, self.mock_notification_service)

    async def test_auto_transition_failure(self) -> None:
        """
        Edge Case: Redis fails during the implicit OPEN -> HALF_OPEN transition in get_state.
        Should handle exception gracefully and default to safe state (CLOSED or OPEN depending on view).
        The code catches exception and returns CLOSED.
        """
        self.mock_redis.get.return_value = b"OPEN"
        self.mock_redis.exists.return_value = 0  # Cooldown expired
        # set_state will be called. Make it fail.
        self.mock_redis.getset.side_effect = Exception("Redis Write Failed")

        state = await self.breaker.get_state()
        self.assertEqual(state, CircuitBreakerState.CLOSED)

    async def test_full_recovery_cycle_with_traffic(self) -> None:
        """
        Complex Scenario: OPEN -> HALF_OPEN -> Healthy Traffic -> CLOSED
        """
        # 1. Start as OPEN
        self.mock_redis.get.return_value = b"OPEN"
        self.mock_redis.exists.return_value = 0  # Cooldown expired

        # 2. Call get_state -> Triggers transition to HALF_OPEN
        state = await self.breaker.get_state()
        self.assertEqual(state, CircuitBreakerState.HALF_OPEN)
        # Verify set_state called
        self.mock_redis.getset.assert_called_with("sentinel:breaker:complex-agent:state", "HALF_OPEN")

        # 3. Simulate Healthy Traffic
        # In HALF_OPEN, we check triggers.
        # Let's say we have metrics that are SAFE (Errors = 2 < 5)
        self.mock_redis.get.return_value = b"HALF_OPEN"
        now = time.time()
        members = [f"{now}:1.0:id{i}".encode("utf-8") for i in range(2)]  # Sum = 2
        self.mock_redis.zrangebyscore.return_value = members

        # 4. Check Triggers
        await self.breaker.check_triggers()

        # 5. Should transition to CLOSED
        self.mock_redis.getset.assert_called_with("sentinel:breaker:complex-agent:state", "CLOSED")

    async def test_relapse_cycle(self) -> None:
        """
        Complex Scenario: OPEN -> HALF_OPEN -> Unhealthy Traffic -> OPEN
        """
        # 1. Start as HALF_OPEN (assuming transition happened)
        self.mock_redis.get.return_value = b"HALF_OPEN"

        # 2. Simulate Unhealthy Traffic (Errors = 10 > 5)
        now = time.time()
        members = [f"{now}:1.0:id{i}".encode("utf-8") for i in range(10)]  # Sum = 10
        self.mock_redis.zrangebyscore.return_value = members

        # 3. Check Triggers
        await self.breaker.check_triggers()

        # 4. Should transition back to OPEN
        self.mock_redis.getset.assert_called_with("sentinel:breaker:complex-agent:state", "OPEN")
        # And reset cooldown
        self.mock_redis.setex.assert_called_with("sentinel:breaker:complex-agent:cooldown", 60, "1")

    async def test_multi_trigger_logic(self) -> None:
        """
        Test that violating ANY trigger trips the breaker.
        """
        self.mock_redis.get.return_value = b"CLOSED"

        # Scenario 1: Latency is fine, Errors high
        # We need to simulate zrangebyscore returning different things for different keys.
        # key format: sentinel:metrics:{agent}:{metric}

        async def zrange_side_effect(key: str, start: float | str, end: float | str) -> list[bytes]:
            if "errors" in key:
                # 10 errors
                return [f"{time.time()}:1.0:id{i}".encode("utf-8") for i in range(10)]
            if "latency" in key:
                # Low latency
                return [f"{time.time()}:100.0:id{i}".encode("utf-8") for i in range(5)]
            return []

        self.mock_redis.zrangebyscore.side_effect = zrange_side_effect

        await self.breaker.check_triggers()
        self.mock_redis.getset.assert_called_with("sentinel:breaker:complex-agent:state", "OPEN")

    async def test_multi_trigger_second_violation(self) -> None:
        """
        Test that violating the SECOND trigger trips it even if first is fine.
        """
        self.mock_redis.get.return_value = b"CLOSED"

        async def zrange_side_effect(key: str, start: float | str, end: float | str) -> list[bytes]:
            if "errors" in key:
                # 0 errors
                return []
            if "latency" in key:
                # High latency (2000 > 1000)
                return [f"{time.time()}:2000.0:id{i}".encode("utf-8") for i in range(5)]
            return []

        self.mock_redis.zrangebyscore.side_effect = zrange_side_effect

        await self.breaker.check_triggers()
        self.mock_redis.getset.assert_called_with("sentinel:breaker:complex-agent:state", "OPEN")
