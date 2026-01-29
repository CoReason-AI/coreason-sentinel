import unittest
from unittest.mock import AsyncMock, MagicMock

import pytest
from coreason_identity.models import UserContext
from redis.asyncio import Redis

from coreason_sentinel.circuit_breaker import CircuitBreaker, CircuitBreakerState
from coreason_sentinel.interfaces import NotificationServiceProtocol
from coreason_sentinel.models import CircuitBreakerTrigger, SentinelConfig


@pytest.mark.asyncio
class TestCircuitBreakerUserContext(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.mock_redis = MagicMock(spec=Redis)
        self.mock_redis.get = AsyncMock(return_value=b"CLOSED")
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
        self.user_context = UserContext(user_id="user123", sub="user123", email="u@e.com")

    async def test_get_state_user(self) -> None:
        """Test getting state for specific user."""
        await self.breaker.get_state(self.user_context)
        self.mock_redis.get.assert_called_with("sentinel:breaker:test-agent:user123:state")

    async def test_set_state_user(self) -> None:
        """Test setting state for specific user."""
        await self.breaker.set_state(CircuitBreakerState.OPEN, reason="bad user", user_context=self.user_context)
        self.mock_redis.getset.assert_called_with("sentinel:breaker:test-agent:user123:state", "OPEN")
        self.mock_redis.setex.assert_called_with("sentinel:breaker:test-agent:user123:cooldown", 60, "1")

        # Verify alert includes user info
        self.mock_notification_service.send_critical_alert.assert_called_with(
            email="test@example.com", agent_id="test-agent", reason="bad user (User: user123)"
        )

    async def test_record_metric_user(self) -> None:
        """Test recording metric for user."""
        await self.breaker.record_metric("cost", 1.0, self.user_context)

        # Should record GLOBAL
        self.mock_redis.zadd.assert_any_call("sentinel:metrics:test-agent:cost", unittest.mock.ANY)
        # Should record USER
        self.mock_redis.zadd.assert_any_call("sentinel:metrics:test-agent:user123:cost", unittest.mock.ANY)

    async def test_check_triggers_user(self) -> None:
        """Test checking triggers for user."""
        # Add a trigger to ensure _evaluate_trigger is called
        self.config.triggers.append(CircuitBreakerTrigger(metric="cost", threshold=10.0, window_seconds=60))
        await self.breaker.check_triggers(self.user_context)
        # Verify redis zrangebyscore called with user key
        args = self.mock_redis.zrangebyscore.call_args[0]
        assert args[0] == "sentinel:metrics:test-agent:user123:cost"

    async def test_trigger_violation_user(self) -> None:
        """Test that a trigger violation for a user trips the breaker."""
        # Setup trigger: Cost > 10
        self.config.triggers.append(CircuitBreakerTrigger(metric="cost", threshold=10.0, window_seconds=60))

        # Mock Redis returning values that violate the trigger
        # zrangebyscore returns list of members: "timestamp:value:uuid"
        # We need sum > 10. Let's return one event with value 15.0
        member = b"1234567890:15.0:uuid"
        self.mock_redis.zrangebyscore.return_value = [member]

        # Call check_triggers
        await self.breaker.check_triggers(self.user_context)

        # Verify transition to OPEN
        # set_state is called. We can check the mock or verify behavior if we mocked the internal methods?
        # Since we are mocking redis, the actual state set involves redis.getset
        # But we want to ensure line 313 is hit.

        # Verify set_state was called with OPEN
        # The first call to set_state might be for global, second for user if global passed?
        # Actually check_triggers calls global then user.
        # If global doesn't trip, user logic runs.
        # We mocked redis to return the violation for the USER key or GLOBAL key?
        # zrangebyscore is called with specific key.
        # We need to make sure when it's called with USER key, it returns violation.

        # Refine mock to return violation only for user key
        def side_effect(key: str, min_score: float, max_score: float) -> list[bytes]:
            if "user123" in key and "cost" in key:
                return [b"1234567890:15.0:uuid"]
            return []

        self.mock_redis.zrangebyscore.side_effect = side_effect

        await self.breaker.check_triggers(self.user_context)

        # Verify set_state called for user
        # We can verify log or redis calls.
        # set_state calls redis.getset(state_key, ...)
        # Check if getset called with user state key and OPEN
        user_state_key = "sentinel:breaker:test-agent:user123:state"

        # Check that getset was called with this key and OPEN
        calls = self.mock_redis.getset.call_args_list
        # Filter for our key
        matching_calls = [c for c in calls if c[0][0] == user_state_key and c[0][1] == CircuitBreakerState.OPEN.value]
        assert len(matching_calls) > 0

    async def test_recovery_user(self) -> None:
        """Test that a user recovers from HALF_OPEN to CLOSED if no violation."""
        # Setup: State is HALF_OPEN
        user_state_key = "sentinel:breaker:test-agent:user123:state"

        async def get_mock(key: str) -> bytes | None:
            if key == user_state_key:
                return b"HALF_OPEN"
            return None

        self.mock_redis.get.side_effect = get_mock

        # No triggers violated (returns empty list by default if not mocked otherwise)
        # Ensure zrangebyscore returns empty list
        self.mock_redis.zrangebyscore.return_value = []

        await self.breaker.check_triggers(self.user_context)

        # Verify transition to CLOSED
        calls = self.mock_redis.getset.call_args_list
        matching_calls = [c for c in calls if c[0][0] == user_state_key and c[0][1] == CircuitBreakerState.CLOSED.value]
        assert len(matching_calls) > 0

    async def test_get_recent_values_user(self) -> None:
        """Test getting recent values for user."""
        await self.breaker.get_recent_values("latency", 10, self.user_context)
        self.mock_redis.zrevrange.assert_called()
        args = self.mock_redis.zrevrange.call_args[0]
        assert args[0] == "sentinel:metrics:test-agent:user123:latency"
        # Should check global (implicit in implementation if called with None first?)
        # Implementation: calls _check_triggers_internal(None) then (user_context)

        # We can verify by looking at zrangebyscore calls if triggers existed
        pass

    async def test_allow_request_user_isolation(self) -> None:
        """Test user isolation logic."""
        # 1. Global CLOSED, User CLOSED -> True
        self.mock_redis.get.return_value = b"CLOSED"
        assert await self.breaker.allow_request(self.user_context) is True

        # 2. Global OPEN -> False (even if User CLOSED)
        self.mock_redis.get.side_effect = lambda k: b"OPEN" if "user123" not in k else b"CLOSED"
        # Wait, get_state(None) calls get(global_key)
        # get_state(user) calls get(user_key)

        # Let's mock side effects properly
        async def get_side_effect(key: str) -> bytes:
            if "user123" in key:
                return b"CLOSED"
            return b"OPEN"

        self.mock_redis.get.side_effect = get_side_effect
        self.mock_redis.exists.return_value = 1  # Cooldown active

        assert await self.breaker.allow_request(self.user_context) is False

        # 3. Global CLOSED, User OPEN -> False
        async def get_side_effect_2(key: str) -> bytes:
            if "user123" in key:
                return b"OPEN"
            return b"CLOSED"

        self.mock_redis.get.side_effect = get_side_effect_2

        assert await self.breaker.allow_request(self.user_context) is False
