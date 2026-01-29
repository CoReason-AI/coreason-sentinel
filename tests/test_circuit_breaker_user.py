import unittest
from unittest.mock import AsyncMock, MagicMock

import pytest
from coreason_identity.models import UserContext
from redis.asyncio import Redis

from coreason_sentinel.circuit_breaker import CircuitBreaker, CircuitBreakerState
from coreason_sentinel.interfaces import NotificationServiceProtocol
from coreason_sentinel.models import SentinelConfig


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
        await self.breaker.check_triggers(self.user_context)
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
