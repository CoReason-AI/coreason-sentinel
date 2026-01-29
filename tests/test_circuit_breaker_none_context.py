import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.asyncio import Redis

from coreason_sentinel.circuit_breaker import CircuitBreaker, CircuitBreakerState
from coreason_sentinel.interfaces import NotificationServiceProtocol
from coreason_sentinel.models import SentinelConfig

@pytest.mark.asyncio
class TestCircuitBreakerNoneContext(unittest.IsolatedAsyncioTestCase):
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
        )
        self.breaker = CircuitBreaker(self.mock_redis, self.config, self.mock_notification_service)

    async def test_get_state_none(self) -> None:
        await self.breaker.get_state(None)
        # Should check global key
        self.mock_redis.get.assert_called_with("sentinel:breaker:test-agent:state")

    async def test_record_metric_none(self) -> None:
        await self.breaker.record_metric("latency", 0.1, None)
        self.mock_redis.zadd.assert_called()
        args = self.mock_redis.zadd.call_args[0]
        # Check key is global
        assert args[0] == "sentinel:metrics:test-agent:latency"

    async def test_check_triggers_none(self) -> None:
        await self.breaker.check_triggers(None)
        # Should access global state
        self.mock_redis.get.assert_called()

    async def test_set_state_none(self) -> None:
        await self.breaker.set_state(CircuitBreakerState.OPEN, reason="test", user_context=None)
        self.mock_redis.getset.assert_called_with("sentinel:breaker:test-agent:state", "OPEN")

    async def test_allow_request_none(self) -> None:
        allowed = await self.breaker.allow_request(None)
        assert allowed is True

    async def test_get_state_empty_context(self) -> None:
        # Mock empty context
        mock_ctx = MagicMock()
        del mock_ctx.user_id
        del mock_ctx.sub
        await self.breaker.get_state(mock_ctx) # type: ignore
        self.mock_redis.get.assert_called_with("sentinel:breaker:test-agent:state")

    async def test_record_metric_empty_context(self) -> None:
        mock_ctx = MagicMock()
        del mock_ctx.user_id
        del mock_ctx.sub
        await self.breaker.record_metric("latency", 0.1, mock_ctx) # type: ignore
        args = self.mock_redis.zadd.call_args[0]
        assert args[0] == "sentinel:metrics:test-agent:latency"

    async def test_set_state_empty_context(self) -> None:
        mock_ctx = MagicMock()
        del mock_ctx.user_id
        del mock_ctx.sub
        await self.breaker.set_state(CircuitBreakerState.OPEN, reason="test", user_context=mock_ctx) # type: ignore
        self.mock_redis.getset.assert_called_with("sentinel:breaker:test-agent:state", "OPEN")
