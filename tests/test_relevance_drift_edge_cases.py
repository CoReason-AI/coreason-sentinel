# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_sentinel

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from coreason_sentinel.circuit_breaker import CircuitBreaker
from coreason_sentinel.drift_engine import DriftEngine
from coreason_sentinel.ingestor import TelemetryIngestor
from coreason_sentinel.interfaces import VeritasEvent
from coreason_sentinel.models import SentinelConfig


class TestRelevanceDriftEdgeCases:
    def test_empty_vectors(self) -> None:
        """
        Test behavior with empty vectors.
        Should probably raise ValueError because cosine of empty vectors is undefined.
        """
        v1: list[float] = []
        v2: list[float] = []

        try:
            DriftEngine.compute_relevance_drift(v1, v2)
        except ValueError:
            pass
        except Exception as e:
            pytest.fail(f"Raised unexpected exception: {e}")

    def test_non_numeric_data(self) -> None:
        """
        Test with non-numeric data in vectors.
        """
        v1 = [1.0, "banana"]
        v2 = [1.0, 0.0]

        with pytest.raises(ValueError):
            DriftEngine.compute_relevance_drift(v1, v2)  # type: ignore

    def test_high_dimensional_vectors(self) -> None:
        """
        Test with large vectors (e.g. OpenAI embedding size).
        """
        dim = 1536
        rng = np.random.default_rng(42)
        v1 = rng.random(dim).tolist()
        v2 = rng.random(dim).tolist()

        drift = DriftEngine.compute_relevance_drift(v1, v2)
        assert 0.0 <= drift <= 2.0
        assert isinstance(drift, float)


class TestComplexRelevanceScenario:
    @pytest.fixture
    def full_stack(self) -> tuple[TelemetryIngestor, CircuitBreaker, MagicMock]:
        # Mock Redis
        redis_mock = MagicMock()
        storage: dict[str, list[tuple[float, str]]] = {}

        async def zadd(key: str, mapping: dict[str, float]) -> int:
            if key not in storage:
                storage[key] = []
            for member, score in mapping.items():
                storage[key].append((score, member))
            return 1

        async def zrangebyscore(key: str, min_s: float | str, max_s: float | str) -> list[str]:
            if key not in storage:
                return []
            return [m for s, m in storage[key]]

        async def zremrangebyscore(key: str, min_s: float | str, max_s: float | str) -> int:
            return 0

        async def expire(key: str, time: int) -> bool:
            return True

        async def get(key: str) -> bytes:
            return b"CLOSED"

        redis_mock.zadd = AsyncMock(side_effect=zadd)
        redis_mock.zrangebyscore = AsyncMock(side_effect=zrangebyscore)
        redis_mock.zremrangebyscore = AsyncMock(side_effect=zremrangebyscore)
        redis_mock.expire = AsyncMock(side_effect=expire)
        redis_mock.get = AsyncMock(side_effect=get)

        config = SentinelConfig(
            agent_id="complex_agent",
            owner_email="admin@example.com",
            phoenix_endpoint="http://localhost",
        )

        notification = MagicMock()
        breaker = CircuitBreaker(redis_mock, config, notification)

        spot_checker = MagicMock()
        baseline_provider = MagicMock()
        veritas_client = MagicMock()

        ingestor = TelemetryIngestor(config, breaker, spot_checker, baseline_provider, veritas_client)

        return ingestor, breaker, redis_mock

    def test_relevance_drift_sequence(self, full_stack: tuple[TelemetryIngestor, CircuitBreaker, MagicMock]) -> None:
        """
        Process a sequence of events with different embedding drifts.
        Verify they are recorded in the circuit breaker.
        """
        ingestor, breaker, redis_mock = full_stack

        # 1. Event with 0 drift (Identical)
        evt1 = VeritasEvent(
            event_id="e1",
            timestamp=datetime.now(),
            agent_id="complex_agent",
            session_id="s1",
            input_text="a",
            output_text="b",
            metrics={},
            metadata={"query_embedding": [1.0, 0.0], "response_embedding": [1.0, 0.0]},
        )

        # 2. Event with 1.0 drift (Orthogonal)
        evt2 = VeritasEvent(
            event_id="e2",
            timestamp=datetime.now(),
            agent_id="complex_agent",
            session_id="s1",
            input_text="a",
            output_text="b",
            metrics={},
            metadata={"query_embedding": [1.0, 0.0], "response_embedding": [0.0, 1.0]},
        )

        # 3. Event with ~0.29 drift (45 degrees)
        evt3 = VeritasEvent(
            event_id="e3",
            timestamp=datetime.now(),
            agent_id="complex_agent",
            session_id="s1",
            input_text="a",
            output_text="b",
            metrics={},
            metadata={"query_embedding": [1.0, 0.0], "response_embedding": [1.0, 1.0]},
        )

        with patch("anyio.to_thread.run_sync", side_effect=lambda func, *args: func(*args)):
            with ingestor:
                ingestor.process_drift(evt1)
                ingestor.process_drift(evt2)
                ingestor.process_drift(evt3)

        key = "sentinel:metrics:complex_agent:relevance_drift"

        assert redis_mock.zadd.call_count >= 3

        calls = [c for c in redis_mock.zadd.call_args_list if c[0][0] == key]
        assert len(calls) == 3

        values = []
        for call in calls:
            mapping = call[0][1]
            for member in mapping.keys():
                val_str = str(member).split(":")[1]
                values.append(float(val_str))

        assert 0.0 in values
        assert 1.0 in values
        assert any(abs(v - 0.29289) < 0.001 for v in values)
