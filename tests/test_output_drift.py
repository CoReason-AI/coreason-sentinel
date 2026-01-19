# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_sentinel

import unittest
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Union
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import anyio
import numpy as np
import pytest

from coreason_sentinel.circuit_breaker import CircuitBreaker
from coreason_sentinel.drift_engine import DriftEngine
from coreason_sentinel.ingestor import TelemetryIngestorAsync
from coreason_sentinel.interfaces import BaselineProviderProtocol, VeritasEvent
from coreason_sentinel.models import CircuitBreakerTrigger, SentinelConfig


@pytest.mark.asyncio
class TestOutputDriftDetection(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.config = SentinelConfig(
            agent_id="test-agent",
            owner_email="test@example.com",
            phoenix_endpoint="http://localhost:6006",
            drift_sample_window=10,
            triggers=[
                CircuitBreakerTrigger(
                    metric="output_drift_kl",
                    threshold=0.5,
                    window_seconds=60,
                    operator=">",
                    aggregation_method="MAX",
                )
            ],
        )
        self.mock_redis = MagicMock()
        self.mock_redis.zadd = AsyncMock()
        self.mock_redis.zremrangebyscore = AsyncMock()
        self.mock_redis.expire = AsyncMock()
        self.mock_redis.zrevrange = AsyncMock()
        self.mock_redis.zrangebyscore = AsyncMock()
        self.mock_redis.get = AsyncMock()
        self.mock_redis.getset = AsyncMock()
        self.mock_redis.setex = AsyncMock()
        self.mock_redis.exists = AsyncMock()

        self.mock_notification_service = MagicMock()

        self.circuit_breaker = CircuitBreaker(self.mock_redis, self.config, self.mock_notification_service)

        self.baseline_provider = MagicMock(spec=BaselineProviderProtocol)
        # Setup a simple baseline: 3 bins [0-10, 10-20, 20-30]
        # Distribution: mostly in middle bin
        # Probs: [0.1, 0.8, 0.1]
        # Edges: [0, 10, 20, 30]
        self.baseline_provider.get_baseline_output_length_distribution.return_value = (
            [0.1, 0.8, 0.1],
            [0.0, 10.0, 20.0, 30.0],
        )
        self.baseline_provider.get_baseline_vectors.return_value = []

        self.spot_checker = Mock()
        self.spot_checker.should_sample.return_value = False
        self.veritas_client = Mock()

        self.ingestor = TelemetryIngestorAsync(
            self.config, self.circuit_breaker, self.spot_checker, self.baseline_provider, self.veritas_client
        )

    async def test_compute_distribution_from_samples(self) -> None:
        """
        Verify that raw samples are correctly histogrammed into probabilities.
        """
        samples = [5.0, 15.0, 15.0, 25.0]  # 1 in bin 0, 2 in bin 1, 1 in bin 2
        edges = [0.0, 10.0, 20.0, 30.0]

        dist = await anyio.to_thread.run_sync(DriftEngine.compute_distribution_from_samples, samples, edges)

        # Total samples = 4
        # Bin 0 (0-10): 1 sample -> 0.25
        # Bin 1 (10-20): 2 samples -> 0.50
        # Bin 2 (20-30): 1 sample -> 0.25
        expected = [0.25, 0.50, 0.25]

        np.testing.assert_array_almost_equal(dist, expected)

    async def test_compute_distribution_empty_samples(self) -> None:
        """
        Verify empty samples return zero distribution.
        """
        edges = [0.0, 10.0, 20.0]
        dist = await anyio.to_thread.run_sync(DriftEngine.compute_distribution_from_samples, [], edges)
        assert dist == [0.0, 0.0]

    async def test_ingestor_process_output_drift_happy_path(self) -> None:
        """
        Test that process_event calculates KL divergence and records it.
        """
        # Mock Redis to return recent samples that match baseline exactly
        samples = [5.0] + [15.0] * 8 + [25.0]
        mock_members = [f"123:{v}:uuid".encode() for v in samples]
        self.mock_redis.zrevrange.return_value = mock_members

        # Create event with output length
        event = VeritasEvent(
            event_id="e1",
            timestamp=datetime.now(timezone.utc),
            agent_id="test-agent",
            session_id="s1",
            input_text="hi",
            output_text="word " * 15,  # length 15
            metrics={"completion_tokens": 15},
            metadata={},
        )

        with patch("anyio.to_thread.run_sync", side_effect=lambda func, *args: func(*args)):
            await self.ingestor.process_drift(event)

        # Verify calls
        drift_call_found = False
        for call in self.mock_redis.zadd.call_args_list:
            args, _ = call
            key = args[0]
            if "output_drift_kl" in key:
                drift_call_found = True
                payload = args[1]  # dict {member: score}
                member = list(payload.keys())[0]
                val = float(member.split(":")[1])
                assert val < 0.01  # KL should be near 0
                break

        assert drift_call_found

    async def test_ingestor_process_output_drift_high_divergence(self) -> None:
        """
        Test that process_event calculates high KL divergence when distributions differ.
        """
        samples = [25.0] * 10
        mock_members = [f"123:{v}:uuid".encode() for v in samples]
        self.mock_redis.zrevrange.return_value = mock_members

        event = VeritasEvent(
            event_id="e1",
            timestamp=datetime.now(timezone.utc),
            agent_id="test-agent",
            session_id="s1",
            input_text="hi",
            output_text="word",
            metrics={"completion_tokens": 25},
            metadata={},
        )

        with patch("anyio.to_thread.run_sync", side_effect=lambda func, *args: func(*args)):
            await self.ingestor.process_drift(event)

        # Verify drift is high
        drift_call_found = False
        for call in self.mock_redis.zadd.call_args_list:
            args, _ = call
            key = args[0]
            if "output_drift_kl" in key:
                drift_call_found = True
                payload = args[1]
                member = list(payload.keys())[0]
                val = float(member.split(":")[1])
                assert val > 1.0
                break

        assert drift_call_found

    async def test_fallback_output_length_calculation(self) -> None:
        """
        Verify fallback to word count if metrics are missing.
        """
        event = VeritasEvent(
            event_id="e1",
            timestamp=datetime.now(timezone.utc),
            agent_id="test-agent",
            session_id="s1",
            input_text="hi",
            output_text="one two three",  # 3 words
            metrics={},
            metadata={},
        )

        self.mock_redis.zrevrange.return_value = [b"123:3.0:uuid"]

        with patch("anyio.to_thread.run_sync", side_effect=lambda func, *args: func(*args)):
            await self.ingestor.process_drift(event)

        length_call_found = False
        for call in self.mock_redis.zadd.call_args_list:
            args, _ = call
            key = args[0]
            if "output_length" in key:
                length_call_found = True
                payload = args[1]
                member = list(payload.keys())[0]
                val = float(member.split(":")[1])
                assert val == 3.0
                break

        assert length_call_found

    async def test_compute_distribution_zero_total_count(self) -> None:
        """
        Verify that if total samples is non-empty but bins capture nothing (e.g. range mismatch),
        it handles division by zero safely.
        """
        samples = [100.0]
        # Bins from 0 to 10. Sample is 100. Counts will be [0]. Total 0.
        edges = [0.0, 10.0]
        dist = await anyio.to_thread.run_sync(DriftEngine.compute_distribution_from_samples, samples, edges)
        assert dist == [0.0]

    async def test_output_length_from_token_count(self) -> None:
        """Verify output length is extracted from 'token_count' metric."""
        event = VeritasEvent(
            event_id="e2",
            timestamp=datetime.now(timezone.utc),
            agent_id="test-agent",
            session_id="s1",
            input_text="hi",
            output_text="word",
            metrics={"token_count": 42},
            metadata={},
        )
        self.mock_redis.zrevrange.return_value = [b"123:42.0:uuid"]

        with patch("anyio.to_thread.run_sync", side_effect=lambda func, *args: func(*args)):
            await self.ingestor.process_drift(event)

        length_call_found = False
        for call in self.mock_redis.zadd.call_args_list:
            args, _ = call
            key = args[0]
            if "output_length" in key:
                length_call_found = True
                payload = args[1]
                member = list(payload.keys())[0]
                val = float(member.split(":")[1])
                assert val == 42.0
                break

        assert length_call_found

    async def test_gradual_drift_scenario(self) -> None:
        """
        Complex Scenario: Gradual Drift.
        Simulate a sequence of events where output length shifts from matching baseline to deviating.
        Verifies that KL divergence increases and eventually trips the breaker.
        Uses a functional Redis mock.
        """
        # Functional mock for Redis
        redis_store: Dict[str, List[Tuple[float, bytes]]] = {}

        async def mock_zadd(key: str, mapping: Dict[Union[str, bytes], float]) -> int:
            if key not in redis_store:
                redis_store[key] = []
            for m, s in mapping.items():
                redis_store[key].append((s, m if isinstance(m, bytes) else m.encode("utf-8")))
            return 1

        async def mock_zrange(key: str, min_s: Union[float, str], max_s: Union[float, str]) -> List[bytes]:
            if key not in redis_store:
                return []
            sorted_items = sorted(redis_store[key], key=lambda x: x[0])
            return [m for s, m in sorted_items]

        async def mock_zrevrange(key: str, start: int, end: int) -> List[bytes]:
            if key not in redis_store:
                return []
            sorted_items = sorted(redis_store[key], key=lambda x: x[0], reverse=True)
            # simulate range
            return [m for s, m in sorted_items[start : end + 1]]

        async def mock_zremrange(key: str, min_s: Union[float, str], max_s: Union[float, str]) -> None:
            pass

        async def mock_get(key: str) -> bytes:
            return b"CLOSED"

        async def mock_getset(key: str, val: Union[str, bytes]) -> bytes:
            return b"CLOSED"

        self.mock_redis.zadd.side_effect = mock_zadd
        self.mock_redis.zrangebyscore.side_effect = mock_zrange
        self.mock_redis.zrevrange.side_effect = mock_zrevrange
        self.mock_redis.zremrangebyscore.side_effect = mock_zremrange
        self.mock_redis.get.side_effect = mock_get
        self.mock_redis.getset.side_effect = mock_getset

        self.config.drift_sample_window = 5
        self.config.triggers[0].threshold = 0.5

        # Baseline: [0.0, 1.0, 0.0] for bins [0, 10, 20, 30]
        self.baseline_provider.get_baseline_output_length_distribution.return_value = (
            [0.0, 1.0, 0.0],
            [0.0, 10.0, 20.0, 30.0],
        )

        good_event = VeritasEvent(
            event_id="good",
            timestamp=datetime.now(timezone.utc),
            agent_id="test-agent",
            session_id="s1",
            input_text="hi",
            output_text="word",
            metrics={"token_count": 15},
            metadata={},
        )

        with patch("anyio.to_thread.run_sync", side_effect=lambda func, *args: func(*args)):
            for _ in range(5):
                await self.ingestor.process_drift(good_event)

        self.mock_redis.getset.assert_not_called()

        bad_event = VeritasEvent(
            event_id="bad",
            timestamp=datetime.now(timezone.utc),
            agent_id="test-agent",
            session_id="s1",
            input_text="hi",
            output_text="word",
            metrics={"token_count": 25},
            metadata={},
        )

        # Override getset to return CLOSED so it can trip
        self.mock_redis.getset.side_effect = None
        self.mock_redis.getset.return_value = b"CLOSED"

        with patch("anyio.to_thread.run_sync", side_effect=lambda func, *args: func(*args)):
            for _ in range(5):
                await self.ingestor.process_drift(bad_event)

        self.mock_redis.getset.assert_any_call("sentinel:breaker:test-agent:state", "OPEN")

    async def test_sparse_data_distribution(self) -> None:
        """Edge Case: Sparse Data."""
        self.mock_redis.zrevrange.return_value = [b"123:15.0:uuid"]

        event = VeritasEvent(
            event_id="e1",
            timestamp=datetime.now(timezone.utc),
            agent_id="test-agent",
            session_id="s1",
            input_text="hi",
            output_text="word",
            metrics={"token_count": 15},
            metadata={},
        )

        with patch("anyio.to_thread.run_sync", side_effect=lambda func, *args: func(*args)):
            await self.ingestor.process_drift(event)

        drift_call_found = False
        for call in self.mock_redis.zadd.call_args_list:
            args, _ = call
            key = args[0]
            if "output_drift_kl" in key:
                drift_call_found = True
                break
        assert drift_call_found

    async def test_samples_completely_outside_bins(self) -> None:
        """Edge Case: Samples far outside baseline bins."""
        self.mock_redis.zrevrange.return_value = [b"123:1000.0:uuid"]

        event = VeritasEvent(
            event_id="e1",
            timestamp=datetime.now(timezone.utc),
            agent_id="test-agent",
            session_id="s1",
            input_text="hi",
            output_text="word",
            metrics={"token_count": 1000},
            metadata={},
        )

        with patch("anyio.to_thread.run_sync", side_effect=lambda func, *args: func(*args)):
            await self.ingestor.process_drift(event)

        drift_val = 0.0
        for call in self.mock_redis.zadd.call_args_list:
            args, _ = call
            key = args[0]
            if "output_drift_kl" in key:
                payload = args[1]
                member = list(payload.keys())[0]
                drift_val = float(member.split(":")[1])
                break

        assert drift_val > 0.4
