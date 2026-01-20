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
from unittest.mock import AsyncMock, MagicMock

import pytest
from redis.asyncio import Redis

from coreason_sentinel.circuit_breaker import CircuitBreaker
from coreason_sentinel.interfaces import NotificationServiceProtocol
from coreason_sentinel.models import HealthReport, SentinelConfig


@pytest.mark.asyncio
class TestCircuitBreakerHealthReport(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.mock_redis = MagicMock(spec=Redis)
        self.mock_redis.get = AsyncMock()
        self.mock_redis.zrangebyscore = AsyncMock()
        self.mock_redis.exists = AsyncMock()

        self.mock_notification_service = MagicMock(spec=NotificationServiceProtocol)
        self.config = SentinelConfig(
            agent_id="test-agent",
            owner_email="test@example.com",
            phoenix_endpoint="http://localhost:6006",
            triggers=[],
        )
        self.breaker = CircuitBreaker(self.mock_redis, self.config, self.mock_notification_service)

    async def test_get_health_report_structure(self) -> None:
        """Test that get_health_report returns a valid HealthReport object."""
        self.mock_redis.get.return_value = b"CLOSED"
        # Mock empty lists for metrics
        self.mock_redis.zrangebyscore.return_value = []

        report = await self.breaker.get_health_report()

        self.assertIsInstance(report, HealthReport)
        self.assertEqual(report.breaker_state, "CLOSED")
        self.assertIsInstance(report.metrics, dict)
        # Check default keys exist
        expected_keys = ["avg_latency", "faithfulness", "cost_per_query", "kl_divergence"]
        for key in expected_keys:
            self.assertIn(key, report.metrics)

    async def test_get_health_report_calculation(self) -> None:
        """Test that metrics are correctly averaged."""
        self.mock_redis.get.return_value = b"CLOSED"
        now = time.time()

        # Setup mock data
        # Latency: 0.1, 0.3 -> Avg 0.2
        # Faithfulness: 0.9, 1.0 -> Avg 0.95
        # Cost: 0.01, 0.03 -> Avg 0.02
        # KL: 0.2, 0.4 -> Avg 0.3

        async def zrange_side_effect(key: str, min_score: float | str, max_score: float | str) -> list[bytes]:
            if "latency" in key:
                return [f"{now}:0.1:id1".encode(), f"{now}:0.3:id2".encode()]
            if "faithfulness" in key:
                return [f"{now}:0.9:id3".encode(), f"{now}:1.0:id4".encode()]
            if "cost" in key:
                return [f"{now}:0.01:id5".encode(), f"{now}:0.03:id6".encode()]
            if "output_drift_kl" in key:
                return [f"{now}:0.2:id7".encode(), f"{now}:0.4:id8".encode()]
            return []

        self.mock_redis.zrangebyscore.side_effect = zrange_side_effect

        report = await self.breaker.get_health_report()

        self.assertAlmostEqual(report.metrics["avg_latency"], 0.2)
        self.assertAlmostEqual(report.metrics["faithfulness"], 0.95)
        self.assertAlmostEqual(report.metrics["cost_per_query"], 0.02)
        self.assertAlmostEqual(report.metrics["kl_divergence"], 0.3)

    async def test_get_health_report_window(self) -> None:
        """Test that metrics respect the 1-hour window."""
        self.mock_redis.get.return_value = b"CLOSED"

        # We need to verify that zrangebyscore is called with the correct start time
        self.mock_redis.zrangebyscore.return_value = []

        now = time.time()
        # Patch time to control 'now'
        with unittest.mock.patch("time.time", return_value=now):
            await self.breaker.get_health_report()

            # Check arguments for one of the calls
            # Expected start time is now - 3600
            expected_start = now - 3600

            # Verify calls
            # args[0] is key, args[1] is min, args[2] is max
            # We iterate through calls to find latency call
            found = False
            for call in self.mock_redis.zrangebyscore.call_args_list:
                args = call.args
                if "latency" in args[0]:
                    self.assertAlmostEqual(args[1], expected_start, delta=1.0)
                    self.assertEqual(args[2], "+inf")
                    found = True
                    break
            self.assertTrue(found, "Latency metric not queried")

    async def test_get_health_report_empty_metrics(self) -> None:
        """Test that empty metrics default to 0.0."""
        self.mock_redis.get.return_value = b"CLOSED"
        self.mock_redis.zrangebyscore.return_value = []

        report = await self.breaker.get_health_report()

        self.assertEqual(report.metrics["avg_latency"], 0.0)
        self.assertEqual(report.metrics["faithfulness"], 0.0)
        self.assertEqual(report.metrics["cost_per_query"], 0.0)
        self.assertEqual(report.metrics["kl_divergence"], 0.0)

    async def test_get_health_report_state_handling(self) -> None:
        """Test that the current state is correctly reflected."""
        self.mock_redis.get.return_value = b"OPEN"
        # If open and cooldown exists
        self.mock_redis.exists.return_value = 1

        report = await self.breaker.get_health_report()
        self.assertEqual(report.breaker_state, "OPEN")


@pytest.mark.asyncio
class TestCircuitBreakerHealthReportEdgeCases(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.mock_redis = MagicMock(spec=Redis)
        self.mock_redis.get = AsyncMock()
        self.mock_redis.zrangebyscore = AsyncMock()
        self.mock_redis.exists = AsyncMock()

        self.mock_notification_service = MagicMock(spec=NotificationServiceProtocol)
        self.config = SentinelConfig(
            agent_id="test-agent",
            owner_email="test@example.com",
            phoenix_endpoint="http://localhost:6006",
            triggers=[],
        )
        self.breaker = CircuitBreaker(self.mock_redis, self.config, self.mock_notification_service)

    async def test_zero_values(self) -> None:
        """
        Verify that valid 0.0 values (e.g. 0 cost) are correctly averaged
        and not treated as missing data.
        """
        self.mock_redis.get.return_value = b"CLOSED"
        now = time.time()

        # 3 events, two are 0.0, one is 3.0. Average should be 1.0.
        # If 0.0 was ignored, average would be 3.0.
        members = [f"{now}:0.0:id1".encode(), f"{now}:0.0:id2".encode(), f"{now}:3.0:id3".encode()]

        # Return these members for 'cost' metric
        async def zrange_side_effect(key: str, *args: float | str, **kwargs: float | str) -> list[bytes]:
            if "cost" in key:
                return members
            return []

        self.mock_redis.zrangebyscore.side_effect = zrange_side_effect

        report = await self.breaker.get_health_report()

        self.assertAlmostEqual(report.metrics["cost_per_query"], 1.0)
        self.assertEqual(len(members), 3)

    async def test_mixed_valid_and_malformed_data(self) -> None:
        """
        Verify behavior when Redis contains malformed data.
        Current implementation fallback is 1.0.
        """
        self.mock_redis.get.return_value = b"CLOSED"
        now = time.time()

        # 1 valid (4.0), 1 malformed (defaults to 1.0). Average -> 2.5
        members = [f"{now}:4.0:id1".encode(), b"malformed_data"]

        async def zrange_side_effect(key: str, *args: float | str, **kwargs: float | str) -> list[bytes]:
            if "faithfulness" in key:
                return members
            return []

        self.mock_redis.zrangebyscore.side_effect = zrange_side_effect

        report = await self.breaker.get_health_report()

        # Expect 2.5 because malformed -> 1.0
        self.assertAlmostEqual(report.metrics["faithfulness"], 2.5)

    async def test_sparse_metrics(self) -> None:
        """
        Verify report generation when only one metric type exists.
        """
        self.mock_redis.get.return_value = b"CLOSED"
        now = time.time()

        async def zrange_side_effect(key: str, *args: float | str, **kwargs: float | str) -> list[bytes]:
            if "avg_latency" in key or "latency" in key:
                return [f"{now}:0.5:id1".encode()]
            # Others return empty
            return []

        self.mock_redis.zrangebyscore.side_effect = zrange_side_effect

        report = await self.breaker.get_health_report()

        self.assertAlmostEqual(report.metrics["avg_latency"], 0.5)
        self.assertEqual(report.metrics["faithfulness"], 0.0)
        self.assertEqual(report.metrics["cost_per_query"], 0.0)

    async def test_large_volume_aggregation(self) -> None:
        """
        Simulate 1000 data points to ensure aggregation logic holds up.
        """
        self.mock_redis.get.return_value = b"CLOSED"
        now = time.time()

        # 1000 items with value 1.0. Average should be 1.0.
        members = [f"{now}:1.0:id{i}".encode() for i in range(1000)]

        async def zrange_side_effect(key: str, *args: float | str, **kwargs: float | str) -> list[bytes]:
            if "latency" in key:
                return members
            return []

        self.mock_redis.zrangebyscore.side_effect = zrange_side_effect

        start_time = time.time()
        report = await self.breaker.get_health_report()
        end_time = time.time()

        self.assertAlmostEqual(report.metrics["avg_latency"], 1.0)
        # Ensure it's reasonably fast (mock overhead exists, but logic is O(N))
        self.assertLess(end_time - start_time, 1.0)

    async def test_boundary_window_inclusion(self) -> None:
        """
        Verify events at exactly `now - 3600` are included (inclusive behavior),
        and `now` are included.
        NOTE: zrangebyscore(min, max) is inclusive.
        """
        self.mock_redis.get.return_value = b"CLOSED"
        now = 10000.0  # Fixed time

        with unittest.mock.patch("time.time", return_value=now):
            await self.breaker.get_health_report()

            # Check the call arguments for latency
            # We assume the implementation uses `now - window`
            expected_start = now - 3600

            for call in self.mock_redis.zrangebyscore.call_args_list:
                args = call.args
                if "latency" in args[0]:
                    # Assert that we are asking for range starting at exactly expected_start
                    self.assertEqual(args[1], expected_start)
