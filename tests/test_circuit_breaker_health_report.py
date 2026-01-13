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

from coreason_sentinel.circuit_breaker import CircuitBreaker
from coreason_sentinel.interfaces import NotificationServiceProtocol
from coreason_sentinel.models import HealthReport, SentinelConfig


class TestCircuitBreakerHealthReport(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_redis = MagicMock(spec=Redis)
        self.mock_notification_service = MagicMock(spec=NotificationServiceProtocol)
        self.config = SentinelConfig(
            agent_id="test-agent",
            owner_email="test@example.com",
            phoenix_endpoint="http://localhost:6006",
            triggers=[],
        )
        self.breaker = CircuitBreaker(self.mock_redis, self.config, self.mock_notification_service)

    def test_get_health_report_structure(self) -> None:
        """Test that get_health_report returns a valid HealthReport object."""
        self.mock_redis.get.return_value = b"CLOSED"
        # Mock empty lists for metrics
        self.mock_redis.zrangebyscore.return_value = []

        report = self.breaker.get_health_report()

        self.assertIsInstance(report, HealthReport)
        self.assertEqual(report.breaker_state, "CLOSED")
        self.assertIsInstance(report.metrics, dict)
        # Check default keys exist
        expected_keys = ["avg_latency", "faithfulness", "cost_per_query", "kl_divergence"]
        for key in expected_keys:
            self.assertIn(key, report.metrics)

    def test_get_health_report_calculation(self) -> None:
        """Test that metrics are correctly averaged."""
        self.mock_redis.get.return_value = b"CLOSED"
        now = time.time()

        # Setup mock data
        # Latency: 0.1, 0.3 -> Avg 0.2
        # Faithfulness: 0.9, 1.0 -> Avg 0.95
        # Cost: 0.01, 0.03 -> Avg 0.02
        # KL: 0.2, 0.4 -> Avg 0.3

        def zrange_side_effect(key: str, min_score: float | str, max_score: float | str) -> list[bytes]:
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

        report = self.breaker.get_health_report()

        self.assertAlmostEqual(report.metrics["avg_latency"], 0.2)
        self.assertAlmostEqual(report.metrics["faithfulness"], 0.95)
        self.assertAlmostEqual(report.metrics["cost_per_query"], 0.02)
        self.assertAlmostEqual(report.metrics["kl_divergence"], 0.3)

    def test_get_health_report_window(self) -> None:
        """Test that metrics respect the 1-hour window."""
        self.mock_redis.get.return_value = b"CLOSED"

        # We need to verify that zrangebyscore is called with the correct start time
        self.mock_redis.zrangebyscore.return_value = []

        now = time.time()
        # Patch time to control 'now'
        with unittest.mock.patch("time.time", return_value=now):
            self.breaker.get_health_report()

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

    def test_get_health_report_empty_metrics(self) -> None:
        """Test that empty metrics default to 0.0."""
        self.mock_redis.get.return_value = b"CLOSED"
        self.mock_redis.zrangebyscore.return_value = []

        report = self.breaker.get_health_report()

        self.assertEqual(report.metrics["avg_latency"], 0.0)
        self.assertEqual(report.metrics["faithfulness"], 0.0)
        self.assertEqual(report.metrics["cost_per_query"], 0.0)
        self.assertEqual(report.metrics["kl_divergence"], 0.0)

    def test_get_health_report_state_handling(self) -> None:
        """Test that the current state is correctly reflected."""
        self.mock_redis.get.return_value = b"OPEN"
        # We need to ensure get_state handles the logic (cooldown etc)
        # But here we mock redis.get, and get_state logic might transition.
        # Let's mock get_state directly if we want to isolate report generation
        # But testing integration with get_state is better.

        # If open and cooldown exists
        self.mock_redis.exists.return_value = 1

        report = self.breaker.get_health_report()
        self.assertEqual(report.breaker_state, "OPEN")
