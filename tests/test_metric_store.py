# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_sentinel

import math
import time
import unittest
from unittest.mock import MagicMock, call

from redis import Redis

from coreason_sentinel.metric_store import MetricStore


class TestMetricStore(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_redis = MagicMock(spec=Redis)
        self.metric_store = MetricStore(self.mock_redis)
        self.agent_id = "test_agent"
        self.metric_name = "latency"

    def test_record_metric_success(self) -> None:
        """Test recording a valid metric."""
        self.metric_store.record_metric(self.agent_id, self.metric_name, 0.5, retention_window=60)

        # Verify zadd called
        self.mock_redis.zadd.assert_called_once()
        args, _ = self.mock_redis.zadd.call_args
        key = args[0]
        mapping = args[1]
        self.assertEqual(key, f"sentinel:metrics:{self.agent_id}:{self.metric_name}")
        self.assertEqual(len(mapping), 1)

        # Verify cleanup
        self.mock_redis.zremrangebyscore.assert_called_once()
        self.mock_redis.expire.assert_called_once()

    def test_record_metric_invalid_value(self) -> None:
        """Test recording NaN or Inf is ignored."""
        self.metric_store.record_metric(self.agent_id, self.metric_name, float("nan"))
        self.mock_redis.zadd.assert_not_called()

        self.metric_store.record_metric(self.agent_id, self.metric_name, float("inf"))
        self.mock_redis.zadd.assert_not_called()

    def test_record_metric_exception_handling(self) -> None:
        """Test that Redis exceptions are caught and logged."""
        self.mock_redis.zadd.side_effect = Exception("Redis down")
        # Should not raise
        self.metric_store.record_metric(self.agent_id, self.metric_name, 0.5)

    def test_get_recent_values(self) -> None:
        """Test retrieving recent values."""
        # Mock zrevrange return
        # Format: timestamp:value:uuid
        now = time.time()
        m1 = f"{now}:1.0:uuid1".encode("utf-8")
        m2 = f"{now}:2.0:uuid2".encode("utf-8")
        self.mock_redis.zrevrange.return_value = [m1, m2]

        values = self.metric_store.get_recent_values(self.agent_id, self.metric_name, limit=2)
        self.assertEqual(values, [1.0, 2.0])

    def test_get_recent_values_empty(self) -> None:
        """Test retrieving when no values exist."""
        self.mock_redis.zrevrange.return_value = []
        values = self.metric_store.get_recent_values(self.agent_id, self.metric_name)
        self.assertEqual(values, [])

    def test_get_recent_values_parsing_error(self) -> None:
        """Test graceful handling of malformed data."""
        self.mock_redis.zrevrange.return_value = [b"malformed_string"]
        values = self.metric_store.get_recent_values(self.agent_id, self.metric_name)
        # Should return 1.0 (fallback)
        self.assertEqual(values, [1.0])

    def test_get_recent_values_exception(self) -> None:
        """Test Redis exception during fetch."""
        self.mock_redis.zrevrange.side_effect = Exception("Redis error")
        values = self.metric_store.get_recent_values(self.agent_id, self.metric_name)
        self.assertEqual(values, [])

    def test_get_values_in_window(self) -> None:
        """Test retrieving values within a time window."""
        now = time.time()
        m1 = f"{now}:10.0:uuid1".encode("utf-8")
        self.mock_redis.zrangebyscore.return_value = [m1]

        values = self.metric_store.get_values_in_window(self.agent_id, self.metric_name, window_seconds=60)
        self.assertEqual(values, [10.0])

    def test_calculate_average(self) -> None:
        """Test average calculation."""
        now = time.time()
        m1 = f"{now}:10.0:uuid1".encode("utf-8")
        m2 = f"{now}:20.0:uuid2".encode("utf-8")
        self.mock_redis.zrangebyscore.return_value = [m1, m2]

        avg = self.metric_store.calculate_average(self.agent_id, self.metric_name)
        self.assertEqual(avg, 15.0)

    def test_calculate_average_no_data(self) -> None:
        """Test average returns 0.0 when no data."""
        self.mock_redis.zrangebyscore.return_value = []
        avg = self.metric_store.calculate_average(self.agent_id, self.metric_name)
        self.assertEqual(avg, 0.0)

    def test_get_values_in_window_exception(self) -> None:
        """Test Redis exception during window fetch."""
        self.mock_redis.zrangebyscore.side_effect = Exception("Redis error")
        values = self.metric_store.get_values_in_window(self.agent_id, self.metric_name, 60)
        self.assertEqual(values, [])

    def test_parse_value_failure(self) -> None:
        """Test parsing failure when value is not a float."""
        # This will call _parse_value_from_member
        now = time.time()
        m1 = f"{now}:not_a_float:uuid".encode("utf-8")
        self.mock_redis.zrevrange.return_value = [m1]

        values = self.metric_store.get_recent_values(self.agent_id, self.metric_name)
        # float("not_a_float") raises ValueError, caught, returns 1.0 (fallback)
        self.assertEqual(values, [1.0])
