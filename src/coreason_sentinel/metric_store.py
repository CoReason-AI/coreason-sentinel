# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_sentinel

from __future__ import annotations

import math
import time
import uuid

from redis import Redis

from coreason_sentinel.utils.logger import logger


class MetricStore:
    """
    Manages the storage and retrieval of time-series metrics using Redis.
    Uses Sorted Sets (ZSET) where score=timestamp and member="{timestamp}:{value}:{uuid}".
    """

    def __init__(self, redis_client: Redis[bytes]):
        self.redis = redis_client

    def record_metric(
        self, agent_id: str, metric_name: str, value: float, retention_window: int = 3600
    ) -> None:
        """
        Records a metric event into a Redis Sorted Set (Sliding Window).
        """
        if not math.isfinite(value):
            logger.warning(f"Ignoring invalid metric value: {value} for {metric_name}")
            return

        key = self._get_key(agent_id, metric_name)
        timestamp = time.time()
        # Unique member to allow multiple events at same timestamp
        member = f"{timestamp}:{value}:{uuid.uuid4()}"

        try:
            # Add event to sorted set
            self.redis.zadd(key, {member: timestamp})

            # Remove elements older than the window
            min_score = timestamp - retention_window
            self.redis.zremrangebyscore(key, "-inf", min_score)

            # Set expiry to ensure cleanup if inactive (2x window is safe buffer)
            self.redis.expire(key, retention_window * 2)
        except Exception as e:
            logger.error(f"Failed to record metric {metric_name}: {e}")

    def get_recent_values(self, agent_id: str, metric_name: str, limit: int = 100) -> list[float]:
        """
        Retrieves the most recent raw values for a given metric.
        """
        key = self._get_key(agent_id, metric_name)
        try:
            # Get the last `limit` elements (newest first)
            events = self.redis.zrevrange(key, 0, limit - 1)
            if not events:
                return []

            return [self._parse_value_from_member(m) for m in events]
        except Exception as e:
            logger.error(f"Failed to fetch recent values for {metric_name}: {e}")
            return []

    def get_values_in_window(self, agent_id: str, metric_name: str, window_seconds: int) -> list[float]:
        """
        Retrieves all values falling within the specified time window ending now.
        """
        key = self._get_key(agent_id, metric_name)
        start_time = time.time() - window_seconds
        try:
            events = self.redis.zrangebyscore(key, start_time, "+inf")
            if not events:
                return []

            return [self._parse_value_from_member(m) for m in events]
        except Exception as e:
            logger.error(f"Failed to fetch values in window for {metric_name}: {e}")
            return []

    def calculate_average(self, agent_id: str, metric_name: str, window_seconds: int = 3600) -> float:
        """
        Calculates the average value of a metric over the specified window.
        Returns 0.0 if no data exists.
        """
        values = self.get_values_in_window(agent_id, metric_name, window_seconds)
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _get_key(self, agent_id: str, metric_name: str) -> str:
        return f"sentinel:metrics:{agent_id}:{metric_name}"

    def _parse_value_from_member(self, member: bytes) -> float:
        """
        Extracts value from "{timestamp}:{value}:{uuid}" member string.
        """
        try:
            s = member.decode("utf-8")
            parts = s.split(":")
            if len(parts) >= 2:
                return float(parts[1])
            # Fallback if format is unexpected but we want to avoid crashing
            return 1.0
        except (ValueError, IndexError, AttributeError):
            return 1.0
