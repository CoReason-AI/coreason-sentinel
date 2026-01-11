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
import uuid
from enum import Enum

from redis import Redis

from coreason_sentinel.models import SentinelConfig, Trigger
from coreason_sentinel.utils.logger import logger


class CircuitBreakerState(str, Enum):
    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"  # Tripped, blocking traffic
    HALF_OPEN = "HALF_OPEN"  # Testing recovery


class CircuitBreaker:
    """
    Manages the state of the Circuit Breaker for a specific agent.
    Uses Redis for persistence to ensure stateless workers see the same state.
    """

    def __init__(self, redis_client: Redis, config: SentinelConfig):
        self.redis = redis_client
        self.config = config
        self.agent_id = config.agent_id
        self._state_key = f"sentinel:breaker:{self.agent_id}:state"

    def get_state(self) -> CircuitBreakerState:
        """
        Retrieves the current state from Redis.
        Defaults to CLOSED if no state is recorded.
        """
        try:
            state_bytes = self.redis.get(self._state_key)
            if state_bytes is None:
                return CircuitBreakerState.CLOSED
            return CircuitBreakerState(state_bytes.decode("utf-8"))
        except Exception as e:
            logger.error(f"Failed to fetch circuit breaker state from Redis: {e}")
            return CircuitBreakerState.CLOSED

    def set_state(self, state: CircuitBreakerState) -> None:
        """
        Explicitly sets the circuit breaker state.
        """
        try:
            self.redis.set(self._state_key, state.value)
            logger.info(f"Circuit Breaker for {self.agent_id} transitioned to {state.value}")
        except Exception as e:
            logger.error(f"Failed to set circuit breaker state in Redis: {e}")
            raise e

    def record_metric(self, metric_name: str, value: float = 1.0) -> None:
        """
        Records a metric event into a Redis Sorted Set (Sliding Window).
        The score is the timestamp, the member is "{timestamp}:{value}:{uuid}".
        """
        # Validate input (NaN/Inf check)
        if not math.isfinite(value):
            logger.warning(f"Ignoring invalid metric value: {value} for {metric_name}")
            return

        key = f"sentinel:metrics:{self.agent_id}:{metric_name}"
        timestamp = time.time()
        # Unique member to allow multiple events at same timestamp
        member = f"{timestamp}:{value}:{uuid.uuid4()}"

        try:
            # Add event to sorted set
            self.redis.zadd(key, {member: timestamp})

            # Prune old metrics to prevent memory leak
            max_window = 3600  # Default 1 hour
            for t in self.config.circuit_breaker_triggers:
                if t.metric_name == metric_name:
                    max_window = max(max_window, t.window_seconds)

            # Remove elements older than the window
            min_score = timestamp - max_window
            self.redis.zremrangebyscore(key, "-inf", min_score)

            self.redis.expire(key, max_window * 2)
        except Exception as e:
            logger.error(f"Failed to record metric {metric_name}: {e}")

    def check_triggers(self) -> None:
        """
        Evaluates all configured triggers against the recorded metrics.
        If a trigger is violated, trips the breaker to OPEN.
        """
        # If already OPEN, no need to check (or maybe we check for recovery? logic usually separates them)
        if self.get_state() == CircuitBreakerState.OPEN:
            return

        now = time.time()
        for trigger in self.config.circuit_breaker_triggers:
            if self._evaluate_trigger(trigger, now):
                logger.warning(
                    f"Trigger violated: {trigger.metric_name} {trigger.operator} {trigger.threshold} "
                    f"in last {trigger.window_seconds}s. Tripping Circuit Breaker."
                )
                self.set_state(CircuitBreakerState.OPEN)
                return

    def _evaluate_trigger(self, trigger: Trigger, now: float) -> bool:
        """
        Returns True if the trigger condition is met (violation).
        """
        key = f"sentinel:metrics:{self.agent_id}:{trigger.metric_name}"
        start_time = now - trigger.window_seconds

        try:
            # Get events within window
            # zrangebyscore returns list of members
            events = self.redis.zrangebyscore(key, start_time, "+inf")
            if not events:
                # If no events, we consider value 0.0 for SUM/COUNT, but for AVG/MIN/MAX it's undefined.
                # Usually we don't trip if no data.
                return False

            values = [self._parse_value_from_member(m) for m in events]

            aggregated_value = 0.0
            if trigger.aggregation_method == "SUM":
                aggregated_value = sum(values)
            elif trigger.aggregation_method == "COUNT":
                aggregated_value = float(len(values))
            elif trigger.aggregation_method == "AVG":
                aggregated_value = sum(values) / len(values)
            elif trigger.aggregation_method == "MIN":
                aggregated_value = min(values)
            elif trigger.aggregation_method == "MAX":
                aggregated_value = max(values)

            # Compare
            if trigger.operator == ">":
                return aggregated_value > trigger.threshold
            elif trigger.operator == "<":
                return aggregated_value < trigger.threshold
            return False

        except Exception as e:
            logger.error(f"Failed to evaluate trigger {trigger.metric_name}: {e}")
            return False

    def _parse_value_from_member(self, member: bytes) -> float:
        """
        Extracts value from "{timestamp}:{value}:{uuid}" member string.
        """
        try:
            s = member.decode("utf-8")
            parts = s.split(":")
            if len(parts) >= 2:
                return float(parts[1])
            return 1.0  # Default fallback
        except (ValueError, IndexError):
            return 1.0
