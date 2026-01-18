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

import random
import time
from datetime import datetime
from enum import Enum

import numpy as np
from redis import Redis

from coreason_sentinel.interfaces import NotificationServiceProtocol
from coreason_sentinel.metric_store import MetricStore
from coreason_sentinel.models import CircuitBreakerTrigger, HealthReport, SentinelConfig
from coreason_sentinel.utils.logger import logger


class CircuitBreakerState(str, Enum):
    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"  # Tripped, blocking traffic
    HALF_OPEN = "HALF_OPEN"  # Testing recovery


class CircuitBreaker:
    """
    Manages the state of the Circuit Breaker for a specific agent.
    Delegates metric storage to MetricStore.
    """

    def __init__(
        self,
        redis_client: Redis[bytes],
        config: SentinelConfig,
        notification_service: NotificationServiceProtocol,
        metric_store: MetricStore | None = None,
    ):
        self.redis = redis_client
        self.config = config
        self.notification_service = notification_service
        self.agent_id = config.agent_id
        # Allow injection of MetricStore, or default to creating one
        self.metric_store = metric_store or MetricStore(redis_client)
        self._state_key = f"sentinel:breaker:{self.agent_id}:state"
        self._cooldown_key = f"sentinel:breaker:{self.agent_id}:cooldown"

    def get_state(self) -> CircuitBreakerState:
        """
        Retrieves the current state from Redis.
        Defaults to CLOSED if no state is recorded.
        AUTO-TRANSITION: If state is OPEN and cooldown has expired, transitions to HALF_OPEN.
        """
        try:
            state_bytes = self.redis.get(self._state_key)
            if state_bytes is None:
                return CircuitBreakerState.CLOSED

            state = CircuitBreakerState(state_bytes.decode("utf-8"))

            if state == CircuitBreakerState.OPEN:
                # Check cooldown
                if not self.redis.exists(self._cooldown_key):
                    logger.info(f"Cooldown expired for {self.agent_id}. Transitioning to HALF_OPEN.")
                    self.set_state(CircuitBreakerState.HALF_OPEN)
                    return CircuitBreakerState.HALF_OPEN

            return state
        except Exception as e:
            logger.error(f"Failed to fetch circuit breaker state from Redis: {e}")
            return CircuitBreakerState.CLOSED

    def set_state(self, state: CircuitBreakerState, reason: str | None = None) -> None:
        """
        Explicitly sets the circuit breaker state.
        """
        try:
            # Atomic set and get old value
            old_state_bytes = self.redis.getset(self._state_key, state.value)

            # Determine if we are effectively transitioning to OPEN
            # We treat None (missing key) as CLOSED.
            was_open = old_state_bytes is not None and old_state_bytes.decode("utf-8") == CircuitBreakerState.OPEN.value

            if state == CircuitBreakerState.OPEN and not was_open:
                # Set cooldown
                self.redis.setex(self._cooldown_key, self.config.recovery_timeout, "1")
                # Send Critical Alert
                if self.config.owner_email:
                    alert_reason = reason or "Circuit Breaker Tripped (Manual or Unknown)"
                    try:
                        self.notification_service.send_critical_alert(
                            email=self.config.owner_email,
                            agent_id=self.agent_id,
                            reason=alert_reason,
                        )
                    except Exception as e:
                        logger.error(f"Failed to send critical alert notification: {e}")

            logger.info(f"Circuit Breaker for {self.agent_id} transitioned to {state.value}")
        except Exception as e:
            logger.error(f"Failed to set circuit breaker state in Redis: {e}")
            raise e

    def allow_request(self) -> bool:
        """
        Determines if a request should be allowed based on the current state.
        CLOSED: Allow all.
        OPEN: Block all.
        HALF_OPEN: Allow 5% probabilistic trickle.
        """
        state = self.get_state()

        if state == CircuitBreakerState.CLOSED:
            return True
        elif state == CircuitBreakerState.OPEN:
            return False
        elif state == CircuitBreakerState.HALF_OPEN:
            # Allow 5% of traffic
            return random.random() < 0.05

        return True  # Fallback

    def record_metric(self, metric_name: str, value: float = 1.0) -> None:
        """
        Records a metric event via the MetricStore.
        Calculates retention window based on triggers.
        """
        max_window = self.config.get_max_window_for_metric(metric_name)
        self.metric_store.record_metric(self.agent_id, metric_name, value, retention_window=max_window)

    def check_triggers(self) -> None:
        """
        Evaluates all configured triggers against the recorded metrics.
        If a trigger is violated, trips the breaker to OPEN.
        If in HALF_OPEN and no trigger is violated, transitions to CLOSED.
        """
        state = self.get_state()

        if state == CircuitBreakerState.OPEN:
            return

        violation = False

        for trigger in self.config.triggers:
            if self._evaluate_trigger(trigger):
                reason = (
                    f"Trigger violated: {trigger.metric} {trigger.operator} {trigger.threshold} "
                    f"in last {trigger.window_seconds}s"
                )
                logger.warning(f"{reason}. Tripping Circuit Breaker.")
                self.set_state(CircuitBreakerState.OPEN, reason=reason)
                violation = True
                return

        # Recovery Logic
        if state == CircuitBreakerState.HALF_OPEN and not violation:
            # If we are here, it means no triggers were violated.
            # We assume the trickle traffic was successful.
            logger.info(f"Circuit Breaker for {self.agent_id} recovering to CLOSED.")
            self.set_state(CircuitBreakerState.CLOSED)

    def _evaluate_trigger(self, trigger: CircuitBreakerTrigger) -> bool:
        """
        Returns True if the trigger condition is met (violation).
        """
        try:
            values = self.metric_store.get_values_in_window(self.agent_id, trigger.metric, trigger.window_seconds)

            if not values:
                return False

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
            elif trigger.aggregation_method.startswith("P"):
                # Handle Percentiles (P50, P90, P95, P99)
                try:
                    percentile = float(trigger.aggregation_method[1:])
                    # numpy.percentile expects range 0-100
                    aggregated_value = float(np.percentile(values, percentile))
                except (ValueError, IndexError):
                    logger.error(f"Invalid percentile format {trigger.aggregation_method} for trigger {trigger.metric}")
                    return False

            # Compare
            if trigger.operator == ">":
                return aggregated_value > trigger.threshold
            elif trigger.operator == "<":
                return aggregated_value < trigger.threshold
            return False

        except Exception as e:
            logger.error(f"Failed to evaluate trigger {trigger.metric}: {e}")
            return False

    def get_recent_values(self, metric_name: str, limit: int = 100) -> list[float]:
        """
        Retrieves the most recent raw values for a given metric.
        Delegates to MetricStore.
        """
        return self.metric_store.get_recent_values(self.agent_id, metric_name, limit)

    def get_health_report(self) -> HealthReport:
        """
        Generates a Health Report for the agent, aggregating metrics over the last hour.
        """
        state = self.get_state()
        metrics: dict[str, float] = {}

        # Default metrics to aggregate
        # 1. Avg Latency (1h)
        metrics["avg_latency"] = self.metric_store.calculate_average(self.agent_id, "latency", window_seconds=3600)

        # 2. Faithfulness (1h) - Average score
        metrics["faithfulness"] = self.metric_store.calculate_average(self.agent_id, "faithfulness", window_seconds=3600)

        # 3. Cost Per Query (1h) - Average cost per event
        metrics["cost_per_query"] = self.metric_store.calculate_average(self.agent_id, "cost", window_seconds=3600)

        # 4. KL Divergence (1h) - Average drift score
        metrics["kl_divergence"] = self.metric_store.calculate_average(
            self.agent_id, "output_drift_kl", window_seconds=3600
        )

        return HealthReport(
            timestamp=datetime.fromtimestamp(time.time()),
            breaker_state=state.value,
            metrics=metrics,
        )
