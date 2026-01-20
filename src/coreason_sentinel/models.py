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
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field


class CircuitBreakerTrigger(BaseModel):
    """
    Defines a condition that triggers the Circuit Breaker.

    Attributes:
        metric: The name of the metric to monitor (e.g., 'faithfulness', 'latency', 'cost').
        threshold: The value threshold that triggers the breaker.
        window_seconds: The time window in seconds to evaluate the metric.
        operator: Comparison operator ('>' or '<').
        aggregation_method: How to aggregate the metric over the window (SUM, AVG, etc.).
    """

    model_config = ConfigDict(extra="forbid")

    metric: str = Field(..., description="The name of the metric to monitor (e.g., 'faithfulness', 'latency', 'cost').")
    threshold: float = Field(..., description="The value threshold that triggers the breaker.")
    window_seconds: int = Field(
        ..., gt=0, description="The time window in seconds to evaluate the metric. Must be positive."
    )
    operator: Literal[">", "<"] = Field(
        ">", description="Comparison operator. Default is '>' (greater than threshold)."
    )
    aggregation_method: Literal["SUM", "AVG", "COUNT", "MIN", "MAX", "P50", "P90", "P95", "P99"] = Field(
        "SUM", description="Aggregation method for the metric over the window. Default is SUM."
    )


class ConditionalSamplingRule(BaseModel):
    """
    Defines a rule for conditional sampling based on event metadata.

    Attributes:
        metadata_key: The key in the metadata dictionary to check.
        operator: The comparison operator (EQUALS, CONTAINS, EXISTS).
        value: The value to compare against.
        sample_rate: The sample rate to apply if the condition is met.
    """

    model_config = ConfigDict(extra="forbid")

    metadata_key: str = Field(..., description="The key in the metadata dictionary to check.")
    operator: Literal["EQUALS", "CONTAINS", "EXISTS"] = Field(..., description="The comparison operator.")
    value: Any = Field(None, description="The value to compare against. Ignored for EXISTS.")
    sample_rate: float = Field(1.0, ge=0.0, le=1.0, description="The sample rate to apply if the condition is met.")


class SentinelConfig(BaseModel):
    """
    Configuration for the Sentinel monitor (The Watchtower).

    Defines the safety limits, sampling rates, and trigger conditions for the agent.

    Attributes:
        agent_id: Unique identifier for the agent being monitored.
        owner_email: Email address for notifications (Critical Alerts).
        phoenix_endpoint: Endpoint URL for Phoenix tracing.
        sampling_rate: Fraction of traffic to sample (0.0 to 1.0).
        drift_threshold_kl: KL Divergence threshold for output drift detection.
        drift_sample_window: Number of recent samples to use for live distribution calculation.
        cost_per_1k_tokens: Cost per 1000 tokens in USD.
        recovery_timeout: Cooldown time in seconds before attempting recovery from OPEN state.
        triggers: List of triggers that can trip the circuit breaker.
        sentiment_regex_patterns: List of regex patterns to detect negative sentiment.
        conditional_sampling_rules: List of rules to override sampling rate based on metadata.
    """

    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(..., description="Unique identifier for the agent being monitored.")
    owner_email: str = Field(..., description="Email address for notifications.")
    phoenix_endpoint: str = Field(..., description="Endpoint URL for Phoenix tracing.")
    sampling_rate: float = Field(
        0.01, ge=0.0, le=1.0, description="Fraction of traffic to sample (0.0 to 1.0). Default 1%."
    )
    drift_threshold_kl: float = Field(0.5, ge=0.0, description="KL Divergence threshold for output drift detection.")
    drift_sample_window: int = Field(
        100, gt=0, description="Number of recent samples to use for live distribution calculation."
    )
    cost_per_1k_tokens: float = Field(
        0.002, ge=0.0, description="Cost per 1000 tokens in USD. Default is 0.002 (approx GPT-3.5)."
    )
    recovery_timeout: int = Field(
        60, gt=0, description="Cooldown time in seconds before attempting recovery from OPEN state."
    )
    triggers: List[CircuitBreakerTrigger] = Field(
        default_factory=list, description="List of triggers that can trip the circuit breaker."
    )
    sentiment_regex_patterns: List[str] = Field(
        default_factory=lambda: ["STOP", "WRONG", "Bad bot"],
        description="List of regex patterns to detect negative sentiment in user input.",
    )
    conditional_sampling_rules: List[ConditionalSamplingRule] = Field(
        default_factory=list, description="List of rules to override the default sampling rate based on metadata."
    )


class HealthReport(BaseModel):
    """
    A snapshot of the agent's health.

    Attributes:
        timestamp: Time of the report.
        breaker_state: Current state of the Circuit Breaker.
        metrics: Key-value pairs of current metrics.
    """

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(..., description="Time of the report.")
    breaker_state: Literal["CLOSED", "OPEN", "HALF_OPEN"] = Field(
        ..., description="Current state of the Circuit Breaker."
    )
    metrics: Dict[str, Any] = Field(
        default_factory=lambda: {
            "avg_latency": "400ms",
            "faithfulness": 0.95,
            "cost_per_query": 0.02,
            "kl_divergence": 0.1,
        },
        description="Key-value pairs of current metrics.",
    )
