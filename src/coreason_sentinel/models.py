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
    """

    model_config = ConfigDict(extra="forbid")

    metadata_key: str = Field(..., description="The key in the metadata dictionary to check.")
    operator: Literal["EQUALS", "CONTAINS", "EXISTS"] = Field(..., description="The comparison operator.")
    value: Any = Field(None, description="The value to compare against. Ignored for EXISTS.")
    sample_rate: float = Field(1.0, ge=0.0, le=1.0, description="The sample rate to apply if the condition is met.")


class SentinelConfig(BaseModel):
    """
    Configuration for the Sentinel monitor.
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
