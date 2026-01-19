# coreason-sentinel

**The "Watchtower" & Circuit Breaker for Production AI Agents**

[![License: Prosperity 3.0](https://img.shields.io/badge/license-Prosperity%203.0-blue)](https://github.com/CoReason-AI/coreason-sentinel/blob/main/LICENSE)
[![CI](https://github.com/CoReason-AI/coreason-sentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/CoReason-AI/coreason-sentinel/actions/workflows/ci.yml)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Docs](https://img.shields.io/badge/docs-product_requirements.md-blue)](docs/product_requirements.md)

**coreason-sentinel** is an automated monitoring service for deployed LLM agents. It goes beyond simple "System Error" tracking to detect "Cognitive Errors" (Hallucinations) and "Business Errors" (Budget Spikes). Acting as a pharmacovigilance unit for AI, it monitors the agent's behavior in the real world, detects side effects like drift or toxicity, and triggers a **Circuit Breaker** to protect users and budgets if safety limits are breached.

## Features

*   **Deep Tracing & Observability:** visualize the entire cognitive chain (Retrieve -> Rerank -> Scout -> Generate) using OpenTelemetry and Arize Phoenix.
*   **Holistic Signal Detection:** Monitors three key signal planes:
    *   **Cognitive:** Hallucination rates, RAG relevance.
    *   **Business:** Token costs, Latency spikes.
    *   **User:** Sentiment analysis, refusal rates, and frustration signals.
*   **The Drift Engine:** Advanced statistical detection for:
    *   **Content Drift:** Changes in retrieved document vectors.
    *   **Style Drift:** "Lazy Agent" syndrome detection using Kullback-Leibler (KL) Divergence.
    *   **Relevance Drift:** Semantic distance between queries and responses.
*   **Automated Circuit Breaker:** A "Dead Man's Switch" that transitions to **OPEN** (blocking traffic) if quality or safety drops below configured thresholds. Includes **HALF-OPEN** self-healing to test recovery.
*   **The Spot Checker:** Automated QA that randomly samples live traffic (or focuses on negative sentiment) and loops it back for grading.

For a full breakdown of requirements and philosophy, see the [Product Requirements Document](docs/product_requirements.md).

## Installation

```bash
pip install coreason-sentinel
```

## Usage

Here is how to initialize the Circuit Breaker to protect your agent:

```python
import time
from redis import Redis
from coreason_sentinel.circuit_breaker import CircuitBreaker
from coreason_sentinel.models import SentinelConfig, CircuitBreakerTrigger

# 1. Configure the Sentinel
config = SentinelConfig(
    agent_id="agent-alpha",
    owner_email="ops@coreason.ai",
    phoenix_endpoint="http://localhost:6006",
    triggers=[
        # Trip if Faithfulness drops below 0.7 in the last hour
        CircuitBreakerTrigger(metric="faithfulness", threshold=0.7, window_seconds=3600, operator="<"),
        # Trip if Latency exceeds 10s in the last minute
        CircuitBreakerTrigger(metric="latency", threshold=10.0, window_seconds=60, operator=">"),
    ]
)

# 2. Initialize the Circuit Breaker
# Note: You must provide a valid Redis client and a NotificationService implementation.
# (Assuming a mock notification service for this example)
class MockNotificationService:
    def notify(self, message):
        print(f"NOTIFICATION: {message}")

breaker = CircuitBreaker(
    redis_client=Redis(),
    config=config,
    notification_service=MockNotificationService()
)

# 3. Protect your Agent
if not breaker.allow_request():
    # Return a maintenance message or failover
    print("Circuit Breaker OPEN: Traffic blocked due to safety violation.")
else:
    # Process your agent request...
    start_time = time.time()
    try:
        # ... agent logic ...
        print("Agent processing request...")
        pass
    finally:
        # Record metrics for the breaker to monitor
        # In a real app, you would calculate actual latency
        breaker.record_metric("latency", time.time() - start_time)
```
