# coreason-sentinel

**The "Watchtower" & Circuit Breaker for Production AI Agents**

[![License: Prosperity 3.0](https://img.shields.io/badge/license-Prosperity%203.0-blue)](https://github.com/CoReason-AI/coreason-sentinel/blob/main/LICENSE)
[![CI](https://github.com/CoReason-AI/coreason-sentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/CoReason-AI/coreason-sentinel/actions/workflows/ci.yml)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Docs](https://img.shields.io/badge/docs-product_requirements.md-blue)](docs/product_requirements.md)

**coreason-sentinel** is a centralized **Observability & Pharmacovigilance Service** for deployed LLM agents. It acts as a "Watchtower," ingesting telemetry (OTEL Spans and Veritas Logs) from distributed agents and managing a global **Circuit Breaker** state in Redis.

It goes beyond simple "System Error" tracking to detect "Cognitive Errors" (Hallucinations) and "Business Errors" (Budget Spikes). If safety limits are breached (e.g., high hallucination rates or budget leaks), Sentinel trips the circuit breaker, instructing agents to block further requests.

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

`coreason-sentinel` is run as a standalone service.

```bash
# Run the service
uvicorn src.coreason_sentinel.main:app --host 0.0.0.0 --port 8000
```

For detailed instructions on configuration, endpoints, and integration, please refer to the [Usage Guide](docs/usage.md).

For a list of dependencies and system requirements, see [Requirements](docs/requirements.md).
