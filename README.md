# CoReason Sentinel

**Production Monitoring & Circuit Breaker Service for AI Agents**

Coreason Sentinel acts as a centralized "Watchtower" for distributed AI agents. It ingests cognitive traces (OTEL spans) and business logs (Veritas events) to monitor agent health in real-time. If safety thresholds are breached (e.g., high hallucination rates, budget leaks, or statistical drift), Sentinel trips a global **Circuit Breaker**, preventing further risky operations.

## Architecture

Sentinel is built as a FastAPI service backed by Redis for state persistence.

*   **Ingestor**: Asynchronously processes telemetry from agents.
*   **Circuit Breaker**: Manages the global health state (CLOSED, OPEN, HALF_OPEN) based on configured triggers.
*   **Drift Engine**: Performs statistical analysis (Cosine Similarity, KL-Divergence) to detect input/output drift.
*   **Spot Checker**: Probabilistically samples traffic for deep evaluation (Assay).

## Key Features

*   **Centralized Circuit Breaker**: Shared health state across all agent instances.
*   **Real-time Monitoring**: Tracks latency, token usage, cost, and custom metrics.
*   **Drift Detection**: Detects vector drift and output distribution shifts using SciPy and NumPy.
*   **Control API**: Simple REST endpoints for agents to check their status before execution.

## Getting Started

Refer to the documentation in `docs/` for detailed instructions.

### Installation

```bash
pip install coreason-sentinel
```

### Quick Start

1.  Start Redis.
2.  Run the Sentinel service:
    ```bash
    uvicorn coreason_sentinel.main:app --reload
    ```
3.  Check the health endpoint:
    ```bash
    curl http://localhost:8000/health
    ```

## License

Prosperity Public License 3.0
