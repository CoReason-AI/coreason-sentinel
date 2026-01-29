# Usage Guide

Coreason Sentinel is a centralized Observability & Pharmacovigilance Service. It acts as a "Watchtower" for your AI agents, monitoring their health, detecting drift, and managing circuit breakers.

## Running the Service

You can run the Sentinel service using `uvicorn`:

```bash
uvicorn coreason_sentinel.main:app --host 0.0.0.0 --port 8000
```

Ensure you have a Redis instance running and configured via the `REDIS_URL` environment variable.

## API Endpoints

### Health Check

Check the liveness of the service:

```http
GET /health
```

**Response:**
```json
{"status": "ok"}
```

### Agent Health Report

Get a detailed health report for a specific agent, including circuit breaker state and aggregated metrics:

```http
GET /health/{agent_id}
```

**Response:**
```json
{
  "timestamp": "2025-01-01T12:00:00",
  "breaker_state": "CLOSED",
  "metrics": {
    "avg_latency": 0.45,
    "faithfulness": 0.98,
    "cost_per_query": 0.002,
    "kl_divergence": 0.05
  }
}
```

### Agent Status Check (Circuit Breaker)

Agents should call this endpoint before processing a request to check if they are allowed to proceed:

```http
GET /status/{agent_id}
```

**Response:** `true` (Allowed) or `false` (Blocked)

### Ingest Veritas Event

Ingest a business log (Veritas Event) for monitoring and drift detection:

```http
POST /ingest/veritas
```

**Payload:**
```json
{
  "event_id": "evt-123",
  "timestamp": "2025-01-01T12:00:00Z",
  "agent_id": "agent-007",
  "session_id": "sess-abc",
  "input_text": "What is the capital of France?",
  "output_text": "Paris",
  "metrics": {
    "latency": 0.5,
    "token_count": 10
  },
  "metadata": {
    "embedding": [0.1, 0.2, ...]
  }
}
```

### Ingest OTEL Span

Ingest an OpenTelemetry span for real-time monitoring:

```http
POST /ingest/otel/span
```

**Payload:**
```json
{
  "trace_id": "32-hex-char-trace-id",
  "span_id": "16-hex-char-span-id",
  "name": "llm_query",
  "start_time_unix_nano": 1700000000000000000,
  "end_time_unix_nano": 1700000001000000000,
  "attributes": {
    "llm.token_count.total": 150
  }
}
```

## Configuration

Configure the service using environment variables:

*   `REDIS_URL`: URL of the Redis instance (default: `redis://localhost:6379`).
*   `AGENT_ID`: Default Agent ID (default: `default_agent`).
*   `OWNER_EMAIL`: Email for critical alerts (default: `admin@coreason.ai`).
*   `PHOENIX_ENDPOINT`: Endpoint for Phoenix tracing (default: `http://localhost:6006`).
