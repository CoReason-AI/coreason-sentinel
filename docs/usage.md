# Usage Guide

`coreason-sentinel` is designed to be run as a centralized **Observability & Pharmacovigilance Service**.

## Configuration

The service is configured using Environment Variables.

| Variable | Default | Description |
| :--- | :--- | :--- |
| `REDIS_URL` | `redis://localhost:6379` | Connection string for the Redis backend. |
| `AGENT_ID` | `agent-001` | The unique ID of the agent being monitored. |
| `OWNER_EMAIL` | `ops@coreason.ai` | Email address for critical alert notifications. |
| `PHOENIX_ENDPOINT` | `http://localhost:6006` | Endpoint for Arize Phoenix tracing. |

## Running the Service

You can run the service using `uvicorn`:

```bash
uvicorn src.coreason_sentinel.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### 1. Health Check
Check if the service is up and running.

- **URL:** `GET /health`
- **Response:** `{"status": "ok"}`

### 2. Agent Status (Circuit Breaker)
Agents should call this endpoint before processing a user request. If it returns `false`, the agent should block the request.

- **URL:** `GET /status/{agent_id}`
- **Response:** `true` (Allowed) or `false` (Blocked)

### 3. Agent Health Report
Get a detailed report of the agent's current metrics and breaker state.

- **URL:** `GET /health/{agent_id}`
- **Response:**
  ```json
  {
    "timestamp": "2023-10-27T10:00:00",
    "breaker_state": "CLOSED",
    "metrics": {
        "avg_latency": 0.45,
        "faithfulness": 0.98,
        "cost_per_query": 0.0012
    }
  }
  ```

### 4. Ingest Telemetry
Send telemetry data to the Sentinel.

#### OTEL Span
- **URL:** `POST /ingest/otel/span`
- **Payload:** `OTELSpan` object.

#### Veritas Event
- **URL:** `POST /ingest/veritas`
- **Payload:** `VeritasEvent` object.

## Client Integration Example

Here is how an agent might integrate with the Sentinel service:

```python
import httpx

SENTINEL_URL = "http://sentinel:8000"
AGENT_ID = "agent-alpha"

def process_user_request(user_input):
    # 1. Check Circuit Breaker
    response = httpx.get(f"{SENTINEL_URL}/status/{AGENT_ID}")
    if not response.json():
        return "I am currently undergoing maintenance. Please try again later."

    # 2. Process Request
    # ... agent logic ...

    # 3. Send Telemetry (Asynchronously)
    # ... send span/event to /ingest endpoints ...
```
