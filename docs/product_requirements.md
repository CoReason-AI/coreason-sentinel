# Product Requirements Document: coreason-sentinel

**Domain:** Production Observability, LLM Tracing, & Pharmacovigilance, Complete Observability
**Architectural Role:** The "Watchtower" & Circuit Breaker
**Core Philosophy:** "Drift is inevitable. You must trace the Cognitive Chain (Technical) and the User Pulse (Sentiment)."
**Dependencies:** arize-phoenix (Tracing), opentelemetry (Instrumentation), coreason-veritas (Data), coreason-assay (Grader), coreason-identity (Notification)

---

## 1. Executive Summary

coreason-sentinel is the automated monitoring service for deployed agents.

It combines **Deep Tracing** (via OpenTelemetry/Phoenix) with **High-Level Signals** (Cost, Sentiment, Drift). It does not just watch for "System Errors" (HTTP 500); it watches for "Cognitive Errors" (Hallucinations) and "Business Errors" (Budget Spikes).

It acts as the **Pharmacovigilance** unit for AI: monitoring the "drug" (Agent) in the real world, detecting side effects (Drift/Toxicity), and triggering a **Circuit Breaker** (Recall) if safety limits are breached.

## 2. Functional Philosophy

The agent must implement the **Trace-Monitor-Sample-Intervene Loop**:

1. **Deep Tracing (SOTA):** We trace the entire chain (Retrieve -> Rerank -> Scout -> Generate). We visualize the *exact* documents used to answer a question.
2. **Holistic Signal Detection:** We monitor three signal planes:
   * *Cognitive:* Hallucination, RAG Relevance (Phoenix).
   * *Business:* Token Cost, Latency (OTEL).
   * *User:* Sentiment, Refusals, "Lazy" responses (Drift).
3. **The Dead Man's Switch:** If safety/quality drops below the "Confidence Interval," Sentinel trips the Circuit Breaker to **OPEN**.
4. **Self-Healing (Half-Open):** After a cooldown, Sentinel automatically tests a trickle of traffic to see if the issue resolved before fully reopening.

---

## 3. Core Functional Requirements (Component Level)

### 3.1 The Omni-Ingestor (The Listener)

**Concept:** A stream processor that ingests both **Traces** (OTEL) and **Logs** (Veritas).

* **Tracing (Via Phoenix):**
  * Captures Retrieval Spans (Query Vector, Top-K Docs).
  * Captures Generation Spans (Prompt Template, Tokens).
* **Metric Extraction (Restored from V1):**
  * **Cost:** Token consumption per session.
  * **Sentiment:** user frustration signals (regex detection for "STOP", "WRONG", "Bad bot").
  * **Refusal Rate:** How often coreason-constitution blocks a response.

### 3.2 The Drift Engine (The Statistician)

**Concept:** Detects both Content Shift and Style Shift.

* **Embedding Drift (Content):** Monitors the distribution of *Retrieved Document Vectors*. If the vectors move to a new cluster, the user intent or data has shifted.
* **Output Drift (Style - Restored from V1):** Uses **Kullback-Leibler (KL) Divergence** to check if the agent's vocabulary or length is shifting (detects "Lazy Agent" syndrome).
* **Relevance Drift:** Uses Phoenix to track the embedding distance between Query and Response.

### 3.3 The Spot Checker (The Auditor)

**Concept:** Automated QA on live traffic.

* **Sampling Strategy:** Configurable (e.g., "1% Random" OR "100% of Negative Sentiment").
* **Loopback:** Sends the Trace to coreason-assay for grading.
  * *Metric:* **Faithfulness** (Did LLM stick to context?).
  * *Metric:* **Retrieval Precision** (Did RAG find the answer?).
* **Integration:** Pushes grades back into Phoenix as Span Attributes for filtering.

### 3.4 The Circuit Breaker (The Enforcer)

**Concept:** An automated kill-switch.

* **States (Restored from V1):**
  * **CLOSED (Normal):** Traffic flows.
  * **OPEN (Tripped):** Traffic blocked. User sees "Maintenance."
  * **HALF-OPEN (Recovery):** Allows 5% of traffic to verify fix.
* **Triggers:**
  * **Cognitive:** Faithfulness < 0.7.
  * **Technical:** P99 Latency > 10s.
  * **Business:** Cost Burn > $100/hr (Restored).
  * **Safety:** > 3 Constitution Violations / min.

---

## 4. Integration Requirements (The Ecosystem)

* **coreason-cortex:** Instrumented with @trace decorators.
* **coreason-identity:** Used to look up the "Agent Owner" email to send Critical Alerts when the Breaker trips.
* **coreason-veritas:** Long-term metric storage (Weeks/Months).
* **arize-phoenix:** Short-term visual debugging (Hours/Days).

---

## 5. User Stories

### Story A: The "Lazy Agent" (Style Drift)

**Context:** An OpenAI update makes the model concise. It stops writing full summaries.
**Detection:** Drift Engine detects Output Length distribution has shifted (KL Divergence high).
**Alert:** SRE notified of "Style Drift." Phoenix shows answers are accurate but too short.
**Action:** SRE updates Prompt Template in coreason-construct.

### Story B: The "Lost in the Middle" (Visual Debugging)

**Context:** Users complain answers are vague.
**Investigation:** SRE opens Phoenix. Filters for "Negative Sentiment."
**Visual:** The Retrieval Span shows the correct doc was found, but the Scout Span filtered it out.
**Action:** SRE lowers the Scout threshold.

### Story C: The "Budget Leak" (Business Trigger)

**Context:** An agent gets stuck in a loop, generating massive tokens.
**Detection:** Omni-Ingestor calculates Tokens/Minute > Threshold.
**Action:** Circuit Breaker trips to OPEN. Admin notified via coreason-identity.
**Result:** Budget saved.

---

## 6. Data Schema

### SentinelConfig

```python
class CircuitBreakerTrigger(BaseModel):
    metric: str           # "faithfulness", "latency", "cost"
    threshold: float
    window_seconds: int

class SentinelConfig(BaseModel):
    agent_id: str
    owner_email: str      # For notifications
    phoenix_endpoint: str
    sampling_rate: float = 0.01
    triggers: List[CircuitBreakerTrigger]
```

### HealthReport

```python
class HealthReport(BaseModel):
    timestamp: datetime
    breaker_state: Literal["CLOSED", "OPEN", "HALF_OPEN"]
    metrics: dict = {
        "avg_latency": "400ms",
        "faithfulness": 0.95,
        "cost_per_query": 0.02,
        "kl_divergence": 0.1  # Style Drift
    }
```

---

## 7. Implementation Directives for the Coding Agent

1. **Dual Ingestion:** The service must accept **OTEL Spans** (gRPC/HTTP) AND poll **Veritas Logs** (SQL/Event Bus).
2. **Redis State Machine:** Use Redis to manage the CLOSED -> OPEN -> HALF-OPEN transitions atomically.
3. **Math Libraries:** Use scipy.stats.entropy for KL Divergence. Use scikit-learn (or vector DB native tools) for Cosine Similarity.
4. **Async Workers:** Drift calculation is heavy. Run it in a background worker (Celery/FastAPI BackgroundTasks), never on the request path.
