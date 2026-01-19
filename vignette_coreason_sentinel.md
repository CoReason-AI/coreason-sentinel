# The Architecture and Utility of coreason-sentinel

### 1. The Philosophy (The Why)
In the traditional software lifecycle, a crash is binary: the server is up (200 OK) or it is down (500 Internal Error). Generative AI introduces a new, insidious failure mode: the **Cognitive Error**. The service is "up"—HTTP 200s are flowing, tokens are generating—but the agent is hallucinating, bleeding budget, or becoming belligerent.

**coreason-sentinel** is built on the premise that standard observability is insufficient for probabilistic systems. It acts as the **Pharmacovigilance** unit for your deployed AI, monitoring the "drug" (the Agent) in the wild for adverse reactions (Drift, Toxicity, Budget Spikes).

The author's insight is that meaningful control requires a **Trace-Monitor-Sample-Intervene** loop. It is not enough to log errors; one must trace the entire cognitive chain—from retrieval to reranking to generation. When the "vital signs" of an agent (faithfulness, cost, sentiment) breach a confidence interval, Sentinel does not just alert a dashboard; it acts. It triggers a **Circuit Breaker**, cutting off the agent to prevent runaway costs or reputational damage, embodying a shift from passive monitoring to active, automated defense.

### 2. Under the Hood (The Dependencies & logic)
The architecture of `coreason-sentinel` is a fusion of distributed state management, deep instrumentation, and statistical rigor.

*   **Deep Tracing (`arize-phoenix`, `opentelemetry`):** The package relies on OpenTelemetry for standard instrumentation but leverages `arize-phoenix` for the specific nuance of LLM tracing—capturing retrieval spans, prompt templates, and embedding vectors. This allows Sentinel to visualize the *exact* context used to generate an answer.
*   **The State Machine (`redis`):** To enforce a circuit breaker across stateless distributed workers, Sentinel uses Redis as a shared brain. It manages the atomic transitions between `CLOSED` (normal), `OPEN` (tripped), and `HALF-OPEN` (recovery) states, ensuring that if one worker detects a toxicity spike, *all* workers stop traffic immediately.
*   **The Statistician (`scipy`, `numpy`):** Drift detection is treated as a mathematical problem.
    *   **Kullback-Leibler (KL) Divergence** is used to detect "Style Drift"—statistical shifts in output length or vocabulary that indicate an agent has become "lazy" or verbose.
    *   **Cosine Similarity** tracks "Content Drift" by monitoring the movement of retrieved document vectors, signaling when user intent shifts away from the knowledge base.
*   **Data Integrity (`pydantic`):** Strict schemas define the `SentinelConfig` and `HealthReport`, ensuring that safety thresholds and triggers are validated before the system ever goes live.

### 3. In Practice (The How)

`coreason-sentinel` is designed to be the wrapper around your agent's critical path.

#### Defines a Safety Constitution
First, you define the "safety constitution" for your agent. Here, we configure Sentinel to trip the breaker if the agent becomes unfaithful to its sources or burns too much budget.

```python
from coreason_sentinel.models import SentinelConfig, CircuitBreakerTrigger

# Define the "Kill Switches"
triggers = [
    CircuitBreakerTrigger(
        metric="faithfulness",
        threshold=0.7,
        operator="<",
        window_seconds=300,  # If faithfulness drops below 0.7 in 5 mins
        aggregation_method="AVG"
    ),
    CircuitBreakerTrigger(
        metric="cost",
        threshold=5.0,
        operator=">",
        window_seconds=60,  # If we burn >$5 in 1 minute
        aggregation_method="SUM"
    )
]

config = SentinelConfig(
    agent_id="finance-bot-v1",
    owner_email="ops@coreason.ai",
    phoenix_endpoint="http://phoenix:6006",
    triggers=triggers
)
```

#### The Circuit Breaker in Action
In your inference loop, Sentinel acts as the gatekeeper. It checks the global state before processing any request and records metrics after completion.

```python
from coreason_sentinel.circuit_breaker import CircuitBreaker

# Injected dependencies (Redis connection not shown)
breaker = CircuitBreaker(redis_client=redis, config=config, notification_service=notifier)

def handle_user_query(query):
    # 1. Check if the Breaker is OPEN
    if not breaker.allow_request():
        return "System is currently undergoing maintenance. Please try again later."

    # 2. Process the request (The Agent's work)
    response = agent.generate(query)

    # 3. Feed the Sentinel (Async in production)
    # Record faithfulness score from a grader
    breaker.record_metric("faithfulness", value=response.faithfulness_score)
    # Record cost
    breaker.record_metric("cost", value=response.token_cost)

    # 4. Evaluate triggers immediately
    breaker.check_triggers()

    return response.text
```

#### Detecting Statistical Drift
Behind the scenes, the `DriftEngine` can be used to compare the "Style" of the live agent against a baseline, detecting issues like the "Lazy Agent" syndrome.

```python
from coreason_sentinel.drift_engine import DriftEngine

# Baseline: Distribution of output lengths from the "Gold Set"
baseline_lengths = [100, 120, 115, 130, 110]
# Live: Recent output lengths (suddenly much shorter)
live_lengths = [20, 25, 22, 18, 24]

# Define bin edges for the histogram
bins = [0, 50, 100, 150, 200]

# Convert raw samples to probability distributions
p_baseline = DriftEngine.compute_distribution_from_samples(baseline_lengths, bins)
q_live = DriftEngine.compute_distribution_from_samples(live_lengths, bins)

# Calculate KL Divergence
drift_score = DriftEngine.compute_kl_divergence(p_baseline, q_live)

if drift_score > 0.5:
    print(f"High Style Drift detected ({drift_score:.2f}). Agent behavior has shifted.")
```
