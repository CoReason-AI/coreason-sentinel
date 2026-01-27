# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_sentinel

import os
from contextlib import asynccontextmanager
from typing import Annotated, AsyncGenerator

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from redis.asyncio import Redis

from coreason_sentinel.circuit_breaker import CircuitBreaker
from coreason_sentinel.ingestor import TelemetryIngestorAsync
from coreason_sentinel.interfaces import OTELSpan, VeritasEvent
from coreason_sentinel.mocks import (
    MockBaselineProvider,
    MockGrader,
    MockNotificationService,
    MockPhoenixClient,
    MockVeritasClient,
)
from coreason_sentinel.models import HealthReport, SentinelConfig
from coreason_sentinel.spot_checker import SpotChecker
from coreason_sentinel.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for initializing and cleaning up resources.
    """
    logger.info("Initializing Sentinel resources...")

    # 1. Initialize Redis
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = Redis.from_url(redis_url, encoding="utf-8", decode_responses=False)

    # 2. Initialize Config
    config = SentinelConfig(
        agent_id=os.getenv("AGENT_ID", "agent-001"),
        owner_email=os.getenv("OWNER_EMAIL", "ops@coreason.ai"),
        phoenix_endpoint=os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006"),
    )

    # 3. Initialize Mocks/Services
    notification_service = MockNotificationService()
    phoenix_client = MockPhoenixClient()
    baseline_provider = MockBaselineProvider()
    veritas_client = MockVeritasClient()

    # 4. Initialize Core Components
    grader = MockGrader()
    spot_checker = SpotChecker(config=config, grader=grader, phoenix_client=phoenix_client)

    circuit_breaker = CircuitBreaker(
        redis_client=redis_client,
        config=config,
        notification_service=notification_service,
    )

    # 5. Initialize Ingestor
    ingestor = TelemetryIngestorAsync(
        config=config,
        circuit_breaker=circuit_breaker,
        spot_checker=spot_checker,
        baseline_provider=baseline_provider,
        veritas_client=veritas_client,
    )

    # Start ingestor (async context)
    await ingestor.__aenter__()

    # Store in app state
    app.state.redis = redis_client
    app.state.circuit_breaker = circuit_breaker
    app.state.ingestor = ingestor
    app.state.config = config

    yield

    logger.info("Shutting down Sentinel resources...")
    await ingestor.__aexit__(None, None, None)
    await redis_client.close()


app = FastAPI(title="CoReason Sentinel", version="0.1.0", lifespan=lifespan)


async def get_telemetry_ingestor() -> TelemetryIngestorAsync:
    """
    Dependency to provide the TelemetryIngestorAsync instance.
    """
    if not hasattr(app.state, "ingestor"):
        # Fallback for tests if lifespan didn't run or state not set
        raise RuntimeError("TelemetryIngestor not initialized in app.state")
    from typing import cast

    return cast(TelemetryIngestorAsync, app.state.ingestor)


@app.get("/health")  # type: ignore[misc]
async def health_check() -> dict[str, str]:
    """
    Health check endpoint.

    Returns:
        dict[str, str]: Status message {"status": "ok"}.
    """
    return {"status": "ok"}


@app.post("/ingest/otel/span", status_code=202)  # type: ignore[misc]
async def ingest_otel_span(
    span: OTELSpan,
    background_tasks: BackgroundTasks,
    ingestor: Annotated[TelemetryIngestorAsync, Depends(get_telemetry_ingestor)],
) -> dict[str, str]:
    """
    Ingests a single OpenTelemetry span.

    Processing is offloaded to a background task to ensure low latency for the tracing client.

    Args:
        span: The OpenTelemetry span to ingest.
        background_tasks: FastAPI BackgroundTasks object.
        ingestor: The TelemetryIngestor instance.

    Returns:
        dict[str, str]: Status message and span ID.
    """
    logger.info(f"Received OTEL span: {span.span_id}")
    background_tasks.add_task(ingestor.process_otel_span, span)
    return {"status": "accepted", "span_id": span.span_id}


@app.get("/health/{agent_id}")  # type: ignore[misc]
async def agent_health_check(
    agent_id: str,
    ingestor: Annotated[TelemetryIngestorAsync, Depends(get_telemetry_ingestor)],
) -> HealthReport:
    """
    Returns the health report for a specific agent.
    """
    if agent_id != ingestor.config.agent_id:
        raise HTTPException(status_code=404, detail="Agent not found")

    return await ingestor.circuit_breaker.get_health_report()


@app.get("/status/{agent_id}")  # type: ignore[misc]
async def agent_status_check(
    agent_id: str,
    ingestor: Annotated[TelemetryIngestorAsync, Depends(get_telemetry_ingestor)],
) -> bool:
    """
    Check if the agent is allowed to process requests (Circuit Breaker status).
    Returns True if allowed, False if blocked.
    """
    if agent_id != ingestor.config.agent_id:
        # If agent unknown, maybe block? or 404?
        # Requirement says "If it returns False, the agent should block".
        # If agent doesn't exist, we probably shouldn't allow it.
        raise HTTPException(status_code=404, detail="Agent not found")

    return await ingestor.circuit_breaker.allow_request()


@app.post("/ingest/veritas", status_code=202)  # type: ignore[misc]
async def ingest_veritas_event(
    event: VeritasEvent,
    background_tasks: BackgroundTasks,
    ingestor: Annotated[TelemetryIngestorAsync, Depends(get_telemetry_ingestor)],
) -> dict[str, str]:
    """
    Ingests a Veritas event.
    Processing is offloaded to background tasks.
    """
    logger.info(f"Received Veritas event: {event.event_id}")
    background_tasks.add_task(ingestor.process_event, event)
    return {"status": "accepted", "event_id": event.event_id}
