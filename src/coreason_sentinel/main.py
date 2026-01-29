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

from fastapi import BackgroundTasks, Depends, FastAPI
from redis.asyncio import Redis

from coreason_sentinel.circuit_breaker import CircuitBreaker
from coreason_sentinel.ingestor import TelemetryIngestorAsync
from coreason_sentinel.interfaces import OTELSpan, VeritasEvent
from coreason_sentinel.mocks import (
    MockAssayGrader,
    MockBaselineProvider,
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
    Lifespan context manager for the FastAPI application.
    Initializes dependencies and cleans them up on shutdown.
    """
    # Load configuration
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    agent_id = os.getenv("AGENT_ID", "default_agent")
    owner_email = os.getenv("OWNER_EMAIL", "admin@coreason.ai")
    phoenix_endpoint = os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006")

    config = SentinelConfig(
        agent_id=agent_id,
        owner_email=owner_email,
        phoenix_endpoint=phoenix_endpoint,
    )

    # Initialize dependencies
    # Using decode_responses=False (default) as CircuitBreaker expects bytes
    redis_client = Redis.from_url(redis_url)
    notification_service = MockNotificationService()
    baseline_provider = MockBaselineProvider()
    veritas_client = MockVeritasClient()
    phoenix_client = MockPhoenixClient()
    grader = MockAssayGrader()

    # Initialize Components
    circuit_breaker = CircuitBreaker(
        redis_client=redis_client,
        config=config,
        notification_service=notification_service,
    )

    spot_checker = SpotChecker(
        config=config,
        grader=grader,
        phoenix_client=phoenix_client,
    )

    ingestor = TelemetryIngestorAsync(
        config=config,
        circuit_breaker=circuit_breaker,
        spot_checker=spot_checker,
        baseline_provider=baseline_provider,
        veritas_client=veritas_client,
    )

    # Initialize ingestor resources (e.g. client)
    await ingestor.__aenter__()

    # Store in app state
    app.state.ingestor = ingestor

    yield

    # Cleanup
    await ingestor.__aexit__(None, None, None)
    await redis_client.close()


app = FastAPI(title="CoReason Sentinel", version="0.3.0", lifespan=lifespan)


async def get_telemetry_ingestor() -> TelemetryIngestorAsync:
    """
    Dependency to provide the TelemetryIngestorAsync instance.
    """
    if not hasattr(app.state, "ingestor"):
        raise RuntimeError("TelemetryIngestor is not initialized in app state.")
    return app.state.ingestor  # type: ignore[no-any-return]


@app.get("/health")  # type: ignore[misc]
async def health_check() -> dict[str, str]:
    """
    Generic Liveness Health check endpoint.
    """
    return {"status": "ok"}


@app.get("/health/{agent_id}")  # type: ignore[misc]
async def get_agent_health(
    agent_id: str,
    ingestor: Annotated[TelemetryIngestorAsync, Depends(get_telemetry_ingestor)],
) -> HealthReport:
    """
    Returns the health report for the specified agent (Circuit Breaker status).
    """
    # In a multi-tenant system, we would look up the correct breaker.
    # For now, we return the single instance's report.
    return await ingestor.circuit_breaker.get_health_report()


@app.get("/status/{agent_id}")  # type: ignore[misc]
async def get_agent_status(
    agent_id: str,
    ingestor: Annotated[TelemetryIngestorAsync, Depends(get_telemetry_ingestor)],
) -> bool:
    """
    Checks if the agent is allowed to process requests.
    Returns True if allowed, False if blocked.
    """
    return await ingestor.circuit_breaker.allow_request()


@app.post("/ingest/veritas", status_code=202)  # type: ignore[misc]
async def ingest_veritas_event(
    event: VeritasEvent,
    background_tasks: BackgroundTasks,
    ingestor: Annotated[TelemetryIngestorAsync, Depends(get_telemetry_ingestor)],
) -> dict[str, str]:
    """
    Ingests a single Veritas event (log).
    """
    logger.info(f"Received Veritas event: {event.event_id}")
    background_tasks.add_task(ingestor.process_event, event)
    return {"status": "accepted", "event_id": event.event_id}


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
