# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_sentinel

from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI

from coreason_sentinel.ingestor import TelemetryIngestor
from coreason_sentinel.interfaces import OTELSpan
from coreason_sentinel.utils.logger import logger

app = FastAPI(title="CoReason Sentinel", version="0.1.0")


def get_telemetry_ingestor() -> TelemetryIngestor:
    """
    Dependency to provide the TelemetryIngestor instance.

    In a real deployment, this would initialize the full object graph
    (Redis, Veritas Client, etc.) based on configuration.
    For now, it raises NotImplementedError to ensure tests override it.

    Raises:
        NotImplementedError: Always, as this is a placeholder dependency.
    """
    raise NotImplementedError("TelemetryIngestor dependency must be overridden.")


@app.get("/health")  # type: ignore[misc]
def health_check() -> dict[str, str]:
    """
    Health check endpoint.

    Returns:
        dict[str, str]: Status message {"status": "ok"}.
    """
    return {"status": "ok"}


@app.post("/ingest/otel/span", status_code=202)  # type: ignore[misc]
def ingest_otel_span(
    span: OTELSpan,
    background_tasks: BackgroundTasks,
    ingestor: Annotated[TelemetryIngestor, Depends(get_telemetry_ingestor)],
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
