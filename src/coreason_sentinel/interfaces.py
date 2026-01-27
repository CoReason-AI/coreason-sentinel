# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_sentinel

from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Tuple

from pydantic import BaseModel, Field


class VeritasEvent(BaseModel):
    """
    Represents a raw event from the Veritas system (Data Source).
    This matches the expected schema from the Telemetry Ingestor description.

    Attributes:
        event_id: Unique identifier for the event.
        timestamp: When the event occurred.
        agent_id: ID of the agent generating the event.
        session_id: Session ID associated with the event.
        input_text: User input.
        output_text: Model output.
        metrics: Dictionary of metrics (latency, tokens, etc.).
        metadata: Additional metadata (embeddings, flags, etc.).
    """

    event_id: str
    timestamp: datetime
    agent_id: str
    session_id: str
    input_text: str
    output_text: str
    metrics: Dict[str, Any]  # e.g., latency, token_usage
    metadata: Dict[str, Any]


class OTELSpan(BaseModel):
    """
    Represents an OpenTelemetry Span for ingestion.
    Simplified model focusing on fields relevant for Sentinel analysis.

    Attributes:
        trace_id: Unique 32-hex-character identifier for the trace.
        span_id: Unique 16-hex-character identifier for the span.
        name: Name of the operation (e.g., 'query_llm').
        start_time_unix_nano: Start time in nanoseconds since epoch.
        end_time_unix_nano: End time in nanoseconds since epoch.
        attributes: Key-value pairs of span attributes.
        status_code: Status code (UNSET, OK, ERROR).
        status_message: Status description if error.
    """

    trace_id: str = Field(..., description="Unique 32-hex-character identifier for the trace.")
    span_id: str = Field(..., description="Unique 16-hex-character identifier for the span.")
    name: str = Field(..., description="Name of the operation (e.g., 'query_llm').")
    start_time_unix_nano: int = Field(..., description="Start time in nanoseconds since epoch.")
    end_time_unix_nano: int = Field(..., description="End time in nanoseconds since epoch.")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Key-value pairs of span attributes.")
    status_code: str = Field("UNSET", description="Status code (UNSET, OK, ERROR).")
    status_message: Optional[str] = Field(None, description="Status description if error.")


class GradeResult(BaseModel):
    """
    Represents the output from the Assay Grader.

    Attributes:
        faithfulness_score: Score (0.0-1.0) indicating how faithful the answer is to context.
        retrieval_precision_score: Score (0.0-1.0) indicating relevance of retrieved docs.
        safety_score: Score (0.0-1.0) indicating safety/toxicity.
        details: Dictionary containing detailed grading info.
    """

    faithfulness_score: float
    retrieval_precision_score: float
    safety_score: float
    details: Dict[str, Any]


class VeritasClientProtocol(Protocol):
    """
    Interface for the Data Source (Veritas).
    """

    def fetch_logs(self, agent_id: str, since: datetime) -> List[VeritasEvent]:
        """
        Fetches logs from the source of truth.

        Args:
            agent_id: The ID of the agent to fetch logs for.
            since: The timestamp to fetch logs starting from.

        Returns:
            List[VeritasEvent]: A list of VeritasEvent objects.
        """
        ...

    def subscribe(self, agent_id: str, callback: Any) -> None:
        """
        Subscribes to live events (optional, depending on push/pull model).
        """
        ...


class AssayGraderProtocol(Protocol):
    """
    Interface for the Grader (Assay).
    """

    def grade_conversation(self, conversation: Dict[str, Any]) -> GradeResult:
        """
        Sends a conversation to the Assay Judge for grading.

        Args:
            conversation: A dictionary containing 'input', 'output', and 'metadata'.

        Returns:
            GradeResult: The result of the grading process.
        """
        ...


class BaselineProviderProtocol(Protocol):
    """
    Interface for retrieving Baseline Signatures.
    """

    def get_baseline_vectors(self, agent_id: str, groups: Optional[List[str]] = None) -> List[List[float]]:
        """
        Retrieves the list of baseline vectors (embeddings) for the agent.

        Args:
            agent_id: The agent ID.
            groups: Optional list of groups/roles to select specific baselines (e.g., 'Doctors').

        Returns:
            List[List[float]]: A list of embedding vectors.
        """
        ...

    def get_baseline_output_length_distribution(
        self, agent_id: str, groups: Optional[List[str]] = None
    ) -> Tuple[List[float], List[float]]:
        """
        Retrieves the baseline output length distribution.

        Args:
            agent_id: The agent ID.
            groups: Optional list of groups/roles to select specific baselines.

        Returns:
            Tuple[List[float], List[float]]: A tuple containing:
                - probabilities: List of float probabilities summing to 1.0.
                - bin_edges: List of floats defining the bin edges (len(probabilities) + 1).
        """
        ...


class NotificationServiceProtocol(Protocol):
    """
    Interface for the Notification Service (Identity).
    Responsible for sending alerts when critical events occur (e.g., Circuit Breaker Trips).
    """

    def send_critical_alert(self, email: str, agent_id: str, reason: str) -> None:
        """
        Sends a critical alert notification.

        Args:
            email: The recipient's email address (from SentinelConfig).
            agent_id: The ID of the agent that triggered the alert.
            reason: The description of why the alert was triggered (e.g., trigger violation).
        """
        ...


class PhoenixClientProtocol(Protocol):
    """
    Interface for the Phoenix Tracing Service.
    Responsible for updating spans with new attributes (e.g., evaluation grades).
    """

    def update_span_attributes(self, trace_id: str, span_id: str, attributes: Dict[str, Any]) -> None:
        """
        Updates an existing span with new attributes.

        Args:
            trace_id: The Trace ID associated with the span.
            span_id: The Span ID to update.
            attributes: A dictionary of attributes to append/update.
        """
        ...
