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
from typing import Any, Dict, List, Protocol

from pydantic import BaseModel


class VeritasEvent(BaseModel):
    """
    Represents a raw event from the Veritas system (Data Source).
    This matches the expected schema from the Telemetry Ingestor description.
    """

    event_id: str
    timestamp: datetime
    agent_id: str
    session_id: str
    input_text: str
    output_text: str
    metrics: Dict[str, Any]  # e.g., latency, token_usage
    metadata: Dict[str, Any]


class GradeResult(BaseModel):
    """
    Represents the output from the Assay Grader.
    """

    faithfulness_score: float
    safety_score: float
    details: Dict[str, Any]


class VeritasClientProtocol(Protocol):
    """
    Interface for the Data Source (Veritas).
    """

    def fetch_logs(self, agent_id: str, since: datetime) -> List[VeritasEvent]:
        """
        Fetches logs from the source of truth.
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
        """
        ...


class BaselineProviderProtocol(Protocol):
    """
    Interface for retrieving Baseline Signatures.
    """

    def get_baseline_vectors(self, agent_id: str) -> List[List[float]]:
        """
        Retrieves the list of baseline vectors (embeddings) for the agent.
        """
        ...
