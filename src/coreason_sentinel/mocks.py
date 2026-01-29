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
from typing import Any, Dict, List, Optional, Tuple

from coreason_sentinel.interfaces import GradeResult, VeritasEvent
from coreason_sentinel.utils.logger import logger


class MockNotificationService:
    """
    Mock implementation of NotificationServiceProtocol.
    """

    def send_critical_alert(self, email: str, agent_id: str, reason: str) -> None:
        logger.info(f"[MOCK] Sending critical alert to {email} for agent {agent_id}: {reason}")


class MockAssayGrader:
    """
    Mock implementation of AssayGraderProtocol.
    """

    def grade_conversation(self, conversation: Dict[str, Any]) -> GradeResult:
        logger.info("[MOCK] Grading conversation...")
        # Return a dummy high score result
        return GradeResult(
            faithfulness_score=0.95,
            retrieval_precision_score=0.9,
            safety_score=1.0,
            details={"mock": True},
        )


class MockPhoenixClient:
    """
    Mock implementation of PhoenixClientProtocol.
    """

    def update_span_attributes(self, trace_id: str, span_id: str, attributes: Dict[str, Any]) -> None:
        logger.info(f"[MOCK] Updating Phoenix span {span_id} (Trace {trace_id}) with {attributes}")


class MockBaselineProvider:
    """
    Mock implementation of BaselineProviderProtocol.
    """

    def get_baseline_vectors(self, agent_id: str, groups: Optional[List[str]] = None) -> List[List[float]]:
        logger.info(f"[MOCK] Fetching baseline vectors for {agent_id} (Groups: {groups})")
        # Return a few dummy vectors (dimension 3 for simplicity in testing, though usually higher)
        return [[0.1, 0.2, 0.3], [0.1, 0.2, 0.4]]

    def get_baseline_output_length_distribution(
        self, agent_id: str, groups: Optional[List[str]] = None
    ) -> Tuple[List[float], List[float]]:
        logger.info(f"[MOCK] Fetching baseline output length distribution for {agent_id}")
        # Return a dummy distribution (probabilities, bin_edges)
        # Probabilities must sum to 1.0. 3 bins.
        probabilities = [0.2, 0.5, 0.3]
        bin_edges = [0.0, 10.0, 50.0, 100.0]
        return probabilities, bin_edges


class MockVeritasClient:
    """
    Mock implementation of VeritasClientProtocol.
    """

    def fetch_logs(self, agent_id: str, since: datetime) -> List[VeritasEvent]:
        logger.info(f"[MOCK] Fetching Veritas logs for {agent_id} since {since}")
        return []

    def subscribe(self, agent_id: str, callback: Any) -> None:
        logger.info(f"[MOCK] Subscribing to Veritas logs for {agent_id}")
