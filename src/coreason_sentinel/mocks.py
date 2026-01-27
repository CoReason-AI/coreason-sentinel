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
from typing import Any, Dict, List, Tuple

from coreason_sentinel.interfaces import (
    AssayGraderProtocol,
    BaselineProviderProtocol,
    GradeResult,
    NotificationServiceProtocol,
    PhoenixClientProtocol,
    VeritasClientProtocol,
    VeritasEvent,
)
from coreason_sentinel.utils.logger import logger


class MockNotificationService(NotificationServiceProtocol):
    """
    Mock implementation of NotificationService for standalone/testing use.
    Logs alerts instead of sending emails.
    """

    def send_critical_alert(self, email: str, agent_id: str, reason: str) -> None:
        logger.warning(f"CRITICAL ALERT to {email} for agent {agent_id}: {reason}")


class MockBaselineProvider(BaselineProviderProtocol):
    """
    Mock implementation of BaselineProvider.
    Returns empty/default baselines.
    """

    def get_baseline_vectors(self, agent_id: str) -> List[List[float]]:
        logger.info(f"MockBaselineProvider: Requesting baseline vectors for {agent_id}")
        return []

    def get_baseline_output_length_distribution(self, agent_id: str) -> Tuple[List[float], List[float]]:
        logger.info(f"MockBaselineProvider: Requesting output length distribution for {agent_id}")
        return [], []


class MockVeritasClient(VeritasClientProtocol):
    """
    Mock implementation of VeritasClient.
    Returns empty log lists.
    """

    def fetch_logs(self, agent_id: str, since: datetime) -> List[VeritasEvent]:
        logger.info(f"MockVeritasClient: Fetching logs for {agent_id} since {since}")
        return []

    def subscribe(self, agent_id: str, callback: Any) -> None:
        logger.info(f"MockVeritasClient: Subscribed to {agent_id}")


class MockPhoenixClient(PhoenixClientProtocol):
    """
    Mock implementation of PhoenixClient.
    Logs span updates.
    """

    def update_span_attributes(self, trace_id: str, span_id: str, attributes: Dict[str, Any]) -> None:
        logger.info(f"MockPhoenixClient: Updating span {span_id} (Trace {trace_id}) with attributes: {attributes}")


class MockGrader(AssayGraderProtocol):
    """
    Mock implementation of AssayGraderProtocol.
    Returns dummy GradeResult.
    """

    def grade_conversation(self, conversation: Dict[str, Any]) -> GradeResult:
        logger.info("MockGrader: Grading conversation")
        return GradeResult(
            faithfulness_score=1.0,
            retrieval_precision_score=1.0,
            safety_score=1.0,
            details={},
        )
