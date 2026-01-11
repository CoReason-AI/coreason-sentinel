# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_sentinel

import random
from typing import Any, Dict, Optional

from coreason_sentinel.interfaces import AssayGraderProtocol, GradeResult
from coreason_sentinel.models import SentinelConfig
from coreason_sentinel.utils.logger import logger


class SpotChecker:
    """
    The Auditor: Responsible for sampling and grading live traffic.
    """

    def __init__(self, config: SentinelConfig, grader: AssayGraderProtocol):
        self.config = config
        self.grader = grader

    def should_sample(self, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Determines if a request should be sampled for grading.
        Uses random sampling based on configured sample_rate.
        Future: Add conditional sampling based on metadata (e.g. sentiment).
        """
        if not metadata:
            metadata = {}

        # 1. Check random sampling
        if random.random() < self.config.sample_rate:
            return True

        # 2. Check conditional sampling (placeholder)
        # e.g. if metadata.get("sentiment") == "negative": return True
        # For now, we only implement random sampling as per Atomic Unit scope.
        return False

    def check_sample(self, conversation: Dict[str, Any]) -> Optional[GradeResult]:
        """
        Sends the conversation to the Assay Grader.
        """
        try:
            logger.info(f"Spot Checking conversation for agent {self.config.agent_id}")
            result = self.grader.grade_conversation(conversation)
            # Log the result
            logger.info(f"Grade Result - Faithfulness: {result.faithfulness_score}, Safety: {result.safety_score}")
            return result
        except Exception as e:
            logger.error(f"Failed to grade conversation: {e}")
            return None
