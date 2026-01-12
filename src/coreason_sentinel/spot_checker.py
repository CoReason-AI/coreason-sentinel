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
from coreason_sentinel.models import ConditionalSamplingRule, SentinelConfig
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
        Calculates effective sample rate based on global config and conditional rules.
        """
        if not metadata:
            metadata = {}

        effective_rate = self.config.sample_rate

        for rule in self.config.conditional_sampling_rules:
            if self._evaluate_rule(rule, metadata):
                effective_rate = max(effective_rate, rule.sample_rate)
                if effective_rate >= 1.0:
                    break

        return random.random() < effective_rate

    def _evaluate_rule(self, rule: ConditionalSamplingRule, metadata: Dict[str, Any]) -> bool:
        """
        Evaluates a single conditional sampling rule against the metadata.
        """
        if rule.operator == "EXISTS":
            return rule.metadata_key in metadata

        if rule.metadata_key not in metadata:
            return False

        value = metadata[rule.metadata_key]

        if rule.operator == "EQUALS":
            return value == rule.value  # type: ignore[no-any-return]

        if rule.operator == "CONTAINS":
            if isinstance(value, (str, list, tuple, dict)):
                return rule.value in value
            # Fallback: convert to string and check?
            # For now, if type doesn't support 'in', return False to be safe
            return False

        return False  # pragma: no cover

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
