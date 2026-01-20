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

from coreason_sentinel.interfaces import (
    AssayGraderProtocol,
    GradeResult,
    PhoenixClientProtocol,
)
from coreason_sentinel.models import ConditionalSamplingRule, SentinelConfig
from coreason_sentinel.utils.logger import logger


class SpotChecker:
    """
    The Auditor: Responsible for sampling and grading live traffic.

    It determines whether a specific request should be audited based on sampling rates
    and conditional rules (e.g., "Sample 100% of negative sentiment").
    If sampled, it sends the conversation to the Assay Grader for detailed evaluation.
    """

    def __init__(
        self,
        config: SentinelConfig,
        grader: AssayGraderProtocol,
        phoenix_client: PhoenixClientProtocol,
    ):
        """
        Initializes the SpotChecker.

        Args:
            config: Configuration defining sampling rates and rules.
            grader: Service to perform grading (Assay).
            phoenix_client: Service to update traces with grade results.
        """
        self.config = config
        self.grader = grader
        self.phoenix_client = phoenix_client

    def should_sample(self, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Determines if a request should be sampled for grading.

        Calculates effective sample rate based on global config and conditional rules.

        Args:
            metadata: Metadata associated with the request (used for conditional rules).

        Returns:
            bool: True if the request should be sampled, False otherwise.
        """
        if not metadata:
            metadata = {}

        effective_rate = self.config.sampling_rate

        for rule in self.config.conditional_sampling_rules:
            if self._evaluate_rule(rule, metadata):
                effective_rate = max(effective_rate, rule.sample_rate)
                if effective_rate >= 1.0:
                    break

        return random.random() < effective_rate

    def _evaluate_rule(self, rule: ConditionalSamplingRule, metadata: Dict[str, Any]) -> bool:
        """
        Evaluates a single conditional sampling rule against the metadata.

        Args:
            rule: The rule to evaluate.
            metadata: The metadata to check against.

        Returns:
            bool: True if the rule condition is met.
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

        If successful, it also updates the corresponding trace in Phoenix with the grade results.

        Args:
            conversation: The conversation data (input, output, metadata).

        Returns:
            Optional[GradeResult]: The grade result if successful, None otherwise.
        """
        try:
            logger.info(f"Spot Checking conversation for agent {self.config.agent_id}")
            result = self.grader.grade_conversation(conversation)
            # Log the result
            logger.info(f"Grade Result - Faithfulness: {result.faithfulness_score}, Safety: {result.safety_score}")

            # Integration: Push grades back to Phoenix if trace info is available
            metadata = conversation.get("metadata", {})
            trace_id = metadata.get("trace_id")
            span_id = metadata.get("span_id")

            if trace_id and span_id:
                try:
                    attributes = {
                        "eval.faithfulness.score": result.faithfulness_score,
                        "eval.retrieval.precision.score": result.retrieval_precision_score,
                        "eval.safety.score": result.safety_score,
                    }
                    # Merge details if they are simple types? Or just dump them?
                    # For now, stick to scores as per PRD requirement.
                    self.phoenix_client.update_span_attributes(
                        trace_id=trace_id, span_id=span_id, attributes=attributes
                    )
                    logger.info(f"Updated Phoenix span {span_id} with evaluation results.")
                except Exception as e:
                    logger.error(f"Failed to update Phoenix span: {e}")

            return result
        except Exception as e:
            logger.error(f"Failed to grade conversation: {e}")
            return None
