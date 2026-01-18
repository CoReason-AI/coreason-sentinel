# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_sentinel

import re
from typing import Any

from coreason_sentinel.utils.logger import logger


class MetricExtractor:
    """
    Utility for extracting custom metrics from event metadata and text.
    Handles Refusal detection and Sentiment analysis.
    """

    def __init__(self, sentiment_regex_patterns: list[str]):
        self.sentiment_regex_patterns = sentiment_regex_patterns

    def extract(self, input_text: str, metadata: dict[str, Any]) -> dict[str, float]:
        """
        Extracts custom metrics based on metadata flags and regex patterns.
        """
        metrics: dict[str, float] = {}

        # 1. Refusal Detection
        # Check "is_refusal" in metadata.
        # It could be a boolean, string "True"/"False", or 1/0.
        is_refusal = metadata.get("is_refusal")
        if self._parse_bool(is_refusal):
            metrics["refusal_count"] = 1.0

        # 2. Sentiment Detection (Regex)
        # We check the input_text for user frustration signals
        for pattern in self.sentiment_regex_patterns:
            try:
                if re.search(pattern, input_text, re.IGNORECASE):
                    metrics["sentiment_frustration_count"] = 1.0
                    break  # Count once per event
            except re.error as e:
                logger.error(f"Invalid regex pattern '{pattern}' in configuration: {e}")
                continue

        return metrics

    def _parse_bool(self, value: Any) -> bool:
        """
        Robust boolean parsing.
        """
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        if isinstance(value, (int, float)):
            return bool(value)
        return False
