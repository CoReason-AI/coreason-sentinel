# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_sentinel

import unittest

from coreason_sentinel.utils.metric_extractor import MetricExtractor


class TestMetricExtractor(unittest.TestCase):
    def setUp(self) -> None:
        self.patterns = ["bad bot", "stupid", "fail"]
        self.extractor = MetricExtractor(self.patterns)

    def test_extract_refusal_bool(self) -> None:
        metrics = self.extractor.extract("hello", {"is_refusal": True})
        self.assertEqual(metrics.get("refusal_count"), 1.0)

    def test_extract_refusal_string_true(self) -> None:
        metrics = self.extractor.extract("hello", {"is_refusal": "True"})
        self.assertEqual(metrics.get("refusal_count"), 1.0)

    def test_extract_refusal_none(self) -> None:
        metrics = self.extractor.extract("hello", {})
        self.assertIsNone(metrics.get("refusal_count"))

    def test_extract_sentiment_match(self) -> None:
        metrics = self.extractor.extract("You are a bad bot", {})
        self.assertEqual(metrics.get("sentiment_frustration_count"), 1.0)

    def test_extract_sentiment_no_match(self) -> None:
        metrics = self.extractor.extract("You are helpful", {})
        self.assertIsNone(metrics.get("sentiment_frustration_count"))

    def test_extract_sentiment_case_insensitive(self) -> None:
        metrics = self.extractor.extract("BAD BOT", {})
        self.assertEqual(metrics.get("sentiment_frustration_count"), 1.0)

    def test_extract_invalid_regex(self) -> None:
        # Invalid regex pattern shouldn't crash
        extractor = MetricExtractor(["["])  # Invalid
        metrics = extractor.extract("hello", {})
        # Should just return empty or log error
        self.assertEqual(metrics, {})

    def test_parse_bool_variants(self) -> None:
        # Test helper directly via extract
        self.assertEqual(self.extractor.extract("", {"is_refusal": 1})["refusal_count"], 1.0)
        self.assertEqual(self.extractor.extract("", {"is_refusal": "yes"})["refusal_count"], 1.0)
        self.assertEqual(self.extractor.extract("", {"is_refusal": 0}), {})
        # Test explicit fallback
        self.assertEqual(self.extractor.extract("", {"is_refusal": object()}), {})
