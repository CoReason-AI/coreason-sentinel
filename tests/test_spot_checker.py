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
from unittest.mock import MagicMock

from coreason_sentinel.interfaces import AssayGraderProtocol, GradeResult, PhoenixClientProtocol
from coreason_sentinel.models import SentinelConfig
from coreason_sentinel.spot_checker import SpotChecker


class TestSpotChecker(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SentinelConfig(
            agent_id="test-agent",
            owner_email="test@example.com",
            phoenix_endpoint="http://localhost:6006",
            sampling_rate=0.1,
            triggers=[],
        )
        self.mock_grader = MagicMock(spec=AssayGraderProtocol)
        self.mock_phoenix = MagicMock(spec=PhoenixClientProtocol)
        self.checker = SpotChecker(self.config, self.mock_grader, self.mock_phoenix)

    def test_should_sample_always(self) -> None:
        """Test with 100% sample rate."""
        self.config.sampling_rate = 1.0
        self.assertTrue(self.checker.should_sample())

    def test_should_sample_never(self) -> None:
        """Test with 0% sample rate."""
        self.config.sampling_rate = 0.0
        self.assertFalse(self.checker.should_sample())

    def test_check_sample_success(self) -> None:
        """Test successful grading without trace info."""
        expected_result = GradeResult(faithfulness_score=0.9, safety_score=1.0, details={"reason": "good"})
        self.mock_grader.grade_conversation.return_value = expected_result

        conversation = {"messages": [{"role": "user", "content": "hello"}]}
        result = self.checker.check_sample(conversation)

        self.assertEqual(result, expected_result)
        self.mock_grader.grade_conversation.assert_called_with(conversation)
        self.mock_phoenix.update_span_attributes.assert_not_called()

    def test_check_sample_success_with_phoenix(self) -> None:
        """Test successful grading WITH trace info -> Phoenix update."""
        expected_result = GradeResult(faithfulness_score=0.9, safety_score=1.0, details={"reason": "good"})
        self.mock_grader.grade_conversation.return_value = expected_result

        conversation = {
            "messages": [{"role": "user", "content": "hello"}],
            "metadata": {"trace_id": "trace-123", "span_id": "span-456"},
        }
        result = self.checker.check_sample(conversation)

        self.assertEqual(result, expected_result)
        self.mock_grader.grade_conversation.assert_called_with(conversation)

        expected_attributes = {
            "eval.faithfulness.score": 0.9,
            "eval.safety.score": 1.0,
        }
        self.mock_phoenix.update_span_attributes.assert_called_with(
            trace_id="trace-123", span_id="span-456", attributes=expected_attributes
        )

    def test_check_sample_failure(self) -> None:
        """Test handling of grader failure."""
        self.mock_grader.grade_conversation.side_effect = Exception("Grader down")

        result = self.checker.check_sample({})
        self.assertIsNone(result)
        self.mock_phoenix.update_span_attributes.assert_not_called()

    def test_check_sample_phoenix_failure(self) -> None:
        """Test successful grading but Phoenix update fails (should still return result)."""
        expected_result = GradeResult(faithfulness_score=0.9, safety_score=1.0, details={"reason": "good"})
        self.mock_grader.grade_conversation.return_value = expected_result
        self.mock_phoenix.update_span_attributes.side_effect = Exception("Phoenix down")

        conversation = {
            "messages": [{"role": "user", "content": "hello"}],
            "metadata": {"trace_id": "trace-123", "span_id": "span-456"},
        }
        result = self.checker.check_sample(conversation)

        self.assertEqual(result, expected_result)  # Should still return grade
        self.mock_phoenix.update_span_attributes.assert_called()
