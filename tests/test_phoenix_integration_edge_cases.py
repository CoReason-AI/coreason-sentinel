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


class TestPhoenixIntegrationEdgeCases(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SentinelConfig(
            agent_id="test-agent",
            owner_email="test@example.com",
            phoenix_endpoint="http://localhost:6006",
            sampling_rate=1.0,  # Always sample for these tests
            triggers=[],
        )
        self.mock_grader = MagicMock(spec=AssayGraderProtocol)
        self.mock_phoenix = MagicMock(spec=PhoenixClientProtocol)
        self.checker = SpotChecker(self.config, self.mock_grader, self.mock_phoenix)

        # Default grader behavior: success
        self.mock_grader.grade_conversation.return_value = GradeResult(
            faithfulness_score=0.9, safety_score=1.0, details={}
        )

    def test_missing_span_id(self) -> None:
        """
        Scenario: Trace ID is present, but Span ID is missing in metadata.
        Expectation: Phoenix client is NOT called (needs both).
        """
        conversation = {
            "messages": [{"role": "user", "content": "hello"}],
            "metadata": {"trace_id": "trace-123"},  # No span_id
        }
        self.checker.check_sample(conversation)

        self.mock_phoenix.update_span_attributes.assert_not_called()

    def test_missing_trace_id(self) -> None:
        """
        Scenario: Span ID is present, but Trace ID is missing.
        Expectation: Phoenix client is NOT called.
        """
        conversation = {
            "messages": [{"role": "user", "content": "hello"}],
            "metadata": {"span_id": "span-456"},  # No trace_id
        }
        self.checker.check_sample(conversation)

        self.mock_phoenix.update_span_attributes.assert_not_called()

    def test_malformed_metadata(self) -> None:
        """
        Scenario: Metadata is present but not a dict, or IDs are not strings.
        Expectation: Code handles it gracefully (or relies on .get safe access).
        """
        # Case 1: metadata is not a dict? ingestor.py ensures it is.
        # But let's check if trace_id is not a string (e.g. None or int)

        conversation = {"messages": [], "metadata": {"trace_id": None, "span_id": "span-123"}}
        self.checker.check_sample(conversation)
        self.mock_phoenix.update_span_attributes.assert_not_called()

    def test_phoenix_down_does_not_block_grading(self) -> None:
        """
        Complex Scenario: 'Phoenix Down'.
        The external observability platform is unavailable (raises Exception).
        The Sentinel MUST still return the grade so that Circuit Breaker triggers can work.
        """
        self.mock_phoenix.update_span_attributes.side_effect = Exception("Connection Timeout")

        conversation = {"messages": [], "metadata": {"trace_id": "t1", "span_id": "s1"}}

        result = self.checker.check_sample(conversation)

        # Verify result is still returned
        self.assertIsNotNone(result)
        if result:
            self.assertEqual(result.faithfulness_score, 0.9)
        # Verify Phoenix was attempted
        self.mock_phoenix.update_span_attributes.assert_called_once()

    def test_mixed_batch_processing(self) -> None:
        """
        Complex Scenario: 'Mixed Batch'.
        Simulates processing a stream of events where some have traces and some don't.
        Verifies correct selective updating.
        """
        # 1. Trace + Span
        c1 = {"metadata": {"trace_id": "t1", "span_id": "s1"}}
        # 2. No Trace
        c2 = {"metadata": {"span_id": "s2"}}
        # 3. Trace + Span
        c3 = {"metadata": {"trace_id": "t3", "span_id": "s3"}}

        batch = [c1, c2, c3]

        for c in batch:
            self.checker.check_sample(c)

        # Verify calls
        # Should be called for t1/s1 and t3/s3
        self.assertEqual(self.mock_phoenix.update_span_attributes.call_count, 2)

        calls = self.mock_phoenix.update_span_attributes.call_args_list
        # Call 1
        self.assertEqual(calls[0].kwargs["trace_id"], "t1")
        self.assertEqual(calls[0].kwargs["span_id"], "s1")
        # Call 2
        self.assertEqual(calls[1].kwargs["trace_id"], "t3")
        self.assertEqual(calls[1].kwargs["span_id"], "s3")
