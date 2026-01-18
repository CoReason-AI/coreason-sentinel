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
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Union
from unittest.mock import MagicMock, patch

from coreason_sentinel.circuit_breaker import CircuitBreaker
from coreason_sentinel.ingestor import TelemetryIngestor
from coreason_sentinel.interfaces import (
    BaselineProviderProtocol,
    VeritasClientProtocol,
    VeritasEvent,
)
from coreason_sentinel.models import SentinelConfig
from coreason_sentinel.spot_checker import SpotChecker


class TestIngestorEdgeCases(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SentinelConfig(
            agent_id="test-agent",
            owner_email="test@example.com",
            phoenix_endpoint="http://localhost:6006",
            sampling_rate=1.0,
            triggers=[],
        )
        self.mock_cb = MagicMock(spec=CircuitBreaker)
        self.mock_sc = MagicMock(spec=SpotChecker)
        self.mock_bp = MagicMock(spec=BaselineProviderProtocol)
        self.mock_veritas = MagicMock(spec=VeritasClientProtocol)
        self.ingestor = TelemetryIngestor(self.config, self.mock_cb, self.mock_sc, self.mock_bp, self.mock_veritas)

        self.event = VeritasEvent(
            event_id="evt-1",
            timestamp=datetime.now(timezone.utc),
            agent_id="test-agent",
            session_id="sess-1",
            input_text="hello",
            output_text="world",
            metrics={"latency": 100, "token_count": 50},
            metadata={},
        )

    @patch("coreason_sentinel.ingestor.DriftEngine.detect_vector_drift")
    @patch("coreason_sentinel.ingestor.logger")
    def test_process_drift_partial_failures(
        self, mock_logger: MagicMock, mock_detect_vector: MagicMock
    ) -> None:
        """
        Verify that if Vector Drift detection fails, Output and Relevance Drift still run.
        """
        # Setup: Vector drift fails with exception
        self.event.metadata["embedding"] = [0.1, 0.2]
        self.event.metadata["query_embedding"] = [1.0]
        self.event.metadata["response_embedding"] = [1.0]
        self.mock_bp.get_baseline_vectors.return_value = [[0.1, 0.2]]

        mock_detect_vector.side_effect = Exception("Vector Failure")

        # Setup: Output drift works
        # Setup: Relevance drift works (implicit since no mock failure)

        self.ingestor.process_drift(self.event)

        # 1. Verify Vector failure logged
        mock_logger.error.assert_any_call("Failed to process vector drift detection: Vector Failure")

        # 2. Verify Output Drift attempted (by checking if metric recorded or internal call made)
        # We can check if _process_output_drift was called if we mocked it, but we didn't.
        # Instead, check if metric recorded for output_length (part of logic)
        self.mock_cb.record_metric.assert_any_call("output_length", 50.0)

        # 3. Verify Relevance Drift recorded
        self.mock_cb.record_metric.assert_any_call("relevance_drift", 0.0)

    def test_process_drift_empty_event(self) -> None:
        """
        Verify robustness against minimal event (no metadata, no metrics).
        """
        empty_event = VeritasEvent(
            event_id="empty",
            timestamp=datetime.now(timezone.utc),
            agent_id="agent",
            session_id="s1",
            input_text="",
            output_text="",
            metrics={},
            metadata={},
        )

        # Should execute without error
        self.ingestor.process_drift(empty_event)

        # Verify no drift metrics recorded
        calls = [c[0][0] for c in self.mock_cb.record_metric.call_args_list]
        self.assertNotIn("vector_drift", calls)
        self.assertNotIn("relevance_drift", calls)
        # Output length might be recorded as 0.0 (fallback to len("") which is 0)
        self.mock_cb.record_metric.assert_any_call("output_length", 0.0)

    def test_drift_lag_scenario(self) -> None:
        """
        Verify that metrics recorded by process_drift are accepted by CircuitBreaker
        even if there is a lag (simulated by sleep is slow, so we rely on logic).
        We just check that record_metric is called.
        """
        # Event happened 1 hour ago
        old_event = VeritasEvent(
            event_id="old",
            timestamp=datetime.fromtimestamp(1000, timezone.utc),
            agent_id="agent",
            session_id="s1",
            input_text="hi",
            output_text="hi",
            metrics={},
            metadata={"query_embedding": [1.0], "response_embedding": [0.0]},
        )

        # Process now
        self.ingestor.process_drift(old_event)

        # Check metric recorded
        self.mock_cb.record_metric.assert_any_call("relevance_drift", 1.0)
