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
from unittest.mock import AsyncMock, MagicMock, patch

from coreason_sentinel.circuit_breaker import CircuitBreaker
from coreason_sentinel.ingestor import TelemetryIngestorAsync
from coreason_sentinel.interfaces import (
    BaselineProviderProtocol,
    VeritasClientProtocol,
    VeritasEvent,
)
from coreason_sentinel.models import SentinelConfig
from coreason_sentinel.spot_checker import SpotChecker


class TestIngestorEdgeCases(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.config = SentinelConfig(
            agent_id="test-agent",
            owner_email="test@example.com",
            phoenix_endpoint="http://localhost:6006",
            sampling_rate=1.0,
            triggers=[],
        )
        self.mock_cb = MagicMock(spec=CircuitBreaker)
        self.mock_sc = MagicMock(spec=SpotChecker)
        self.mock_sc.should_sample.return_value = False
        self.mock_bp = MagicMock(spec=BaselineProviderProtocol)
        self.mock_veritas = MagicMock(spec=VeritasClientProtocol)
        self.ingestor = TelemetryIngestorAsync(self.config, self.mock_cb, self.mock_sc, self.mock_bp, self.mock_veritas)

        self.event = VeritasEvent(
            event_id="evt-1",
            timestamp=datetime.now(timezone.utc),
            agent_id="test-agent",
            session_id="sess-1",
            input_text="hello",
            output_text="world",
            metrics={"latency": 0.1},
            metadata={"embedding": [0.1, 0.2]},
        )

    async def test_ingest_process_event_failure_skips_drift(self) -> None:
        """
        Edge Case: If process_event fails, process_drift should NOT be called.
        """
        # Setup: process_event raises exception
        with patch.object(self.ingestor, "process_event", side_effect=Exception("Processing Failed")):
            with patch.object(self.ingestor, "process_drift", new_callable=AsyncMock) as mock_drift:
                # Need to mock anyio.to_thread.run_sync used in ingest_from_veritas_since
                with patch("anyio.to_thread.run_sync", side_effect=lambda func, *args: func(*args)):
                    self.mock_veritas.fetch_logs.return_value = [self.event]
                    count = await self.ingestor.ingest_from_veritas_since(datetime.now(timezone.utc))

                # Should not have processed successfully
                self.assertEqual(count, 0)
                # process_drift should NOT have been called
                mock_drift.assert_not_called()

    async def test_ingest_process_drift_partial_failure_counts_success(self) -> None:
        """
        Edge Case: If process_drift fails (externally) or swallows errors,
        but process_event succeeded, does it count?

        Current Implementation: process_drift handles its own errors for specific drift types,
        but if it raises a top-level exception (e.g. check_triggers fails),
        it would bubble up and cause the event to be skipped in the loop (count not incremented).

        Let's test the scenario where process_drift RAISES an exception (simulating catastrophic failure).
        """
        with patch.object(self.ingestor, "process_drift", side_effect=Exception("Drift Boom")):
            # Mock anyio.to_thread.run_sync if needed, but here fetch_logs is sync.
            # But ingest_from_veritas_since wraps fetch_logs in run_sync.
            with patch("anyio.to_thread.run_sync", side_effect=lambda func, *args: func(*args)):
                self.mock_veritas.fetch_logs.return_value = [self.event]

                # The exception is caught in the loop in ingestor.py
                count = await self.ingestor.ingest_from_veritas_since(datetime.now(timezone.utc))

                # Since the try/except block wraps both calls, if process_drift fails,
                # the 'count += 1' line is skipped.
                # So expected count is 0.
                self.assertEqual(count, 0)

        # Verify metrics from process_event WERE recorded (since it ran before the crash)
        self.mock_cb.record_metric.assert_any_call("latency", 0.1, None)

    async def test_ingest_drift_lag_simulation(self) -> None:
        """
        Complex Scenario: Drift calculation is slow.
        Ensures strict ordering: process_event -> process_drift -> next event.
        """
        call_order = []

        async def mock_process_event(evt: VeritasEvent) -> None:
            call_order.append(f"event_{evt.event_id}")

        async def mock_process_drift(evt: VeritasEvent) -> None:
            call_order.append(f"drift_{evt.event_id}")

        with patch.object(self.ingestor, "process_event", side_effect=mock_process_event):
            with patch.object(self.ingestor, "process_drift", side_effect=mock_process_drift):
                with patch("anyio.to_thread.run_sync", side_effect=lambda func, *args: func(*args)):
                    evt1 = self.event.model_copy(update={"event_id": "1"})
                    evt2 = self.event.model_copy(update={"event_id": "2"})
                    self.mock_veritas.fetch_logs.return_value = [evt1, evt2]

                    await self.ingestor.ingest_from_veritas_since(datetime.now(timezone.utc))

                    # Strict sequential order
                    expected = ["event_1", "drift_1", "event_2", "drift_2"]
                    self.assertEqual(call_order, expected)

    async def test_ingest_empty_event_fields(self) -> None:
        """
        Edge Case: Event with empty strings/dicts.
        Should not crash.
        """
        weird_event = VeritasEvent(
            event_id="evt-empty",
            timestamp=datetime.now(timezone.utc),
            agent_id="test-agent",
            session_id="",
            input_text="",
            output_text="",
            metrics={},
            metadata={},
        )

        with patch("anyio.to_thread.run_sync", side_effect=lambda func, *args: func(*args)):
            self.mock_veritas.fetch_logs.return_value = [weird_event]
            count = await self.ingestor.ingest_from_veritas_since(datetime.now(timezone.utc))

        self.assertEqual(count, 1)
        # Verify no crash in process_drift
        self.mock_bp.get_baseline_vectors.assert_not_called()
