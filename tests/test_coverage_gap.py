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

import pytest

from coreason_sentinel.circuit_breaker import CircuitBreaker
from coreason_sentinel.ingestor import TelemetryIngestor, TelemetryIngestorAsync
from coreason_sentinel.interfaces import (
    BaselineProviderProtocol,
    VeritasClientProtocol,
    VeritasEvent,
)
from coreason_sentinel.models import SentinelConfig
from coreason_sentinel.spot_checker import SpotChecker


@pytest.mark.asyncio
class TestCoverageGap(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.config = SentinelConfig(
            agent_id="test-agent",
            owner_email="test@example.com",
            phoenix_endpoint="http://localhost:6006",
            sampling_rate=1.0,
            triggers=[],
        )
        self.mock_cb = MagicMock(spec=CircuitBreaker)
        self.mock_cb.record_metric = AsyncMock()
        self.mock_cb.check_triggers = AsyncMock()
        self.mock_cb.get_recent_values = AsyncMock()

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
            metadata={},
        )

    async def test_ingest_from_veritas_fetch_logs_exception(self) -> None:
        """Cover 147-149: Exception handling in fetch_logs."""
        self.mock_veritas.fetch_logs.side_effect = Exception("Fetch Error")

        with patch("anyio.to_thread.run_sync", side_effect=lambda func, *args: func(*args)):
            count = await self.ingestor.ingest_from_veritas_since(datetime.now(timezone.utc))

        self.assertEqual(count, 0)

    async def test_ingest_from_veritas_empty_logs(self) -> None:
        """Cover 152: if not events: return 0."""
        self.mock_veritas.fetch_logs.return_value = []
        with patch("anyio.to_thread.run_sync", side_effect=lambda func, *args: func(*args)):
            count = await self.ingestor.ingest_from_veritas_since(datetime.now(timezone.utc))
        self.assertEqual(count, 0)

    async def test_process_event_custom_metrics_loop(self) -> None:
        """Cover 178: loop over custom metrics."""
        self.config.sentiment_regex_patterns = ["hello"]
        self.event.input_text = "hello world"

        await self.ingestor.process_event(self.event)

        self.mock_cb.record_metric.assert_any_call("sentiment_frustration_count", 1.0)

    async def test_process_drift_vector_exception(self) -> None:
        """Cover 221-222: Exception in vector drift."""
        self.event.metadata["embedding"] = [0.1]

        with patch("anyio.to_thread.run_sync", side_effect=lambda func, *args: func(*args)):
            # Force get_baseline_vectors to raise
            self.mock_bp.get_baseline_vectors.side_effect = Exception("Vector DB Error")

            await self.ingestor.process_drift(self.event)
            # Should not crash

    async def test_regex_error(self) -> None:
        """Cover 266-268: Regex error."""
        self.config.sentiment_regex_patterns = ["["]  # Invalid regex
        self.event.input_text = "test"

        metrics = self.ingestor._extract_custom_metrics("test", {})
        self.assertEqual(metrics, {})

    async def test_output_drift_early_returns(self) -> None:
        """Cover 292, 295, 304."""
        # 292: AttributeError/NotImplementedError
        self.mock_bp.get_baseline_output_length_distribution.side_effect = AttributeError
        with patch("anyio.to_thread.run_sync", side_effect=lambda func, *args: func(*args)):
            await self.ingestor._process_output_drift(self.event)

        # 295: not baseline_dist
        self.mock_bp.get_baseline_output_length_distribution.side_effect = None
        self.mock_bp.get_baseline_output_length_distribution.return_value = ([], [])
        with patch("anyio.to_thread.run_sync", side_effect=lambda func, *args: func(*args)):
            await self.ingestor._process_output_drift(self.event)

        # 304: not recent_samples
        self.mock_bp.get_baseline_output_length_distribution.return_value = ([0.1], [0, 1])
        self.mock_cb.get_recent_values.return_value = []
        with patch("anyio.to_thread.run_sync", side_effect=lambda func, *args: func(*args)):
            await self.ingestor._process_output_drift(self.event)

    async def test_kl_divergence_error(self) -> None:
        """Cover 314-315: KL error."""
        # Setup valid flow until KL computation
        self.mock_bp.get_baseline_output_length_distribution.return_value = ([0.5, 0.5], [0, 10, 20])
        self.mock_cb.get_recent_values.return_value = [5, 15]

        with patch("anyio.to_thread.run_sync", side_effect=lambda func, *args: func(*args)):
            with patch(
                "coreason_sentinel.drift_engine.DriftEngine.compute_kl_divergence", side_effect=ValueError("Math Error")
            ):
                await self.ingestor._process_output_drift(self.event)


class TestSyncFacadeCoverage(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.config = SentinelConfig(
            agent_id="test-agent",
            owner_email="test@example.com",
            phoenix_endpoint="http://localhost:6006",
            sampling_rate=1.0,
            triggers=[],
        )
        self.mock_cb = MagicMock(spec=CircuitBreaker)
        self.mock_cb.record_metric = AsyncMock()
        self.mock_cb.check_triggers = AsyncMock()

        self.mock_sc = MagicMock(spec=SpotChecker)
        self.mock_sc.should_sample.return_value = False
        self.mock_bp = MagicMock(spec=BaselineProviderProtocol)
        self.mock_veritas = MagicMock(spec=VeritasClientProtocol)
        self.ingestor = TelemetryIngestor(self.config, self.mock_cb, self.mock_sc, self.mock_bp, self.mock_veritas)

    def test_ingest_from_veritas_since_sync(self) -> None:
        """Cover 354: ingest_from_veritas_since facade."""
        with patch("anyio.run") as mock_run:
            self.ingestor.ingest_from_veritas_since(datetime.now(timezone.utc))
            mock_run.assert_called_once()
            # Assert arguments
            args = mock_run.call_args
            self.assertEqual(args[0][0], self.ingestor._async.ingest_from_veritas_since)
