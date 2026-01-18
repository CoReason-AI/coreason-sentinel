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
from datetime import datetime
from unittest.mock import MagicMock, patch

from coreason_sentinel.drift_monitor import DriftMonitor
from coreason_sentinel.interfaces import BaselineProviderProtocol, VeritasEvent
from coreason_sentinel.metric_store import MetricStore
from coreason_sentinel.models import SentinelConfig


class TestDriftMonitor(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SentinelConfig(
            agent_id="test_agent",
            owner_email="test@example.com",
            phoenix_endpoint="http://locahost",
            drift_sample_window=10,
        )
        self.mock_baseline_provider = MagicMock(spec=BaselineProviderProtocol)
        self.mock_metric_store = MagicMock(spec=MetricStore)
        self.monitor = DriftMonitor(self.config, self.mock_baseline_provider, self.mock_metric_store)

        self.event = VeritasEvent(
            event_id="evt1",
            timestamp=datetime.now(),
            agent_id="test_agent",
            session_id="sess1",
            input_text="input",
            output_text="output text",
            metrics={"completion_tokens": 10},
            metadata={},
        )

    def test_process_event_calls_sub_methods(self) -> None:
        """Test that process_event triggers all checks."""
        with (
            patch.object(self.monitor, "_check_vector_drift") as mock_vector,
            patch.object(self.monitor, "_check_output_drift") as mock_output,
            patch.object(self.monitor, "_check_relevance_drift") as mock_relevance,
        ):
            self.monitor.process_event(self.event)
            mock_vector.assert_called_once_with(self.event)
            mock_output.assert_called_once_with(self.event)
            mock_relevance.assert_called_once_with(self.event)

    def test_check_vector_drift_success(self) -> None:
        """Test vector drift calculation and recording."""
        self.event.metadata["embedding"] = [0.1, 0.2]
        self.mock_baseline_provider.get_baseline_vectors.return_value = [[0.1, 0.2]]

        # DriftEngine.detect_vector_drift mocked or real?
        # Let's mock it to isolate DriftMonitor logic
        with patch("coreason_sentinel.drift_monitor.DriftEngine.detect_vector_drift", return_value=0.5) as mock_drift:
            self.monitor._check_vector_drift(self.event)
            mock_drift.assert_called()
            # Verify metric recorded
            self.mock_metric_store.record_metric.assert_called_with(
                "test_agent", "vector_drift", 0.5, retention_window=3600
            )

    def test_check_vector_drift_no_embedding(self) -> None:
        """Test vector drift skipped if no embedding."""
        self.event.metadata = {}  # No embedding
        self.monitor._check_vector_drift(self.event)
        self.mock_metric_store.record_metric.assert_not_called()

    def test_check_vector_drift_invalid_embedding(self) -> None:
        """Test vector drift skipped if embedding is invalid type."""
        self.event.metadata["embedding"] = "not_a_list"
        self.monitor._check_vector_drift(self.event)
        self.mock_metric_store.record_metric.assert_not_called()

    def test_check_vector_drift_exception(self) -> None:
        """Test vector drift exception handling."""
        self.event.metadata["embedding"] = [0.1]
        self.mock_baseline_provider.get_baseline_vectors.side_effect = Exception("DB Error")
        # Should not raise
        self.monitor._check_vector_drift(self.event)

    def test_check_output_drift_success(self) -> None:
        """Test output drift calculation."""
        # 1. Output length calc
        self.event.metrics = {"completion_tokens": 100}

        # 2. Baseline setup
        baseline_dist = [0.5, 0.5]
        bin_edges = [0, 50, 100]
        self.mock_baseline_provider.get_baseline_output_length_distribution.return_value = (baseline_dist, bin_edges)

        # 3. Recent samples setup
        self.mock_metric_store.get_recent_values.return_value = [10, 20]

        # 4. Mock DriftEngine
        with (
            patch("coreason_sentinel.drift_monitor.DriftEngine.compute_distribution_from_samples") as mock_dist,
            patch("coreason_sentinel.drift_monitor.DriftEngine.compute_kl_divergence", return_value=0.1),
        ):
            mock_dist.return_value = [0.6, 0.4]

            self.monitor._check_output_drift(self.event)

            # Verify length recorded
            self.mock_metric_store.record_metric.assert_any_call(
                "test_agent", "output_length", 100.0, retention_window=3600
            )
            # Verify KL recorded
            self.mock_metric_store.record_metric.assert_any_call(
                "test_agent", "output_drift_kl", 0.1, retention_window=3600
            )

    def test_check_output_drift_token_count(self) -> None:
        """Test output length from token_count key."""
        self.event.metrics = {"token_count": 50}
        self.mock_baseline_provider.get_baseline_output_length_distribution.return_value = None

        self.monitor._check_output_drift(self.event)
        self.mock_metric_store.record_metric.assert_called_with(
            "test_agent", "output_length", 50.0, retention_window=3600
        )

    def test_check_output_drift_fallback_length(self) -> None:
        """Test fallback output length calculation."""
        self.event.metrics = {}  # No tokens
        self.event.output_text = "hello world"  # 2 words

        # Abort drift calculation early to just test length extraction
        self.mock_baseline_provider.get_baseline_output_length_distribution.return_value = None

        self.monitor._check_output_drift(self.event)
        self.mock_metric_store.record_metric.assert_called_with(
            "test_agent", "output_length", 2.0, retention_window=3600
        )

    def test_check_output_drift_empty_baseline(self) -> None:
        """Test empty baseline distribution."""
        self.event.metrics = {"token_count": 50}
        # Returns empty lists
        self.mock_baseline_provider.get_baseline_output_length_distribution.return_value = ([], [])

        self.monitor._check_output_drift(self.event)
        # Should record length but stop there
        self.assertEqual(self.mock_metric_store.record_metric.call_count, 1)

    def test_check_output_drift_no_baseline(self) -> None:
        """Test output drift skipped if no baseline."""
        self.mock_baseline_provider.get_baseline_output_length_distribution.return_value = None
        self.monitor._check_output_drift(self.event)
        # Should record length but NOT kl divergence
        calls = self.mock_metric_store.record_metric.call_args_list
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0][1], "output_length")

    def test_check_output_drift_no_recent_samples(self) -> None:
        """Test output drift skipped if no recent samples."""
        self.mock_baseline_provider.get_baseline_output_length_distribution.return_value = ([0.1], [0, 10])
        self.mock_metric_store.get_recent_values.return_value = []

        self.monitor._check_output_drift(self.event)
        # Should record length but NOT kl divergence
        # 1 call for output_length
        self.assertEqual(self.mock_metric_store.record_metric.call_count, 1)

    def test_check_relevance_drift_success(self) -> None:
        """Test relevance drift calculation."""
        self.event.metadata = {"query_embedding": [0.1, 0.0], "response_embedding": [0.0, 0.1]}

        with patch(
            "coreason_sentinel.drift_monitor.DriftEngine.compute_relevance_drift", return_value=1.0
        ) as mock_calc:
            self.monitor._check_relevance_drift(self.event)
            mock_calc.assert_called()
            self.mock_metric_store.record_metric.assert_called_with(
                "test_agent", "relevance_drift", 1.0, retention_window=3600
            )

    def test_check_relevance_drift_missing_data(self) -> None:
        """Test relevance drift skipped if embeddings missing."""
        self.event.metadata = {"query_embedding": [0.1]}
        self.monitor._check_relevance_drift(self.event)
        self.mock_metric_store.record_metric.assert_not_called()

    def test_check_relevance_drift_invalid_type(self) -> None:
        """Test relevance drift skipped if invalid type."""
        self.event.metadata = {"query_embedding": "invalid", "response_embedding": [0.1]}
        self.monitor._check_relevance_drift(self.event)
        self.mock_metric_store.record_metric.assert_not_called()

    def test_check_relevance_drift_exception(self) -> None:
        """Test exception handling."""
        self.event.metadata = {"query_embedding": [0.1], "response_embedding": [0.1]}
        with patch(
            "coreason_sentinel.drift_monitor.DriftEngine.compute_relevance_drift", side_effect=Exception("Math Error")
        ):
            self.monitor._check_relevance_drift(self.event)
            self.mock_metric_store.record_metric.assert_not_called()

    def test_check_output_drift_baseline_error(self) -> None:
        """Test handling of NotImplementedError from provider."""
        self.event.metrics = {"completion_tokens": 100}
        self.mock_baseline_provider.get_baseline_output_length_distribution.side_effect = NotImplementedError(
            "Not supported"
        )

        self.monitor._check_output_drift(self.event)
        # Should catch and log debug
        # Should record length but not drift
        self.mock_metric_store.record_metric.assert_called_with(
            "test_agent", "output_length", 100.0, retention_window=3600
        )

    def test_check_output_drift_generic_exception(self) -> None:
        """Test generic exception handling."""
        self.event.metrics = {"completion_tokens": 100}
        self.mock_baseline_provider.get_baseline_output_length_distribution.side_effect = Exception("Boom")

        self.monitor._check_output_drift(self.event)
        # Should catch and log error
        # Should record length
        self.mock_metric_store.record_metric.assert_called_with(
            "test_agent", "output_length", 100.0, retention_window=3600
        )
