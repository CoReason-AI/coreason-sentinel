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
    GradeResult,
    PhoenixClientProtocol,
    VeritasClientProtocol,
    VeritasEvent,
)
from coreason_sentinel.models import CircuitBreakerTrigger, SentinelConfig
from coreason_sentinel.spot_checker import SpotChecker


class TestTelemetryIngestor(unittest.TestCase):
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
        self.mock_sc.should_sample.return_value = False  # Default to no sampling
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
            metadata={"user_tier": "free"},
        )

    def test_ingest_from_veritas_since_success(self) -> None:
        """Test fetching and processing logs from Veritas."""
        since_time = datetime.now(timezone.utc)
        events = [self.event, self.event]
        self.mock_veritas.fetch_logs.return_value = events

        count = self.ingestor.ingest_from_veritas_since(since_time)

        self.mock_veritas.fetch_logs.assert_called_with("test-agent", since_time)
        self.assertEqual(count, 2)
        # Should have called process_event 2 times
        # Since process_event calls check_triggers, it should be called 2 times.
        self.assertEqual(self.mock_cb.check_triggers.call_count, 2)

    def test_ingest_from_veritas_since_empty(self) -> None:
        """Test fetching no logs from Veritas."""
        since_time = datetime.now(timezone.utc)
        self.mock_veritas.fetch_logs.return_value = []

        count = self.ingestor.ingest_from_veritas_since(since_time)

        self.assertEqual(count, 0)
        self.mock_cb.check_triggers.assert_not_called()

    def test_ingest_from_veritas_since_exception(self) -> None:
        """Test handling exception during fetch."""
        since_time = datetime.now(timezone.utc)
        self.mock_veritas.fetch_logs.side_effect = Exception("API Error")

        count = self.ingestor.ingest_from_veritas_since(since_time)

        self.assertEqual(count, 0)
        self.mock_cb.check_triggers.assert_not_called()

    def test_ingest_from_veritas_returns_none(self) -> None:
        """
        Edge Case: veritas client returns None instead of list.
        Should handle gracefully (return 0).
        """
        since_time = datetime.now(timezone.utc)
        self.mock_veritas.fetch_logs.return_value = None

        count = self.ingestor.ingest_from_veritas_since(since_time)
        self.assertEqual(count, 0)

    def test_ingest_from_veritas_partial_failure(self) -> None:
        """
        Edge Case: One event fails to process, loop continues.
        """
        since_time = datetime.now(timezone.utc)
        # 3 events
        events = [self.event, self.event, self.event]
        self.mock_veritas.fetch_logs.return_value = events

        # Mock process_event to fail on 2nd call
        # We need to mock self.ingestor.process_event?
        # But we are testing the ingestor logic.
        # We can patch the method on the instance.
        with patch.object(self.ingestor, "process_event", side_effect=[None, Exception("Processing Failed"), None]):
            count = self.ingestor.ingest_from_veritas_since(since_time)

            # Expecting 2 successful processings
            self.assertEqual(count, 2)

    def test_batch_trip_scenario(self) -> None:
        """
        Complex Scenario: 'Batch Trip'.
        A batch of events where early events cause a trip, but processing continues.
        """
        since_time = datetime.now(timezone.utc)
        events = [self.event, self.event, self.event]
        self.mock_veritas.fetch_logs.return_value = events

        count = self.ingestor.ingest_from_veritas_since(since_time)

        self.assertEqual(count, 3)
        self.assertEqual(self.mock_cb.check_triggers.call_count, 3)

        # Expected Metrics per event:
        # 1. latency
        # 2. tokens
        # Drift is NO LONGER called in process_event
        # So only 2 metrics per event = 6 total
        self.assertEqual(self.mock_cb.record_metric.call_count, 6)

    def test_process_event_records_metrics(self) -> None:
        """Test that event metrics are sent to CircuitBreaker."""
        self.mock_sc.should_sample.return_value = False  # Skip sampling
        self.ingestor.process_event(self.event)

        # Check record_metric calls
        self.mock_cb.record_metric.assert_any_call("latency", 100.0)
        self.mock_cb.record_metric.assert_any_call("token_count", 50.0)
        # Check trigger evaluation - called ONCE at the end
        self.mock_cb.check_triggers.assert_called_once()

    def test_process_event_samples_and_records_grade(self) -> None:
        """Test that sampled events are graded and scores recorded."""
        self.mock_sc.should_sample.return_value = True
        self.mock_sc.check_sample.return_value = GradeResult(faithfulness_score=0.9, safety_score=0.8, details={})

        self.ingestor.process_event(self.event)

        # Verify sampling
        self.mock_sc.check_sample.assert_called_once()

        # Verify grade metrics recorded
        self.mock_cb.record_metric.assert_any_call("faithfulness_score", 0.9)
        self.mock_cb.record_metric.assert_any_call("safety_score", 0.8)

        # Check triggers called ONCE at the end
        self.mock_cb.check_triggers.assert_called_once()

    def test_process_event_sample_failure(self) -> None:
        """Test handling when spot checker returns None."""
        self.mock_sc.should_sample.return_value = True
        self.mock_sc.check_sample.return_value = None  # Grading failed

        self.ingestor.process_event(self.event)

        # Verify no grade metrics recorded
        calls = [c[0][0] for c in self.mock_cb.record_metric.call_args_list]
        self.assertNotIn("faithfulness_score", calls)

    @patch("coreason_sentinel.ingestor.DriftEngine.detect_vector_drift")
    def test_process_event_does_not_call_drift(self, mock_detect_drift: MagicMock) -> None:
        """Test that process_event does NOT call drift detection anymore."""
        self.mock_sc.should_sample.return_value = False

        embedding = [0.1, 0.2, 0.3]
        self.event.metadata["embedding"] = embedding

        self.ingestor.process_event(self.event)

        self.mock_bp.get_baseline_vectors.assert_not_called()
        mock_detect_drift.assert_not_called()

        # Also ensure Output drift logic is skipped (checking metric)
        calls = [c[0][0] for c in self.mock_cb.record_metric.call_args_list]
        self.assertNotIn("vector_drift", calls)
        self.assertNotIn("output_length", calls)

    @patch("coreason_sentinel.ingestor.DriftEngine.detect_vector_drift")
    def test_process_drift_vector(self, mock_detect_drift: MagicMock) -> None:
        """Test that process_drift calls vector drift logic."""
        embedding = [0.1, 0.2, 0.3]
        baseline = [[0.1, 0.2, 0.3]]
        self.event.metadata["embedding"] = embedding
        self.mock_bp.get_baseline_vectors.return_value = baseline
        mock_detect_drift.return_value = 0.5

        self.ingestor.process_drift(self.event)

        self.mock_bp.get_baseline_vectors.assert_called_with("test-agent")
        mock_detect_drift.assert_called_with(baseline, [embedding])
        self.mock_cb.record_metric.assert_any_call("vector_drift", 0.5)

    def test_process_drift_output(self) -> None:
        """Test that process_drift calls output drift logic."""
        # Setup mocks for output drift
        self.mock_bp.get_baseline_output_length_distribution.return_value = ([0.5, 0.5], [0, 10, 20])
        self.mock_cb.get_recent_values.return_value = [10.0]

        self.ingestor.process_drift(self.event)

        # Verify output_length recorded
        # event metrics had "tokens": 50. If output_drift checks "completion_tokens" or "token_count" (50).
        self.mock_cb.record_metric.assert_any_call("output_length", 50.0)
        # Verify KL recorded (assuming DriftEngine works)
        calls = [c[0][0] for c in self.mock_cb.record_metric.call_args_list]
        self.assertIn("output_drift_kl", calls)

    def test_process_drift_relevance(self) -> None:
        """Test that process_drift calls relevance drift logic."""
        self.event.metadata["query_embedding"] = [1.0]
        self.event.metadata["response_embedding"] = [1.0]

        self.ingestor.process_drift(self.event)

        self.mock_cb.record_metric.assert_any_call("relevance_drift", 0.0)

    def test_drift_detection_exception(self) -> None:
        """Test error handling in drift detection (process_drift)."""
        self.event.metadata["embedding"] = [0.1]
        # Simulate exception
        self.mock_bp.get_baseline_vectors.side_effect = Exception("DB Error")

        # Should not raise exception
        self.ingestor.process_drift(self.event)
        # And should log error (implicitly covered by execution flow)

    def test_drift_detection_no_embedding(self) -> None:
        """Test drift detection skipped when no embedding."""
        self.event.metadata = {}  # No embedding
        self.ingestor.process_drift(self.event)
        self.mock_bp.get_baseline_vectors.assert_not_called()

    def test_drift_detection_no_baseline(self) -> None:
        """Test drift detection skipped when no baseline."""
        self.event.metadata["embedding"] = [0.1]
        self.mock_bp.get_baseline_vectors.return_value = []
        self.ingestor.process_drift(self.event)
        # drift detection not called
        calls = [c[0][0] for c in self.mock_cb.record_metric.call_args_list]
        self.assertNotIn("vector_drift", calls)

    def test_drift_dimension_mismatch(self) -> None:
        """Test behavior when embedding dimension mismatches baseline."""
        embedding = [0.1, 0.2]  # 2D
        baseline = [[0.1, 0.2, 0.3]]  # 3D
        self.event.metadata["embedding"] = embedding
        self.mock_bp.get_baseline_vectors.return_value = baseline

        self.ingestor.process_drift(self.event)

        # Should not raise.
        calls = [c[0][0] for c in self.mock_cb.record_metric.call_args_list]
        self.assertNotIn("vector_drift", calls)

    def test_drift_storm_scenario(self) -> None:
        """
        Complex Scenario: 'Drift Storm'.
        High vector drift causes Circuit Breaker to trip.
        Validates full flow using process_drift.
        """
        from redis import Redis

        from coreason_sentinel.interfaces import NotificationServiceProtocol

        mock_redis = MagicMock(spec=Redis)
        mock_notification_service = MagicMock(spec=NotificationServiceProtocol)

        # Trigger: Vector Drift > 0.5 (AVG) in 60s
        trigger = CircuitBreakerTrigger(
            metric="vector_drift", threshold=0.5, window_seconds=60, operator=">", aggregation_method="AVG"
        )
        config = SentinelConfig(
            agent_id="drift-bot",
            owner_email="test@example.com",
            phoenix_endpoint="http://localhost:6006",
            sampling_rate=0.0,
            triggers=[trigger],
        )

        real_cb = CircuitBreaker(mock_redis, config, mock_notification_service)
        mock_sc = MagicMock(spec=SpotChecker)
        mock_sc.should_sample.return_value = False
        mock_bp = MagicMock(spec=BaselineProviderProtocol)
        mock_veritas = MagicMock(spec=VeritasClientProtocol)

        # Baseline: Unit vector on X axis
        mock_bp.get_baseline_vectors.return_value = [[1.0, 0.0]]

        ingestor = TelemetryIngestor(config, real_cb, mock_sc, mock_bp, mock_veritas)

        # Setup Mock Redis
        redis_store: Dict[str, List[Tuple[float, bytes]]] = {}

        def mock_zadd(key: str, mapping: Dict[Union[str, bytes], float]) -> None:
            if key not in redis_store:
                redis_store[key] = []
            for m, s in mapping.items():
                redis_store[key].append((s, m if isinstance(m, bytes) else m.encode("utf-8")))

        def mock_zrange(key: str, min_s: Union[float, str], max_s: Union[float, str]) -> List[bytes]:
            if key not in redis_store:
                return []
            return [m for s, m in redis_store[key]]

        def mock_zremrange(key: str, min_s: Union[float, str], max_s: Union[float, str]) -> None:
            pass

        def mock_get(key: str) -> bytes:
            return b"CLOSED"

        mock_redis.zadd.side_effect = mock_zadd
        mock_redis.zrangebyscore.side_effect = mock_zrange
        mock_redis.zremrangebyscore.side_effect = mock_zremrange
        mock_redis.get.side_effect = mock_get

        # Feed events with Orthogonal embeddings (Unit vector on Y axis)
        event = VeritasEvent(
            event_id="evt-drift",
            timestamp=datetime.now(timezone.utc),
            agent_id="drift-bot",
            session_id="sess-1",
            input_text="foo",
            output_text="bar",
            metrics={},
            metadata={"embedding": [0.0, 1.0]},
        )

        # Mock getset to return CLOSED so the breaker actually trips
        mock_redis.getset.return_value = b"CLOSED"

        for _ in range(3):
            # MUST CALL process_drift
            ingestor.process_drift(event)

        # Drift scores recorded: 1.0, 1.0, 1.0
        # Avg: 1.0.
        # Threshold: 0.5.
        # 1.0 > 0.5 -> Trip.

        mock_redis.getset.assert_any_call("sentinel:breaker:drift-bot:state", "OPEN")

    def test_hallucination_storm_scenario(self) -> None:
        """
        Story B: The 'Hallucination Storm'.
        Simulates a sequence where low faithfulness scores trigger the Circuit Breaker.
        NOTE: This test uses REAL Logic classes with MOCKED Redis/Grader.
        """
        # 1. Setup Logic with Mocks
        from redis import Redis

        from coreason_sentinel.interfaces import NotificationServiceProtocol

        mock_redis = MagicMock(spec=Redis)
        mock_notification_service = MagicMock(spec=NotificationServiceProtocol)

        # Trigger: Faithfulness < 0.5 (AVG).
        trigger = CircuitBreakerTrigger(
            metric="faithfulness_score", threshold=0.5, window_seconds=60, operator="<", aggregation_method="AVG"
        )
        config = SentinelConfig(
            agent_id="storm-agent",
            owner_email="test@example.com",
            phoenix_endpoint="http://localhost:6006",
            sampling_rate=1.0,
            triggers=[trigger],
        )

        real_cb = CircuitBreaker(mock_redis, config, mock_notification_service)

        from coreason_sentinel.interfaces import AssayGraderProtocol

        mock_grader = MagicMock(spec=AssayGraderProtocol)
        mock_phoenix = MagicMock(spec=PhoenixClientProtocol)
        mock_bp = MagicMock(spec=BaselineProviderProtocol)
        mock_veritas = MagicMock(spec=VeritasClientProtocol)
        real_sc = SpotChecker(config, mock_grader, mock_phoenix)

        ingestor = TelemetryIngestor(config, real_cb, real_sc, mock_bp, mock_veritas)

        # 2. Simulate "Bad" Traffic (Mock Redis Storage)
        redis_store: Dict[str, List[Tuple[float, bytes]]] = {}

        def mock_zadd(key: str, mapping: Dict[Union[str, bytes], float]) -> None:
            if key not in redis_store:
                redis_store[key] = []
            for m, s in mapping.items():
                redis_store[key].append((s, m if isinstance(m, bytes) else m.encode("utf-8")))

        def mock_zrange(key: str, min_s: Union[float, str], max_s: Union[float, str]) -> List[bytes]:
            if key not in redis_store:
                return []
            return [m for s, m in redis_store[key]]

        def mock_zremrange(key: str, min_s: Union[float, str], max_s: Union[float, str]) -> None:
            pass

        def mock_get(key: str) -> bytes:
            return b"CLOSED"

        mock_redis.zadd.side_effect = mock_zadd
        mock_redis.zrangebyscore.side_effect = mock_zrange
        mock_redis.zremrangebyscore.side_effect = mock_zremrange
        mock_redis.get.side_effect = mock_get

        # 3. Process Events
        mock_grader.grade_conversation.return_value = GradeResult(faithfulness_score=0.1, safety_score=1.0, details={})
        mock_redis.getset.return_value = b"CLOSED"

        for _i in range(5):
            ingestor.process_event(self.event)

        mock_redis.getset.assert_any_call("sentinel:breaker:storm-agent:state", "OPEN")

    def test_output_drift_missing_methods(self) -> None:
        """Test graceful handling when provider is missing distribution methods (process_drift)."""
        self.mock_bp.get_baseline_output_length_distribution.side_effect = AttributeError("Old Provider")

        # Use process_drift
        self.ingestor.process_drift(self.event)

        calls = [c[0][0] for c in self.mock_cb.record_metric.call_args_list]
        self.assertNotIn("output_drift_kl", calls)

    def test_output_drift_empty_baseline(self) -> None:
        """Test graceful handling of empty baseline (process_drift)."""
        self.mock_bp.get_baseline_output_length_distribution.return_value = ([], [])
        self.ingestor.process_drift(self.event)
        calls = [c[0][0] for c in self.mock_cb.record_metric.call_args_list]
        self.assertNotIn("output_drift_kl", calls)

    def test_output_drift_no_recent_samples(self) -> None:
        """Test graceful handling when no recent samples in Redis (process_drift)."""
        self.mock_bp.get_baseline_output_length_distribution.return_value = ([0.5, 0.5], [0, 10, 20])
        self.mock_cb.get_recent_values.return_value = []

        self.ingestor.process_drift(self.event)

        calls = [c[0][0] for c in self.mock_cb.record_metric.call_args_list]
        self.assertNotIn("output_drift_kl", calls)

    @patch("coreason_sentinel.ingestor.DriftEngine.compute_kl_divergence")
    def test_output_drift_kl_error(self, mock_kl: MagicMock) -> None:
        """Test handling of KL computation error."""
        self.mock_bp.get_baseline_output_length_distribution.return_value = ([0.5, 0.5], [0, 10, 20])
        self.mock_cb.get_recent_values.return_value = [5.0, 15.0]

        mock_kl.side_effect = ValueError("KL Error")

        self.ingestor.process_drift(self.event)

        # Should log warning but continue
        # Triggers checked
        self.mock_cb.check_triggers.assert_called_once()

    @patch("coreason_sentinel.ingestor.logger")
    def test_output_drift_general_exception(self, mock_logger: MagicMock) -> None:
        """Test catching unexpected exception in drift logic."""
        with patch.object(self.ingestor, "_process_output_drift", side_effect=Exception("Boom")) as mock_method:
            self.ingestor.process_drift(self.event)
            mock_method.assert_called_once()
            mock_logger.error.assert_called_with("Failed to process output drift detection: Boom")

    def test_extract_refusal_metric(self) -> None:
        """Test that refusal flag in metadata records a metric."""
        self.event.metadata["is_refusal"] = True
        self.mock_sc.should_sample.return_value = False

        self.ingestor.process_event(self.event)

        self.mock_cb.record_metric.assert_any_call("refusal_count", 1.0)

    def test_extract_sentiment_metric(self) -> None:
        """Test that configured regex patterns trigger sentiment metric."""
        self.event.input_text = "Please STOP this now."
        self.mock_sc.should_sample.return_value = False

        self.ingestor.process_event(self.event)

        self.mock_cb.record_metric.assert_any_call("sentiment_frustration_count", 1.0)

    def test_extract_sentiment_metric_custom_config(self) -> None:
        """Test that custom regex patterns work."""
        self.config.sentiment_regex_patterns = ["^HATE"]
        self.event.input_text = "HATE this result"
        self.mock_sc.should_sample.return_value = False

        self.ingestor.process_event(self.event)

        self.mock_cb.record_metric.assert_any_call("sentiment_frustration_count", 1.0)

    def test_extract_sentiment_no_match(self) -> None:
        """Test that no metric is recorded if no match."""
        self.event.input_text = "I love this result"
        self.mock_sc.should_sample.return_value = False

        self.ingestor.process_event(self.event)

        calls = [c[0][0] for c in self.mock_cb.record_metric.call_args_list]
        self.assertNotIn("sentiment_frustration_count", calls)

    @patch("coreason_sentinel.ingestor.logger")
    def test_extract_sentiment_invalid_regex(self, mock_logger: MagicMock) -> None:
        """Test that invalid regex patterns are handled gracefully."""
        self.config.sentiment_regex_patterns = ["[", "STOP"]
        self.event.input_text = "Please STOP"
        self.mock_sc.should_sample.return_value = False

        self.ingestor.process_event(self.event)

        found_call = False
        for call_args in mock_logger.error.call_args_list:
            arg = call_args[0][0]
            if "Invalid regex pattern '[' in configuration:" in arg:
                found_call = True
                break
        self.assertTrue(found_call, "Did not find expected error log for invalid regex")

        self.mock_cb.record_metric.assert_any_call("sentiment_frustration_count", 1.0)

    def test_extract_sentiment_empty_input(self) -> None:
        """Test behavior with empty input text."""
        self.event.input_text = ""
        self.mock_sc.should_sample.return_value = False
        self.ingestor.process_event(self.event)
        calls = [c[0][0] for c in self.mock_cb.record_metric.call_args_list]
        self.assertNotIn("sentiment_frustration_count", calls)

    def test_extract_refusal_false_metadata(self) -> None:
        """Test that is_refusal=False records nothing."""
        self.event.metadata["is_refusal"] = False
        self.mock_sc.should_sample.return_value = False
        self.ingestor.process_event(self.event)
        calls = [c[0][0] for c in self.mock_cb.record_metric.call_args_list]
        self.assertNotIn("refusal_count", calls)

    def test_extract_mixed_signals(self) -> None:
        """Test 'Mixed Signals': Event is BOTH a refusal AND has negative sentiment."""
        self.event.metadata["is_refusal"] = True
        self.event.input_text = "STOP replying badly"
        self.mock_sc.should_sample.return_value = False

        self.ingestor.process_event(self.event)

        self.mock_cb.record_metric.assert_any_call("refusal_count", 1.0)
        self.mock_cb.record_metric.assert_any_call("sentiment_frustration_count", 1.0)

    def test_extract_overlapping_regex(self) -> None:
        """Test that multiple matching regexes only record ONE metric per event."""
        self.config.sentiment_regex_patterns = ["STOP", "BAD"]
        self.event.input_text = "STOP being BAD"
        self.mock_sc.should_sample.return_value = False

        self.ingestor.process_event(self.event)

        sentiment_calls = [
            c for c in self.mock_cb.record_metric.call_args_list if c[0][0] == "sentiment_frustration_count"
        ]
        self.assertEqual(len(sentiment_calls), 1)

    def test_complex_meltdown_scenario(self) -> None:
        """
        Scenario: 'The Meltdown'.
        Simulates a stream of events with high refusal and frustration rates.
        """
        self.config.sentiment_regex_patterns = ["FAIL"]
        self.mock_sc.should_sample.return_value = False

        events = []
        for i in range(10):
            evt = VeritasEvent(
                event_id=f"evt-{i}",
                timestamp=datetime.now(timezone.utc),
                agent_id="test-agent",
                session_id="sess-1",
                input_text="System FAIL" if i < 5 else "Normal request",
                output_text="...",
                metrics={},
                metadata={"is_refusal": i >= 3},
            )
            events.append(evt)

        for evt in events:
            self.ingestor.process_event(evt)

        sentiment_calls = [
            c for c in self.mock_cb.record_metric.call_args_list if c[0][0] == "sentiment_frustration_count"
        ]
        self.assertEqual(len(sentiment_calls), 5)

        refusal_calls = [c for c in self.mock_cb.record_metric.call_args_list if c[0][0] == "refusal_count"]
        self.assertEqual(len(refusal_calls), 7)
