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
from coreason_sentinel.interfaces import BaselineProviderProtocol, GradeResult, VeritasEvent
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
        self.mock_bp = MagicMock(spec=BaselineProviderProtocol)
        self.ingestor = TelemetryIngestor(self.config, self.mock_cb, self.mock_sc, self.mock_bp)

        self.event = VeritasEvent(
            event_id="evt-1",
            timestamp=datetime.now(timezone.utc),
            agent_id="test-agent",
            session_id="sess-1",
            input_text="hello",
            output_text="world",
            metrics={"latency": 100, "tokens": 50},
            metadata={"user_tier": "free"},
        )

    def test_process_event_records_metrics(self) -> None:
        """Test that event metrics are sent to CircuitBreaker."""
        self.mock_sc.should_sample.return_value = False  # Skip sampling
        self.ingestor.process_event(self.event)

        # Check record_metric calls
        self.mock_cb.record_metric.assert_any_call("latency", 100.0)
        self.mock_cb.record_metric.assert_any_call("tokens", 50.0)
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
        # We need to ensure record_metric was NOT called with score keys
        calls = [c[0][0] for c in self.mock_cb.record_metric.call_args_list]
        self.assertNotIn("faithfulness_score", calls)

    @patch("coreason_sentinel.ingestor.DriftEngine.detect_vector_drift")
    def test_drift_detection(self, mock_detect_drift: MagicMock) -> None:
        """Test drift detection flow when embedding is present."""
        self.mock_sc.should_sample.return_value = False  # Ensure no sampling triggers

        embedding = [0.1, 0.2, 0.3]
        baseline = [[0.1, 0.2, 0.3]]
        self.event.metadata["embedding"] = embedding
        self.mock_bp.get_baseline_vectors.return_value = baseline
        mock_detect_drift.return_value = 0.5

        self.ingestor.process_event(self.event)

        self.mock_bp.get_baseline_vectors.assert_called_with("test-agent")
        mock_detect_drift.assert_called_with(baseline, [embedding])
        self.mock_cb.record_metric.assert_any_call("vector_drift", 0.5)
        # Check triggers called ONCE at the end
        self.mock_cb.check_triggers.assert_called_once()

    def test_drift_detection_exception(self) -> None:
        """Test error handling in drift detection."""
        self.mock_sc.should_sample.return_value = False
        self.event.metadata["embedding"] = [0.1]
        # Simulate exception
        self.mock_bp.get_baseline_vectors.side_effect = Exception("DB Error")

        # Should not raise exception
        self.ingestor.process_event(self.event)
        # And should log error (implicitly covered by execution flow)

    def test_drift_detection_no_embedding(self) -> None:
        """Test drift detection skipped when no embedding."""
        self.event.metadata = {}  # No embedding
        self.ingestor.process_event(self.event)
        self.mock_bp.get_baseline_vectors.assert_not_called()

    def test_drift_detection_no_baseline(self) -> None:
        """Test drift detection skipped when no baseline."""
        self.event.metadata["embedding"] = [0.1]
        self.mock_bp.get_baseline_vectors.return_value = []
        self.ingestor.process_event(self.event)
        # drift detection not called
        # checking record_metric not called with vector_drift
        calls = [c[0][0] for c in self.mock_cb.record_metric.call_args_list]
        self.assertNotIn("vector_drift", calls)

    def test_drift_dimension_mismatch(self) -> None:
        """
        Test behavior when embedding dimension mismatches baseline.
        DriftEngine would raise ValueError, which should be caught.
        """
        self.mock_sc.should_sample.return_value = False
        embedding = [0.1, 0.2]  # 2D
        baseline = [[0.1, 0.2, 0.3]]  # 3D
        self.event.metadata["embedding"] = embedding
        self.mock_bp.get_baseline_vectors.return_value = baseline

        # Use the REAL DriftEngine (not patched) to verify it raises ValueError and we catch it.
        # But we need to ensure the DriftEngine is importable and works.
        # Since we patched it in other tests, here we rely on the implementation in src.
        # However, the class `DriftEngine` is used in `ingestor.py`.
        # To test the `try...except` block in `ingestor.py` specifically catching ValueError from DriftEngine:

        # We can just let the real DriftEngine run.
        self.ingestor.process_event(self.event)

        # Should not raise.
        # Should NOT have recorded vector_drift.
        calls = [c[0][0] for c in self.mock_cb.record_metric.call_args_list]
        self.assertNotIn("vector_drift", calls)

    def test_drift_storm_scenario(self) -> None:
        """
        Complex Scenario: 'Drift Storm'.
        High vector drift causes Circuit Breaker to trip.
        Validates full flow: Event -> Embedding -> Drift Calc -> Metric -> Trigger.
        """
        from redis import Redis

        mock_redis = MagicMock(spec=Redis)

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

        real_cb = CircuitBreaker(mock_redis, config)
        mock_sc = MagicMock(spec=SpotChecker)
        mock_sc.should_sample.return_value = False
        mock_bp = MagicMock(spec=BaselineProviderProtocol)

        # Baseline: Unit vector on X axis
        mock_bp.get_baseline_vectors.return_value = [[1.0, 0.0]]

        ingestor = TelemetryIngestor(config, real_cb, mock_sc, mock_bp)

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
        # Cosine Similarity (1,0) . (0,1) = 0.
        # Drift = 1 - 0 = 1.0.
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

        for _ in range(3):
            ingestor.process_event(event)

        # Drift scores recorded: 1.0, 1.0, 1.0
        # Avg: 1.0.
        # Threshold: 0.5.
        # 1.0 > 0.5 -> Trip.

        mock_redis.set.assert_called_with("sentinel:breaker:drift-bot:state", "OPEN")

    def test_hallucination_storm_scenario(self) -> None:
        """
        Story B: The 'Hallucination Storm'.
        Simulates a sequence where low faithfulness scores trigger the Circuit Breaker.
        NOTE: This test uses REAL Logic classes with MOCKED Redis/Grader to verify the integration logic.
        """
        # 1. Setup Logic with Mocks
        from redis import Redis

        mock_redis = MagicMock(spec=Redis)

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

        real_cb = CircuitBreaker(mock_redis, config)

        from coreason_sentinel.interfaces import AssayGraderProtocol

        mock_grader = MagicMock(spec=AssayGraderProtocol)
        mock_bp = MagicMock(spec=BaselineProviderProtocol)
        real_sc = SpotChecker(config, mock_grader)

        ingestor = TelemetryIngestor(config, real_cb, real_sc, mock_bp)

        # 2. Simulate "Bad" Traffic (Mock Redis Storage)

        # State var to hold our "Redis" data: {key: [(score, member_bytes)]}
        redis_store: Dict[str, List[Tuple[float, bytes]]] = {}

        def mock_zadd(key: str, mapping: Dict[Union[str, bytes], float]) -> None:
            # mapping is {member: score}
            if key not in redis_store:
                redis_store[key] = []
            for m, s in mapping.items():
                redis_store[key].append((s, m if isinstance(m, bytes) else m.encode("utf-8")))

        def mock_zrange(key: str, min_s: Union[float, str], max_s: Union[float, str]) -> List[bytes]:
            # Return list of members for THIS key
            # Also mock pruning? For this test we don't prune, just append.
            if key not in redis_store:
                return []
            return [m for s, m in redis_store[key]]

        def mock_zremrange(key: str, min_s: Union[float, str], max_s: Union[float, str]) -> None:
            pass  # No-op for test

        def mock_get(key: str) -> bytes:
            return b"CLOSED"

        mock_redis.zadd.side_effect = mock_zadd
        mock_redis.zrangebyscore.side_effect = mock_zrange
        mock_redis.zremrangebyscore.side_effect = mock_zremrange
        mock_redis.get.side_effect = mock_get

        # 3. Process Events
        # Grader returns 0.1 (Low faithfulness).
        mock_grader.grade_conversation.return_value = GradeResult(faithfulness_score=0.1, safety_score=1.0, details={})

        # Send 5 events.
        # faithfulness_score accumulator: 0.1, 0.1, ...
        # AVG will be 0.1.
        # Threshold: 0.5. Operator: <.
        # 0.1 < 0.5 is TRUE. Violation!
        for _i in range(5):
            ingestor.process_event(self.event)

        # Verify set_state("OPEN") was called.
        mock_redis.set.assert_called_with("sentinel:breaker:storm-agent:state", "OPEN")

    def test_output_drift_missing_methods(self) -> None:
        """Test graceful handling when provider is missing distribution methods."""
        # Provider raises AttributeError
        self.mock_bp.get_baseline_output_length_distribution.side_effect = AttributeError("Old Provider")
        self.mock_sc.should_sample.return_value = False

        # Should not crash
        self.ingestor.process_event(self.event)

        calls = [c[0][0] for c in self.mock_cb.record_metric.call_args_list]
        self.assertNotIn("output_drift_kl", calls)

    def test_output_drift_empty_baseline(self) -> None:
        """Test graceful handling of empty baseline."""
        self.mock_bp.get_baseline_output_length_distribution.return_value = ([], [])
        self.mock_sc.should_sample.return_value = False
        self.ingestor.process_event(self.event)
        calls = [c[0][0] for c in self.mock_cb.record_metric.call_args_list]
        self.assertNotIn("output_drift_kl", calls)

    def test_output_drift_no_recent_samples(self) -> None:
        """Test graceful handling when no recent samples in Redis."""
        self.mock_bp.get_baseline_output_length_distribution.return_value = ([0.5, 0.5], [0, 10, 20])
        self.mock_sc.should_sample.return_value = False
        # Redis returns empty list for get_recent_values
        self.mock_cb.get_recent_values.return_value = []

        self.ingestor.process_event(self.event)

        calls = [c[0][0] for c in self.mock_cb.record_metric.call_args_list]
        self.assertNotIn("output_drift_kl", calls)

    @patch("coreason_sentinel.ingestor.DriftEngine.compute_kl_divergence")
    def test_output_drift_kl_error(self, mock_kl: MagicMock) -> None:
        """Test handling of KL computation error."""
        self.mock_bp.get_baseline_output_length_distribution.return_value = ([0.5, 0.5], [0, 10, 20])
        self.mock_sc.should_sample.return_value = False
        self.mock_cb.get_recent_values.return_value = [5.0, 15.0]

        mock_kl.side_effect = ValueError("KL Error")

        self.ingestor.process_event(self.event)

        # Should log warning but continue
        self.mock_cb.check_triggers.assert_called_once()

    @patch("coreason_sentinel.ingestor.logger")
    def test_output_drift_general_exception(self, mock_logger: MagicMock) -> None:
        """Test catching unexpected exception in drift logic."""
        # Patch the method on the instance directly to ensure it raises
        with patch.object(self.ingestor, "_process_output_drift", side_effect=Exception("Boom")) as mock_method:
            self.mock_sc.should_sample.return_value = False

            self.ingestor.process_event(self.event)

            # Verify the mock was called
            mock_method.assert_called_once()
            # Verify logger error was called
            mock_logger.error.assert_called_with("Failed to process output drift detection: Boom")

    def test_extract_refusal_metric(self) -> None:
        """Test that refusal flag in metadata records a metric."""
        self.event.metadata["is_refusal"] = True
        self.mock_sc.should_sample.return_value = False

        self.ingestor.process_event(self.event)

        self.mock_cb.record_metric.assert_any_call("refusal_count", 1.0)

    def test_extract_sentiment_metric(self) -> None:
        """Test that configured regex patterns trigger sentiment metric."""
        # Default config has "STOP"
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

        # Ensure sentiment metric was NOT recorded
        calls = [c[0][0] for c in self.mock_cb.record_metric.call_args_list]
        self.assertNotIn("sentiment_frustration_count", calls)

    @patch("coreason_sentinel.ingestor.logger")
    def test_extract_sentiment_invalid_regex(self, mock_logger: MagicMock) -> None:
        """Test that invalid regex patterns are handled gracefully."""
        self.config.sentiment_regex_patterns = ["[", "STOP"]  # First invalid, second valid
        self.event.input_text = "Please STOP"
        self.mock_sc.should_sample.return_value = False

        self.ingestor.process_event(self.event)

        # Verify invalid regex logged error
        # The error message varies by Python version ("unterminated character set" vs "bad character range")
        # We check that the start of the message is correct.
        found_call = False
        for call_args in mock_logger.error.call_args_list:
            arg = call_args[0][0]
            if "Invalid regex pattern '[' in configuration:" in arg:
                found_call = True
                break
        self.assertTrue(found_call, "Did not find expected error log for invalid regex")

        # Verify valid pattern still worked
        self.mock_cb.record_metric.assert_any_call("sentiment_frustration_count", 1.0)

    def test_extract_sentiment_empty_input(self) -> None:
        """Test behavior with empty input text."""
        self.event.input_text = ""
        self.mock_sc.should_sample.return_value = False
        self.ingestor.process_event(self.event)
        # Should not record metric
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
        # Both "STOP" and "BAD" match
        self.config.sentiment_regex_patterns = ["STOP", "BAD"]
        self.event.input_text = "STOP being BAD"
        self.mock_sc.should_sample.return_value = False

        self.ingestor.process_event(self.event)

        # Verify only ONE sentiment call (plus latency/tokens)
        sentiment_calls = [
            c for c in self.mock_cb.record_metric.call_args_list if c[0][0] == "sentiment_frustration_count"
        ]
        self.assertEqual(len(sentiment_calls), 1)

    def test_complex_meltdown_scenario(self) -> None:
        """
        Scenario: 'The Meltdown'.
        Simulates a stream of events with high refusal and frustration rates.
        Verifies that metrics are consistently recorded.
        """
        self.config.sentiment_regex_patterns = ["FAIL"]
        self.mock_sc.should_sample.return_value = False

        # 10 events: 5 refusals, 5 frustration, 3 overlapping
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
                metadata={"is_refusal": i >= 3},  # Refusals for i=3..9 (7 total actually)
            )
            events.append(evt)

        for evt in events:
            self.ingestor.process_event(evt)

        # Verification
        # Sentiment: i=0,1,2,3,4 -> 5 calls
        sentiment_calls = [
            c for c in self.mock_cb.record_metric.call_args_list if c[0][0] == "sentiment_frustration_count"
        ]
        self.assertEqual(len(sentiment_calls), 5)

        # Refusals: i=3..9 -> 7 calls
        refusal_calls = [c for c in self.mock_cb.record_metric.call_args_list if c[0][0] == "refusal_count"]
        self.assertEqual(len(refusal_calls), 7)
