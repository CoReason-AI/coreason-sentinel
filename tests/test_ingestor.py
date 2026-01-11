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
from coreason_sentinel.models import SentinelConfig, Trigger
from coreason_sentinel.spot_checker import SpotChecker


class TestTelemetryIngestor(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SentinelConfig(agent_id="test-agent", sample_rate=1.0, circuit_breaker_triggers=[])
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
        # Check trigger evaluation
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

        # Check triggers called (once after metrics, once after grading)
        self.assertEqual(self.mock_cb.check_triggers.call_count, 2)

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
        self.assertEqual(self.mock_cb.check_triggers.call_count, 2)  # once metrics, once drift

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
        trigger = Trigger(
            metric_name="vector_drift", threshold=0.5, window_seconds=60, operator=">", aggregation_method="AVG"
        )
        config = SentinelConfig(agent_id="drift-bot", sample_rate=0.0, circuit_breaker_triggers=[trigger])

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
        trigger = Trigger(
            metric_name="faithfulness_score", threshold=0.5, window_seconds=60, operator="<", aggregation_method="AVG"
        )
        config = SentinelConfig(agent_id="storm-agent", sample_rate=1.0, circuit_breaker_triggers=[trigger])

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
