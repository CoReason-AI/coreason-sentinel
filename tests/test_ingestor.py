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
from unittest.mock import MagicMock

from coreason_sentinel.circuit_breaker import CircuitBreaker
from coreason_sentinel.ingestor import TelemetryIngestor
from coreason_sentinel.interfaces import GradeResult, VeritasEvent
from coreason_sentinel.models import SentinelConfig, Trigger
from coreason_sentinel.spot_checker import SpotChecker


class TestTelemetryIngestor(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SentinelConfig(agent_id="test-agent", sample_rate=1.0, circuit_breaker_triggers=[])
        self.mock_cb = MagicMock(spec=CircuitBreaker)
        self.mock_sc = MagicMock(spec=SpotChecker)
        self.ingestor = TelemetryIngestor(self.config, self.mock_cb, self.mock_sc)

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
        real_sc = SpotChecker(config, mock_grader)

        ingestor = TelemetryIngestor(config, real_cb, real_sc)

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
