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

from redis import Redis

from coreason_sentinel.circuit_breaker import CircuitBreaker
from coreason_sentinel.ingestor import TelemetryIngestor
from coreason_sentinel.interfaces import (
    AssayGraderProtocol,
    BaselineProviderProtocol,
    GradeResult,
    NotificationServiceProtocol,
    PhoenixClientProtocol,
    VeritasClientProtocol,
    VeritasEvent,
)
from coreason_sentinel.models import CircuitBreakerTrigger, SentinelConfig
from coreason_sentinel.spot_checker import SpotChecker


class TestRetrievalPrecisionScenarios(unittest.TestCase):
    def setUp(self) -> None:
        # Mock Redis for Circuit Breaker
        self.mock_redis = MagicMock(spec=Redis)
        self.redis_store: Dict[str, List[Tuple[float, bytes]]] = {}
        self.redis_kv: Dict[str, bytes] = {}

        def mock_zadd(key: str, mapping: Dict[Union[str, bytes], float]) -> None:
            if key not in self.redis_store:
                self.redis_store[key] = []
            for m, s in mapping.items():
                self.redis_store[key].append((s, m if isinstance(m, bytes) else m.encode("utf-8")))

        def mock_zrange(key: str, min_s: Union[float, str], max_s: Union[float, str]) -> List[bytes]:
            if key not in self.redis_store:
                return []
            return [m for s, m in self.redis_store[key]]

        def mock_zrangebyscore(key: str, min_s: Union[float, str], max_s: Union[float, str]) -> List[bytes]:
            # Simple mock: ignore timestamps for now and return all,
            # or ideally implement simple filtering if needed.
            # For these tests, we just assume all events are relevant.
            if key not in self.redis_store:
                return []
            return [m for s, m in self.redis_store[key]]

        def mock_zremrange(key: str, min_s: Union[float, str], max_s: Union[float, str]) -> None:
            pass

        def mock_get(key: str) -> bytes | None:
            val = self.redis_kv.get(key)
            if isinstance(val, str):
                return val.encode("utf-8")
            return val

        def mock_set(key: str, value: bytes | str) -> None:
            if isinstance(value, str):
                value = value.encode("utf-8")
            self.redis_kv[key] = value

        def mock_getset(key: str, value: bytes | str) -> bytes | None:
            old = self.redis_kv.get(key)
            if isinstance(old, str):
                old = old.encode("utf-8")

            if isinstance(value, str):
                value = value.encode("utf-8")
            self.redis_kv[key] = value
            return old

        def mock_exists(key: str) -> int:
            return 1 if key in self.redis_kv else 0

        def mock_setex(key: str, time: int, value: bytes | str) -> None:
            if isinstance(value, str):
                value = value.encode("utf-8")
            self.redis_kv[key] = value

        self.mock_redis.zadd.side_effect = mock_zadd
        self.mock_redis.zrangebyscore.side_effect = mock_zrangebyscore
        self.mock_redis.zremrangebyscore.side_effect = mock_zremrange
        self.mock_redis.get.side_effect = mock_get
        self.mock_redis.set.side_effect = mock_set
        self.mock_redis.getset.side_effect = mock_getset
        self.mock_redis.exists.side_effect = mock_exists
        self.mock_redis.setex.side_effect = mock_setex

        self.mock_notification_service = MagicMock(spec=NotificationServiceProtocol)

        # Config with a trigger on retrieval_precision_score
        # Trigger if AVG score < 0.5 in last 60 seconds
        self.trigger = CircuitBreakerTrigger(
            metric="retrieval_precision_score",
            threshold=0.5,
            window_seconds=60,
            operator="<",
            aggregation_method="AVG",
        )
        self.config = SentinelConfig(
            agent_id="rag-agent",
            owner_email="test@example.com",
            phoenix_endpoint="http://localhost:6006",
            sampling_rate=1.0,  # 100% sampling
            triggers=[self.trigger],
        )

        self.circuit_breaker = CircuitBreaker(self.mock_redis, self.config, self.mock_notification_service)

        self.mock_grader = MagicMock(spec=AssayGraderProtocol)
        self.mock_phoenix = MagicMock(spec=PhoenixClientProtocol)
        self.spot_checker = SpotChecker(self.config, self.mock_grader, self.mock_phoenix)

        self.mock_bp = MagicMock(spec=BaselineProviderProtocol)
        self.mock_veritas = MagicMock(spec=VeritasClientProtocol)

        self.ingestor = TelemetryIngestor(
            self.config, self.circuit_breaker, self.spot_checker, self.mock_bp, self.mock_veritas
        )

        self.base_event = VeritasEvent(
            event_id="evt-1",
            timestamp=datetime.now(timezone.utc),
            agent_id="rag-agent",
            session_id="sess-1",
            input_text="query",
            output_text="answer",
            metrics={"latency": 0.1},
            metadata={"trace_id": "t1", "span_id": "s1"},
        )

    def test_rag_failure_storm(self) -> None:
        """
        Complex Scenario: 'RAG Failure Storm'.
        The grader returns low retrieval precision scores consistently.
        The Circuit Breaker tracks this metric and trips to OPEN when the average drops below threshold.
        """
        # 1. Setup Grader to return low precision (0.2)
        # Note: faithfulness is high (model answered well based on bad context?), but precision is low (bad context).
        self.mock_grader.grade_conversation.return_value = GradeResult(
            faithfulness_score=0.9, retrieval_precision_score=0.2, safety_score=1.0, details={}
        )

        # 2. Process events
        # First event
        self.ingestor.process_event(self.base_event)

        # Verify metric recorded
        # Key: sentinel:metrics:rag-agent:retrieval_precision_score
        key = "sentinel:metrics:rag-agent:retrieval_precision_score"
        self.assertIn(key, self.redis_store)
        self.assertEqual(len(self.redis_store[key]), 1)

        # 3. Process enough events to form a representative average
        for _ in range(4):
            self.ingestor.process_event(self.base_event)

        # Total 5 events. Avg precision = 0.2. Threshold < 0.5.
        # Should have tripped.

        # Verify breaker state
        # The CircuitBreaker.set_state uses getset.
        # Check if set_state was called with OPEN.
        # We can check the kv store or the mock calls (though calls might be messy with redis internals).

        state_key = "sentinel:breaker:rag-agent:state"
        self.assertEqual(self.redis_kv.get(state_key), b"OPEN")

        # Verify Critical Alert sent
        self.mock_notification_service.send_critical_alert.assert_called()
        call_args = self.mock_notification_service.send_critical_alert.call_args
        self.assertIn("Trigger violated: retrieval_precision_score < 0.5", call_args.kwargs["reason"])

    def test_retrieval_precision_boundary_values(self) -> None:
        """
        Edge Case: Test boundaries 0.0 and 1.0.
        Verify they are correctly passed to Phoenix and Redis.
        """
        # Case A: Perfect Score (1.0)
        self.mock_grader.grade_conversation.return_value = GradeResult(
            faithfulness_score=1.0, retrieval_precision_score=1.0, safety_score=1.0, details={}
        )
        self.ingestor.process_event(self.base_event)

        # Check Phoenix
        expected_attributes_1 = {
            "eval.faithfulness.score": 1.0,
            "eval.retrieval.precision.score": 1.0,
            "eval.safety.score": 1.0,
        }
        self.mock_phoenix.update_span_attributes.assert_called_with(
            trace_id="t1", span_id="s1", attributes=expected_attributes_1
        )

        # Case B: Zero Score (0.0)
        self.mock_grader.grade_conversation.return_value = GradeResult(
            faithfulness_score=1.0, retrieval_precision_score=0.0, safety_score=1.0, details={}
        )
        self.ingestor.process_event(self.base_event)

        # Check Phoenix
        expected_attributes_0 = {
            "eval.faithfulness.score": 1.0,
            "eval.retrieval.precision.score": 0.0,
            "eval.safety.score": 1.0,
        }
        self.mock_phoenix.update_span_attributes.assert_called_with(
            trace_id="t1", span_id="s1", attributes=expected_attributes_0
        )
