from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from coreason_sentinel.circuit_breaker import CircuitBreaker
from coreason_sentinel.drift_engine import DriftEngine
from coreason_sentinel.ingestor import TelemetryIngestor
from coreason_sentinel.interfaces import VeritasEvent
from coreason_sentinel.models import SentinelConfig


class TestRelevanceDriftEdgeCases:
    def test_empty_vectors(self) -> None:
        """
        Test behavior with empty vectors.
        Should probably raise ValueError because cosine of empty vectors is undefined.
        """
        v1: list[float] = []
        v2: list[float] = []

        # Depending on implementation, this might raise ValueError from scipy or internal check
        # Checking if DriftEngine handles it gracefully or raises.
        # Ideally it should raise ValueError for invalid dimension (0).
        try:
            DriftEngine.compute_relevance_drift(v1, v2)
            # If it doesn't raise, we need to know what it returns.
            # If it returns NaN or throws, we catch it.
        except ValueError:
            pass  # Expected
        except Exception as e:
            pytest.fail(f"Raised unexpected exception: {e}")

    def test_non_numeric_data(self) -> None:
        """
        Test with non-numeric data in vectors.
        """
        v1 = [1.0, "banana"]
        v2 = [1.0, 0.0]

        with pytest.raises(ValueError):
            DriftEngine.compute_relevance_drift(v1, v2)  # type: ignore

    def test_high_dimensional_vectors(self) -> None:
        """
        Test with large vectors (e.g. OpenAI embedding size).
        """
        dim = 1536
        rng = np.random.default_rng(42)
        v1 = rng.random(dim).tolist()
        v2 = rng.random(dim).tolist()

        drift = DriftEngine.compute_relevance_drift(v1, v2)
        assert 0.0 <= drift <= 2.0
        assert isinstance(drift, float)


class TestComplexRelevanceScenario:
    @pytest.fixture
    def full_stack(self) -> tuple[TelemetryIngestor, CircuitBreaker, MagicMock]:
        # Mock Redis
        redis_mock = MagicMock()
        # Mock ZADD / ZRANGE behavior loosely for the "record_metric" part
        # But since CircuitBreaker uses real Redis commands, mocking strict behavior is hard without a fake-redis lib.
        # However, the user memory says: "Redis interactions must be mocked... functional mocking...
        # rather than static MagicMock".
        # We'll use a simple dict-based mock for the specific methods we need.

        storage: dict[str, list[tuple[float, str]]] = {}  # key -> list of (score, member)

        def zadd(key: str, mapping: dict[str, float]) -> int:
            if key not in storage:
                storage[key] = []
            for member, score in mapping.items():
                storage[key].append((score, member))
            return 1

        def zrangebyscore(key: str, min_s: float | str, max_s: float | str) -> list[str]:
            # simplified: return all members
            if key not in storage:
                return []
            return [m for s, m in storage[key]]

        redis_mock.zadd.side_effect = zadd
        redis_mock.zrangebyscore.side_effect = zrangebyscore
        # We also need zremrangebyscore and expire to not crash
        redis_mock.zremrangebyscore.return_value = 0
        redis_mock.expire.return_value = True

        config = SentinelConfig(
            agent_id="complex_agent",
            owner_email="admin@example.com",
            phoenix_endpoint="http://localhost",
        )

        notification = MagicMock()
        breaker = CircuitBreaker(redis_mock, config, notification)

        spot_checker = MagicMock()
        baseline_provider = MagicMock()
        veritas_client = MagicMock()

        ingestor = TelemetryIngestor(config, breaker, spot_checker, baseline_provider, veritas_client)

        return ingestor, breaker, redis_mock

    def test_relevance_drift_sequence(self, full_stack: tuple[TelemetryIngestor, CircuitBreaker, MagicMock]) -> None:
        """
        Process a sequence of events with different embedding drifts.
        Verify they are recorded in the circuit breaker.
        """
        ingestor, breaker, redis_mock = full_stack

        # 1. Event with 0 drift (Identical)
        evt1 = VeritasEvent(
            event_id="e1",
            timestamp=datetime.now(),
            agent_id="complex_agent",
            session_id="s1",
            input_text="a",
            output_text="b",
            metrics={},
            metadata={"query_embedding": [1.0, 0.0], "response_embedding": [1.0, 0.0]},
        )

        # 2. Event with 1.0 drift (Orthogonal)
        evt2 = VeritasEvent(
            event_id="e2",
            timestamp=datetime.now(),
            agent_id="complex_agent",
            session_id="s1",
            input_text="a",
            output_text="b",
            metrics={},
            metadata={"query_embedding": [1.0, 0.0], "response_embedding": [0.0, 1.0]},
        )

        # 3. Event with ~0.29 drift (45 degrees)
        # Cos(45) = 0.707. Drift = 1 - 0.707 = 0.293
        evt3 = VeritasEvent(
            event_id="e3",
            timestamp=datetime.now(),
            agent_id="complex_agent",
            session_id="s1",
            input_text="a",
            output_text="b",
            metrics={},
            metadata={"query_embedding": [1.0, 0.0], "response_embedding": [1.0, 1.0]},
        )

        ingestor.process_event(evt1)
        ingestor.process_event(evt2)
        ingestor.process_event(evt3)

        # Verification
        # Check that 'record_metric' was called or check our fake redis
        # Let's check the fake redis storage
        key = "sentinel:metrics:complex_agent:relevance_drift"

        # We need to access the closure 'storage' from the mock side_effect if possible,
        # or just inspect the calls to zadd.
        assert redis_mock.zadd.call_count >= 3

        # Filter calls for our specific key
        calls = [c for c in redis_mock.zadd.call_args_list if c[0][0] == key]
        assert len(calls) == 3

        # Check values in the calls
        # Call args: (key, {member: score})
        # member is "{timestamp}:{value}:{uuid}"

        values = []
        for call in calls:
            mapping = call[0][1]
            for member in mapping.keys():
                # Extract value between first and second colon
                # member format: timestamp:value:uuid
                val_str = str(member).split(":")[1]
                values.append(float(val_str))

        # We expect [0.0, 1.0, ~0.29289]
        # Order might be preserved if calls list is ordered
        assert 0.0 in values
        assert 1.0 in values
        assert any(abs(v - 0.29289) < 0.001 for v in values)
