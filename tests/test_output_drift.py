# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_sentinel

from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
from coreason_sentinel.circuit_breaker import CircuitBreaker
from coreason_sentinel.drift_engine import DriftEngine
from coreason_sentinel.ingestor import TelemetryIngestor
from coreason_sentinel.interfaces import BaselineProviderProtocol, VeritasEvent
from coreason_sentinel.models import CircuitBreakerTrigger, SentinelConfig


class TestOutputDriftDetection:
    @pytest.fixture
    def config(self) -> SentinelConfig:
        return SentinelConfig(
            agent_id="test-agent",
            owner_email="test@example.com",
            phoenix_endpoint="http://localhost:6006",
            drift_sample_window=10,
            triggers=[
                CircuitBreakerTrigger(
                    metric="output_drift_kl",
                    threshold=0.5,
                    window_seconds=60,
                    operator=">",
                    aggregation_method="MAX",
                )
            ],
        )

    @pytest.fixture
    def mock_redis(self) -> MagicMock:
        redis = MagicMock()
        # Mock zadd to do nothing
        redis.zadd.return_value = 1
        # Mock zremrangebyscore
        redis.zremrangebyscore.return_value = 0
        # Mock expire
        redis.expire.return_value = True
        return redis

    @pytest.fixture
    def circuit_breaker(self, mock_redis: MagicMock, config: SentinelConfig) -> CircuitBreaker:
        from coreason_sentinel.interfaces import NotificationServiceProtocol

        mock_notification_service = MagicMock(spec=NotificationServiceProtocol)
        cb = CircuitBreaker(mock_redis, config, mock_notification_service)
        # Mock check_triggers to avoid redis calls in unit test if desired,
        # but we want to test interaction.
        # We will mock get_recent_values instead of the redis call inside it for simplicity
        # or we can mock zrevrange. Let's mock zrevrange.
        return cb

    @pytest.fixture
    def baseline_provider(self) -> MagicMock:
        provider = MagicMock(spec=BaselineProviderProtocol)
        # Setup a simple baseline: 3 bins [0-10, 10-20, 20-30]
        # Distribution: mostly in middle bin
        # Probs: [0.1, 0.8, 0.1]
        # Edges: [0, 10, 20, 30]
        provider.get_baseline_output_length_distribution.return_value = (
            [0.1, 0.8, 0.1],
            [0.0, 10.0, 20.0, 30.0],
        )
        provider.get_baseline_vectors.return_value = []
        return provider

    @pytest.fixture
    def ingestor(
        self,
        config: SentinelConfig,
        circuit_breaker: CircuitBreaker,
        baseline_provider: MagicMock,
    ) -> TelemetryIngestor:
        spot_checker = Mock()
        spot_checker.should_sample.return_value = False
        veritas_client = Mock()  # Added mock for veritas_client
        return TelemetryIngestor(config, circuit_breaker, spot_checker, baseline_provider, veritas_client)

    def test_compute_distribution_from_samples(self) -> None:
        """
        Verify that raw samples are correctly histogrammed into probabilities.
        """
        samples = [5.0, 15.0, 15.0, 25.0]  # 1 in bin 0, 2 in bin 1, 1 in bin 2
        edges = [0.0, 10.0, 20.0, 30.0]

        dist = DriftEngine.compute_distribution_from_samples(samples, edges)

        # Total samples = 4
        # Bin 0 (0-10): 1 sample -> 0.25
        # Bin 1 (10-20): 2 samples -> 0.50
        # Bin 2 (20-30): 1 sample -> 0.25
        expected = [0.25, 0.50, 0.25]

        np.testing.assert_array_almost_equal(dist, expected)

    def test_compute_distribution_empty_samples(self) -> None:
        """
        Verify empty samples return zero distribution.
        """
        edges = [0.0, 10.0, 20.0]
        dist = DriftEngine.compute_distribution_from_samples([], edges)
        assert dist == [0.0, 0.0]

    def test_ingestor_process_output_drift_happy_path(
        self,
        ingestor: TelemetryIngestor,
        circuit_breaker: CircuitBreaker,
        mock_redis: MagicMock,
        baseline_provider: MagicMock,
    ) -> None:
        """
        Test that process_event calculates KL divergence and records it.
        """
        # Mock Redis to return recent samples that match baseline exactly
        # Baseline: [0.1, 0.8, 0.1] for bins [0, 10, 20, 30]
        # Let's say samples are [5, 15, 15, 15, 15, 15, 15, 15, 15, 25] (10 samples)
        # 1 in bin 0, 8 in bin 1, 1 in bin 2 => Matches baseline exactly.

        # We need to mock what redis.zrevrange returns.
        # It returns list of member bytes: b"{ts}:{val}:{uuid}"
        samples = [5.0] + [15.0] * 8 + [25.0]
        mock_members = [f"123:{v}:uuid".encode() for v in samples]
        mock_redis.zrevrange.return_value = mock_members

        # Create event with output length
        event = VeritasEvent(
            event_id="e1",
            timestamp=datetime.now(timezone.utc),
            agent_id="test-agent",
            session_id="s1",
            input_text="hi",
            output_text="word " * 15,  # length 15
            metrics={"completion_tokens": 15},
            metadata={},
        )

        ingestor.process_drift(event)

        # Verify calls
        # 1. output_length recorded
        # 2. output_drift_kl recorded

        # Check that zadd was called for 'output_drift_kl'
        # We iterate over calls to find the one for the metric
        drift_call_found = False
        for call in mock_redis.zadd.call_args_list:
            args, _ = call
            key = args[0]
            if "output_drift_kl" in key:
                drift_call_found = True
                # The value recorded should be close to 0.0 because distribution matches baseline
                payload = args[1]  # dict {member: score}
                member = list(payload.keys())[0]
                val = float(member.split(":")[1])
                assert val < 0.01  # KL should be near 0
                break

        assert drift_call_found

    def test_ingestor_process_output_drift_high_divergence(
        self,
        ingestor: TelemetryIngestor,
        circuit_breaker: CircuitBreaker,
        mock_redis: MagicMock,
        baseline_provider: MagicMock,
    ) -> None:
        """
        Test that process_event calculates high KL divergence when distributions differ.
        """
        # Baseline: [0.1, 0.8, 0.1] (Centered)
        # Live Samples: All in last bin [25, 25, 25...] -> [0, 0, 1]

        samples = [25.0] * 10
        mock_members = [f"123:{v}:uuid".encode() for v in samples]
        mock_redis.zrevrange.return_value = mock_members

        event = VeritasEvent(
            event_id="e1",
            timestamp=datetime.now(timezone.utc),
            agent_id="test-agent",
            session_id="s1",
            input_text="hi",
            output_text="word",
            metrics={"completion_tokens": 25},
            metadata={},
        )

        ingestor.process_drift(event)

        # Verify drift is high
        drift_call_found = False
        for call in mock_redis.zadd.call_args_list:
            args, _ = call
            key = args[0]
            if "output_drift_kl" in key:
                drift_call_found = True
                payload = args[1]
                member = list(payload.keys())[0]
                val = float(member.split(":")[1])
                # KL([0.1, 0.8, 0.1] || [0, 0, 1]) is high
                # Actually KL is computed as KL(Baseline || Live)
                # KL(P || Q) = sum P * log(P/Q)
                # P=[0.1, 0.8, 0.1], Q=[e, e, 1.0] (with smoothing)
                # 0.1*log(0.1/e) + 0.8*log(0.8/e) + 0.1*log(0.1/1)
                # This should be large positive number.
                assert val > 1.0
                break

        assert drift_call_found

    def test_fallback_output_length_calculation(
        self,
        ingestor: TelemetryIngestor,
        mock_redis: MagicMock,
        baseline_provider: MagicMock,
    ) -> None:
        """
        Verify fallback to word count if metrics are missing.
        """
        # Event with no metrics, just text
        event = VeritasEvent(
            event_id="e1",
            timestamp=datetime.now(timezone.utc),
            agent_id="test-agent",
            session_id="s1",
            input_text="hi",
            output_text="one two three",  # 3 words
            metrics={},
            metadata={},
        )

        # Mock recent samples to avoid None errors
        mock_redis.zrevrange.return_value = [b"123:3.0:uuid"]

        ingestor.process_drift(event)

        # Verify 'output_length' recorded with value 3.0
        length_call_found = False
        for call in mock_redis.zadd.call_args_list:
            args, _ = call
            key = args[0]
            if "output_length" in key:
                length_call_found = True
                payload = args[1]
                member = list(payload.keys())[0]
                val = float(member.split(":")[1])
                assert val == 3.0
                break

        assert length_call_found

    def test_compute_distribution_zero_total_count(self) -> None:
        """
        Verify that if total samples is non-empty but bins capture nothing (e.g. range mismatch),
        it handles division by zero safely.
        """
        samples = [100.0]
        # Bins from 0 to 10. Sample is 100. Counts will be [0]. Total 0.
        edges = [0.0, 10.0]
        dist = DriftEngine.compute_distribution_from_samples(samples, edges)
        assert dist == [0.0]

    def test_output_length_from_token_count(
        self,
        ingestor: TelemetryIngestor,
        mock_redis: MagicMock,
    ) -> None:
        """Verify output length is extracted from 'token_count' metric."""
        event = VeritasEvent(
            event_id="e2",
            timestamp=datetime.now(timezone.utc),
            agent_id="test-agent",
            session_id="s1",
            input_text="hi",
            output_text="word",
            metrics={"token_count": 42},
            metadata={},
        )
        mock_redis.zrevrange.return_value = [b"123:42.0:uuid"]

        ingestor.process_drift(event)

        # Verify 'output_length' recorded with value 42.0
        length_call_found = False
        for call in mock_redis.zadd.call_args_list:
            args, _ = call
            key = args[0]
            if "output_length" in key:
                length_call_found = True
                payload = args[1]
                member = list(payload.keys())[0]
                val = float(member.split(":")[1])
                assert val == 42.0
                break

        assert length_call_found

    def test_gradual_drift_scenario(self, config: SentinelConfig, baseline_provider: MagicMock) -> None:
        """
        Complex Scenario: Gradual Drift.
        Simulate a sequence of events where output length shifts from matching baseline to deviating.
        Verifies that KL divergence increases and eventually trips the breaker.
        Uses a functional Redis mock.
        """
        # 1. Setup Logic with Functional Mocks
        from typing import Dict, List, Tuple, Union

        from coreason_sentinel.interfaces import NotificationServiceProtocol

        mock_redis = MagicMock()
        mock_notification_service = MagicMock(spec=NotificationServiceProtocol)
        redis_store: Dict[str, List[Tuple[float, bytes]]] = {}

        def mock_zadd(key: str, mapping: Dict[Union[str, bytes], float]) -> None:
            if key not in redis_store:
                redis_store[key] = []
            for m, s in mapping.items():
                redis_store[key].append((s, m if isinstance(m, bytes) else m.encode("utf-8")))

        def mock_zrange(key: str, min_s: Union[float, str], max_s: Union[float, str]) -> List[bytes]:
            # Return list of members for THIS key
            if key not in redis_store:
                return []
            # Sort by score just in case, though zadd appends
            sorted_items = sorted(redis_store[key], key=lambda x: x[0])
            return [m for s, m in sorted_items]

        def mock_zrevrange(key: str, start: int, end: int) -> List[bytes]:
            if key not in redis_store:
                return []
            sorted_items = sorted(redis_store[key], key=lambda x: x[0], reverse=True)
            # Simulate slice (end is inclusive in redis-py for zrange but start/end are ranks here)
            # zrevrange(0, N) gets N+1 items if inclusive? Redis docs say stop is inclusive.
            # But python slice is exclusive.
            # Usually redis-py zrevrange(0, 99) gets 100 items.
            # Let's assume input 'limit' logic in CircuitBreaker uses zrevrange(0, limit - 1).
            # So we return up to limit items.
            # Logic: events = self.redis.zrevrange(key, 0, limit - 1)
            # If limit is 10, args are 0, 9. We want indices 0..9 inclusive.
            # Python slice [0:10] is correct.
            # start=0, end=9. slice is [0 : 9+1].
            return [m for s, m in sorted_items[start : end + 1]]

        def mock_zremrange(key: str, min_s: Union[float, str], max_s: Union[float, str]) -> None:
            pass

        def mock_get(key: str) -> bytes:
            return b"CLOSED"

        mock_redis.zadd.side_effect = mock_zadd
        mock_redis.zrangebyscore.side_effect = mock_zrange
        mock_redis.zrevrange.side_effect = mock_zrevrange
        mock_redis.zremrangebyscore.side_effect = mock_zremrange
        mock_redis.get.side_effect = mock_get

        # Update config to have a small window for faster drift
        config.drift_sample_window = 5
        # Trigger: output_drift_kl > 0.5

        real_cb = CircuitBreaker(mock_redis, config, mock_notification_service)
        spot_checker = Mock()
        spot_checker.should_sample.return_value = False
        veritas_client = Mock()  # Added veritas_client mock

        # Baseline: [0.0, 1.0, 0.0] for bins [0, 10, 20, 30]
        # This matches "Good" samples (15.0) perfectly, so KL should be ~0.
        baseline_provider.get_baseline_output_length_distribution.return_value = (
            [0.0, 1.0, 0.0],
            [0.0, 10.0, 20.0, 30.0],
        )

        ingestor = TelemetryIngestor(config, real_cb, spot_checker, baseline_provider, veritas_client)

        # 2. Feed "Good" Events (Length 15)
        # Should match baseline, low drift.
        good_event = VeritasEvent(
            event_id="good",
            timestamp=datetime.now(timezone.utc),
            agent_id="test-agent",
            session_id="s1",
            input_text="hi",
            output_text="word",
            metrics={"token_count": 15},
            metadata={},
        )

        for _ in range(5):
            ingestor.process_drift(good_event)

        # Verify NO trip
        mock_redis.set.assert_not_called()

        # 3. Feed "Bad" Events (Length 25)
        # Drift should increase.
        # Window is 5. As we add bad events, they replace good ones in the calculation window.
        bad_event = VeritasEvent(
            event_id="bad",
            timestamp=datetime.now(timezone.utc),
            agent_id="test-agent",
            session_id="s1",
            input_text="hi",
            output_text="word",
            metrics={"token_count": 25},
            metadata={},
        )

        # 1st bad event: Window [25, 15, 15, 15, 15] -> Some drift
        # ...
        # 5th bad event: Window [25, 25, 25, 25, 25] -> High drift (All in Bin 2, Baseline expects 10% there)

        # Mock getset return value
        mock_redis.getset.return_value = b"CLOSED"

        for _ in range(5):
            ingestor.process_drift(bad_event)

        # Verify TRIP
        mock_redis.getset.assert_any_call("sentinel:breaker:test-agent:state", "OPEN")

    def test_sparse_data_distribution(self, config: SentinelConfig, baseline_provider: MagicMock) -> None:
        """
        Edge Case: Sparse Data.
        Verify KL calculation works with only 1 recent sample.
        """
        mock_redis = MagicMock()
        # Return only 1 sample [15.0] (Perfect match to baseline center)
        mock_redis.zrevrange.return_value = [b"123:15.0:uuid"]

        from coreason_sentinel.interfaces import NotificationServiceProtocol

        mock_notification_service = MagicMock(spec=NotificationServiceProtocol)

        real_cb = CircuitBreaker(mock_redis, config, mock_notification_service)
        veritas_client = Mock()  # Added mock for veritas_client
        ingestor = TelemetryIngestor(
            config, real_cb, Mock(should_sample=Mock(return_value=False)), baseline_provider, veritas_client
        )

        event = VeritasEvent(
            event_id="e1",
            timestamp=datetime.now(timezone.utc),
            agent_id="test-agent",
            session_id="s1",
            input_text="hi",
            output_text="word",
            metrics={"token_count": 15},
            metadata={},
        )

        ingestor.process_drift(event)

        # Check drift recorded. Should be low but not failing.
        drift_call_found = False
        for call in mock_redis.zadd.call_args_list:
            args, _ = call
            key = args[0]
            if "output_drift_kl" in key:
                drift_call_found = True
                break
        assert drift_call_found

    def test_samples_completely_outside_bins(self, config: SentinelConfig, baseline_provider: MagicMock) -> None:
        """
        Edge Case: Samples far outside baseline bins.
        """
        mock_redis = MagicMock()
        # Sample 1000.0 (Far outside bins 0-30)
        mock_redis.zrevrange.return_value = [b"123:1000.0:uuid"]

        from coreason_sentinel.interfaces import NotificationServiceProtocol

        mock_notification_service = MagicMock(spec=NotificationServiceProtocol)

        real_cb = CircuitBreaker(mock_redis, config, mock_notification_service)
        veritas_client = Mock()  # Added mock for veritas_client
        ingestor = TelemetryIngestor(
            config, real_cb, Mock(should_sample=Mock(return_value=False)), baseline_provider, veritas_client
        )

        event = VeritasEvent(
            event_id="e1",
            timestamp=datetime.now(timezone.utc),
            agent_id="test-agent",
            session_id="s1",
            input_text="hi",
            output_text="word",
            metrics={"token_count": 1000},
            metadata={},
        )

        ingestor.process_drift(event)

        # Check drift recorded.
        # Distribution will be [0, 0, 0].
        # KL(Baseline || Live)
        # Baseline [0.1, 0.8, 0.1]. Live [e, e, e].
        # Should be high.

        drift_val = 0.0
        for call in mock_redis.zadd.call_args_list:
            args, _ = call
            key = args[0]
            if "output_drift_kl" in key:
                payload = args[1]
                member = list(payload.keys())[0]
                drift_val = float(member.split(":")[1])
                break

        # With zero matching bins, drift should be substantial
        assert drift_val > 0.1
