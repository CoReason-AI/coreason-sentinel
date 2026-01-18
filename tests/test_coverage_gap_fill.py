import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from redis import Redis

from coreason_sentinel.circuit_breaker import CircuitBreaker, CircuitBreakerState
from coreason_sentinel.models import CircuitBreakerTrigger, SentinelConfig


class TestCoverageGapFill(unittest.TestCase):
    def test_circuit_breaker_percentile_exception(self) -> None:
        """
        Target circuit_breaker.py lines 208-210 (exception block).
        We mock numpy.percentile to raise ValueError.
        """
        mock_redis = MagicMock(spec=Redis)
        mock_redis.zrangebyscore.return_value = [b"123.0:1.0:uuid"]  # Return something so logic proceeds

        trigger = CircuitBreakerTrigger(metric="latency", threshold=10.0, window_seconds=60, aggregation_method="P99")
        config = SentinelConfig(
            agent_id="test", owner_email="test@test.com", phoenix_endpoint="http://localhost", triggers=[trigger]
        )
        breaker = CircuitBreaker(mock_redis, config, MagicMock())

        # We need to mock _parse_value_from_member if we mock metric store or rely on real one?
        # The breaker uses self.metric_store.get_values_in_window.
        # Let's mock the metric_store on the breaker instance directly to control return values easily.
        breaker.metric_store.get_values_in_window = MagicMock(return_value=[1.0, 2.0])  # type: ignore

        with patch("numpy.percentile", side_effect=ValueError("Test Error")):
            # This should trigger the except block
            breaker.check_triggers()

        # If we didn't crash, we're good. Logic catches exception and returns False.
        # Ensure state didn't change (default closed).
        self.assertEqual(breaker.get_state(), CircuitBreakerState.CLOSED)

    def test_drift_engine_zero_total(self) -> None:
        """
        Target drift_engine.py line 170 (if total == 0).
        """
        from coreason_sentinel.drift_engine import DriftEngine

        # Construct scenario where histogram returns counts summing to 0.
        # Empty samples? No, that returns early.
        # Samples outside bin range?
        samples = [10.0, 20.0]
        bin_edges = [0.0, 5.0]
        # np.histogram(samples, bins=bin_edges) -> counts [0], sum 0.

        dist = DriftEngine.compute_distribution_from_samples(samples, bin_edges)
        self.assertEqual(dist, [0.0])

    def test_otel_handler_token_parsing_exception(self) -> None:
        """
        Target otel_handler.py line 85 (except ValueError, TypeError).
        """
        from coreason_sentinel.handlers.otel_handler import OtelIngestionHandler
        from coreason_sentinel.metric_store import MetricStore
        from coreason_sentinel.utils.metric_extractor import MetricExtractor

        handler = OtelIngestionHandler(
            MagicMock(spec=SentinelConfig),
            MagicMock(spec=MetricStore),
            MagicMock(spec=CircuitBreaker),
            MagicMock(spec=MetricExtractor),
        )

        # Pass a dict where token count is not a number
        attrs: dict[str, Any] = {"llm.token_count.total": "invalid"}

        # This calls _extract_token_count
        count = handler._extract_token_count(attrs)
        self.assertEqual(count, 0.0)
