import unittest
from unittest.mock import MagicMock

from coreason_sentinel.circuit_breaker import CircuitBreaker
from coreason_sentinel.drift_engine import DriftEngine
from coreason_sentinel.handlers.otel_handler import OtelIngestionHandler
from coreason_sentinel.models import SentinelConfig


class TestZCoverage(unittest.TestCase):
    def test_circuit_breaker_percentile_exception_direct(self) -> None:
        """
        Directly force exception in _evaluate_trigger percentile logic.
        Target: circuit_breaker.py:208-210
        """
        mock_redis = MagicMock()
        config = SentinelConfig(agent_id="test", owner_email="a@b.c", phoenix_endpoint="http://h", triggers=[])
        cb = CircuitBreaker(mock_redis, config, MagicMock())

        # Mock metric_store to return data
        cb.metric_store.get_values_in_window = MagicMock(return_value=[10.0])  # type: ignore

        # USE MOCK TRIGGER to bypass Pydantic validation
        trigger = MagicMock()
        trigger.metric = "latency"
        trigger.window_seconds = 60
        trigger.aggregation_method = "PXX"  # Starts with P but invalid float
        trigger.threshold = 1.0
        trigger.operator = ">"

        # _evaluate_trigger should catch ValueError and return False
        result = cb._evaluate_trigger(trigger)
        self.assertFalse(result)

    def test_drift_engine_zero_total_direct(self) -> None:
        """
        Target: drift_engine.py:170
        Ensure if total == 0 return path is taken.
        """
        samples = [10.0]
        bin_edges = [0.0, 5.0]
        # np.histogram([10], bins=[0,5]) -> count [0] -> total 0
        dist = DriftEngine.compute_distribution_from_samples(samples, bin_edges)
        self.assertEqual(dist, [0.0])

    def test_otel_handler_token_exception_direct(self) -> None:
        """
        Target: otel_handler.py:85
        Force exception during float conversion.
        """
        handler = OtelIngestionHandler(MagicMock(), MagicMock(), MagicMock(), MagicMock())

        # Attributes with invalid float string
        attrs = {"llm.token_count.total": "invalid"}

        # Should catch ValueError and return 0.0
        count = handler._extract_token_count(attrs)
        self.assertEqual(count, 0.0)

        # Test TypeError case (e.g. None or object that can't be converted)
        attrs_type = {"llm.token_count.total": None}
        count_type = handler._extract_token_count(attrs_type)
        self.assertEqual(count_type, 0.0)
