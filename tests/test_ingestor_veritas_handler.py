import unittest
from datetime import datetime
from unittest.mock import MagicMock

from coreason_sentinel.circuit_breaker import CircuitBreaker
from coreason_sentinel.drift_monitor import DriftMonitor
from coreason_sentinel.handlers.veritas_handler import VeritasIngestionHandler
from coreason_sentinel.interfaces import GradeResult, VeritasClientProtocol, VeritasEvent
from coreason_sentinel.metric_store import MetricStore
from coreason_sentinel.models import SentinelConfig
from coreason_sentinel.spot_checker import SpotChecker
from coreason_sentinel.utils.metric_extractor import MetricExtractor


class TestVeritasIngestionHandler(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SentinelConfig(
            agent_id="test_agent", owner_email="test@example.com", phoenix_endpoint="http://localhost", triggers=[]
        )
        self.mock_client = MagicMock(spec=VeritasClientProtocol)
        self.mock_metric_store = MagicMock(spec=MetricStore)
        self.mock_breaker = MagicMock(spec=CircuitBreaker)
        self.mock_spot_checker = MagicMock(spec=SpotChecker)
        self.mock_drift_monitor = MagicMock(spec=DriftMonitor)
        self.mock_extractor = MagicMock(spec=MetricExtractor)

        self.handler = VeritasIngestionHandler(
            self.config,
            self.mock_client,
            self.mock_metric_store,
            self.mock_breaker,
            self.mock_spot_checker,
            self.mock_drift_monitor,
            self.mock_extractor,
        )

        self.event = VeritasEvent(
            event_id="e1",
            timestamp=datetime.now(),
            agent_id="test_agent",
            session_id="s1",
            input_text="in",
            output_text="out",
            metrics={"latency": 0.5},
            metadata={"meta": "data"},
        )

    def test_ingest_since_success(self) -> None:
        """Test fetching and processing logs."""
        self.mock_client.fetch_logs.return_value = [self.event]
        self.mock_extractor.extract.return_value = {}

        count = self.handler.ingest_since(datetime.now())
        self.assertEqual(count, 1)
        self.mock_client.fetch_logs.assert_called_once()
        self.mock_drift_monitor.process_event.assert_called_with(self.event)
        self.mock_breaker.check_triggers.assert_called_once()

    def test_ingest_since_fetch_error(self) -> None:
        """Test error handling during fetch."""
        self.mock_client.fetch_logs.side_effect = Exception("API Error")
        count = self.handler.ingest_since(datetime.now())
        self.assertEqual(count, 0)

    def test_ingest_since_empty(self) -> None:
        """Test empty log fetch."""
        self.mock_client.fetch_logs.return_value = []
        count = self.handler.ingest_since(datetime.now())
        self.assertEqual(count, 0)

    def test_process_event_metrics_recording(self) -> None:
        """Test recording of standard metrics."""
        self.mock_extractor.extract.return_value = {}
        self.handler.process_event(self.event)
        self.mock_metric_store.record_metric.assert_any_call("test_agent", "latency", 0.5, retention_window=3600)

    def test_process_event_custom_metrics(self) -> None:
        """Test recording of custom metrics."""
        self.mock_extractor.extract.return_value = {"refusal_count": 1.0}
        self.handler.process_event(self.event)
        self.mock_metric_store.record_metric.assert_any_call("test_agent", "refusal_count", 1.0, retention_window=3600)

    def test_process_event_spot_check(self) -> None:
        """Test spot checking triggering."""
        self.mock_extractor.extract.return_value = {}
        self.mock_spot_checker.should_sample.return_value = True
        self.mock_spot_checker.check_sample.return_value = GradeResult(
            faithfulness_score=0.9, retrieval_precision_score=0.8, safety_score=1.0, details={}
        )

        self.handler.process_event(self.event)

        self.mock_spot_checker.check_sample.assert_called()
        self.mock_metric_store.record_metric.assert_any_call(
            "test_agent", "faithfulness_score", 0.9, retention_window=3600
        )

    def test_process_event_drift_detection(self) -> None:
        """Test drift monitor invocation."""
        self.mock_extractor.extract.return_value = {}
        self.handler.process_event(self.event)
        self.mock_drift_monitor.process_event.assert_called_with(self.event)

    def test_process_event_triggers(self) -> None:
        """Test trigger check invocation."""
        self.mock_extractor.extract.return_value = {}
        self.handler.process_event(self.event)
        self.mock_breaker.check_triggers.assert_called_once()

    def test_process_event_exception_handling_in_loop(self) -> None:
        """Test individual event failure doesn't stop loop."""
        bad_event = VeritasEvent(
            event_id="bad",
            timestamp=datetime.now(),
            agent_id="test_agent",
            session_id="s1",
            input_text="in",
            output_text="out",
            metrics={"latency": 0.5},
            metadata={},
        )
        self.mock_client.fetch_logs.return_value = [bad_event, self.event]

        # Make processing first event fail
        # Mock record_metric to fail only for first event logic?
        # Easier: Mock metric_extractor to fail for first event
        self.mock_extractor.extract.side_effect = [Exception("Extract Error"), {}]

        count = self.handler.ingest_since(datetime.now())
        self.assertEqual(count, 1)  # Only 1 succeeded

    def test_record_metrics_trigger_window(self) -> None:
        """Test trigger window max logic."""
        # Set up a trigger
        t = MagicMock()
        t.metric = "latency"
        t.window_seconds = 7200  # 2 hours
        self.config.triggers = [t]

        self.handler.process_event(self.event)

        # Verify retention window is max(3600, 7200) = 7200
        self.mock_metric_store.record_metric.assert_any_call("test_agent", "latency", 0.5, retention_window=7200)
