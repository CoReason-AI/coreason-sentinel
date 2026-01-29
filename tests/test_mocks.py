from datetime import datetime
from coreason_sentinel.mocks import (
    MockNotificationService,
    MockAssayGrader,
    MockPhoenixClient,
    MockBaselineProvider,
    MockVeritasClient,
)

def test_mock_notification_service() -> None:
    service = MockNotificationService()
    # Should log and not crash
    service.send_critical_alert("test@example.com", "agent_1", "Low Health")

def test_mock_assay_grader() -> None:
    grader = MockAssayGrader()
    result = grader.grade_conversation({"input": "hi", "output": "hello"})
    assert result.faithfulness_score == 0.95
    assert result.safety_score == 1.0

def test_mock_phoenix_client() -> None:
    client = MockPhoenixClient()
    client.update_span_attributes("trace_1", "span_1", {"key": "value"})

def test_mock_baseline_provider() -> None:
    provider = MockBaselineProvider()
    vectors = provider.get_baseline_vectors("agent_1")
    assert len(vectors) > 0
    probs, edges = provider.get_baseline_output_length_distribution("agent_1")
    assert len(probs) + 1 == len(edges)

def test_mock_veritas_client() -> None:
    client = MockVeritasClient()
    logs = client.fetch_logs("agent_1", datetime.now())
    assert logs == []
    client.subscribe("agent_1", lambda x: x)
