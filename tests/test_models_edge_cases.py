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

from coreason_sentinel.models import (
    CircuitBreakerTrigger,
    ConditionalSamplingRule,
    HealthReport,
    SentinelConfig,
)
from pydantic import ValidationError


class TestModelsEdgeCases(unittest.TestCase):
    def test_migration_safety_renamed_fields(self) -> None:
        """
        Verify that using old field names raises a ValidationError due to extra='forbid'.
        This prevents silent defaults being used when users think they are configuring a value.
        """
        # Case 1: SentinelConfig - sample_rate (old) vs sampling_rate (new)
        with self.assertRaises(ValidationError) as cm:
            SentinelConfig(  # type: ignore[call-arg]
                agent_id="test",
                owner_email="admin@example.com",
                phoenix_endpoint="http://localhost",
                sample_rate=0.5,  # OLD NAME
            )
        self.assertIn("sample_rate", str(cm.exception))
        self.assertIn("Extra inputs are not permitted", str(cm.exception))

        # Case 2: CircuitBreakerTrigger - metric_name (old) vs metric (new)
        with self.assertRaises(ValidationError) as cm:
            CircuitBreakerTrigger(  # type: ignore[call-arg]
                metric_name="errors",  # OLD NAME
                threshold=1,
                window_seconds=60,
            )
        self.assertIn("metric_name", str(cm.exception))

        # Case 3: HealthReport - agent_status (old) vs breaker_state (new)
        with self.assertRaises(ValidationError) as cm:
            HealthReport(  # type: ignore[call-arg]
                timestamp=datetime.now(timezone.utc),
                agent_status="CLOSED",  # OLD NAME
            )
        self.assertIn("agent_status", str(cm.exception))

    def test_forbid_extra_fields(self) -> None:
        """
        Verify that unknown fields are rejected.
        """
        with self.assertRaises(ValidationError):
            SentinelConfig(  # type: ignore[call-arg]
                agent_id="test",
                owner_email="a",
                phoenix_endpoint="b",
                unexpected_field="should_fail",
            )

    def test_missing_required_fields(self) -> None:
        """
        Verify strict requirements for new fields.
        """
        # Missing owner_email
        with self.assertRaises(ValidationError) as cm:
            SentinelConfig(  # type: ignore[call-arg]
                agent_id="test",
                phoenix_endpoint="http://localhost",
            )
        self.assertIn("owner_email", str(cm.exception))

        # Missing phoenix_endpoint
        with self.assertRaises(ValidationError) as cm:
            SentinelConfig(  # type: ignore[call-arg]
                agent_id="test",
                owner_email="a",
            )
        self.assertIn("phoenix_endpoint", str(cm.exception))

    def test_complex_config_serialization(self) -> None:
        """
        Verify that a fully populated, complex configuration serializes and deserializes correctly.
        """
        trigger1 = CircuitBreakerTrigger(metric="cost", threshold=10, window_seconds=60)
        trigger2 = CircuitBreakerTrigger(
            metric="latency", threshold=500, window_seconds=60, operator=">", aggregation_method="MAX"
        )

        rule1 = ConditionalSamplingRule(metadata_key="vip", operator="EXISTS", sample_rate=1.0)
        rule2 = ConditionalSamplingRule(metadata_key="tier", operator="EQUALS", value="free", sample_rate=0.01)

        config = SentinelConfig(
            agent_id="complex-agent",
            owner_email="ops@coreason.ai",
            phoenix_endpoint="https://phoenix.coreason.ai",
            sampling_rate=0.05,
            drift_threshold_kl=0.9,
            drift_sample_window=200,
            cost_per_1k_tokens=0.05,
            recovery_timeout=120,
            triggers=[trigger1, trigger2],
            sentiment_regex_patterns=["ERROR", "FAIL"],
            conditional_sampling_rules=[rule1, rule2],
        )

        # Serialize
        json_str = config.model_dump_json()

        # Deserialize
        restored = SentinelConfig.model_validate_json(json_str)

        # Deep equality check
        self.assertEqual(config, restored)

        # Check nested fields
        self.assertEqual(len(restored.triggers), 2)
        self.assertEqual(restored.triggers[1].aggregation_method, "MAX")
        self.assertEqual(len(restored.conditional_sampling_rules), 2)
        self.assertEqual(restored.conditional_sampling_rules[1].value, "free")

    def test_trigger_window_must_be_positive(self) -> None:
        """
        Verify strict logic validation for Trigger.
        """
        with self.assertRaises(ValidationError):
            CircuitBreakerTrigger(metric="foo", threshold=1, window_seconds=0)

    def test_sampling_rate_bounds(self) -> None:
        """
        Verify bounds for sampling rate.
        """
        with self.assertRaises(ValidationError):
            SentinelConfig(agent_id="a", owner_email="b", phoenix_endpoint="c", sampling_rate=-0.1)
        with self.assertRaises(ValidationError):
            SentinelConfig(agent_id="a", owner_email="b", phoenix_endpoint="c", sampling_rate=1.1)
