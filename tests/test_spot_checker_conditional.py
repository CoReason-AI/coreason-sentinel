# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_sentinel

import unittest.mock
from unittest.mock import MagicMock

import pytest
from coreason_sentinel.models import ConditionalSamplingRule, SentinelConfig
from coreason_sentinel.spot_checker import SpotChecker


@pytest.fixture
def mock_grader() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_phoenix() -> MagicMock:
    return MagicMock()


@pytest.fixture
def base_config() -> SentinelConfig:
    return SentinelConfig(
        agent_id="test-agent",
        sampling_rate=0.0,  # Default to 0 to verify overrides
        phoenix_endpoint="http://localhost:6006",
        owner_email="test@example.com",
    )


def test_should_sample_default_false(
    base_config: SentinelConfig, mock_grader: MagicMock, mock_phoenix: MagicMock
) -> None:
    checker = SpotChecker(base_config, mock_grader, mock_phoenix)
    # Rate is 0.0, no rules
    assert checker.should_sample({"some": "data"}) is False


def test_should_sample_default_true(
    base_config: SentinelConfig, mock_grader: MagicMock, mock_phoenix: MagicMock
) -> None:
    base_config.sampling_rate = 1.0
    checker = SpotChecker(base_config, mock_grader, mock_phoenix)
    assert checker.should_sample({"some": "data"}) is True


def test_rule_equals_match(base_config: SentinelConfig, mock_grader: MagicMock, mock_phoenix: MagicMock) -> None:
    rule = ConditionalSamplingRule(
        metadata_key="sentiment",
        operator="EQUALS",
        value="negative",
        sample_rate=1.0,
    )
    base_config.conditional_sampling_rules = [rule]
    checker = SpotChecker(base_config, mock_grader, mock_phoenix)

    # Match
    assert checker.should_sample({"sentiment": "negative"}) is True
    # No match
    assert checker.should_sample({"sentiment": "positive"}) is False


def test_rule_contains_match(base_config: SentinelConfig, mock_grader: MagicMock, mock_phoenix: MagicMock) -> None:
    rule = ConditionalSamplingRule(metadata_key="tags", operator="CONTAINS", value="vip", sample_rate=1.0)
    base_config.conditional_sampling_rules = [rule]
    checker = SpotChecker(base_config, mock_grader, mock_phoenix)

    # Match in list
    assert checker.should_sample({"tags": ["user", "vip"]}) is True
    # Match in string
    assert checker.should_sample({"tags": "vip_user"}) is True
    # No match
    assert checker.should_sample({"tags": ["user"]}) is False
    # Wrong type (int) should result in False, not error
    assert checker.should_sample({"tags": 123}) is False


def test_rule_exists_match(base_config: SentinelConfig, mock_grader: MagicMock, mock_phoenix: MagicMock) -> None:
    rule = ConditionalSamplingRule(metadata_key="error_trace", operator="EXISTS", sample_rate=1.0)
    base_config.conditional_sampling_rules = [rule]
    checker = SpotChecker(base_config, mock_grader, mock_phoenix)

    assert checker.should_sample({"error_trace": "stack..."}) is True
    assert checker.should_sample({"other": "value"}) is False


def test_multiple_rules_precedence(
    base_config: SentinelConfig, mock_grader: MagicMock, mock_phoenix: MagicMock
) -> None:
    # Rule A: 100% if "urgent"
    # Rule B: 50% if "vip"
    # Global: 0%
    rule_a = ConditionalSamplingRule(metadata_key="priority", operator="EQUALS", value="urgent", sample_rate=1.0)
    rule_b = ConditionalSamplingRule(metadata_key="user_type", operator="EQUALS", value="vip", sample_rate=0.5)

    base_config.conditional_sampling_rules = [rule_a, rule_b]
    checker = SpotChecker(base_config, mock_grader, mock_phoenix)

    # Only Urgent (100%)
    assert checker.should_sample({"priority": "urgent"}) is True

    # Only VIP (50%) - mock random to test
    # If we set random to 0.6, it should be False (0.6 > 0.5)
    # If we set random to 0.4, it should be True (0.4 < 0.5)

    # We rely on statistical testing or patching random
    # Let's patch random for deterministic behavior
    with unittest.mock.patch("random.random", return_value=0.4):
        assert checker.should_sample({"user_type": "vip"}) is True

    with unittest.mock.patch("random.random", return_value=0.6):
        assert checker.should_sample({"user_type": "vip"}) is False

    # Both match: Max(1.0, 0.5) = 1.0. random=0.9 should pass.
    with unittest.mock.patch("random.random", return_value=0.9):
        assert checker.should_sample({"priority": "urgent", "user_type": "vip"}) is True


def test_missing_metadata_handling(
    base_config: SentinelConfig, mock_grader: MagicMock, mock_phoenix: MagicMock
) -> None:
    rule = ConditionalSamplingRule(metadata_key="missing", operator="EQUALS", value="value", sample_rate=1.0)
    base_config.conditional_sampling_rules = [rule]
    checker = SpotChecker(base_config, mock_grader, mock_phoenix)

    # Metadata None
    assert checker.should_sample(None) is False
    # Metadata missing key
    assert checker.should_sample({}) is False


def test_rule_matches_derived_metric(
    base_config: SentinelConfig, mock_grader: MagicMock, mock_phoenix: MagicMock
) -> None:
    # Simulate the integration scenario where Ingestor adds "refusal_count"
    rule = ConditionalSamplingRule(metadata_key="refusal_count", operator="EXISTS", sample_rate=1.0)
    base_config.conditional_sampling_rules = [rule]
    checker = SpotChecker(base_config, mock_grader, mock_phoenix)

    # Metadata from ingestor
    metadata = {"refusal_count": 1.0, "other": "info"}
    assert checker.should_sample(metadata) is True


def test_complex_rule_interaction_intermediate_rates(
    base_config: SentinelConfig, mock_grader: MagicMock, mock_phoenix: MagicMock
) -> None:
    # Rule A: 0.2
    # Rule B: 0.5
    # Rule C: 0.8
    # Global: 0.1
    # Expected: 0.8
    rule_a = ConditionalSamplingRule(metadata_key="k1", operator="EXISTS", sample_rate=0.2)
    rule_b = ConditionalSamplingRule(metadata_key="k2", operator="EXISTS", sample_rate=0.5)
    rule_c = ConditionalSamplingRule(metadata_key="k3", operator="EXISTS", sample_rate=0.8)

    base_config.conditional_sampling_rules = [rule_a, rule_b, rule_c]
    checker = SpotChecker(base_config, mock_grader, mock_phoenix)

    metadata = {"k1": 1, "k2": 1, "k3": 1}

    # If random is 0.79, should sample (0.79 < 0.8)
    with unittest.mock.patch("random.random", return_value=0.79):
        assert checker.should_sample(metadata) is True

    # If random is 0.81, should NOT sample
    with unittest.mock.patch("random.random", return_value=0.81):
        assert checker.should_sample(metadata) is False


def test_edge_case_types(base_config: SentinelConfig, mock_grader: MagicMock, mock_phoenix: MagicMock) -> None:
    # EQUALS None
    rule_none = ConditionalSamplingRule(metadata_key="data", operator="EQUALS", value=None, sample_rate=1.0)
    base_config.conditional_sampling_rules = [rule_none]
    checker = SpotChecker(base_config, mock_grader, mock_phoenix)

    assert checker.should_sample({"data": None}) is True
    assert checker.should_sample({"data": "some"}) is False

    # EQUALS Boolean
    rule_bool = ConditionalSamplingRule(metadata_key="is_test", operator="EQUALS", value=True, sample_rate=1.0)
    base_config.conditional_sampling_rules = [rule_bool]
    checker = SpotChecker(base_config, mock_grader, mock_phoenix)

    assert checker.should_sample({"is_test": True}) is True
    assert checker.should_sample({"is_test": False}) is False

    # CONTAINS on None (should be safe False)
    rule_contains = ConditionalSamplingRule(metadata_key="tags", operator="CONTAINS", value="x", sample_rate=1.0)
    base_config.conditional_sampling_rules = [rule_contains]
    checker = SpotChecker(base_config, mock_grader, mock_phoenix)

    assert checker.should_sample({"tags": None}) is False


def test_case_sensitivity(base_config: SentinelConfig, mock_grader: MagicMock, mock_phoenix: MagicMock) -> None:
    rule = ConditionalSamplingRule(metadata_key="status", operator="EQUALS", value="ERROR", sample_rate=1.0)
    base_config.conditional_sampling_rules = [rule]
    checker = SpotChecker(base_config, mock_grader, mock_phoenix)

    # Match exact
    assert checker.should_sample({"status": "ERROR"}) is True
    # Mismatch case
    assert checker.should_sample({"status": "error"}) is False
