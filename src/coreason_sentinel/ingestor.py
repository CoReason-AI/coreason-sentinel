# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_sentinel


import re
from datetime import datetime
from types import TracebackType
from typing import Any, ContextManager, Dict, Optional, Type, cast

import anyio
import anyio.from_thread
import httpx
from coreason_identity.models import UserContext

from coreason_sentinel.circuit_breaker import CircuitBreaker
from coreason_sentinel.drift_engine import DriftEngine
from coreason_sentinel.interfaces import BaselineProviderProtocol, OTELSpan, VeritasClientProtocol, VeritasEvent
from coreason_sentinel.models import SentinelConfig
from coreason_sentinel.spot_checker import SpotChecker
from coreason_sentinel.utils.logger import logger


class TelemetryIngestorAsync:
    """
    The Listener: Orchestrates the monitoring pipeline (Async Version).

    The Omni-Ingestor ingests both OTEL Spans (real-time traces) and Veritas Logs (long-term data).
    It routes events to the Circuit Breaker for metric tracking and trigger evaluation,
    to the Spot Checker for auditing, and to the Drift Engine for statistical analysis.
    """

    def __init__(
        self,
        config: SentinelConfig,
        circuit_breaker: CircuitBreaker,
        spot_checker: SpotChecker,
        baseline_provider: BaselineProviderProtocol,
        veritas_client: VeritasClientProtocol,
        client: Optional[httpx.AsyncClient] = None,
    ):
        """
        Initializes the TelemetryIngestorAsync.

        Args:
            config: Configuration for ingestion rules.
            circuit_breaker: Instance of CircuitBreaker to record metrics and check triggers.
            spot_checker: Instance of SpotChecker to audit samples.
            baseline_provider: Provider for baseline vectors and distributions for drift detection.
            veritas_client: Client to fetch historical/batched logs.
            client: Optional external httpx.AsyncClient for connection pooling.
        """
        self.config = config
        self.circuit_breaker = circuit_breaker
        self.spot_checker = spot_checker
        self.baseline_provider = baseline_provider
        self.veritas_client = veritas_client
        self._internal_client = client is None
        if client:
            self._client = client
        else:
            self._client = httpx.AsyncClient()  # pragma: no cover

    async def __aenter__(self) -> "TelemetryIngestorAsync":
        # Ensure client is usable.
        # If internal client was closed, recreate it? No, assumed single use or proper lifecycle.
        if self._internal_client and self._client.is_closed:
            self._client = httpx.AsyncClient()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if self._internal_client:
            await self._client.aclose()
        # Close other resources if needed

    async def process_otel_span(self, span: OTELSpan, user_context: Optional[UserContext] = None) -> None:
        """
        Processes a single OpenTelemetry Span asynchronously.
        """
        logger.info(f"Processing OTEL Span {span.span_id} - {span.name}")

        # Map user context to span attributes if available
        if user_context:
            user_id = getattr(user_context, "user_id", getattr(user_context, "sub", None))
            if user_id:
                span.attributes["enduser.id"] = user_id

            groups = getattr(user_context, "groups", getattr(user_context, "permissions", None))
            if groups:
                # Map permissions/groups to role. Taking the first one or joining them.
                # Requirement: Map user_context.groups/permissions to enduser.role
                span.attributes["enduser.role"] = ",".join(groups)

        # 1. Calculate Latency (seconds)
        if span.end_time_unix_nano > span.start_time_unix_nano:
            latency_sec = (span.end_time_unix_nano - span.start_time_unix_nano) / 1e9
            await self.circuit_breaker.record_metric("latency", latency_sec, user_context)

        attributes = span.attributes or {}

        # 2. Extract Token Counts
        token_count = 0.0

        try:
            if "llm.token_count.total" in attributes:
                token_count = float(attributes["llm.token_count.total"])
            elif "gen_ai.usage.total_tokens" in attributes:
                token_count = float(attributes["gen_ai.usage.total_tokens"])
            elif "llm.usage.total_tokens" in attributes:
                token_count = float(attributes["llm.usage.total_tokens"])
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse token count from span attributes: {e}")
            token_count = 0.0

        if token_count > 0:
            await self.circuit_breaker.record_metric("token_count", token_count, user_context)

            if self.config.cost_per_1k_tokens > 0:
                cost = (token_count / 1000.0) * self.config.cost_per_1k_tokens
                await self.circuit_breaker.record_metric("cost", cost, user_context)

        # 4. Extract Custom Metrics
        input_text = ""
        if "gen_ai.prompt" in attributes:
            input_text = str(attributes["gen_ai.prompt"])
        elif "llm.input_messages" in attributes:
            input_text = str(attributes["llm.input_messages"])

        custom_metrics = self._extract_custom_metrics(input_text, attributes)
        for metric_name, value in custom_metrics.items():
            await self.circuit_breaker.record_metric(metric_name, value, user_context)

        # Security Check: If high severity metric detected (e.g. from custom_metrics via regex)
        # Note: 'refusal_count' or 'sentiment_frustration_count' might trigger this if severe?
        # The prompt says: "If the event is a 'Security Violation' (detected attack)..."
        # We assume if specific metrics are present.
        # For now, let's assume 'security_violation' key in custom metrics or attributes.
        if "security_violation" in custom_metrics or attributes.get("security_violation"):
            user_id = "unknown"
            if user_context:
                user_id = getattr(user_context, "user_id", getattr(user_context, "sub", "unknown"))
            logger.critical(f"SECURITY VIOLATION detected for User ID: {user_id}")

        # 5. Check Triggers
        await self.circuit_breaker.check_triggers(user_context)

    async def ingest_from_veritas_since(self, since: datetime) -> int:
        """
        Polls Veritas for logs since the given timestamp and processes them asynchronously.
        """
        try:
            events = await anyio.to_thread.run_sync(self.veritas_client.fetch_logs, self.config.agent_id, since)
        except Exception as e:
            logger.error(f"Failed to fetch logs from Veritas: {e}")
            return 0

        if not events:
            return 0

        count = 0
        for event in events:
            try:
                # Note: ingest_from_veritas_since doesn't have UserContext passed in.
                # Assuming batch ingestion might not have user context or it's embedded in event metadata?
                # For now, we pass None.
                await self.process_event(event)
                count += 1
            except Exception as e:
                logger.error(f"Failed to process event {event.event_id}: {e}")
                continue

        return count

    async def process_event(self, event: VeritasEvent, user_context: Optional[UserContext] = None) -> None:
        """
        Processes a single telemetry event from Veritas asynchronously.
        """
        logger.info(f"Processing event {event.event_id} for agent {event.agent_id}")

        for metric_name, value in event.metrics.items():
            if isinstance(value, (int, float)):
                await self.circuit_breaker.record_metric(metric_name, float(value), user_context)

        custom_metrics = self._extract_custom_metrics(event.input_text, event.metadata)
        for metric_name, value in custom_metrics.items():
            await self.circuit_breaker.record_metric(metric_name, value, user_context)

        combined_metadata = event.metadata.copy()
        combined_metadata.update(custom_metrics)

        conversation = {
            "input": event.input_text,
            "output": event.output_text,
            "metadata": combined_metadata,
        }

        if self.spot_checker.should_sample(combined_metadata):
            grade = await anyio.to_thread.run_sync(self.spot_checker.check_sample, conversation)
            if grade:
                await self.circuit_breaker.record_metric("faithfulness_score", grade.faithfulness_score, user_context)
                await self.circuit_breaker.record_metric(
                    "retrieval_precision_score", grade.retrieval_precision_score, user_context
                )
                await self.circuit_breaker.record_metric("safety_score", grade.safety_score, user_context)

        # 6. Drift Detection
        # Note: process_drift records metrics and calls check_triggers internally.
        await self.process_drift(event, user_context)

    async def process_drift(self, event: VeritasEvent, user_context: Optional[UserContext] = None) -> None:
        """
        Processes Drift Detection for a single event asynchronously.
        """
        logger.info(f"Processing drift for event {event.event_id}")

        groups = None
        if user_context:
            groups = getattr(user_context, "groups", getattr(user_context, "permissions", None))

        # 1. Drift Detection (Vector)
        embedding = event.metadata.get("embedding")
        if embedding and isinstance(embedding, list):
            try:
                # DB call -> likely IO (but using run_sync for safety if provider is sync)
                baselines = await anyio.to_thread.run_sync(
                    self.baseline_provider.get_baseline_vectors, event.agent_id, groups
                )
                if baselines:
                    # Calculation -> CPU bound
                    drift_score = await anyio.to_thread.run_sync(
                        DriftEngine.detect_vector_drift, baselines, [embedding]
                    )
                    await self.circuit_breaker.record_metric("vector_drift", drift_score, user_context)
            except Exception as e:
                logger.error(f"Failed to process vector drift detection: {e}")

        # 2. Drift Detection (Output Length)
        try:
            await self._process_output_drift(event, user_context)
        except Exception as e:
            logger.error(f"Failed to process output drift detection: {e}")

        # 3. Drift Detection (Relevance - Query vs Response)
        query_embedding = event.metadata.get("query_embedding")
        response_embedding = event.metadata.get("response_embedding")

        if (
            query_embedding
            and isinstance(query_embedding, list)
            and response_embedding
            and isinstance(response_embedding, list)
        ):
            try:
                relevance_drift = await anyio.to_thread.run_sync(
                    DriftEngine.compute_relevance_drift, query_embedding, response_embedding
                )
                await self.circuit_breaker.record_metric("relevance_drift", relevance_drift, user_context)
            except Exception as e:
                logger.error(f"Failed to process relevance drift detection: {e}")

        await self.circuit_breaker.check_triggers(user_context)

    def _extract_custom_metrics(self, input_text: str, metadata: Dict[str, Any]) -> Dict[str, float]:
        """
        Extracts custom metrics based on metadata flags and regex patterns.
        """
        metrics: Dict[str, float] = {}

        if metadata.get("is_refusal"):
            metrics["refusal_count"] = 1.0

        for pattern in self.config.sentiment_regex_patterns:
            try:
                if re.search(pattern, input_text, re.IGNORECASE):
                    metrics["sentiment_frustration_count"] = 1.0
                    break
            except re.error as e:
                logger.error(f"Invalid regex pattern '{pattern}' in configuration: {e}")
                continue

        return metrics

    async def _process_output_drift(self, event: VeritasEvent, user_context: Optional[UserContext] = None) -> None:
        """
        Detects drift in output length (token count) using KL Divergence.
        """
        output_length = 0.0
        if "completion_tokens" in event.metrics:
            output_length = float(event.metrics["completion_tokens"])
        elif "token_count" in event.metrics:
            output_length = float(event.metrics["token_count"])
        else:
            output_length = float(len(event.output_text.split()))

        await self.circuit_breaker.record_metric("output_length", output_length, user_context)

        groups = None
        if user_context:
            groups = getattr(user_context, "groups", getattr(user_context, "permissions", None))
        try:
            # DB call
            baseline_dist, bin_edges = await anyio.to_thread.run_sync(
                self.baseline_provider.get_baseline_output_length_distribution, event.agent_id, groups
            )
        except (AttributeError, NotImplementedError):
            return

        if not baseline_dist or not bin_edges:
            return

        recent_samples = await self.circuit_breaker.get_recent_values(
            "output_length", self.config.drift_sample_window, user_context
        )
        if not recent_samples:
            return

        # Computation
        live_dist = await anyio.to_thread.run_sync(
            DriftEngine.compute_distribution_from_samples, recent_samples, bin_edges
        )

        try:
            kl_divergence = await anyio.to_thread.run_sync(DriftEngine.compute_kl_divergence, baseline_dist, live_dist)
            await self.circuit_breaker.record_metric("output_drift_kl", kl_divergence, user_context)
        except ValueError as e:
            logger.warning(f"Skipping KL calculation due to validation error: {e}")


class TelemetryIngestor:
    """
    The Listener: Orchestrates the monitoring pipeline.
    (Sync Facade for TelemetryIngestorAsync)
    """

    def __init__(
        self,
        config: SentinelConfig,
        circuit_breaker: CircuitBreaker,
        spot_checker: SpotChecker,
        baseline_provider: BaselineProviderProtocol,
        veritas_client: VeritasClientProtocol,
    ):
        self._async = TelemetryIngestorAsync(config, circuit_breaker, spot_checker, baseline_provider, veritas_client)
        self._portal: Optional[anyio.from_thread.BlockingPortal] = None
        self._portal_cm: Optional[ContextManager[anyio.from_thread.BlockingPortal]] = None

    def __enter__(self) -> "TelemetryIngestor":
        # Start the portal to maintain the event loop and context
        self._portal_cm = anyio.from_thread.start_blocking_portal()
        self._portal = self._portal_cm.__enter__()
        # Enter the async context
        self._portal.call(self._async.__aenter__)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if self._portal:
            try:
                self._portal.call(self._async.__aexit__, exc_type, exc_val, exc_tb)
            finally:
                if self._portal_cm:
                    self._portal_cm.__exit__(exc_type, exc_val, exc_tb)
                self._portal = None
                self._portal_cm = None

    def process_otel_span(self, span: OTELSpan, user_context: Optional[UserContext] = None) -> None:
        if not self._portal:
            raise RuntimeError("TelemetryIngestor must be used within a context manager (with ... as svc:)")
        self._portal.call(self._async.process_otel_span, span, user_context)  # pragma: no cover

    def ingest_from_veritas_since(self, since: datetime) -> int:
        if not self._portal:
            raise RuntimeError("TelemetryIngestor must be used within a context manager (with ... as svc:)")
        return cast(int, self._portal.call(self._async.ingest_from_veritas_since, since))  # pragma: no cover

    def process_event(self, event: VeritasEvent, user_context: Optional[UserContext] = None) -> None:
        if not self._portal:
            raise RuntimeError("TelemetryIngestor must be used within a context manager (with ... as svc:)")
        self._portal.call(self._async.process_event, event, user_context)  # pragma: no cover

    def process_drift(self, event: VeritasEvent, user_context: Optional[UserContext] = None) -> None:
        if not self._portal:
            raise RuntimeError("TelemetryIngestor must be used within a context manager (with ... as svc:)")
        self._portal.call(self._async.process_drift, event, user_context)  # pragma: no cover
