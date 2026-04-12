"""
server/metrics.py — Prometheus metrics for the DataCleaner OpenEnv server.

Exposes instrument families via prometheus_client:

  Phase 1 (deterministic grader):
    data_cleaner_step_latency_seconds  — Histogram: per-task step processing time
    data_cleaner_reward                — Histogram: per-task step reward distribution
    data_cleaner_task_score            — Histogram: per-task final episode score

  Phase 2 (LLM judge):
    data_cleaner_judge_latency_seconds — Histogram: judge LLM round-trip time
    data_cleaner_judge_fallback_total  — Counter: how often judge fell back to det

Endpoint:
  GET /metrics  →  text/plain; version=0.0.4  (standard Prometheus scrape format)

Usage inside environment.py::

    from server.metrics import MetricsCollector
    mc = MetricsCollector.get()
    with mc.step_timer(task_name="field-extraction"):
        score, feedback = grader(payload)
    mc.record_reward(task_name="field-extraction", reward=reward_delta)
    mc.record_score(task_name="field-extraction", score=score)

    # Phase 2 judge logging:
    mc.record_judge_latency(task_name="field-extraction", latency=0.45)
    mc.record_judge_fallback(task_name="field-extraction")
"""

import time
from contextlib import contextmanager
from typing import Generator

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Histogram,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    _PROM_OK = True
except ImportError:  # pragma: no cover — prometheus_client is in requirements.txt
    _PROM_OK = False

# ── Bucket boundaries ─────────────────────────────────────────────────────────
_LATENCY_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
_REWARD_BUCKETS  = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99)
_SCORE_BUCKETS   = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99)
# Phase 2 — judge latency: 0.1 s … 5 s (2 s hard timeout)
_JUDGE_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0)

_TASK_NAMES = ("field-extraction", "dedup-merge", "multi-source-reconciliation")


class MetricsCollector:
    """
    Singleton metrics collector.

    All Histogram instruments are registered against a private CollectorRegistry
    (not the global default) so multiple test instantiations never clash on the
    prometheus_client duplicate-metric guard.
    """

    _instance: "MetricsCollector | None" = None

    def __init__(self) -> None:
        if _PROM_OK:
            self._registry = CollectorRegistry()
            self._step_latency = Histogram(
                "data_cleaner_step_latency_seconds",
                "Wall-clock time (seconds) for a single grader call, by task.",
                labelnames=["task"],
                buckets=_LATENCY_BUCKETS,
                registry=self._registry,
            )
            self._reward = Histogram(
                "data_cleaner_reward",
                "Per-step reward delta returned to the agent, by task.",
                labelnames=["task"],
                buckets=_REWARD_BUCKETS,
                registry=self._registry,
            )
            self._task_score = Histogram(
                "data_cleaner_task_score",
                "Final clamped score [0.01, 0.99] after each grader call, by task.",
                labelnames=["task"],
                buckets=_SCORE_BUCKETS,
                registry=self._registry,
            )
            # ── Phase 2: judge instrumentation ────────────────────────────────
            self._judge_latency = Histogram(
                "data_cleaner_judge_latency_seconds",
                "Wall-clock time (seconds) for a single LLM judge call, by task.",
                labelnames=["task"],
                buckets=_JUDGE_LATENCY_BUCKETS,
                registry=self._registry,
            )
            self._judge_fallback = Counter(
                "data_cleaner_judge_fallback_total",
                "Number of judge calls that fell back to the deterministic score.",
                labelnames=["task"],
                registry=self._registry,
            )
        else:  # prometheus_client unavailable — all methods become no-ops
            self._registry = None
            self._step_latency = None
            self._reward = None
            self._task_score = None
            self._judge_latency = None
            self._judge_fallback = None

    # ── Singleton access ──────────────────────────────────────────────────────
    @classmethod
    def get(cls) -> "MetricsCollector":
        """Return the process-wide singleton, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Discard the singleton (test isolation only — not for production use)."""
        cls._instance = None

    # ── Instrumentation helpers ───────────────────────────────────────────────
    @contextmanager
    def step_timer(self, task_name: str) -> Generator[None, None, None]:
        """Context manager: measure and record step latency for *task_name*."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            if _PROM_OK and self._step_latency is not None:
                self._step_latency.labels(task=task_name).observe(elapsed)

    def record_reward(self, task_name: str, reward: float) -> None:
        """Record a reward delta observation for *task_name*."""
        if _PROM_OK and self._reward is not None:
            self._reward.labels(task=task_name).observe(max(0.0, float(reward)))

    def record_score(self, task_name: str, score: float) -> None:
        """Record a grader score observation for *task_name*."""
        if _PROM_OK and self._task_score is not None:
            self._task_score.labels(task=task_name).observe(float(score))

    # ── Phase 2: judge instrumentation ───────────────────────────────────────

    def record_judge_latency(self, task_name: str, latency: float) -> None:
        """Record the wall-clock time of an LLM judge call for *task_name*."""
        if _PROM_OK and self._judge_latency is not None:
            self._judge_latency.labels(task=task_name).observe(max(0.0, float(latency)))

    def record_judge_fallback(self, task_name: str) -> None:
        """Increment the fallback counter when the judge was unavailable/timed out."""
        if _PROM_OK and self._judge_fallback is not None:
            self._judge_fallback.labels(task=task_name).inc()

    # ── Prometheus text exposition ────────────────────────────────────────────
    def generate_text(self) -> tuple[str, str]:
        """
        Return (body, content_type) suitable for an HTTP response.

        Falls back to a minimal text payload when prometheus_client is absent.
        """
        if _PROM_OK and self._registry is not None:
            body = generate_latest(self._registry).decode("utf-8")
            return body, CONTENT_TYPE_LATEST
        # Degraded mode: static comment so /metrics always returns 200
        body = (
            "# prometheus_client not installed — install prometheus-client>=0.16.0\n"
            "# Metrics unavailable.\n"
        )
        return body, "text/plain; version=0.0.4; charset=utf-8"
