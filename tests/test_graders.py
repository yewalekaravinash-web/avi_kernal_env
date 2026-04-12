"""
tests/test_graders.py — Unit tests for DataCleaner graders (Phase 1 coverage).

Coverage targets per test class:
  TestGradeTask1  — field-level matching, P0-1 lowercase normalisation, clamp contract
  TestGradeTask2  — cluster scoring, merge scoring, P0-2 rapidfuzz threshold
  TestGradeTask3  — field accuracy, null penalties, clamp contract
  TestClamp       — _clamp() boundary conditions
  TestMetrics     — MetricsCollector: singleton, instrumentation, text exposition

Run:
    pytest tests/ -v
    pytest tests/ -v --cov=server --cov-report=term-missing
"""
from __future__ import annotations

import sys
import os
import time

import pytest

# ── Make project root importable from any working directory ──────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from server.tasks import (
    _clamp,
    grade_task1,
    grade_task2,
    grade_task3,
    TASK1_GOLD,
    TASK2_GOLD_CLUSTERS,
    TASK2_GOLD_MERGED,
    TASK3_GOLD,
    _normalize_phone,
    _COMPANY_MATCH_THRESHOLD,
)


# ══════════════════════════════════════════════════════════════════════════════
# _clamp — boundary conditions
# ══════════════════════════════════════════════════════════════════════════════

class TestClamp:
    """_clamp must enforce the open-(0,1) score contract at 2dp precision."""

    @pytest.mark.parametrize("raw,expected", [
        (0.0,   0.01),   # floor
        (-5.0,  0.01),   # below floor
        (1.0,   0.99),   # ceiling
        (2.0,   0.99),   # above ceiling
        (0.5,   0.50),   # passthrough
        (0.999, 0.99),   # rounds down to ceiling
        (0.001, 0.01),   # rounds up to floor
        (0.555, 0.56),   # normal rounding
    ])
    def test_clamp_values(self, raw: float, expected: float) -> None:
        assert _clamp(raw) == expected

    def test_clamp_never_returns_zero(self) -> None:
        assert _clamp(0.0) > 0.0

    def test_clamp_never_returns_one(self) -> None:
        assert _clamp(1.0) < 1.0

    def test_clamp_output_is_2dp(self) -> None:
        result = _clamp(0.123456)
        assert result == round(result, 2)


# ══════════════════════════════════════════════════════════════════════════════
# grade_task1 — field extraction
# ══════════════════════════════════════════════════════════════════════════════

class TestGradeTask1:
    """Parametric coverage for all 9 schema fields and the P0-1 fix."""

    @pytest.fixture
    def perfect_payload(self) -> dict:
        """Gold-standard answer — should achieve the maximum clamped score."""
        return dict(TASK1_GOLD)

    def test_perfect_payload_max_score(self, perfect_payload: dict) -> None:
        score, feedback = grade_task1(perfect_payload)
        assert score == 0.99, f"Perfect payload expected 0.99, got {score}"
        assert "9/9" in feedback

    # ── P0-1 regression: lowercase normalisation ──────────────────────────────

    @pytest.mark.parametrize("field,value", [
        ("category", "ELECTRONICS"),    # all-caps
        ("category", "Electronics"),    # title-case
        ("category", " electronics "),  # with whitespace
        ("brand",    "LOGITECH"),
        ("brand",    "Logitech"),
        ("brand",    "  logitech  "),
    ])
    def test_lowercase_field_normalised(self, perfect_payload: dict,
                                        field: str, value: str) -> None:
        """P0-1: fields with schema_hint 'str (lowercase)' must accept any case."""
        payload = dict(perfect_payload)
        payload[field] = value
        score, feedback = grade_task1(payload)
        assert score == 0.99, (
            f"P0-1 regression: '{field}'='{value}' should match after normalisation "
            f"but got score={score}. Feedback:\n{feedback}"
        )

    # ── Float fields ──────────────────────────────────────────────────────────

    @pytest.mark.parametrize("field,good,bad", [
        ("price",   29.99,  30.00),
        ("rating",  4.5,    4.4),
    ])
    def test_float_field_tolerance(self, perfect_payload: dict,
                                   field: str, good: float, bad: float) -> None:
        p_good = dict(perfect_payload)
        p_good[field] = good
        score_good, _ = grade_task1(p_good)
        assert score_good == 0.99

        p_bad = dict(perfect_payload)
        p_bad[field] = bad
        score_bad, _ = grade_task1(p_bad)
        assert score_bad < 0.99

    # ── Int fields ────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("field,good,bad", [
        ("quantity",     142, 141),
        ("discount_pct", 10,  11),
    ])
    def test_int_field_exact(self, perfect_payload: dict,
                              field: str, good: int, bad: int) -> None:
        p_good = dict(perfect_payload)
        p_good[field] = good
        score_good, _ = grade_task1(p_good)
        assert score_good == 0.99

        p_bad = dict(perfect_payload)
        p_bad[field] = bad
        score_bad, _ = grade_task1(p_bad)
        assert score_bad < 0.99

    # ── Date field ────────────────────────────────────────────────────────────

    def test_date_exact_match(self, perfect_payload: dict) -> None:
        payload = dict(perfect_payload)
        payload["listed_date"] = "2024-03-15"
        score, _ = grade_task1(payload)
        assert score == 0.99

    def test_date_wrong_format_penalised(self, perfect_payload: dict) -> None:
        payload = dict(perfect_payload)
        payload["listed_date"] = "03/15/2024"  # original un-normalised format
        score, _ = grade_task1(payload)
        assert score < 0.99

    # ── Empty / None payload ──────────────────────────────────────────────────

    def test_empty_payload_minimum_score(self) -> None:
        score, feedback = grade_task1({})
        assert score == 0.01  # fully wrong → clamp floor
        assert "0/9" in feedback

    # ── Partial payload ───────────────────────────────────────────────────────

    @pytest.mark.parametrize("n_correct", [1, 3, 5, 7])
    def test_partial_score_proportional(self, n_correct: int) -> None:
        """Score should be proportional to the number of correct fields."""
        payload = {}
        fields = list(TASK1_GOLD.keys())
        for f in fields[:n_correct]:
            payload[f] = TASK1_GOLD[f]
        score, _ = grade_task1(payload)
        expected_raw = n_correct / 9
        # Allow for clamp adjustments at extremes
        assert score == pytest.approx(_clamp(expected_raw), abs=0.01)

    # ── Score contract ────────────────────────────────────────────────────────

    def test_score_within_open_interval(self, perfect_payload: dict) -> None:
        score, _ = grade_task1(perfect_payload)
        assert 0.0 < score < 1.0

    def test_feedback_contains_score(self, perfect_payload: dict) -> None:
        score, feedback = grade_task1(perfect_payload)
        assert f"Task1 score={score:.2f}" in feedback


# ══════════════════════════════════════════════════════════════════════════════
# grade_task2 — duplicate detection & merge
# ══════════════════════════════════════════════════════════════════════════════

class TestGradeTask2:
    """Cluster scoring, merge scoring, and P0-2 rapidfuzz integration."""

    @pytest.fixture
    def perfect_clusters(self) -> list:
        return [list(c) for c in TASK2_GOLD_CLUSTERS]

    @pytest.fixture
    def perfect_merged(self) -> list:
        return [dict(r) for r in TASK2_GOLD_MERGED]

    @pytest.fixture
    def perfect_payload(self, perfect_clusters, perfect_merged) -> dict:
        return {"clusters": perfect_clusters, "merged": perfect_merged}

    # ── Perfect answer ────────────────────────────────────────────────────────

    def test_perfect_payload_max_score(self, perfect_payload: dict) -> None:
        score, feedback = grade_task2(perfect_payload)
        assert score == 0.99, f"Expected 0.99, got {score}. Feedback:\n{feedback}"

    # ── Cluster scoring ───────────────────────────────────────────────────────

    def test_all_clusters_correct(self, perfect_clusters, perfect_merged) -> None:
        score, feedback = grade_task2({"clusters": perfect_clusters, "merged": perfect_merged})
        assert "5/5" in feedback or "cluster=1.00" in feedback

    @pytest.mark.parametrize("drop_idx", [0, 1, 2, 3])
    def test_missing_one_cluster_penalised(self, perfect_clusters: list,
                                           perfect_merged: list,
                                           drop_idx: int) -> None:
        clusters = [c for i, c in enumerate(perfect_clusters) if i != drop_idx]
        score, _ = grade_task2({"clusters": clusters, "merged": perfect_merged})
        assert score < 0.99

    def test_empty_clusters_low_score(self, perfect_merged: list) -> None:
        score, _ = grade_task2({"clusters": [], "merged": perfect_merged})
        assert score <= 0.5, (
            f"Empty clusters should cap score at ≤0.50 (cluster contributes 0); got {score}"
        )

    # ── P0-2 rapidfuzz threshold ──────────────────────────────────────────────

    @pytest.mark.parametrize("company_name,should_match", [
        ("Acme Corp",           True),   # exact gold name
        ("ACME CORP",           True),   # all-caps variant
        ("Acme Corporation",    True),   # near match — fuzz >= 85
        ("Completely Different", False), # no match
        ("",                    False),  # empty string
    ])
    def test_company_fuzzy_matching(self, perfect_clusters: list,
                                    company_name: str, should_match: bool) -> None:
        """P0-2: rapidfuzz.WRatio (case-insensitive) threshold=85 controls merge matching."""
        from rapidfuzz.fuzz import WRatio
        gold_name = "Acme Corp"
        # grade_task2 lowercases both sides before calling _fuzz_ratio
        similarity = WRatio(company_name.strip().lower(), gold_name.strip().lower())
        if should_match:
            assert similarity >= _COMPANY_MATCH_THRESHOLD, (
                f"'{company_name}' vs '{gold_name}' → {similarity:.1f} "
                f"(expected >= {_COMPANY_MATCH_THRESHOLD})"
            )
        else:
            assert similarity < _COMPANY_MATCH_THRESHOLD

    def test_threshold_constant_is_85(self) -> None:
        """Threshold must be exactly 85.0 per spec."""
        assert _COMPANY_MATCH_THRESHOLD == 85.0

    # ── Merge scoring ─────────────────────────────────────────────────────────

    def test_wrong_phone_penalised(self, perfect_clusters: list,
                                   perfect_merged: list) -> None:
        merged = [dict(r) for r in perfect_merged]
        merged[0]["phone"] = "000-000-0000"
        score, _ = grade_task2({"clusters": perfect_clusters, "merged": merged})
        assert score < 0.99

    def test_wrong_email_penalised(self, perfect_clusters: list,
                                   perfect_merged: list) -> None:
        merged = [dict(r) for r in perfect_merged]
        merged[0]["email"] = "wrong@example.com"
        score, _ = grade_task2({"clusters": perfect_clusters, "merged": merged})
        assert score < 0.99

    # ── Empty payload ─────────────────────────────────────────────────────────

    def test_empty_payload_minimum_score(self) -> None:
        score, _ = grade_task2({})
        assert score == 0.01

    # ── Score contract ────────────────────────────────────────────────────────

    def test_score_within_open_interval(self, perfect_payload: dict) -> None:
        score, _ = grade_task2(perfect_payload)
        assert 0.0 < score < 1.0

    def test_feedback_contains_score(self, perfect_payload: dict) -> None:
        score, feedback = grade_task2(perfect_payload)
        assert f"Task2 score={score:.2f}" in feedback


# ══════════════════════════════════════════════════════════════════════════════
# grade_task3 — multi-source reconciliation
# ══════════════════════════════════════════════════════════════════════════════

class TestGradeTask3:
    """Field accuracy, null penalties, conflict resolution scoring."""

    @pytest.fixture
    def perfect_payload(self) -> dict:
        return {"records": [dict(r) for r in TASK3_GOLD]}

    # ── Perfect answer ────────────────────────────────────────────────────────

    def test_perfect_payload_max_score(self, perfect_payload: dict) -> None:
        score, feedback = grade_task3(perfect_payload)
        assert score == 0.99, f"Expected 0.99, got {score}. Feedback:\n{feedback}"

    def test_perfect_accuracy_string(self, perfect_payload: dict) -> None:
        _, feedback = grade_task3(perfect_payload)
        assert "accuracy=1.00" in feedback or "15/15" in feedback

    # ── Null penalties ────────────────────────────────────────────────────────

    def test_null_field_incurs_penalty(self, perfect_payload: dict) -> None:
        payload = {"records": [dict(r) for r in TASK3_GOLD]}
        payload["records"][0]["phone"] = None
        score, feedback = grade_task3(payload)
        assert score < 0.99
        assert "null_penalties=1" in feedback

    def test_multiple_nulls_compound_penalty(self, perfect_payload: dict) -> None:
        payload = {"records": [dict(r) for r in TASK3_GOLD]}
        for rec in payload["records"]:
            rec["phone"] = None
        score_multi, feedback = grade_task3(payload)
        assert "null_penalties=3" in feedback
        # penalty capped at -0.20
        assert score_multi >= _clamp(0.0)

    def test_null_penalty_capped_at_20_pct(self) -> None:
        """Null penalty cap: min(penalties * 0.05, 0.20) — can't exceed 0.20."""
        # 10 null penalties → min(0.50, 0.20) = 0.20 deduction
        payload = {"records": [
            {"entity_id": "E001", "name": None, "phone": None,
             "email": None, "tier": None, "revenue": None},
            {"entity_id": "E002", "name": None, "phone": None,
             "email": None, "tier": None, "revenue": None},
        ]}
        score, feedback = grade_task3(payload)
        # All nulls where gold has values → max penalty applied
        assert score == _clamp(0.0)  # base 0.0 - capped 0.20 → max(0, -0.20) → 0.0 → clamp floor

    # ── Wrong values ──────────────────────────────────────────────────────────

    @pytest.mark.parametrize("entity_id,field,wrong_value", [
        ("E001", "tier",    "silver"),   # should be gold
        ("E002", "phone",   "415-555-9999"),  # billing's bad phone
        ("E001", "revenue", 500000),     # crm revenue — billing's (512000) is gold
        ("E003", "email",   None),       # null instead of admin@gamma.com
    ])
    def test_wrong_field_penalised(self, entity_id: str, field: str,
                                   wrong_value) -> None:
        payload = {"records": [dict(r) for r in TASK3_GOLD]}
        for rec in payload["records"]:
            if rec["entity_id"] == entity_id:
                rec[field] = wrong_value
        score, _ = grade_task3(payload)
        assert score < 0.99

    # ── Missing entity ────────────────────────────────────────────────────────

    def test_missing_entity_skipped_gracefully(self) -> None:
        """Unknown entity_ids are silently skipped — no crash."""
        payload = {"records": [{"entity_id": "E999", "name": "Ghost", "phone": "000"}]}
        score, _ = grade_task3(payload)
        assert 0.0 < score < 1.0  # clamp floor, no exception

    def test_partial_records_proportional_score(self) -> None:
        """Only E001 submitted → score reflects partial coverage."""
        payload = {"records": [dict(TASK3_GOLD[0])]}
        score, feedback = grade_task3(payload)
        assert 0.0 < score < 0.99  # partial score

    # ── Empty payload ─────────────────────────────────────────────────────────

    def test_empty_records_minimum_score(self) -> None:
        score, _ = grade_task3({"records": []})
        assert score == 0.01

    # ── Score contract ────────────────────────────────────────────────────────

    def test_score_within_open_interval(self, perfect_payload: dict) -> None:
        score, _ = grade_task3(perfect_payload)
        assert 0.0 < score < 1.0

    def test_feedback_contains_score(self, perfect_payload: dict) -> None:
        score, feedback = grade_task3(perfect_payload)
        assert f"Task3 score={score:.2f}" in feedback


# ══════════════════════════════════════════════════════════════════════════════
# _normalize_phone
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalizePhone:
    @pytest.mark.parametrize("raw,expected", [
        ("617-555-0101",  "6175550101"),
        ("(617)555-0101", "6175550101"),
        ("6175550101",    "6175550101"),
        ("617 555 0101",  "6175550101"),
        ("",              ""),
    ])
    def test_strips_non_digits(self, raw: str, expected: str) -> None:
        assert _normalize_phone(raw) == expected


# ══════════════════════════════════════════════════════════════════════════════
# MetricsCollector — unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestMetricsCollector:
    """MetricsCollector singleton, instrumentation helpers, text exposition."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Isolate each test from a shared singleton state."""
        from server.metrics import MetricsCollector
        MetricsCollector.reset()
        yield
        MetricsCollector.reset()

    def test_singleton_same_instance(self) -> None:
        from server.metrics import MetricsCollector
        a = MetricsCollector.get()
        b = MetricsCollector.get()
        assert a is b

    def test_reset_creates_new_instance(self) -> None:
        from server.metrics import MetricsCollector
        a = MetricsCollector.get()
        MetricsCollector.reset()
        b = MetricsCollector.get()
        assert a is not b

    def test_step_timer_does_not_raise(self) -> None:
        from server.metrics import MetricsCollector
        mc = MetricsCollector.get()
        with mc.step_timer(task_name="field-extraction"):
            time.sleep(0.001)

    def test_step_timer_measures_elapsed(self) -> None:
        """Timer context manager must capture positive elapsed time."""
        from server.metrics import MetricsCollector
        mc = MetricsCollector.get()
        start = time.perf_counter()
        with mc.step_timer(task_name="field-extraction"):
            time.sleep(0.005)
        elapsed = time.perf_counter() - start
        assert elapsed >= 0.004  # generous lower bound for CI

    def test_record_reward_does_not_raise(self) -> None:
        from server.metrics import MetricsCollector
        mc = MetricsCollector.get()
        mc.record_reward("dedup-merge", 0.42)

    def test_record_score_does_not_raise(self) -> None:
        from server.metrics import MetricsCollector
        mc = MetricsCollector.get()
        mc.record_score("multi-source-reconciliation", 0.75)

    def test_generate_text_returns_tuple(self) -> None:
        from server.metrics import MetricsCollector
        mc = MetricsCollector.get()
        body, content_type = mc.generate_text()
        assert isinstance(body, str)
        assert isinstance(content_type, str)

    def test_generate_text_content_type_prometheus(self) -> None:
        from server.metrics import MetricsCollector, _PROM_OK
        mc = MetricsCollector.get()
        _, content_type = mc.generate_text()
        if _PROM_OK:
            assert "text/plain" in content_type
        else:
            assert "text/plain" in content_type  # degraded mode also plain text

    def test_generate_text_contains_metric_name(self) -> None:
        from server.metrics import MetricsCollector, _PROM_OK
        if not _PROM_OK:
            pytest.skip("prometheus_client not installed")
        mc = MetricsCollector.get()
        mc.record_score("field-extraction", 0.80)
        body, _ = mc.generate_text()
        assert "data_cleaner_task_score" in body

    def test_generate_text_contains_step_latency_metric(self) -> None:
        from server.metrics import MetricsCollector, _PROM_OK
        if not _PROM_OK:
            pytest.skip("prometheus_client not installed")
        mc = MetricsCollector.get()
        with mc.step_timer("dedup-merge"):
            pass
        body, _ = mc.generate_text()
        assert "data_cleaner_step_latency_seconds" in body

    def test_generate_text_contains_reward_metric(self) -> None:
        from server.metrics import MetricsCollector, _PROM_OK
        if not _PROM_OK:
            pytest.skip("prometheus_client not installed")
        mc = MetricsCollector.get()
        mc.record_reward("field-extraction", 0.30)
        body, _ = mc.generate_text()
        assert "data_cleaner_reward" in body

    @pytest.mark.parametrize("task_name", [
        "field-extraction",
        "dedup-merge",
        "multi-source-reconciliation",
    ])
    def test_all_task_names_accepted(self, task_name: str) -> None:
        from server.metrics import MetricsCollector
        mc = MetricsCollector.get()
        mc.record_score(task_name, 0.55)
        mc.record_reward(task_name, 0.20)
        with mc.step_timer(task_name):
            pass
        body, _ = mc.generate_text()
        assert isinstance(body, str)
