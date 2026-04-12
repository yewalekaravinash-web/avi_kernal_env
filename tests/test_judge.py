"""
tests/test_judge.py — Phase 2 LLM Judge Integration tests.

Coverage
--------
TestJudgePromptBuilding   (6 cases)  — _build_messages() correctness for all 3 tasks
TestJudgeBlend            (8 cases)  — JudgeClient.blend() correctness & edge cases
TestJudgeFallback         (3 cases)  — fallback paths: no httpx, no URL, no key
TestAlphaCalibration      (3 cases)  — calibrate_alpha() MSE and Pearson modes

Total: 20 handcrafted test cases — no LLM calls required.

All tests are fully offline: they mock/stub the HTTP layer so the CI never
needs a live LLM endpoint.  The judge prompt content is validated structurally
(presence of key terms, JSON schema, gold/agent fields) rather than
semantically, which is the correct contract for a unit test.

Run:
    pytest tests/test_judge.py -v
    pytest tests/ -v --cov=server
"""
from __future__ import annotations

import json
import sys
import os
from typing import Any, Dict, List, Tuple

import pytest

# ── Make project root importable ─────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from server.judge import (
    JudgeClient,
    _clamp,
    calibrate_alpha,
    JUDGE_ALPHA,
    _TASK1_JUDGE_TEMPLATE,
    _TASK2_JUDGE_TEMPLATE,
    _TASK3_JUDGE_TEMPLATE,
)
from server.tasks import (
    TASK1_GOLD,
    TASK2_GOLD_CLUSTERS,
    TASK2_GOLD_MERGED,
    TASK3_GOLD,
)


# ═════════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def client() -> JudgeClient:
    """JudgeClient with no real API credentials — safe for offline tests."""
    return JudgeClient(api_url="", api_key="", model="test-model")


@pytest.fixture
def perfect_t1() -> Dict[str, Any]:
    return dict(TASK1_GOLD)


@pytest.fixture
def wrong_t1() -> Dict[str, Any]:
    return {
        "product_id": "WRONG",
        "name": "Wrong Item",
        "price": 0.0,
        "listed_date": "1900-01-01",
        "quantity": 0,
        "category": "unknown",
        "brand": "unknown",
        "rating": 0.0,
        "discount_pct": 0,
    }


@pytest.fixture
def perfect_t2() -> Dict[str, Any]:
    return {
        "clusters": [list(c) for c in TASK2_GOLD_CLUSTERS],
        "merged":   TASK2_GOLD_MERGED,
    }


@pytest.fixture
def perfect_t3() -> Dict[str, Any]:
    return {"records": TASK3_GOLD}


# ═════════════════════════════════════════════════════════════════════════════
# TestJudgePromptBuilding  (6 cases)
# ═════════════════════════════════════════════════════════════════════════════

class TestJudgePromptBuilding:
    """
    Validate that _build_messages() produces well-formed prompts for each task.
    These tests exercise the string-formatting paths without any LLM or HTTP calls.
    """

    # Case 1 — Task 1 system message contains key schema terms
    def test_task1_system_contains_schema_fields(self, client, perfect_t1):
        system, user = client._build_messages(1, perfect_t1, TASK1_GOLD)
        for keyword in ("product_id", "price", "listed_date", "LOWERCASE"):
            assert keyword in system or keyword in user, (
                f"'{keyword}' missing from Task 1 judge prompt"
            )

    # Case 2 — Task 1 gold answer is embedded in the user message
    def test_task1_user_contains_gold_value(self, client, perfect_t1):
        _, user = client._build_messages(1, perfect_t1, TASK1_GOLD)
        assert "P-00192" in user, "GOLD product_id missing from Task 1 judge prompt"
        assert "29.99" in user,   "GOLD price missing from Task 1 judge prompt"

    # Case 3 — Task 1 wrong agent payload is embedded
    def test_task1_user_contains_agent_answer(self, client, wrong_t1):
        _, user = client._build_messages(1, wrong_t1, TASK1_GOLD)
        assert "WRONG" in user, "Agent product_id 'WRONG' missing from Task 1 prompt"

    # Case 4 — Task 2 prompt contains cluster and merge sections
    def test_task2_prompt_structure(self, client, perfect_t2):
        system, user = client._build_messages(2, perfect_t2, None)
        assert "GOLD CLUSTERS" in user,         "Task 2 prompt missing GOLD CLUSTERS section"
        assert "GOLD MERGED" in user,            "Task 2 prompt missing GOLD MERGED section"
        assert "AGENT CLUSTERS" in user,         "Task 2 prompt missing AGENT CLUSTERS section"
        assert "AGENT MERGED" in user,           "Task 2 prompt missing AGENT MERGED section"
        assert "Acme Corp" in user,              "Known company name missing from Task 2 prompt"

    # Case 5 — Task 3 prompt contains all 3 entity IDs
    def test_task3_prompt_contains_entity_ids(self, client, perfect_t3):
        _, user = client._build_messages(3, perfect_t3, None)
        for eid in ("E001", "E002", "E003"):
            assert eid in user, f"Entity ID '{eid}' missing from Task 3 judge prompt"

    # Case 6 — Task 3 prompt mentions source priority policy terms
    def test_task3_prompt_contains_policy(self, client, perfect_t3):
        system, user = client._build_messages(3, perfect_t3, None)
        for term in ("crm", "billing", "marketing", "null"):
            assert term in system or term in user, (
                f"Policy term '{term}' missing from Task 3 judge prompt"
            )


# ═════════════════════════════════════════════════════════════════════════════
# TestJudgeBlend  (8 cases)
# ═════════════════════════════════════════════════════════════════════════════

class TestJudgeBlend:
    """
    Validate JudgeClient.blend() arithmetic, clamp enforcement, and α edge cases.
    Industry contract: hybrid must always be in (0.01, 0.99) at 2 dp.
    """

    # Case 7 — α=0 means pure deterministic
    def test_alpha_zero_returns_det(self):
        result = JudgeClient.blend(judge_score=0.8, det_score=0.5, alpha=0.0)
        assert result == 0.50

    # Case 8 — α=1 means pure judge
    def test_alpha_one_returns_judge(self):
        result = JudgeClient.blend(judge_score=0.8, det_score=0.5, alpha=1.0)
        assert result == 0.80

    # Case 9 — α=0.3 default blends correctly: 0.3*0.8 + 0.7*0.5 = 0.59
    def test_default_alpha_blend(self):
        result = JudgeClient.blend(judge_score=0.8, det_score=0.5, alpha=0.3)
        assert abs(result - 0.59) < 0.005, f"Expected ~0.59, got {result}"

    # Case 10 — blend result is always clamped ≥ 0.01
    def test_blend_clamp_floor(self):
        result = JudgeClient.blend(judge_score=0.0, det_score=0.0, alpha=0.5)
        assert result >= 0.01

    # Case 11 — blend result is always clamped ≤ 0.99
    def test_blend_clamp_ceiling(self):
        result = JudgeClient.blend(judge_score=1.0, det_score=1.0, alpha=0.5)
        assert result <= 0.99

    # Case 12 — blend output has at most 2 decimal places
    def test_blend_output_is_2dp(self):
        result = JudgeClient.blend(judge_score=0.77, det_score=0.33, alpha=0.3)
        assert result == round(result, 2)

    # Case 13 — α out-of-range is silently clamped (blend called with α > 1)
    def test_blend_alpha_clamped_above_one(self):
        result = JudgeClient.blend(judge_score=0.6, det_score=0.4, alpha=5.0)
        # α clamped to 1.0 → result = judge_score = 0.6
        assert result == 0.60

    # Case 14 — equal scores → hybrid equals the same score regardless of α
    def test_blend_equal_scores_stable(self):
        for alpha in (0.0, 0.3, 0.5, 0.99):
            result = JudgeClient.blend(judge_score=0.7, det_score=0.7, alpha=alpha)
            assert result == 0.70, f"Stable blend failed at alpha={alpha}: got {result}"


# ═════════════════════════════════════════════════════════════════════════════
# TestJudgeFallback  (3 cases)
# ═════════════════════════════════════════════════════════════════════════════

class TestJudgeFallback:
    """
    Verify that judge_sync() degrades gracefully to det_score when the judge
    is unconfigured or unavailable.  No HTTP calls are made in these tests.
    """

    # Case 15 — no api_url → immediate fallback
    def test_fallback_when_no_api_url(self, perfect_t1):
        c = JudgeClient(api_url="", api_key="sk-test", model="x", timeout=2.0)
        score, latency, used_fallback = c.judge_sync(1, perfect_t1, TASK1_GOLD, det_score=0.77)
        assert used_fallback is True,    "Should fall back when api_url is empty"
        assert score == 0.77,            "Fallback should return det_score unchanged"
        assert latency == 0.0,           "Fallback latency should be 0.0 (no call made)"

    # Case 16 — no api_key → immediate fallback
    def test_fallback_when_no_api_key(self, perfect_t1):
        c = JudgeClient(api_url="http://localhost/v1", api_key="", model="x", timeout=2.0)
        score, latency, used_fallback = c.judge_sync(1, perfect_t1, TASK1_GOLD, det_score=0.55)
        assert used_fallback is True,    "Should fall back when api_key is empty"
        assert score == 0.55

    # Case 17 — fallback score is always clamped to [0.01, 0.99]
    def test_fallback_clamps_det_score(self, perfect_t1):
        c = JudgeClient(api_url="", api_key="", model="x", timeout=2.0)
        for raw_det in (0.0, 1.0, -0.5, 2.0):
            score, _, _ = c.judge_sync(1, perfect_t1, TASK1_GOLD, det_score=raw_det)
            assert 0.01 <= score <= 0.99, (
                f"Fallback returned out-of-range score {score} for det={raw_det}"
            )


# ═════════════════════════════════════════════════════════════════════════════
# TestAlphaCalibration  (3 cases)
# ═════════════════════════════════════════════════════════════════════════════

class TestAlphaCalibration:
    """
    Validate calibrate_alpha() returns sensible α values over synthetic pair sets.

    Industry standard for α calibration: when judge == det (perfect correlation),
    α→0 minimises MSE. When judge is constant noise, α→0 is also optimal.
    Non-zero α is only warranted when the judge provides signal beyond det.
    """

    # Case 18 — perfect judge/det correlation → α near 0 (MSE mode)
    def test_perfect_correlation_gives_low_alpha(self):
        # Judge exactly equals det → any α is MSE-neutral → calibrate to minimum
        pairs: List[Tuple[float, float]] = [(v, v) for v in [0.2, 0.4, 0.6, 0.8, 0.5]]
        alpha = calibrate_alpha(pairs, steps=10, target="mse")
        # MSE = 0 for any α when judge==det; function returns earliest (smallest α)
        assert 0.0 < alpha <= 0.5, f"Expected low alpha for perfect correlation, got {alpha}"

    # Case 19 — empty pairs → returns default JUDGE_ALPHA
    def test_empty_pairs_returns_default(self):
        alpha = calibrate_alpha([], steps=10)
        assert alpha == JUDGE_ALPHA, (
            f"Empty pairs should return JUDGE_ALPHA={JUDGE_ALPHA}, got {alpha}"
        )

    # Case 20 — Pearson mode returns a valid α in (0, 1)
    def test_pearson_mode_returns_valid_alpha(self):
        # Synthetic pairs where judge is slightly higher than det
        pairs = [(d + 0.1, d) for d in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]
        alpha = calibrate_alpha(pairs, steps=10, target="pearson")
        assert 0.0 < alpha < 1.0, f"Pearson calibration returned invalid alpha {alpha}"
        assert alpha == round(alpha, 2), "Alpha should be rounded to 2dp"
