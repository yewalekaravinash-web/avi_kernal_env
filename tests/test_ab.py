"""
tests/test_ab.py — Phase 2 A/B Test: deterministic-only vs hybrid reward.

Protocol (matches Phase 2 spec)
--------------------------------
  Group A — 50 episodes, JUDGE_ENABLED=0  → score = det_score
  Group B — 50 episodes, JUDGE_ENABLED=1  → score = blend(mock_judge, det_score)

The mock judge returns det_score + Gaussian noise (σ=0.08) clamped to
[0.01, 0.99].  This simulates a real judge that agrees with the deterministic
grader but adds stochastic signal — the industry-standard assumption for
calibration experiments.

Outputs
-------
  Per-task mean / std / min / max for both groups.
  Pearson correlation between judge and det scores (calibration proxy).
  Recommended α using calibrate_alpha().
  Statistical assertion: hybrid mean ± 3σ overlaps det mean (sanity guard).

Run:
    pytest tests/test_ab.py -v -s          # full tabular output
    pytest tests/test_ab.py -v             # summary only
"""
from __future__ import annotations

import math
import random
import sys
import os
import statistics
from typing import Dict, List, Tuple

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from server.tasks import (
    grade_task1, grade_task2, grade_task3,
    TASK1_GOLD,
    TASK2_GOLD_CLUSTERS, TASK2_GOLD_MERGED,
    TASK3_GOLD,
    TASK1_INPUT, TASK2_RECORDS, TASK3_SOURCES, TASK3_POLICY,
)
from server.judge import JudgeClient, _clamp, calibrate_alpha, JUDGE_ALPHA

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED         = 42
N_EPISODES   = 50       # episodes per group
JUDGE_NOISE  = 0.08     # σ for mock judge Gaussian noise
ALPHA        = 0.30     # blend factor used in Group B

# ── Synthetic agent payloads for 50 episodes per task ─────────────────────────
# Each "episode" degrades the gold answer by randomly zeroing out fields,
# producing a realistic distribution of partial scores.


def _perturb_t1(rng: random.Random, corruption_rate: float) -> dict:
    """Return a Task-1 payload with `corruption_rate` fraction of fields corrupted."""
    payload = dict(TASK1_GOLD)
    fields  = list(payload.keys())
    k = max(0, int(len(fields) * corruption_rate))
    for f in rng.sample(fields, k):
        payload[f] = None  # corrupt → grade will give 0 for this field
    return payload


def _perturb_t2(rng: random.Random, corruption_rate: float) -> dict:
    """Return a Task-2 payload with some clusters randomly split."""
    clusters = [list(c) for c in TASK2_GOLD_CLUSTERS]
    merged   = [dict(m) for m in TASK2_GOLD_MERGED]
    if corruption_rate > 0.5 and clusters:
        # Split the first cluster to simulate a missed duplicate
        big = clusters[0]
        if len(big) > 1:
            clusters[0] = big[:1]
            clusters.append(big[1:])
    return {"clusters": clusters, "merged": merged}


def _perturb_t3(rng: random.Random, corruption_rate: float) -> dict:
    """Return a Task-3 payload with some field values corrupted."""
    records = [dict(r) for r in TASK3_GOLD]
    if corruption_rate > 0.3:
        for r in records:
            if rng.random() < corruption_rate:
                field = rng.choice(["phone", "email", "tier", "revenue"])
                r[field] = None
    return {"records": records}


def generate_episodes(
    task_id: int,
    n: int,
    rng: random.Random,
) -> List[Tuple[dict, float]]:
    """
    Generate n (payload, det_score) pairs for task_id with varying corruption.
    Returns list of (payload, det_score).
    """
    grader = {1: grade_task1, 2: grade_task2, 3: grade_task3}[task_id]
    perturber = {1: _perturb_t1, 2: _perturb_t2, 3: _perturb_t3}[task_id]

    results = []
    for i in range(n):
        # corruption linearly sweeps 0..0.9 across episodes for variety
        corruption = (i / max(n - 1, 1)) * 0.9
        payload    = perturber(rng, corruption)
        score, _   = grader(payload)
        results.append((payload, score))
    return results


def mock_judge_score(det_score: float, rng: random.Random) -> float:
    """Simulate an LLM judge: det + Gaussian noise (σ=JUDGE_NOISE), clamped."""
    noise = rng.gauss(0, JUDGE_NOISE)
    return _clamp(det_score + noise)


# ═════════════════════════════════════════════════════════════════════════════
# A/B test fixture — runs once for all test methods
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def ab_results() -> Dict:
    """
    Run 50-episode A/B test for all 3 tasks.
    Returns nested dict:
      {task_id: {"det": [scores], "hybrid": [scores], "judge": [scores]}}
    """
    rng = random.Random(SEED)
    data: Dict[int, Dict[str, List[float]]] = {}

    for task_id in (1, 2, 3):
        episodes = generate_episodes(task_id, N_EPISODES, rng)

        det_scores:    List[float] = []
        hybrid_scores: List[float] = []
        judge_scores:  List[float] = []

        for payload, det in episodes:
            judge = mock_judge_score(det, rng)
            hybrid = JudgeClient.blend(judge, det, alpha=ALPHA)

            det_scores.append(det)
            judge_scores.append(judge)
            hybrid_scores.append(hybrid)

        data[task_id] = {
            "det":    det_scores,
            "hybrid": hybrid_scores,
            "judge":  judge_scores,
        }

    return data


# ═════════════════════════════════════════════════════════════════════════════
# Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestABScoreDistributions:
    """Compare det-only (Group A) vs hybrid (Group B) score distributions."""

    @pytest.mark.parametrize("task_id", [1, 2, 3])
    def test_det_scores_in_valid_range(self, ab_results, task_id):
        for s in ab_results[task_id]["det"]:
            assert 0.01 <= s <= 0.99, f"Task {task_id} det score out of range: {s}"

    @pytest.mark.parametrize("task_id", [1, 2, 3])
    def test_hybrid_scores_in_valid_range(self, ab_results, task_id):
        for s in ab_results[task_id]["hybrid"]:
            assert 0.01 <= s <= 0.99, f"Task {task_id} hybrid score out of range: {s}"

    @pytest.mark.parametrize("task_id", [1, 2, 3])
    def test_hybrid_mean_within_3sigma_of_det(self, ab_results, task_id):
        """
        Industry sanity check: hybrid mean must be within 3σ of det mean.
        A larger drift indicates the judge is adversarially biased.
        """
        det_vals    = ab_results[task_id]["det"]
        hybrid_vals = ab_results[task_id]["hybrid"]
        det_mean    = statistics.mean(det_vals)
        det_std     = statistics.stdev(det_vals) if len(det_vals) > 1 else 0.01
        hybrid_mean = statistics.mean(hybrid_vals)
        assert abs(hybrid_mean - det_mean) <= 3 * det_std, (
            f"Task {task_id}: hybrid mean {hybrid_mean:.3f} deviates more than "
            f"3σ ({3*det_std:.3f}) from det mean {det_mean:.3f}"
        )

    @pytest.mark.parametrize("task_id", [1, 2, 3])
    def test_det_episode_count(self, ab_results, task_id):
        assert len(ab_results[task_id]["det"]) == N_EPISODES

    @pytest.mark.parametrize("task_id", [1, 2, 3])
    def test_hybrid_episode_count(self, ab_results, task_id):
        assert len(ab_results[task_id]["hybrid"]) == N_EPISODES


class TestAlphaCalibrationFromAB:
    """
    Use the A/B data to calibrate α and verify calibrate_alpha() produces a
    sensible recommendation.  This mirrors the week-3 calibration protocol.
    """

    def test_calibrated_alpha_in_valid_range(self, ab_results):
        """Calibrated α must be in (0, 1)."""
        pairs: List[Tuple[float, float]] = []
        for task_id in (1, 2, 3):
            j_list = ab_results[task_id]["judge"]
            d_list = ab_results[task_id]["det"]
            pairs.extend(zip(j_list, d_list))

        alpha = calibrate_alpha(pairs, steps=20, target="mse")
        assert 0.0 < alpha < 1.0, f"Calibrated alpha out of range: {alpha}"

    def test_calibrated_alpha_is_2dp(self, ab_results):
        pairs = [
            (j, d)
            for task_id in (1, 2, 3)
            for j, d in zip(
                ab_results[task_id]["judge"],
                ab_results[task_id]["det"],
            )
        ]
        alpha = calibrate_alpha(pairs, steps=20)
        assert alpha == round(alpha, 2)

    def test_score_table_printed(self, ab_results, capsys):
        """Print the score table for human review (--capture=no to see it)."""
        print("\n\n" + "=" * 68)
        print(f"  Phase 2 A/B Test — {N_EPISODES} episodes per group | α={ALPHA}")
        print("=" * 68)
        header = f"  {'Task':<30}  {'Det μ':>7}  {'Det σ':>7}  {'Hyb μ':>7}  {'Hyb σ':>7}"
        print(header)
        print("  " + "-" * 64)
        task_names = {1: "field-extraction", 2: "dedup-merge", 3: "multi-source-reconciliation"}
        for task_id in (1, 2, 3):
            det    = ab_results[task_id]["det"]
            hybrid = ab_results[task_id]["hybrid"]
            dm = statistics.mean(det);    ds = statistics.stdev(det)
            hm = statistics.mean(hybrid); hs = statistics.stdev(hybrid)
            row = (
                f"  {task_names[task_id]:<30}  "
                f"{dm:>7.3f}  {ds:>7.3f}  {hm:>7.3f}  {hs:>7.3f}"
            )
            print(row)
        print("=" * 68)

        # Calibration summary
        pairs = [
            (j, d)
            for task_id in (1, 2, 3)
            for j, d in zip(
                ab_results[task_id]["judge"],
                ab_results[task_id]["det"],
            )
        ]
        alpha_mse     = calibrate_alpha(pairs, steps=20, target="mse")
        alpha_pearson = calibrate_alpha(pairs, steps=20, target="pearson")
        print(f"\n  Calibrated α (MSE)    : {alpha_mse}")
        print(f"  Calibrated α (Pearson): {alpha_pearson}")
        print(f"  Current JUDGE_ALPHA   : {JUDGE_ALPHA}")
        print()

        # No assert needed — this test validates the pipeline runs end-to-end
        out, _ = capsys.readouterr()
        assert "A/B Test" in out  # confirms output was produced
