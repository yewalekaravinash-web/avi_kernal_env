"""
server/judge.py — LLM-as-Judge integration for DataCleaner OpenEnv (Phase 2).

Architecture
------------
JudgeClient
  • Async primary path   : httpx.AsyncClient → chat/completions → parse JSON score
  • Synchronous path     : httpx.Client (used by environment.step() which is sync)
  • Hard timeout         : 2 s (configurable via JUDGE_TIMEOUT)
  • Deterministic fallback: returns det_score unchanged on any error / timeout
  • Alpha blend          : hybrid = α * judge_score + (1-α) * det_score
                            Default α = 0.3  (tuned via calibrate_alpha())

Environment variables
---------------------
  JUDGE_ENABLED    1|true|yes  activates judge calls (default off)
  JUDGE_API_URL    OpenAI-compatible base URL  (falls back to API_BASE_URL)
  JUDGE_API_KEY    Bearer token              (falls back to HF_TOKEN)
  JUDGE_MODEL      Model identifier          (default: Qwen/Qwen2.5-72B-Instruct)
  JUDGE_TIMEOUT    Float seconds             (default: 2.0)
  JUDGE_ALPHA      Float ∈ (0,1)             (default: 0.3)

Public API
----------
  JudgeClient.judge_sync(task_id, agent_payload, gold_payload, det_score)
      → (judge_score, latency_s, used_fallback)

  JudgeClient.blend(judge_score, det_score, alpha)
      → float  (clamped hybrid reward)

  calibrate_alpha(pairs, steps=10)
      → alpha  (minimises MSE between hybrid and det over the provided pairs)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# ── httpx ─────────────────────────────────────────────────────────────────────
try:
    import httpx
    _HTTPX_OK = True
except ImportError:
    _HTTPX_OK = False

# ── Configuration ─────────────────────────────────────────────────────────────
JUDGE_ENABLED: bool  = os.getenv("JUDGE_ENABLED", "").lower() in ("1", "true", "yes")
JUDGE_TIMEOUT: float = float(os.getenv("JUDGE_TIMEOUT", "2.0"))
JUDGE_API_URL: str   = (
    os.getenv("JUDGE_API_URL", "")
    or os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
)
JUDGE_API_KEY: str = (
    os.getenv("JUDGE_API_KEY", "")
    or os.getenv("HF_TOKEN", "")
    or os.getenv("API_KEY", "")
)
JUDGE_MODEL: str   = os.getenv("JUDGE_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
JUDGE_ALPHA: float = max(0.01, min(0.99, float(os.getenv("JUDGE_ALPHA", "0.3"))))

# ── Clamp (mirrors tasks._clamp) ──────────────────────────────────────────────
_EPS: float = 0.01

def _clamp(score: float) -> float:
    """Enforce open-(0,1) score contract at 2 dp, matching tasks._clamp."""
    return round(max(_EPS, min(float(score), 1.0 - _EPS)), 2)


# ═════════════════════════════════════════════════════════════════════════════
# Judge prompts — one per task
# ═════════════════════════════════════════════════════════════════════════════

_TASK1_JUDGE_SYSTEM = (
    "You are a strict automated data-quality evaluator. "
    "You return ONLY a JSON object with a single key 'score' (float in [0.01, 0.99]). "
    "No markdown, no explanation."
)

_TASK1_JUDGE_TEMPLATE = """\
Evaluate this field-extraction result.

TARGET SCHEMA:
  product_id : str (verbatim)
  name       : str (whitespace trimmed, original casing preserved)
  price      : float (2dp)
  listed_date: str YYYY-MM-DD
  quantity   : int
  category   : str LOWERCASE
  brand      : str LOWERCASE
  rating     : float
  discount_pct: int

GOLD ANSWER:
{gold}

AGENT ANSWER:
{agent}

Scoring rubric (each field worth 1/9 ≈ 0.111):
  + Full credit if value matches gold exactly (types and values).
  + Case-insensitive match for 'category' and 'brand'.
  + Float match within 0.001 tolerance for price/rating.
  + Integer match required for quantity / discount_pct.
  + Date must be YYYY-MM-DD.
  0 credit if field is missing, null, or wrong type.

Return: {{"score": <float>}}"""


_TASK2_JUDGE_SYSTEM = (
    "You are a strict automated data-quality evaluator for entity deduplication. "
    "You return ONLY a JSON object with a single key 'score' (float in [0.01, 0.99]). "
    "No markdown, no explanation."
)

_TASK2_JUDGE_TEMPLATE = """\
Evaluate this deduplication and merge result.

GOLD CLUSTERS (each inner list = IDs of duplicate records):
{gold_clusters}

GOLD MERGED CANONICAL RECORDS:
{gold_merged}

AGENT CLUSTERS:
{agent_clusters}

AGENT MERGED RECORDS:
{agent_merged}

Scoring rubric:
  Cluster score (50%): 0.1 per gold cluster perfectly reproduced as a set.
  Merge score   (50%):
    Per merged record: find best fuzzy match on company name (threshold ≥ 85).
    Then 0.025 per correct field (address, phone, email, company) = 0.1/record.
  Deduct 0.05 per null field where a non-null value should exist.
  Clamp final score to [0.01, 0.99].

Return: {{"score": <float>}}"""


_TASK3_JUDGE_SYSTEM = (
    "You are a strict automated data-quality evaluator for multi-source reconciliation. "
    "You return ONLY a JSON object with a single key 'score' (float in [0.01, 0.99]). "
    "No markdown, no explanation."
)

_TASK3_JUDGE_TEMPLATE = """\
Evaluate this multi-source record reconciliation.

SOURCE PRIORITY POLICY (apply field-by-field):
  name   : crm > billing > marketing (prefer shorter/cleaner)
  phone  : crm > marketing > billing (billing phone for E002 is known bad)
  email  : crm (unless null) → billing → marketing
  tier   : crm > marketing > billing
  revenue: billing > crm > marketing
  rule   : Never set a field to null when any source has a non-null value.

GOLD RECONCILED RECORDS:
{gold}

AGENT SUBMITTED RECORDS:
{agent}

Scoring rubric (15 fields total — 3 entities × 5 fields):
  + 1/15 per field matching gold exactly (case-insensitive str, int exact).
  - 0.05 per null field where a non-null value existed in some source.
  - 0.05 per field violating source priority policy.
  Clamp to [0.01, 0.99].

Return: {{"score": <float>}}"""


# ═════════════════════════════════════════════════════════════════════════════
# JudgeClient
# ═════════════════════════════════════════════════════════════════════════════

class JudgeClient:
    """
    LLM-as-Judge with synchronous and asynchronous call paths.

    The *synchronous* path (judge_sync) is used by environment.step().
    The *async* path (judge) is available for batch/pipeline callers.

    Both paths enforce self.timeout and return det_score on any failure.
    """

    def __init__(
        self,
        api_url:  str   = JUDGE_API_URL,
        api_key:  str   = JUDGE_API_KEY,
        model:    str   = JUDGE_MODEL,
        timeout:  float = JUDGE_TIMEOUT,
        alpha:    float = JUDGE_ALPHA,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model   = model
        self.timeout = timeout
        self.alpha   = alpha

    # ── Public: synchronous (used by environment.step) ────────────────────────

    def judge_sync(
        self,
        task_id:       int,
        agent_payload: Dict[str, Any],
        gold_payload:  Any,
        det_score:     float = 0.5,
    ) -> Tuple[float, float, bool]:
        """
        Score agent_payload synchronously.

        Returns
        -------
        (judge_score, latency_seconds, used_fallback)
          used_fallback=True  → judge unavailable; judge_score == det_score
        """
        if not _HTTPX_OK or not self.api_url or not self.api_key:
            return _clamp(det_score), 0.0, True

        system_msg, user_msg = self._build_messages(task_id, agent_payload, gold_payload)
        t0 = time.perf_counter()
        try:
            score = self._sync_call(system_msg, user_msg)
            latency = time.perf_counter() - t0
            return _clamp(score), round(latency, 4), False
        except Exception as exc:  # timeout, parse, network
            latency = time.perf_counter() - t0
            return _clamp(det_score), round(latency, 4), True

    # ── Public: async (for batch callers / pipelines) ─────────────────────────

    async def judge(
        self,
        task_id:       int,
        agent_payload: Dict[str, Any],
        gold_payload:  Any,
        det_score:     float = 0.5,
    ) -> Tuple[float, float, bool]:
        """
        Async version of judge_sync.  Enforces self.timeout via asyncio.wait_for.
        """
        if not _HTTPX_OK or not self.api_url or not self.api_key:
            return _clamp(det_score), 0.0, True

        system_msg, user_msg = self._build_messages(task_id, agent_payload, gold_payload)
        t0 = time.perf_counter()
        try:
            score = await asyncio.wait_for(
                self._async_call(system_msg, user_msg),
                timeout=self.timeout,
            )
            latency = time.perf_counter() - t0
            return _clamp(score), round(latency, 4), False
        except Exception:
            latency = time.perf_counter() - t0
            return _clamp(det_score), round(latency, 4), True

    # ── Public: blend ─────────────────────────────────────────────────────────

    @staticmethod
    def blend(
        judge_score: float,
        det_score:   float,
        alpha:       float = JUDGE_ALPHA,
    ) -> float:
        """
        Compute hybrid reward: α*judge + (1-α)*det, clamped to [0.01, 0.99].

        Parameters
        ----------
        judge_score : LLM judge score in [0.01, 0.99]
        det_score   : Deterministic grader score in [0.01, 0.99]
        alpha       : Weight on judge (0 = pure deterministic, 1 = pure judge)
        """
        alpha = max(0.0, min(1.0, float(alpha)))
        raw   = alpha * float(judge_score) + (1.0 - alpha) * float(det_score)
        return _clamp(raw)

    # ── Internal: prompt building ─────────────────────────────────────────────

    def _build_messages(
        self,
        task_id:       int,
        agent_payload: Dict[str, Any],
        gold_payload:  Any,
    ) -> Tuple[str, str]:
        """Return (system_message, user_message) for the given task."""
        if task_id == 1:
            system = _TASK1_JUDGE_SYSTEM
            user   = _TASK1_JUDGE_TEMPLATE.format(
                gold=json.dumps(gold_payload, indent=2, ensure_ascii=False),
                agent=json.dumps(agent_payload, indent=2, ensure_ascii=False),
            )
        elif task_id == 2:
            # Import gold data lazily to avoid circular imports
            from server.tasks import TASK2_GOLD_CLUSTERS, TASK2_GOLD_MERGED
            system = _TASK2_JUDGE_SYSTEM
            user   = _TASK2_JUDGE_TEMPLATE.format(
                gold_clusters=json.dumps(
                    [sorted(c) for c in TASK2_GOLD_CLUSTERS], indent=2
                ),
                gold_merged=json.dumps(TASK2_GOLD_MERGED, indent=2, ensure_ascii=False),
                agent_clusters=json.dumps(
                    agent_payload.get("clusters", []), indent=2
                ),
                agent_merged=json.dumps(
                    agent_payload.get("merged", []), indent=2, ensure_ascii=False
                ),
            )
        elif task_id == 3:
            from server.tasks import TASK3_GOLD
            system = _TASK3_JUDGE_SYSTEM
            user   = _TASK3_JUDGE_TEMPLATE.format(
                gold=json.dumps(TASK3_GOLD, indent=2, ensure_ascii=False),
                agent=json.dumps(
                    agent_payload.get("records", []), indent=2, ensure_ascii=False
                ),
            )
        elif task_id == 4:
            system = _TASK4_JUDGE_SYSTEM
            user   = _TASK4_JUDGE_TEMPLATE.format(
                gold=json.dumps(gold_payload, indent=2, ensure_ascii=False),
                agent=json.dumps(agent_payload, indent=2, ensure_ascii=False),
            )
        elif task_id == 5:
            system = _TASK5_JUDGE_SYSTEM
            user   = _TASK5_JUDGE_TEMPLATE.format(
                gold=json.dumps(gold_payload, indent=2, ensure_ascii=False),
                agent=json.dumps(agent_payload, indent=2, ensure_ascii=False),
            )
        elif task_id == 6:
            system = _TASK6_JUDGE_SYSTEM
            user   = _TASK6_JUDGE_TEMPLATE.format(
                gold=json.dumps(gold_payload, indent=2, ensure_ascii=False),
                agent=json.dumps(agent_payload, indent=2, ensure_ascii=False),
            )
        elif task_id == 7:
            system = _TASK7_JUDGE_SYSTEM
            user   = _TASK7_JUDGE_TEMPLATE.format(
                gold=json.dumps(gold_payload, indent=2, ensure_ascii=False),
                agent=json.dumps(agent_payload, indent=2, ensure_ascii=False),
            )
        elif task_id == 8:
            system = _TASK8_JUDGE_SYSTEM
            user   = _TASK8_JUDGE_TEMPLATE.format(
                gold=json.dumps(gold_payload, indent=2, ensure_ascii=False),
                agent=json.dumps(agent_payload, indent=2, ensure_ascii=False),
            )
        elif task_id == 9:
            system = _TASK9_JUDGE_SYSTEM
            user   = _TASK9_JUDGE_TEMPLATE.format(
                gold=json.dumps(gold_payload, indent=2, ensure_ascii=False),
                agent=json.dumps(agent_payload, indent=2, ensure_ascii=False),
            )
        else:  # task 10
            system = _TASK10_JUDGE_SYSTEM
            user   = _TASK10_JUDGE_TEMPLATE.format(
                gold=json.dumps(gold_payload, indent=2, ensure_ascii=False),
                agent=json.dumps(agent_payload, indent=2, ensure_ascii=False),
            )
        return system, user

    # ── Internal: HTTP calls ──────────────────────────────────────────────────

    def _request_body(self, system_msg: str, user_msg: str) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            "max_tokens":  64,
            "temperature": 0.0,
        }

    @staticmethod
    def _parse_response(data: Dict[str, Any]) -> float:
        text = data["choices"][0]["message"]["content"].strip()
        # Strip optional markdown fences
        if text.startswith("```"):
            lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()
        parsed = json.loads(text)
        return float(parsed["score"])

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _sync_call(self, system_msg: str, user_msg: str) -> float:
        """Blocking HTTP POST with self.timeout enforced by httpx."""
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.api_url}/chat/completions",
                headers=self._headers(),
                json=self._request_body(system_msg, user_msg),
            )
            resp.raise_for_status()
            return self._parse_response(resp.json())

    async def _async_call(self, system_msg: str, user_msg: str) -> float:
        """Non-blocking HTTP POST with self.timeout enforced by httpx."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.api_url}/chat/completions",
                headers=self._headers(),
                json=self._request_body(system_msg, user_msg),
            )
            resp.raise_for_status()
            return self._parse_response(resp.json())


# ═════════════════════════════════════════════════════════════════════════════
# Alpha calibration
# ═════════════════════════════════════════════════════════════════════════════

def calibrate_alpha(
    pairs:  List[Tuple[float, float]],
    steps:  int = 20,
    target: str = "mse",
) -> float:
    """
    Choose α that minimises |hybrid - det| MSE over the provided (judge, det) pairs.

    Rationale: when judge and det are perfectly correlated we want α=0 (trust det).
    When they diverge, a non-zero α allows the judge signal to expand the reward
    signal space.  MSE minimisation keeps the hybrid close to the calibrated det.

    Parameters
    ----------
    pairs  : list of (judge_score, det_score) tuples from a calibration run
    steps  : number of α candidates to sweep in (0, 1)
    target : 'mse' (default) or 'pearson' (maximise correlation between hybrid & det)

    Returns
    -------
    best_alpha : float in (0, 1) rounded to 2 dp
    """
    if not pairs:
        return JUDGE_ALPHA

    alphas = [round(i / steps, 2) for i in range(1, steps)]  # 0.05 … 0.95

    if target == "pearson":
        import statistics
        best_alpha = JUDGE_ALPHA
        best_corr  = -2.0
        j_vals = [p[0] for p in pairs]
        d_vals = [p[1] for p in pairs]
        for a in alphas:
            h_vals = [a * j + (1 - a) * d for j, d in zip(j_vals, d_vals)]
            try:
                corr = statistics.correlation(h_vals, d_vals)
            except Exception:
                corr = 0.0
            if corr > best_corr:
                best_corr  = corr
                best_alpha = a
        return best_alpha

    # default: MSE
    best_alpha = JUDGE_ALPHA
    best_mse   = float("inf")
    for a in alphas:
        mse = sum(
            (a * j + (1 - a) * d - d) ** 2
            for j, d in pairs
        ) / len(pairs)
        if mse < best_mse:
            best_mse   = mse
            best_alpha = a
    return best_alpha
