"""
DataCleaner Environment — OpenEnv-compliant implementation.

Three tasks (Easy → Medium → Hard) cycling on successive reset() calls.
All graders are deterministic. Partial rewards accumulate per step.
"""
import os
import sys

# ── Ensure /app is on sys.path so absolute imports always resolve ─────────────
_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from models import DataCleanerAction, DataCleanerObservation
    from server.tasks import TASKS
    from server.tasks_extended import TASKS_EXTENDED, TASK_GOLD_EXTENDED
    from server.metrics import MetricsCollector
    from server.judge import JudgeClient, JUDGE_ENABLED, JUDGE_ALPHA
except ImportError:
    from ..models import DataCleanerAction, DataCleanerObservation  # type: ignore[import]
    from .tasks import TASKS  # type: ignore[import]
    from .tasks_extended import TASKS_EXTENDED, TASK_GOLD_EXTENDED  # type: ignore[import]
    from .metrics import MetricsCollector  # type: ignore[import]
    from .judge import JudgeClient, JUDGE_ENABLED, JUDGE_ALPHA  # type: ignore[import]

MAX_TASK_ID = 10

# Merge base tasks and extended tasks into a single registry
_ALL_TASKS: dict = {}
try:
    from server.tasks import TASKS as _BASE_TASKS
    from server.tasks_extended import TASKS_EXTENDED as _EXT_TASKS
    _ALL_TASKS = {**_BASE_TASKS, **_EXT_TASKS}
except Exception:
    _ALL_TASKS = {}

# ── Process-wide JudgeClient singleton (JUDGE_ENABLED gates every call) ───────
_JUDGE_CLIENT: JudgeClient = JudgeClient()

# Gold payloads used by the judge for reference (imported lazily per task)
from server.tasks import TASK1_GOLD  # noqa: E402  (after sys.path setup)


class DataCleanerEnvironment(Environment):
    """Structured Data Extraction & Cleaning Environment.

    Episode lifecycle:
        reset() → returns task observation (task cycles 1→2→3→1→...)
        step(action) → grades payload, returns partial/final reward
        state() → current episode metadata
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode_count = 0
        self._current_task_id = 1
        self._steps_in_episode = 0
        self._cumulative_score = 0.0
        self._done = False

    def reset(self) -> DataCleanerObservation:
        # Cycle tasks 1 → 2 → 3 → 1 ...
        self._episode_count += 1
        self._current_task_id = ((self._episode_count - 1) % MAX_TASK_ID) + 1
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._steps_in_episode = 0
        self._cumulative_score = 0.0
        self._done = False

        task = _ALL_TASKS.get(self._current_task_id, list(TASKS.values())[0])
        return DataCleanerObservation(
            task_id=self._current_task_id,
            task_name=task["name"],
            instruction=task["instruction"],
            input_data=task["input_data"],
            schema_hint=task.get("schema_hint"),
            step_feedback="Episode started. Submit your answer via action.payload.",
            cumulative_score=0.0,
            done=False,
            reward=0.0,
        )

    def step(self, action: DataCleanerAction) -> DataCleanerObservation:  # type: ignore[override]
        self._state.step_count += 1
        self._steps_in_episode += 1
        task = _ALL_TASKS.get(self._current_task_id, list(TASKS.values())[0])

        # Validate task_id matches
        if action.task_id != self._current_task_id:
            feedback = (
                f"⚠ task_id mismatch: action.task_id={action.task_id}, "
                f"current={self._current_task_id}. Grading anyway."
            )
        else:
            feedback = ""

        # ── Deterministic grade ────────────────────────────────────────────────
        _mc = MetricsCollector.get()
        task_name = task["name"]
        with _mc.step_timer(task_name=task_name):
            det_score, grade_feedback = task["grader"](action.payload)
        feedback = (feedback + "\n" + grade_feedback).strip()

        # ── Phase 2: LLM judge (optional hybrid reward) ────────────────────────
        # Gated by JUDGE_ENABLED env var so det-only and hybrid modes are
        # toggled without code changes (A/B test requirement).
        if JUDGE_ENABLED:
            gold_payload = self._gold_for_task(self._current_task_id)
            judge_score, judge_latency, used_fallback = _JUDGE_CLIENT.judge_sync(
                task_id=self._current_task_id,
                agent_payload=action.payload,
                gold_payload=gold_payload,
                det_score=det_score,
            )
            _mc.record_judge_latency(task_name=task_name, latency=judge_latency)
            if used_fallback:
                _mc.record_judge_fallback(task_name=task_name)
            score = JudgeClient.blend(judge_score, det_score, alpha=JUDGE_ALPHA)
            feedback += (
                f"\n[judge] score={judge_score:.2f} latency={judge_latency:.3f}s"
                f" fallback={used_fallback} alpha={JUDGE_ALPHA}"
                f" hybrid={score:.2f}"
            )
        else:
            score = det_score

        # Partial reward: score delta above previous cumulative
        reward_delta = max(0.0, score - self._cumulative_score)
        self._cumulative_score = score

        # Record reward and score observations
        _mc.record_reward(task_name=task_name, reward=reward_delta)
        _mc.record_score(task_name=task_name, score=score)

        # Episode ends when max_steps reached or perfect (clamped-max) score.
        max_steps = task["max_steps"]
        done = (self._steps_in_episode >= max_steps) or (score >= 0.99)
        self._done = done

        return DataCleanerObservation(
            task_id=self._current_task_id,
            task_name=task["name"],
            instruction=task["instruction"],
            input_data=task["input_data"],
            schema_hint=task.get("schema_hint"),
            step_feedback=feedback,
            cumulative_score=score,
            done=done,
            reward=reward_delta,
            metadata={
                "step":        self._steps_in_episode,
                "max_steps":   max_steps,
                "final_score": score if done else None,
                "judge_enabled": JUDGE_ENABLED,
            },
        )

    @staticmethod
    def _gold_for_task(task_id: int):
        """Return the gold reference payload for the judge, keyed by task_id."""
        from server.tasks import TASK1_GOLD, TASK2_GOLD_MERGED, TASK2_GOLD_CLUSTERS, TASK3_GOLD
        if task_id == 1:
            return TASK1_GOLD
        elif task_id == 2:
            return {"clusters": [list(c) for c in TASK2_GOLD_CLUSTERS],
                    "merged":   TASK2_GOLD_MERGED}
        elif task_id == 3:
            return TASK3_GOLD
        else:
            # Tasks 4-10: gold from TASK_GOLD_EXTENDED
            try:
                from server.tasks_extended import TASK_GOLD_EXTENDED
                return TASK_GOLD_EXTENDED.get(task_id, {})
            except ImportError:
                return {}

    @property
    def state(self) -> State:
        return self._state
