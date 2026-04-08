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
except ImportError:
    from ..models import DataCleanerAction, DataCleanerObservation  # type: ignore[import]
    from .tasks import TASKS  # type: ignore[import]

MAX_TASK_ID = 3


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

        task = TASKS[self._current_task_id]
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
        task = TASKS[self._current_task_id]

        # Validate task_id matches
        if action.task_id != self._current_task_id:
            feedback = (
                f"⚠ task_id mismatch: action.task_id={action.task_id}, "
                f"current={self._current_task_id}. Grading anyway."
            )
        else:
            feedback = ""

        # Grade
        score, grade_feedback = task["grader"](action.payload)
        feedback = (feedback + "\n" + grade_feedback).strip()

        # Partial reward: score delta above previous cumulative
        reward_delta = max(0.0, score - self._cumulative_score)
        self._cumulative_score = score

        # Episode ends when max_steps reached or perfect score
        max_steps = task["max_steps"]
        done = (self._steps_in_episode >= max_steps) or (score >= 1.0)
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
                "step": self._steps_in_episode,
                "max_steps": max_steps,
                "final_score": score if done else None,
            },
        )

    @property
    def state(self) -> State:
        return self._state
