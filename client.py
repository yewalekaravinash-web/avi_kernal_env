"""
DataCleaner Environment Client.

Thin EnvClient wrapper for the DataCleaner OpenEnv environment.
Used by external callers who want a typed Python client rather
than raw HTTP/WebSocket calls.
"""
from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from models import DataCleanerAction, DataCleanerObservation
except ImportError:
    from .models import DataCleanerAction, DataCleanerObservation  # type: ignore[import]


class DataCleanerEnvClient(
    EnvClient[DataCleanerAction, DataCleanerObservation, State]
):
    """
    Client for the DataCleaner environment server.

    Maintains a persistent WebSocket connection to the environment server.
    Each instance has its own dedicated session.

    Example::

        with DataCleanerEnvClient(base_url="http://localhost:8000") as client:
            result = client.reset()
            result = client.step(
                DataCleanerAction(task_id=1, payload={"product_id": "P-001", ...})
            )
            print(result.observation.step_feedback)
    """

    def _step_payload(self, action) -> Dict:
        """
        Build the wire payload for a step request.

        Accepts both a DataCleanerAction object (normal path) and a plain dict
        (fallback path used when inference.py cannot import the models).
        Attribute access on a plain dict raises AttributeError, which causes
        env_client.step() to throw, the exception handler in run_task to fire,
        and every task to score 0.000 with steps=1. This guard prevents that.
        """
        if isinstance(action, dict):
            return {"task_id": action.get("task_id"), "payload": action.get("payload", {})}
        return {"task_id": action.task_id, "payload": action.payload}

    def _parse_result(self, payload: Dict) -> StepResult[DataCleanerObservation]:
        obs_raw = payload.get("observation", {})
        done = bool(payload.get("done", False))
        reward = payload.get("reward")
        obs = DataCleanerObservation(
            task_id=obs_raw.get("task_id", 1),
            task_name=obs_raw.get("task_name", ""),
            instruction=obs_raw.get("instruction", ""),
            input_data=obs_raw.get("input_data", {}),
            schema_hint=obs_raw.get("schema_hint"),
            step_feedback=obs_raw.get("step_feedback", ""),
            cumulative_score=float(obs_raw.get("cumulative_score", 0.0)),
            done=done,
            reward=reward,
            metadata=obs_raw.get("metadata", {}),
        )
        return StepResult(observation=obs, reward=reward, done=done)

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
