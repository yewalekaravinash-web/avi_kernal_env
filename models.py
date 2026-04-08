"""
Data models for the Data Cleaner Environment.
Three tasks: field extraction, deduplication, multi-source reconciliation.
"""
import json as _json
from typing import Any, Dict, List, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field, field_validator


class DataCleanerAction(Action):
    """Action: submit a cleaned/extracted data payload."""
    task_id: int = Field(..., description="Task index: 1=easy, 2=medium, 3=hard")
    payload: Dict[str, Any] = Field(..., description="Agent's answer payload")

    @field_validator("payload", mode="before")
    @classmethod
    def coerce_payload(cls, v: Any) -> Dict[str, Any]:
        """
        Robustly accept payload in multiple forms:
          1. Already a dict                             → pass through
          2. A JSON string (Playground textarea paste)  → parse, then unwrap
          3. {"payload": {...}} single-key wrapper      → unwrap inner dict

        Root cause this guards against:
          Pydantic error: 'Input should be a valid dictionary [input_type=str]'
          Triggered when the Playground UI forwards the raw textarea text as the
          payload value instead of first JSON-parsing it.
        """
        # ── Step 1: if it arrived as a raw JSON string, decode it first ──────
        if isinstance(v, str):
            try:
                v = _json.loads(v)
            except _json.JSONDecodeError as exc:
                raise ValueError(
                    f"payload must be a JSON object or a valid JSON string; "
                    f"got unparseable string: {exc}"
                ) from exc

        # ── Step 2: unwrap accidental {"payload": {...}} single-key wrapper ──
        # Happens when the Playground sample (which wraps the inner dict in a
        # "payload" key) is pasted verbatim into the Payload field.
        if isinstance(v, dict) and list(v.keys()) == ["payload"]:
            v = v["payload"]

        return v


class DataCleanerObservation(Observation):
    """Observation returned after each step."""
    task_id: int = Field(default=1, description="Current task index")
    task_name: str = Field(default="", description="Human-readable task name")
    instruction: str = Field(default="", description="What the agent must do")
    input_data: Any = Field(default=None, description="Raw data to process")
    schema_hint: Optional[Dict[str, str]] = Field(default=None, description="Target schema if applicable")
    step_feedback: str = Field(default="", description="Incremental feedback")
    cumulative_score: float = Field(default=0.0, description="Score so far [0,1]")
