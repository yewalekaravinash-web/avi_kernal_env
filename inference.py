"""
Inference Script — DataCleaner OpenEnv Environment
====================================================
MANDATORY ENV VARS:
  API_BASE_URL      LLM endpoint      (default: https://router.huggingface.co/v1)
  MODEL_NAME        Model identifier  (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN          API / HF key      (inject at runtime: docker run -e HF_TOKEN=hf_...)
  LOCAL_IMAGE_NAME  Docker image name (if using from_docker_image())

STDOUT FORMAT (strictly enforced — deviation breaks evaluation scoring):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import json
import os
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ── openenv-core imports (correct public API — no submodule paths) ─────────────
try:
    from openenv.core import EnvClient, SyncEnvClient                     # noqa: F401
    from openenv.core.client_types import StepResult
    from openenv.core.env_server.types import Observation, Action
    _OPENENV_OK = True
except ImportError as _oe_err:
    _OPENENV_OK = False
    _OPENENV_ERR = str(_oe_err)

# ── DataCleaner models (two fallback paths) ───────────────────────────────────
try:
    from data_cleaner_env.models import DataCleanerAction, DataCleanerObservation
except ImportError:
    try:
        from models import DataCleanerAction, DataCleanerObservation
    except ImportError:
        DataCleanerAction = None
        DataCleanerObservation = None

# ── Configuration ─────────────────────────────────────────────────────────────
IMAGE_NAME        = os.getenv("LOCAL_IMAGE_NAME", "")
API_KEY           = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL      = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME        = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
OPENENV_URL       = os.getenv("OPENENV_SERVER_URL", "http://localhost:8000")
BENCHMARK         = "data_cleaner_env"
MAX_STEPS         = 6
TEMPERATURE       = 0.2
MAX_TOKENS        = 1024
SUCCESS_THRESHOLD = 0.6
SERVER_RETRY_MAX  = 10         # attempts to wait for server readiness
SERVER_RETRY_WAIT = 3.0        # seconds between retries


# ════════════════════════════════════════════════════════════════════════════════
# Concrete EnvClient — EnvClient is abstract; _step_payload + _parse_result
# must be implemented before it can be instantiated.
# ════════════════════════════════════════════════════════════════════════════════

class DataCleanerEnvClient(EnvClient):
    """Concrete client for the DataCleaner environment server."""

    def _step_payload(self, action: Any) -> Dict[str, Any]:
        """Serialize action → wire dict for the server."""
        if isinstance(action, dict):
            return action
        # DataCleanerAction pydantic model
        return {"task_id": action.task_id, "payload": action.payload}

    def _parse_result(self, payload: Dict[str, Any]) -> "StepResult":
        """Deserialize server response → StepResult[DataCleanerObservation]."""
        obs_raw  = payload.get("observation", {})
        done     = bool(payload.get("done", False))
        reward   = payload.get("reward")

        # Build observation — use the pydantic model when available,
        # fall back to a plain namespace object so the rest of the script
        # still works even if the model import failed.
        if DataCleanerObservation is not None:
            try:
                obs = DataCleanerObservation(
                    task_id         = obs_raw.get("task_id", 1),
                    task_name       = obs_raw.get("task_name", ""),
                    instruction     = obs_raw.get("instruction", ""),
                    input_data      = obs_raw.get("input_data", {}),
                    schema_hint     = obs_raw.get("schema_hint"),
                    step_feedback   = obs_raw.get("step_feedback", ""),
                    cumulative_score= float(obs_raw.get("cumulative_score", 0.0)),
                    done            = done,
                    reward          = reward,
                    metadata        = obs_raw.get("metadata", {}),
                )
            except Exception:
                obs = _SimpleNamespace(**obs_raw, done=done, reward=reward)
        else:
            obs = _SimpleNamespace(**obs_raw, done=done, reward=reward)

        return StepResult(observation=obs, reward=reward, done=done)


class _SimpleNamespace:
    """Minimal stand-in when DataCleanerObservation cannot be imported."""
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


# ════════════════════════════════════════════════════════════════════════════════
# Structured stdout log helpers (field order strictly enforced)
# ════════════════════════════════════════════════════════════════════════════════

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rstr = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rstr}",
        flush=True,
    )


# ════════════════════════════════════════════════════════════════════════════════
# System prompt
# ════════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = textwrap.dedent("""
You are a precise data-cleaning agent. You receive a task observation as JSON.
You MUST respond with a single valid JSON object containing exactly one key: "payload".
The value of "payload" is your answer to the data task described in the observation.

Task 1 (field-extraction): payload = dict matching the target schema.
Task 2 (dedup-merge):      payload = {"clusters": [[id,...],...], "merged": [{...},...]}
Task 3 (multi-source-reconciliation): payload = {"records": [{"entity_id":...,...},...]}

Rules:
- Output ONLY the JSON object. No markdown, no explanation.
- Never set a field to null when a value can be inferred.
- For dates use YYYY-MM-DD. For phones keep digits with dashes.
""").strip()


# ════════════════════════════════════════════════════════════════════════════════
# LLM call  (OpenAI client — required by submission rules)
# ════════════════════════════════════════════════════════════════════════════════

def call_llm(client: OpenAI, observation_json: str) -> Dict[str, Any]:
    """Call LLM and return parsed payload dict. Never raises."""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": observation_json},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (resp.choices[0].message.content or "").strip()

        # Strip markdown fences if the model wrapped its JSON
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(
                l for l in lines
                if not l.strip().startswith("```")
            ).strip()

        parsed = json.loads(text)
        return parsed.get("payload", parsed)

    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        return {}


# ════════════════════════════════════════════════════════════════════════════════
# Server readiness probe  (retries before failing)
# ════════════════════════════════════════════════════════════════════════════════

def wait_for_server(url: str, retries: int = SERVER_RETRY_MAX,
                    wait: float = SERVER_RETRY_WAIT) -> bool:
    """Return True once /health responds 200, False after all retries."""
    import urllib.request
    import urllib.error
    health_url = url.rstrip("/") + "/health"
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(health_url, timeout=5) as r:
                if r.status == 200:
                    print(f"[DEBUG] Server ready at {url} (attempt {attempt})", flush=True)
                    return True
        except Exception as exc:
            print(f"[DEBUG] Server not ready yet ({attempt}/{retries}): {exc}", flush=True)
        time.sleep(wait)
    return False


# ════════════════════════════════════════════════════════════════════════════════
# Episode runner  (fully synchronous via SyncEnvClient wrapper)
# ════════════════════════════════════════════════════════════════════════════════

def run_task(llm_client: OpenAI, env_client, task_num: int) -> Dict[str, Any]:
    """Run one task episode. Always emits [START] … [STEP]* … [END]."""
    task_name  = f"task{task_num}"
    task_id    = task_num
    rewards: List[float] = []
    steps_taken = 0
    score  = 0.0
    success = False

    try:
        result = env_client.reset()
        obs    = result.observation

        task_name = getattr(obs, "task_name", task_name)
        task_id   = getattr(obs, "task_id",   task_id)

        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            obs_dict = {
                "task_id":         task_id,
                "task_name":       task_name,
                "instruction":     getattr(obs, "instruction",      ""),
                "input_data":      getattr(obs, "input_data",       {}),
                "schema_hint":     getattr(obs, "schema_hint",      None),
                "step_feedback":   getattr(obs, "step_feedback",    ""),
                "cumulative_score":getattr(obs, "cumulative_score", 0.0),
            }
            obs_json = json.dumps(obs_dict, default=str)

            payload = call_llm(llm_client, obs_json)

            action_data = {"task_id": task_id, "payload": payload}
            error: Optional[str] = None

            try:
                result = env_client.step(action_data)
                obs    = result.observation
                reward = float(result.reward or 0.0)
                done   = bool(result.done)
                score  = float(getattr(obs, "cumulative_score", 0.0))
            except Exception as exc:
                error  = str(exc)[:120]
                reward = 0.0
                done   = True
                print(f"[DEBUG] step error: {exc}", flush=True)

            rewards.append(reward)
            steps_taken = step
            action_summary = (
                f"payload_keys={list(payload.keys())}"
                if isinstance(payload, dict) else "payload=list"
            )
            log_step(step=step, action=action_summary,
                     reward=reward, done=done, error=error)

            if done:
                break

        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        # Unexpected error mid-episode — log_start may or may not have fired;
        # [END] is always emitted in the finally block below.
        print(f"[DEBUG] run_task unhandled: {exc}", flush=True)
        if not rewards:                     # ensure [START] was emitted
            log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_name": task_name, "score": score,
            "rewards": rewards, "steps": steps_taken}


# ════════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Guard: openenv-core must be importable ────────────────────────────────
    if not _OPENENV_OK:
        print(f"[DEBUG] FATAL: openenv-core not installed: {_OPENENV_ERR}", flush=True)
        # Emit mandatory [END] before exiting so evaluator has something to parse
        log_end(success=False, steps=0, score=0.0, rewards=[])
        sys.exit(1)

    # ── Guard: HF_TOKEN warning ───────────────────────────────────────────────
    if not API_KEY:
        print(
            "[DEBUG] WARNING: HF_TOKEN / API_KEY not set. "
            "LLM calls will fail. Pass -e HF_TOKEN=hf_... to docker run.",
            flush=True,
        )

    # ── Build LLM client (OpenAI-compatible, required by rules) ──────────────
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")

    # ── Build environment client ──────────────────────────────────────────────
    if IMAGE_NAME:
        print(f"[DEBUG] Launching env from docker image: {IMAGE_NAME}", flush=True)
        import asyncio
        async def _launch():
            return await DataCleanerEnvClient.from_docker_image(IMAGE_NAME)
        async_client = asyncio.run(_launch())
    else:
        # Wait for the server to be reachable before connecting
        if not wait_for_server(OPENENV_URL):
            print(
                f"[DEBUG] FATAL: server at {OPENENV_URL} did not become ready. "
                "Ensure the env container is running and OPENENV_SERVER_URL is correct.",
                flush=True,
            )
            log_end(success=False, steps=0, score=0.0, rewards=[])
            sys.exit(1)

        print(f"[DEBUG] Connecting to server at {OPENENV_URL}", flush=True)
        async_client = DataCleanerEnvClient(base_url=OPENENV_URL)

    # Use the synchronous wrapper — no asyncio in the hot path
    env_client = async_client.sync()

    all_scores: List[float] = []
    try:
        with env_client:                    # guarantees WebSocket cleanup
            for task_num in range(1, 4):    # tasks 1, 2, 3
                result = run_task(llm_client, env_client, task_num)
                all_scores.append(result["score"])
                print(
                    f"[DEBUG] Task {task_num} ({result['task_name']}) "
                    f"score: {result['score']:.3f}",
                    flush=True,
                )
    except Exception as exc:
        print(f"[DEBUG] main loop error: {exc}", flush=True)

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"[DEBUG] Average score across {len(all_scores)} tasks: {avg:.3f}", flush=True)


if __name__ == "__main__":
    main()
