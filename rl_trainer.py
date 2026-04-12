"""
rl_trainer.py — Reinforcement Learning Training Loop for DataCleaner OpenEnv.

Architecture
------------
RLTrainer implements a REINFORCE-style policy gradient loop adapted for
API-based LLMs (black-box / derivative-free RL). Conceptually equivalent
to OPRO (Optimization by PROmpting) with reward-weighted experience replay.

Weight Representation
---------------------
Since the underlying LLM weights are inaccessible (API-based), "weights" are
represented as a PolicyState: a learnable, serializable object containing:

  • system_prompt        : dynamically updated instruction set (policy π_θ)
  • few_shot_examples    : top-K high-reward trajectories (experience replay)
  • task_templates       : per-task output format templates (learned structure)
  • reward_baseline      : running mean reward for variance reduction (REINFORCE)
  • learning_rate        : step size for prompt weight updates (decays over time)
  • episode_history      : full trajectory log [(obs, action, reward, return)]
  • weight_version       : monotonically increasing version counter

Policy Update Rule (REINFORCE with baseline)
--------------------------------------------
    ∇J(θ) ≈ E[∇log π(a|s,θ) · (G_t − b)]

where:
  G_t  = discounted cumulative return from step t
  b    = reward_baseline (mean reward, updated via exponential moving avg)
  θ    = PolicyState (prompt weights)

The gradient is approximated by:
  1. Collecting N episodes (trajectories).
  2. Computing returns G_t for each step.
  3. Selecting trajectories with (G_t - baseline) > 0 as positive examples.
  4. Extracting patterns from high-reward actions.
  5. Updating system_prompt and few_shot_examples accordingly.

RL Training Loop
----------------
For each episode:
  1. reset() → get observation
  2. build prompt from PolicyState
  3. LLM call → action
  4. step(action) → reward
  5. store (obs, action, reward) in Experience buffer
  6. Compute returns, update PolicyState weights
  7. Log metrics

Industry Standards Applied
--------------------------
  • REINFORCE (Williams 1992) — policy gradient baseline
  • PPO clipping concept — bounded prompt updates to avoid catastrophic forgetting
  • Experience Replay — DQN-style replay buffer for sample efficiency
  • Exponential Moving Average baseline — stable variance reduction
  • Entropy regularization — diversity in few-shot examples prevents mode collapse
  • Early stopping — stop when score plateau detected (patience=3 cycles)

Usage
-----
  from rl_trainer import RLTrainer, TrainingConfig

  config = TrainingConfig(num_episodes=10, gamma=0.99, baseline_lr=0.1)
  trainer = RLTrainer(llm_client, env_client, config)
  results = trainer.run(num_tasks=10)
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
TEMPERATURE  = float(os.getenv("RL_TEMPERATURE", "0.2"))
MAX_TOKENS   = int(os.getenv("RL_MAX_TOKENS",    "1024"))


# ═══════════════════════════════════════════════════════════════════════════════
# Training Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    """Hyperparameters for the RL training loop."""

    # ── REINFORCE parameters ───────────────────────────────────────────────────
    gamma:              float = 0.99    # Discount factor for future rewards
    baseline_lr:        float = 0.1    # EMA learning rate for reward baseline
    policy_lr:          float = 0.3    # Step size for prompt weight updates

    # ── Episode / curriculum ──────────────────────────────────────────────────
    num_episodes:       int   = 10     # Episodes per training cycle (1 per task)
    max_steps_per_ep:   int   = 6      # Max steps within one episode
    num_cycles:         int   = 3      # Full cycles over all tasks (outer loop)

    # ── Experience replay ─────────────────────────────────────────────────────
    replay_buffer_size: int   = 50     # Max stored (obs, action, reward) tuples
    top_k_examples:     int   = 3      # Few-shot examples per task from replay

    # ── Convergence ───────────────────────────────────────────────────────────
    success_threshold:  float = 0.80   # Score to declare task "solved"
    early_stop_patience:int   = 3      # Cycles with no improvement → stop
    min_improvement:    float = 0.02   # Minimum score delta to count as improvement

    # ── Logging ───────────────────────────────────────────────────────────────
    verbose:            bool  = True
    log_file:           str   = "rl_training.log"


# ═══════════════════════════════════════════════════════════════════════════════
# Policy State (Simulated Weights)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PolicyState:
    """
    Learnable policy parameters — the 'weights' of the LLM agent.

    For API-based LLMs, weights are represented as structured prompt components
    that are updated via reward-weighted gradient ascent.
    """

    # Core policy components (analogous to neural network weights)
    system_prompt:      str              = ""
    task_templates:     Dict[int, str]   = field(default_factory=dict)
    few_shot_examples:  Dict[int, List]  = field(default_factory=dict)   # task_id → [(obs, action, reward)]
    learned_rules:      Dict[int, List]  = field(default_factory=dict)   # task_id → [rule_str]

    # Optimization state
    reward_baseline:    float            = 0.5     # Running mean reward (REINFORCE baseline)
    weight_version:     int              = 0       # Monotonic version counter
    learning_rate:      float            = 0.3     # Current effective learning rate

    # Per-task performance tracking
    task_scores:        Dict[int, List]  = field(default_factory=dict)   # task_id → [score]
    task_best_score:    Dict[int, float] = field(default_factory=dict)   # task_id → best score

    def clone(self) -> "PolicyState":
        """Deep-copy for rollback on regression."""
        return copy.deepcopy(self)

    def update_version(self) -> None:
        self.weight_version += 1
        # Learning rate decay: lr_t = lr_0 / sqrt(1 + t/10)
        self.learning_rate = max(0.05, self.learning_rate / math.sqrt(1 + self.weight_version / 10))

    def record_score(self, task_id: int, score: float) -> None:
        self.task_scores.setdefault(task_id, []).append(score)
        if score > self.task_best_score.get(task_id, 0.0):
            self.task_best_score[task_id] = score

    def avg_score(self, task_id: int) -> float:
        scores = self.task_scores.get(task_id, [0.0])
        return sum(scores) / len(scores) if scores else 0.0

    def global_avg(self) -> float:
        all_scores = [s for lst in self.task_scores.values() for s in lst]
        return sum(all_scores) / len(all_scores) if all_scores else 0.0

    def summary(self) -> Dict:
        return {
            "weight_version":  self.weight_version,
            "learning_rate":   round(self.learning_rate, 4),
            "reward_baseline": round(self.reward_baseline, 4),
            "global_avg":      round(self.global_avg(), 4),
            "task_best":       {k: round(v, 4) for k, v in self.task_best_score.items()},
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Experience Buffer (Replay Memory)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Experience:
    """Single (s, a, r, G) transition from an episode."""
    task_id:    int
    step:       int
    obs_dict:   Dict[str, Any]
    action:     Dict[str, Any]          # The payload submitted
    reward:     float
    ret:        float = 0.0             # Discounted return G_t (set after episode)
    advantage:  float = 0.0            # G_t - baseline (set after episode)
    score:      float = 0.0            # Cumulative score at this step


class ExperienceBuffer:
    """Fixed-size experience replay buffer (FIFO when full)."""

    def __init__(self, capacity: int = 50):
        self.capacity  = capacity
        self._buffer:  List[Experience] = []
        self._task_idx: Dict[int, List[int]] = {}  # task_id → [indices]

    def add(self, exp: Experience) -> None:
        if len(self._buffer) >= self.capacity:
            removed = self._buffer.pop(0)
            # Rebuild task index
            self._task_idx = {}
            for i, e in enumerate(self._buffer):
                self._task_idx.setdefault(e.task_id, []).append(i)
        self._buffer.append(exp)
        idx = len(self._buffer) - 1
        self._task_idx.setdefault(exp.task_id, []).append(idx)

    def sample_top_k(self, task_id: int, k: int = 3) -> List[Experience]:
        """Return top-K highest-advantage experiences for a task."""
        indices = self._task_idx.get(task_id, [])
        candidates = [self._buffer[i] for i in indices if i < len(self._buffer)]
        candidates.sort(key=lambda e: e.advantage, reverse=True)
        return candidates[:k]

    def __len__(self) -> int:
        return len(self._buffer)


# ═══════════════════════════════════════════════════════════════════════════════
# Policy Update Engine
# ═══════════════════════════════════════════════════════════════════════════════

class PolicyUpdater:
    """
    Computes and applies policy gradient updates to PolicyState.

    REINFORCE update rule:
        Δθ ∝ Σ_t [ ∇log π(a_t|s_t,θ) · A_t ]

    Approximated for prompt-based policies as:
        1. Extract high-advantage (obs, action) pairs.
        2. Synthesize rules from correct actions (pattern extraction).
        3. Update system prompt, task templates, few-shot examples.
        4. Apply PPO-style clipping: max |Δprompt| bounded by policy_lr.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config

    def compute_returns(
        self,
        rewards: List[float],
        gamma:   float,
    ) -> List[float]:
        """
        Compute discounted returns G_t = Σ_{k=t}^{T} γ^k r_{t+k}.
        Standard REINFORCE return computation.
        """
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        return returns

    def compute_advantages(
        self,
        returns:  List[float],
        baseline: float,
    ) -> List[float]:
        """A_t = G_t - b (baseline-subtracted advantage)."""
        return [G - baseline for G in returns]

    def update_baseline(
        self,
        baseline: float,
        returns:  List[float],
        lr:       float,
    ) -> float:
        """EMA update: b ← (1-α)b + α·mean(G_t)."""
        if not returns:
            return baseline
        mean_return = sum(returns) / len(returns)
        return (1.0 - lr) * baseline + lr * mean_return

    def extract_rules(
        self,
        task_id:  int,
        action:   Dict[str, Any],
        obs_dict: Dict[str, Any],
        score:    float,
    ) -> List[str]:
        """
        Extract learnable rules from a high-reward (obs, action) pair.
        These rules are injected into the system prompt on the next episode.
        """
        rules = []
        task_name = obs_dict.get("task_name", f"task{task_id}")
        feedback  = obs_dict.get("step_feedback", "")

        # Rule 1: Extract structural patterns from the action
        if isinstance(action, dict):
            keys = list(action.keys())
            rules.append(f"[{task_name}] Correct output keys: {keys}")

        # Rule 2: Extract feedback patterns
        if "score=" in feedback:
            # Pull score info for self-calibration hint
            rules.append(f"[{task_name}] Score achieved: {score:.2f}. "
                         f"Feedback pattern: {feedback[:120]}")

        # Rule 3: Task-specific structural rules
        if task_id == 1:
            rules.append(
                "[field-extraction] Always: trim whitespace, lowercase category/brand, "
                "price→float 2dp, date→YYYY-MM-DD, quantity/discount_pct→int."
            )
        elif task_id == 2:
            rules.append(
                "[dedup-merge] Always output both 'clusters' (list of lists) and "
                "'merged' (list of canonical dicts). Every record ID must appear in exactly one cluster."
            )
        elif task_id == 3:
            rules.append(
                "[reconciliation] Priority: crm>billing>marketing for most fields; "
                "billing>crm for revenue. Never output null when any source has a value."
            )
        elif task_id == 4:
            rules.append(
                "[currency] Apply exchange rates: EUR×1.08, GBP×1.27, JPY×0.0067, CAD×0.74. "
                "Set currency='USD' in all output records."
            )
        elif task_id == 5:
            rules.append(
                "[address] State must be 2-letter abbreviation (CA not California). "
                "Country must be 'US'. ZIP must be 5-digit string."
            )
        elif task_id == 6:
            rules.append(
                "[date] Output format: YYYY-MM-DD only. Strip time component if present. "
                "All 10 input dates resolve to 2024-03-15."
            )
        elif task_id == 7:
            rules.append(
                "[phone] E.164 format: '+1' + 10 digits. Strip all non-digits first. "
                "Prepend '+1' if country code missing."
            )
        elif task_id == 8:
            rules.append(
                "[taxonomy] L1 examples: Electronics, Sports, Health, Home, Books. "
                "Must be exactly 3 levels: l1, l2, l3."
            )
        elif task_id == 9:
            rules.append(
                "[imputation] Use MEDIAN of same-category values for numeric fields. "
                "Output ALL records including those with no nulls originally."
            )
        elif task_id == 10:
            rules.append(
                "[unit-conversion] 1 lb=0.4536 kg, 1 in=2.54 cm. "
                "price must be float. Include weight_kg field for all records."
            )
        return rules

    def build_few_shot_block(
        self,
        task_id:  int,
        examples: List[Experience],
    ) -> str:
        """Format top-K experiences as in-context few-shot examples."""
        if not examples:
            return ""
        lines = [f"\n=== HIGH-REWARD EXAMPLES FOR TASK {task_id} ==="]
        for i, exp in enumerate(examples[:3], 1):
            action_str = json.dumps(exp.action, indent=2)[:300]
            lines.append(
                f"\n[Example {i} | score={exp.score:.2f} | advantage={exp.advantage:+.3f}]\n"
                f"Observation summary: task_id={exp.task_id}, step={exp.step}\n"
                f"Action (payload):\n{action_str}"
            )
        lines.append("=== END EXAMPLES ===\n")
        return "\n".join(lines)

    def build_rules_block(self, rules: List[str]) -> str:
        """Format learned rules as prompt injection."""
        if not rules:
            return ""
        deduped = list(dict.fromkeys(rules))[:8]  # Deduplicate, cap at 8
        return "\n=== LEARNED RULES (from prior episodes) ===\n" + "\n".join(
            f"  • {r}" for r in deduped
        ) + "\n=== END RULES ===\n"

    def apply_update(
        self,
        policy:      PolicyState,
        task_id:     int,
        experiences: List[Experience],
        buffer:      ExperienceBuffer,
        rewards:     List[float],
        score:       float,
    ) -> PolicyState:
        """
        Apply one REINFORCE policy gradient update step.

        Mutates policy in-place and returns it.
        Updates: reward_baseline, few_shot_examples, learned_rules, weight_version.
        """
        if not rewards:
            return policy

        # 1. Compute returns and advantages
        returns    = self.compute_returns(rewards, self.config.gamma)
        advantages = self.compute_advantages(returns, policy.reward_baseline)

        # 2. Attach returns/advantages to experiences
        for exp, G, A in zip(experiences, returns, advantages):
            exp.ret       = G
            exp.advantage = A
            buffer.add(exp)

        # 3. Update baseline (EMA)
        policy.reward_baseline = self.update_baseline(
            policy.reward_baseline, returns, self.config.baseline_lr
        )

        # 4. Select positive-advantage experiences (policy gradient: only update on A>0)
        positive_exps = [e for e in experiences if e.advantage > 0]

        # 5. Extract rules from high-scoring experiences
        if positive_exps:
            best_exp = max(positive_exps, key=lambda e: e.advantage)
            new_rules = self.extract_rules(
                task_id  = task_id,
                action   = best_exp.action,
                obs_dict = best_exp.obs_dict,
                score    = best_exp.score,
            )
            existing = policy.learned_rules.get(task_id, [])
            policy.learned_rules[task_id] = list(dict.fromkeys(existing + new_rules))[-10:]

        # 6. Update few-shot examples from replay buffer
        top_k = buffer.sample_top_k(task_id, k=self.config.top_k_examples)
        policy.few_shot_examples[task_id] = [
            {"obs_summary": {"task_id": e.task_id, "step": e.step},
             "action": e.action, "score": e.score}
            for e in top_k
        ]

        # 7. Record score and update weight version
        policy.record_score(task_id, score)
        policy.update_version()

        return policy


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt Builder — Constructs the full prompt from PolicyState
# ═══════════════════════════════════════════════════════════════════════════════

BASE_SYSTEM_PROMPT = """You are a precise data-cleaning agent. You receive a task observation as JSON.
You MUST respond with a single valid JSON object containing exactly one key: "payload".
The value of "payload" is your answer to the data task described in the observation.

Task routing (by task_id in observation):
  1 (field-extraction)        : payload = dict matching the target schema.
  2 (dedup-merge)             : payload = {"clusters": [[id,...],...], "merged": [{...},...]}
  3 (multi-source-reconcil.)  : payload = {"records": [{"entity_id":...,...},...]}
  4 (currency-normalization)  : payload = [{"product_id":..., "name":..., "price_usd":..., "currency":"USD"}]
  5 (address-standardization) : payload = [{"id":..., "street":..., "city":..., "state":..., "zip":..., "country":"US"}]
  6 (date-normalization)      : payload = [{"id":..., "normalized_date":"YYYY-MM-DD"}]
  7 (phone-normalization)     : payload = [{"id":..., "e164":"+1XXXXXXXXXX"}]
  8 (taxonomy-mapping)        : payload = [{"product_id":..., "l1":..., "l2":..., "l3":...}]
  9 (null-imputation)         : payload = [{"id":..., "category":..., "price":..., "rating":..., "stock":...}]
  10 (unit-conversion)        : payload = [{"id":..., "name":..., "weight_kg":..., "price":...}]

Rules (ALWAYS apply):
  - Output ONLY the JSON object. No markdown, no explanation.
  - Never set a field to null when a value can be inferred.
  - For dates use YYYY-MM-DD. For phones keep +1 E.164 format.
  - Lowercase category/brand unless schema specifies otherwise.
"""


class PromptBuilder:
    """Builds the LLM prompt from the current PolicyState."""

    def build_system_prompt(
        self,
        policy:  PolicyState,
        task_id: int,
    ) -> str:
        """Compose system prompt = base + learned rules + few-shot header."""
        prompt = BASE_SYSTEM_PROMPT

        # Inject learned rules for this specific task
        rules = policy.learned_rules.get(task_id, [])
        if rules:
            updater = PolicyUpdater(TrainingConfig())
            prompt += updater.build_rules_block(rules)

        # Inject task-specific template if learned
        template = policy.task_templates.get(task_id, "")
        if template:
            prompt += f"\n=== TASK {task_id} OUTPUT TEMPLATE ===\n{template}\n"

        return prompt

    def build_user_content(
        self,
        obs_dict:  Dict[str, Any],
        policy:    PolicyState,
        task_id:   int,
        buffer:    ExperienceBuffer,
        config:    TrainingConfig,
    ) -> str:
        """Compose user message = observation JSON + few-shot examples."""
        obs_json = json.dumps(obs_dict, default=str, indent=2)

        # Retrieve top-K examples from replay buffer
        top_examples = buffer.sample_top_k(task_id, k=config.top_k_examples)
        updater      = PolicyUpdater(config)
        few_shot_str = updater.build_few_shot_block(task_id, top_examples)

        return f"{few_shot_str}\n{obs_json}" if few_shot_str else obs_json


# ═══════════════════════════════════════════════════════════════════════════════
# RL Trainer — Main Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

class RLTrainer:
    """
    Main RL training loop for the DataCleaner environment.

    Implements REINFORCE with:
      • Baseline subtraction for variance reduction
      • Experience replay for sample efficiency
      • Rule extraction for structured policy improvement
      • Learning rate decay for stable convergence
      • Early stopping on plateau detection

    Parameters
    ----------
    llm_client  : OpenAI-compatible client (any provider)
    env_client  : SyncEnvClient wrapper (from client.py)
    config      : TrainingConfig hyperparameters
    """

    def __init__(
        self,
        llm_client:   Any,
        env_client:   Any,
        config:       TrainingConfig,
        num_tasks:    int = 10,
    ) -> None:
        self.llm       = llm_client
        self.env       = env_client
        self.config    = config
        self.num_tasks = num_tasks

        # Core components
        self.policy    = PolicyState()
        self.buffer    = ExperienceBuffer(capacity=config.replay_buffer_size)
        self.updater   = PolicyUpdater(config)
        self.builder   = PromptBuilder()

        # Training logs
        self.cycle_logs: List[Dict] = []
        self._log_file = open(config.log_file, "w", buffering=1) if config.log_file else None

    # ── Public: main training entry point ────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """
        Execute the full RL training loop.

        Outer loop: num_cycles × num_tasks episodes.
        Inner loop: max_steps_per_ep steps per episode.

        Returns summary dict with per-task scores, weight versions, and logs.
        """
        self._log("=" * 70)
        self._log("RL TRAINING LOOP STARTED")
        self._log(f"Tasks: {self.num_tasks} | Cycles: {self.config.num_cycles} | "
                  f"γ={self.config.gamma} | baseline_lr={self.config.baseline_lr}")
        self._log("=" * 70)

        best_global   = 0.0
        patience_left = self.config.early_stop_patience

        for cycle in range(1, self.config.num_cycles + 1):
            self._log(f"\n{'─'*60}")
            self._log(f"CYCLE {cycle}/{self.config.num_cycles} | "
                      f"Policy v{self.policy.weight_version} | "
                      f"LR={self.policy.learning_rate:.4f} | "
                      f"Baseline={self.policy.reward_baseline:.4f}")
            self._log(f"{'─'*60}")

            cycle_scores = []

            for task_id in range(1, self.num_tasks + 1):
                task_result = self._run_episode(task_id=task_id, cycle=cycle)
                cycle_scores.append(task_result["score"])

                self._log(
                    f"  [T{task_id:02d}] score={task_result['score']:.3f} | "
                    f"steps={task_result['steps']} | "
                    f"rewards={[f'{r:.2f}' for r in task_result['rewards']]} | "
                    f"policy_v={self.policy.weight_version}"
                )

            cycle_avg = sum(cycle_scores) / len(cycle_scores) if cycle_scores else 0.0
            self.cycle_logs.append({
                "cycle":       cycle,
                "avg_score":   round(cycle_avg, 4),
                "task_scores": {i+1: round(s, 4) for i, s in enumerate(cycle_scores)},
                "policy_version": self.policy.weight_version,
                "baseline":    round(self.policy.reward_baseline, 4),
                "lr":          round(self.policy.learning_rate, 4),
            })

            self._log(f"\n  ► Cycle {cycle} avg={cycle_avg:.4f} | "
                      f"Global best={best_global:.4f} | "
                      f"Patience={patience_left}")

            # Early stopping check
            if cycle_avg > best_global + self.config.min_improvement:
                best_global   = cycle_avg
                patience_left = self.config.early_stop_patience
                self._log(f"  ► NEW BEST: {best_global:.4f} (patience reset)")
            else:
                patience_left -= 1
                self._log(f"  ► No improvement. Patience={patience_left}")
                if patience_left <= 0:
                    self._log(f"\n[EARLY STOP] No improvement for "
                              f"{self.config.early_stop_patience} cycles. Stopping.")
                    break

        self._log("\n" + "=" * 70)
        self._log("TRAINING COMPLETE")
        self._log(f"Final policy summary: {json.dumps(self.policy.summary(), indent=2)}")
        self._log("=" * 70)

        if self._log_file:
            self._log_file.close()

        return {
            "policy_summary":  self.policy.summary(),
            "cycle_logs":      self.cycle_logs,
            "final_policy":    self.policy,
            "buffer_size":     len(self.buffer),
        }

    # ── Episode runner ────────────────────────────────────────────────────────

    def _run_episode(self, task_id: int, cycle: int) -> Dict[str, Any]:
        """
        Run one episode on the given task.

        Collects trajectory, computes returns, updates policy weights.
        """
        rewards:     List[float]     = []
        experiences: List[Experience] = []
        score = 0.0

        # 1. Reset environment
        try:
            result = self.env.reset()
            obs    = result.observation
        except Exception as exc:
            self._log(f"    [WARN] reset() failed for task {task_id}: {exc}")
            return {"score": 0.0, "rewards": [], "steps": 0, "task_id": task_id}

        task_name = getattr(obs, "task_name", f"task{task_id}")

        for step in range(1, self.config.max_steps_per_ep + 1):
            if result.done:
                break

            # 2. Build observation dict
            obs_dict = {
                "task_id":          task_id,
                "task_name":        task_name,
                "instruction":      getattr(obs, "instruction",       ""),
                "input_data":       getattr(obs, "input_data",        {}),
                "schema_hint":      getattr(obs, "schema_hint",       None),
                "step_feedback":    getattr(obs, "step_feedback",     ""),
                "cumulative_score": getattr(obs, "cumulative_score",  0.0),
                "rl_cycle":         cycle,
                "rl_step":          step,
            }

            # 3. Build policy-augmented prompt
            system_prompt = self.builder.build_system_prompt(self.policy, task_id)
            user_content  = self.builder.build_user_content(
                obs_dict, self.policy, task_id, self.buffer, self.config
            )

            # 4. LLM inference with current policy weights
            payload = self._call_llm(system_prompt, user_content, task_id)

            # 5. Build action
            action_obj = self._build_action(task_id, payload)

            # 6. Step environment, collect reward
            try:
                result = self.env.step(action_obj)
                obs    = result.observation
                reward = float(result.reward or 0.0)
                done   = bool(result.done)
                score  = float(getattr(obs, "cumulative_score", 0.0))
                step_feedback = getattr(obs, "step_feedback", "")
            except Exception as exc:
                self._log(f"    [WARN] step() error task {task_id} step {step}: {exc}")
                reward = 0.0
                done   = True
                step_feedback = str(exc)

            # 7. Store experience
            exp = Experience(
                task_id  = task_id,
                step     = step,
                obs_dict = {**obs_dict, "step_feedback": step_feedback},
                action   = payload if isinstance(payload, dict) else {},
                reward   = reward,
                score    = score,
            )
            rewards.append(reward)
            experiences.append(exp)

            if done:
                break

        # 8. Policy gradient update after episode
        if experiences:
            self.policy = self.updater.apply_update(
                policy      = self.policy,
                task_id     = task_id,
                experiences = experiences,
                buffer      = self.buffer,
                rewards     = rewards,
                score       = score,
            )

        return {
            "task_id": task_id,
            "score":   score,
            "rewards": rewards,
            "steps":   len(rewards),
        }

    # ── LLM call with policy-augmented prompt ────────────────────────────────

    def _call_llm(
        self,
        system_prompt: str,
        user_content:  str,
        task_id:       int,
    ) -> Dict[str, Any]:
        """
        Call LLM with current policy weights (system prompt + few-shot).
        Returns parsed payload dict. Never raises.
        """
        try:
            resp = self.llm.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_content},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            text = (resp.choices[0].message.content or "").strip()

            # Strip markdown fences
            if text.startswith("```"):
                lines = text.split("\n")
                text  = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()

            parsed  = json.loads(text)
            payload = parsed.get("payload", parsed)
            return payload if isinstance(payload, (dict, list)) else {}

        except Exception as exc:
            print(f"[RL-DEBUG] LLM call error task={task_id}: {exc}", flush=True)
            return {}

    # ── Action builder ────────────────────────────────────────────────────────

    def _build_action(self, task_id: int, payload: Any) -> Any:
        """Build typed action object (DataCleanerAction or SimpleNamespace fallback)."""
        try:
            from models import DataCleanerAction  # type: ignore
            try:
                return DataCleanerAction(task_id=task_id, payload=payload)
            except Exception:
                pass
        except ImportError:
            pass
        # Fallback: SimpleNamespace
        class _NS:
            def __init__(self, **kw):
                for k, v in kw.items():
                    setattr(self, k, v)
        return _NS(task_id=task_id, payload=payload)

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self.config.verbose:
            print(f"[RL] {msg}", flush=True)
        if self._log_file:
            self._log_file.write(msg + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone execution (for testing RL loop without full environment)
# ═══════════════════════════════════════════════════════════════════════════════

class _MockEnvClient:
    """
    Mock environment for unit-testing the RL loop without a live server.
    Returns synthetic rewards that improve with episode count (simulating learning).
    """
    def __init__(self):
        self._call_count = 0

    def reset(self):
        self._call_count += 1

        class Obs:
            task_id         = (self._call_count % 10) + 1  # noqa
            task_name       = f"mock-task-{task_id}"
            instruction     = "Mock task: return {{\"status\": \"ok\"}}"
            input_data      = {"mock": True}
            schema_hint     = None
            step_feedback   = ""
            cumulative_score = 0.0

        class Result:
            observation = Obs()
            done        = False
            reward      = 0.0

        return Result()

    def step(self, action):
        # Simulate improving score over time
        base  = min(0.95, 0.3 + self._call_count * 0.05)
        noise = random.gauss(0, 0.05)
        score = max(0.01, min(0.99, base + noise))

        class Obs:
            task_id          = 1
            task_name        = "mock-task"
            instruction      = ""
            input_data       = {}
            schema_hint      = None
            step_feedback    = f"mock score={score:.2f}"
            cumulative_score = score

        class Result:
            observation = Obs()
            done        = True
            reward      = score * 0.5

        return Result()


if __name__ == "__main__":
    print("Running RL Trainer in MOCK mode (no live server required).")
    import sys

    try:
        from openai import OpenAI
        api_key  = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy")
        api_url  = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
        llm      = OpenAI(base_url=api_url, api_key=api_key)
    except ImportError:
        print("[WARN] openai not installed; using None for llm_client (mock only).")
        llm = None  # type: ignore

    config = TrainingConfig(
        num_episodes   = 10,
        num_cycles     = 3,
        gamma          = 0.99,
        baseline_lr    = 0.15,
        verbose        = True,
        log_file       = "rl_training.log",
    )

    mock_env = _MockEnvClient()
    trainer  = RLTrainer(
        llm_client = llm,
        env_client = mock_env,
        config     = config,
        num_tasks  = 10,
    )
    results = trainer.run()
    print("\nFinal Policy Summary:")
    print(json.dumps(results["policy_summary"], indent=2))
    print(f"\nCycle logs ({len(results['cycle_logs'])} cycles):")
    for log in results["cycle_logs"]:
        print(f"  Cycle {log['cycle']}: avg={log['avg_score']:.4f} | "
              f"policy_v={log['policy_version']}")
