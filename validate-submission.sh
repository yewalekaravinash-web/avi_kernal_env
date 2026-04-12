#!/usr/bin/env bash
#
# validate-submission.sh — OpenEnv Submission Validator
#
# Checks that your HF Space is live, Docker image builds, and openenv validate passes.
#
# Prerequisites:
#   - Docker:       https://docs.docker.com/get-docker/
#   - openenv-core: pip install openenv-core
#   - curl (usually pre-installed)
#
# Arguments:
#   ping_url   Your HuggingFace Space URL. Two accepted formats:
#                Runtime URL:  https://owner-space-name.hf.space
#                Web UI URL:   https://huggingface.co/spaces/owner/space_name
#              The script auto-converts the web UI format to the runtime format.
#   repo_dir   Path to your repo (default: current directory)
#
# Examples:
#   ./validate-submission.sh https://ayewalekar2000-avi-kernal-env.hf.space
#   ./validate-submission.sh https://huggingface.co/spaces/ayewalekar2000/avi_kernal_env
#   ./validate-submission.sh https://huggingface.co/spaces/ayewalekar2000/avi_kernal_env ./my-repo
#

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout &>/dev/null; then
    timeout "$secs" "$@"
  elif command -v gtimeout &>/dev/null; then
    gtimeout "$secs" "$@"
  else
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    local watcher=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    kill "$watcher" 2>/dev/null
    wait "$watcher" 2>/dev/null
    return $rc
  fi
}

portable_mktemp() {
  local prefix="${1:-validate}"
  mktemp "${TMPDIR:-/tmp}/${prefix}-XXXXXX" 2>/dev/null || mktemp
}

CLEANUP_FILES=()
cleanup() { rm -f "${CLEANUP_FILES[@]+"${CLEANUP_FILES[@]}"}"; }
trap cleanup EXIT

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  printf "\n"
  printf "  ping_url   Your HuggingFace Space URL. Two accepted formats:\n"
  printf "               Runtime URL:  https://owner-space-name.hf.space\n"
  printf "               Web UI URL:   https://huggingface.co/spaces/owner/space_name\n"
  printf "  repo_dir   Path to your repo (default: current directory)\n"
  exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf "Error: directory '%s' not found\n" "${2:-.}"
  exit 1
fi

# ── Auto-convert huggingface.co/spaces/<owner>/<space> → <owner>-<space>.hf.space
# HuggingFace replaces underscores with hyphens in the subdomain of the runtime URL.
# Example: huggingface.co/spaces/ayewalekar2000/avi_kernal_env
#       →  ayewalekar2000-avi-kernal-env.hf.space
if printf '%s' "$PING_URL" | grep -qE '^https?://huggingface\.co/spaces/'; then
  RAW_PATH="${PING_URL#*huggingface.co/spaces/}"   # owner/space_name[/...]
  RAW_PATH="${RAW_PATH%%\?*}"                       # strip query string
  RAW_PATH="${RAW_PATH%/}"                          # strip trailing slash
  OWNER="${RAW_PATH%%/*}"
  SPACE="${RAW_PATH#*/}"
  # Underscores → hyphens (HF subdomain convention)
  OWNER_SLUG="${OWNER//_/-}"
  SPACE_SLUG="${SPACE//_/-}"
  CONVERTED="https://${OWNER_SLUG}-${SPACE_SLUG}.hf.space"
  printf "${YELLOW}Note:${NC} Detected huggingface.co/spaces URL.\n"
  printf "      Auto-converted to runtime URL: ${BOLD}%s${NC}\n\n" "$CONVERTED"
  PING_URL="$CONVERTED"
fi

PING_URL="${PING_URL%/}"
export PING_URL
PASS=0

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; PASS=$((PASS + 1)); }
fail() { log "${RED}FAILED${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
stop_at() {
  printf "\n"
  printf "${RED}${BOLD}Validation stopped at %s.${NC} Fix the above before continuing.\n" "$1"
  exit 1
}

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${BOLD}  OpenEnv Submission Validator${NC}\n"
printf "${BOLD}========================================${NC}\n"
log "Repo:     $REPO_DIR"
log "Ping URL: $PING_URL"
printf "\n"

log "${BOLD}Step 1/3: Pinging HF Space${NC} ($PING_URL/reset) ..."

CURL_OUTPUT=$(portable_mktemp "validate-curl")
CLEANUP_FILES+=("$CURL_OUTPUT")
HTTP_CODE=$(curl -s -o "$CURL_OUTPUT" -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30 2>"$CURL_OUTPUT" || printf "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space is live and responds to /reset"
elif [ "$HTTP_CODE" = "000" ]; then
  fail "HF Space not reachable (connection failed or timed out)"
  hint "Check your network connection and that the Space is running."
  hint "Try: curl -s -o /dev/null -w '%%{http_code}' -X POST $PING_URL/reset"
  stop_at "Step 1"
else
  fail "HF Space /reset returned HTTP $HTTP_CODE (expected 200)"
  hint "Make sure your Space is running and the URL is correct."
  hint "Runtime URL format: https://<owner>-<space-name>.hf.space"
  hint "Try opening $PING_URL in your browser first."
  stop_at "Step 1"
fi

log "${BOLD}Step 2/3: Running docker build${NC} ..."

if ! command -v docker &>/dev/null; then
  fail "docker command not found"
  hint "Install Docker: https://docs.docker.com/get-docker/"
  stop_at "Step 2"
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found in repo root or server/ directory"
  stop_at "Step 2"
fi

log "  Found Dockerfile in $DOCKER_CONTEXT"

BUILD_OK=false
BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  printf "%s\n" "$BUILD_OUTPUT" | tail -20
  stop_at "Step 2"
fi

log "${BOLD}Step 3/3: Running openenv validate${NC} ..."

if ! command -v openenv &>/dev/null; then
  fail "openenv command not found"
  hint "Install it: pip install openenv-core"
  stop_at "Step 3"
fi

VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && openenv validate 2>&1) && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUTPUT" ] && log "  $VALIDATE_OUTPUT"
else
  fail "openenv validate failed"
  printf "%s\n" "$VALIDATE_OUTPUT"
  stop_at "Step 3"
fi

log "${BOLD}Step 4/7: Phase 1 — code integrity checks${NC} ..."

PHASE1_OK=true

# P0-1: .strip().lower() for 'str (lowercase)' schema fields
if grep -q "lowercase.*in.*schema_type" "$REPO_DIR/server/tasks.py" 2>/dev/null && \
   grep -q "strip().lower()" "$REPO_DIR/server/tasks.py" 2>/dev/null; then
  pass "P0-1: grade_task1 applies .strip().lower() for 'str (lowercase)' fields"
else
  fail "P0-1: grade_task1 missing .strip().lower() normalisation for lowercase schema fields"
  hint "In grade_task1, add: elif 'lowercase' in schema_type: match = str(agent_val).strip().lower() == str(gold_val).strip().lower()"
  PHASE1_OK=false
fi

# P0-2: rapidfuzz replaces _company_sim
if grep -q "def _company_sim" "$REPO_DIR/server/tasks.py" 2>/dev/null; then
  fail "P0-2: _company_sim() still present — must be replaced with rapidfuzz.fuzz.WRatio()"
  hint "Remove _company_sim and import: from rapidfuzz.fuzz import WRatio as _fuzz_ratio"
  PHASE1_OK=false
else
  pass "P0-2: _company_sim() removed"
fi
if grep -q "WRatio" "$REPO_DIR/server/tasks.py" 2>/dev/null && \
   grep -q "85" "$REPO_DIR/server/tasks.py" 2>/dev/null; then
  pass "P0-2: rapidfuzz.fuzz.WRatio with threshold >= 85 present"
else
  fail "P0-2: rapidfuzz.fuzz.WRatio / threshold 85 not found in server/tasks.py"
  PHASE1_OK=false
fi

# P0-3: no duplicate DataCleanerEnvClient in inference.py
if grep -q "class DataCleanerEnvClient(EnvClient):" "$REPO_DIR/inference.py" 2>/dev/null; then
  fail "P0-3: Duplicate DataCleanerEnvClient class still present in inference.py"
  hint "Remove the class definition and add: from client import DataCleanerEnvClient"
  PHASE1_OK=false
else
  pass "P0-3: No duplicate DataCleanerEnvClient in inference.py"
fi
if grep -q "from client import DataCleanerEnvClient" "$REPO_DIR/inference.py" 2>/dev/null; then
  pass "P0-3: DataCleanerEnvClient correctly imported from client.py"
else
  fail "P0-3: 'from client import DataCleanerEnvClient' missing in inference.py"
  PHASE1_OK=false
fi

# Phase 1: prometheus-client dependency
if grep -q "prometheus-client" "$REPO_DIR/server/requirements.txt" 2>/dev/null; then
  pass "Phase 1: prometheus-client present in server/requirements.txt"
else
  fail "Phase 1: prometheus-client MISSING from server/requirements.txt"
  hint "Add: prometheus-client>=0.16.0"
  PHASE1_OK=false
fi

# Phase 1: server/metrics.py exists
if [ -f "$REPO_DIR/server/metrics.py" ]; then
  pass "Phase 1: server/metrics.py exists"
else
  fail "Phase 1: server/metrics.py MISSING"
  PHASE1_OK=false
fi

# Phase 1: /metrics endpoint in app.py
if grep -q 'path == "/metrics"' "$REPO_DIR/server/app.py" 2>/dev/null || \
   grep -q '"/metrics"' "$REPO_DIR/server/app.py" 2>/dev/null; then
  pass "Phase 1: /metrics endpoint present in server/app.py"
else
  fail "Phase 1: /metrics endpoint MISSING from server/app.py"
  PHASE1_OK=false
fi

# Phase 1: test file exists
if [ -f "$REPO_DIR/tests/test_graders.py" ]; then
  pass "Phase 1: tests/test_graders.py exists"
else
  fail "Phase 1: tests/test_graders.py MISSING"
  PHASE1_OK=false
fi

# RC-3: openai import must be guarded — bare import crashes before main() runs
# A bare `from openai import OpenAI` exits before any [START]/[END] line is
# emitted; the evaluator scores the entire run as 0.
if grep -q "^from openai import OpenAI" "$REPO_DIR/inference.py" 2>/dev/null; then
  fail "RC-3: bare 'from openai import OpenAI' at module level in inference.py"
  hint "Wrap in: try: from openai import OpenAI; _OPENAI_OK=True  except ImportError: _OPENAI_OK=False"
  PHASE1_OK=false
else
  pass "RC-3: openai import is guarded (not bare module-level)"
fi
if grep -q "_OPENAI_OK" "$REPO_DIR/inference.py" 2>/dev/null; then
  pass "RC-3: _OPENAI_OK guard present in inference.py"
else
  fail "RC-3: _OPENAI_OK guard MISSING — main() will not emit [END] on missing openai"
  hint "Add _OPENAI_OK flag and guard at top of main()"
  PHASE1_OK=false
fi

# RC-4: rapidfuzz must appear in server/requirements.txt so it is installed
# in the Docker image. Without it, tasks.py fails to import and the server
# returns HTTP 500 on every /step request.
if grep -q "^rapidfuzz" "$REPO_DIR/server/requirements.txt" 2>/dev/null; then
  pass "RC-4: rapidfuzz present in server/requirements.txt"
else
  fail "RC-4: rapidfuzz MISSING from server/requirements.txt"
  hint "Add: rapidfuzz>=3.0.0"
  PHASE1_OK=false
fi

[ "$PHASE1_OK" = false ] && stop_at "Step 4 (Phase 1 integrity)"

log "${BOLD}Step 4b/7: Phase 2 — LLM Judge code integrity checks${NC} ..."

PHASE2_OK=true

# Phase 2: server/judge.py exists
if [ -f "$REPO_DIR/server/judge.py" ]; then
  pass "Phase 2: server/judge.py exists"
else
  fail "Phase 2: server/judge.py MISSING — implement JudgeClient class"
  PHASE2_OK=false
fi

# Phase 2: JudgeClient class
if grep -q "class JudgeClient" "$REPO_DIR/server/judge.py" 2>/dev/null; then
  pass "Phase 2: JudgeClient class present in server/judge.py"
else
  fail "Phase 2: JudgeClient class MISSING"
  PHASE2_OK=false
fi

# Phase 2: 2s timeout
if grep -q "JUDGE_TIMEOUT\|timeout.*2" "$REPO_DIR/server/judge.py" 2>/dev/null; then
  pass "Phase 2: 2s timeout (JUDGE_TIMEOUT) configured in JudgeClient"
else
  fail "Phase 2: 2s timeout MISSING from server/judge.py"
  PHASE2_OK=false
fi

# Phase 2: deterministic fallback
if grep -q "used_fallback\|fallback" "$REPO_DIR/server/judge.py" 2>/dev/null; then
  pass "Phase 2: deterministic fallback implemented in JudgeClient"
else
  fail "Phase 2: deterministic fallback path MISSING from server/judge.py"
  PHASE2_OK=false
fi

# Phase 2: alpha blend
if grep -q "def blend\|JUDGE_ALPHA" "$REPO_DIR/server/judge.py" 2>/dev/null; then
  pass "Phase 2: alpha blend factor (JUDGE_ALPHA) present"
else
  fail "Phase 2: alpha blend factor MISSING from server/judge.py"
  PHASE2_OK=false
fi

# Phase 2: JUDGE_ENABLED in environment.py
if grep -q "JUDGE_ENABLED" "$REPO_DIR/server/environment.py" 2>/dev/null; then
  pass "Phase 2: JUDGE_ENABLED env var gating present in server/environment.py"
else
  fail "Phase 2: JUDGE_ENABLED not found in server/environment.py"
  PHASE2_OK=false
fi

# Phase 2: judge latency in metrics.py
if grep -q "judge_latency\|data_cleaner_judge" "$REPO_DIR/server/metrics.py" 2>/dev/null; then
  pass "Phase 2: judge latency metric present in server/metrics.py"
else
  fail "Phase 2: judge latency metric MISSING from server/metrics.py"
  PHASE2_OK=false
fi

# Phase 2: httpx in requirements
if grep -q "^httpx" "$REPO_DIR/server/requirements.txt" 2>/dev/null; then
  pass "Phase 2: httpx present in server/requirements.txt"
else
  fail "Phase 2: httpx MISSING from server/requirements.txt"
  hint "Add: httpx>=0.25.0"
  PHASE2_OK=false
fi

# Phase 2: test files
if [ -f "$REPO_DIR/tests/test_judge.py" ]; then
  pass "Phase 2: tests/test_judge.py exists (20 handcrafted cases)"
else
  fail "Phase 2: tests/test_judge.py MISSING"
  PHASE2_OK=false
fi
if [ -f "$REPO_DIR/tests/test_ab.py" ]; then
  pass "Phase 2: tests/test_ab.py exists (A/B 50 det vs 50 hybrid episodes)"
else
  fail "Phase 2: tests/test_ab.py MISSING"
  PHASE2_OK=false
fi

# Phase 2: calibrate_alpha()
if grep -q "def calibrate_alpha" "$REPO_DIR/server/judge.py" 2>/dev/null; then
  pass "Phase 2: calibrate_alpha() implemented for judge-vs-deterministic calibration"
else
  fail "Phase 2: calibrate_alpha() MISSING from server/judge.py"
  PHASE2_OK=false
fi

[ "$PHASE2_OK" = false ] && stop_at "Step 4b (Phase 2 integrity)"

log "${BOLD}Step 4c/8: Phase 3 — Extended tasks (4-10) code integrity checks${NC} ..."

PHASE3_OK=true

# Phase 3: server/tasks_extended.py exists
if [ -f "$REPO_DIR/server/tasks_extended.py" ]; then
  pass "Phase 3: server/tasks_extended.py exists (Tasks 4-10)"
else
  fail "Phase 3: server/tasks_extended.py MISSING — implement tasks 4-10"
  hint "Create server/tasks_extended.py with grade_task4..grade_task10 and TASKS_EXTENDED dict"
  PHASE3_OK=false
fi

# Phase 3: grade functions for tasks 4-10
for TN in 4 5 6 7 8 9 10; do
  if grep -q "def grade_task${TN}" "$REPO_DIR/server/tasks_extended.py" 2>/dev/null; then
    pass "Phase 3: grade_task${TN}() defined in server/tasks_extended.py"
  else
    fail "Phase 3: grade_task${TN}() MISSING from server/tasks_extended.py"
    PHASE3_OK=false
  fi
done

# Phase 3: TASKS_EXTENDED registry
if grep -q "^TASKS_EXTENDED" "$REPO_DIR/server/tasks_extended.py" 2>/dev/null; then
  pass "Phase 3: TASKS_EXTENDED registry dict defined"
else
  fail "Phase 3: TASKS_EXTENDED registry MISSING from server/tasks_extended.py"
  hint "Add: TASKS_EXTENDED = {4: {...}, 5: {...}, ..., 10: {...}}"
  PHASE3_OK=false
fi

# Phase 3: TASK_GOLD_EXTENDED
if grep -q "^TASK_GOLD_EXTENDED" "$REPO_DIR/server/tasks_extended.py" 2>/dev/null; then
  pass "Phase 3: TASK_GOLD_EXTENDED dict defined"
else
  fail "Phase 3: TASK_GOLD_EXTENDED MISSING from server/tasks_extended.py"
  hint "Add: TASK_GOLD_EXTENDED = {4: TASK4_GOLD, ..., 10: TASK10_GOLD}"
  PHASE3_OK=false
fi

# Phase 3: MAX_TASK_ID = 10 in environment.py
if grep -q "MAX_TASK_ID.*=.*10" "$REPO_DIR/server/environment.py" 2>/dev/null; then
  pass "Phase 3: MAX_TASK_ID = 10 in server/environment.py (cycles tasks 1→10)"
else
  fail "Phase 3: MAX_TASK_ID = 10 MISSING from server/environment.py"
  hint "Set MAX_TASK_ID = 10 so the environment cycles through all extended tasks"
  PHASE3_OK=false
fi

# Phase 3: environment.py imports tasks_extended
if grep -q "tasks_extended" "$REPO_DIR/server/environment.py" 2>/dev/null; then
  pass "Phase 3: server/environment.py imports from server/tasks_extended.py"
else
  fail "Phase 3: tasks_extended import MISSING in server/environment.py"
  hint "Add: from server.tasks_extended import TASKS_EXTENDED, TASK_GOLD_EXTENDED"
  PHASE3_OK=false
fi

# Phase 3: gold grader direct verification
if REPO_DIR="$REPO_DIR" python3 - <<'PYEOF' 2>/dev/null
import sys, os
sys.path.insert(0, os.environ.get("REPO_DIR", "."))
try:
    from server.tasks_extended import (
        grade_task4, TASK4_GOLD, grade_task5, TASK5_GOLD,
        grade_task6, TASK6_GOLD, grade_task7, TASK7_GOLD,
        grade_task8, TASK8_GOLD, grade_task9, TASK9_GOLD,
        grade_task10, TASK10_GOLD,
    )
    failed = []
    for tid, grader, gold in [
        (4, grade_task4, TASK4_GOLD), (5, grade_task5, TASK5_GOLD),
        (6, grade_task6, TASK6_GOLD), (7, grade_task7, TASK7_GOLD),
        (8, grade_task8, TASK8_GOLD), (9, grade_task9, TASK9_GOLD),
        (10, grade_task10, TASK10_GOLD),
    ]:
        score, _ = grader(gold)
        if score < 0.95:
            failed.append((tid, score))
    sys.exit(1 if failed else 0)
except Exception as e:
    print(f"error: {e}", file=sys.stderr); sys.exit(2)
PYEOF
then
  pass "Phase 3: grade_task4..10 all score gold ≥ 0.95 (graders verified)"
else
  fail "Phase 3: extended grader(s) scored below 0.95 on gold — check grade_task4..10"
  hint "Run: python3 -c \"from server.tasks_extended import grade_task4, TASK4_GOLD; print(grade_task4(TASK4_GOLD))\""
  PHASE3_OK=false
fi

[ "$PHASE3_OK" = false ] && stop_at "Step 4c (Phase 3 integrity)"

log "${BOLD}Step 5/8: Phase 1 + Phase 2 + Phase 3 — pytest unit tests${NC} ..."

if command -v pytest &>/dev/null || python3 -m pytest --version &>/dev/null 2>&1; then
  PYTEST_CMD="python3 -m pytest"
  command -v pytest &>/dev/null && PYTEST_CMD="pytest"
  ALL_TEST_FILES="tests/test_graders.py tests/test_judge.py tests/test_ab.py"
  PYTEST_OUTPUT=$(cd "$REPO_DIR" && PYTHONPATH="." ${PYTEST_CMD} ${ALL_TEST_FILES} \
    -v --tb=short 2>&1) && PYTEST_OK=true || PYTEST_OK=false
  if [ "$PYTEST_OK" = true ]; then
    pass "Phase 1+2+3: All pytest unit tests passed"
    PASSED_LINES=$(echo "$PYTEST_OUTPUT" | grep -c " PASSED" || echo "?")
    log "  ${PASSED_LINES} test(s) passed"
  else
    fail "Phase 1+2+3: pytest unit tests FAILED"
    printf "%s\n" "$PYTEST_OUTPUT" | tail -30
    stop_at "Step 5 (pytest)"
  fi
else
  printf "${YELLOW}Warning:${NC} pytest not installed — unit tests skipped.\n"
  hint "Install with: pip install pytest pytest-cov"
fi

log "${BOLD}Step 6/8: Phase 1 — /metrics endpoint live check${NC} ..."

METRICS_OUTPUT=$(curl -s "$PING_URL/metrics" --max-time 15 2>/dev/null || true)
if [ -z "$METRICS_OUTPUT" ]; then
  fail "Phase 1: /metrics endpoint returned empty response or timed out"
  hint "Ensure server/app.py routes GET /metrics and server is running."
  stop_at "Step 6 (/metrics)"
fi

if echo "$METRICS_OUTPUT" | grep -q "data_cleaner_"; then
  pass "Phase 1: /metrics returns Prometheus data_cleaner_* metric families"
else
  fail "Phase 1: /metrics did not return 'data_cleaner_' metric names"
  printf "  First 10 lines of /metrics response:\n"
  echo "$METRICS_OUTPUT" | head -10 | while IFS= read -r line; do
    printf "    %s\n" "$line"
  done
  hint "Verify server/metrics.py is imported and MetricsCollector.get().generate_text() is called."
  stop_at "Step 6 (/metrics)"
fi

for METRIC in data_cleaner_step_latency_seconds data_cleaner_reward data_cleaner_task_score; do
  if echo "$METRICS_OUTPUT" | grep -q "$METRIC"; then
    pass "Phase 1: /metrics contains ${METRIC}"
  else
    fail "Phase 1: /metrics MISSING metric family '${METRIC}'"
  fi
done

log "${BOLD}Step 7/8: Phase 1 — live P0-1 uppercase normalisation check${NC} ..."

# RC-2 fix validation: the environment cycles tasks 1→2→3→1 on each reset().
# We must land on task 1 before testing the uppercase normalisation, or the
# task-1 payload is graded by a different grader and always returns score=0.
_live_reset_to_task1() {
  local body code attempt=0 max=12 task_id
  while [ "${attempt}" -lt "${max}" ]; do
    body=$(curl -s -w "\n__CODE__%{http_code}" -X POST \
      -H "Content-Type: application/json" -d '{}' \
      "$PING_URL/reset" --max-time 15 2>/dev/null)
    code=$(printf '%s' "${body}" | grep '__CODE__' | sed 's/__CODE__//')
    body=$(printf '%s' "${body}" | grep -v '__CODE__')
    task_id=$(printf '%s' "${body}" | grep -oP '(?<="task_id":)\s*\d+' | tr -d ' ' || echo "")
    [ "${task_id}" = "1" ] && { printf '%s' "${body}"; return 0; }
    attempt=$((attempt + 1))
  done
  printf '%s' "${body}"
}

_live_reset_to_task1 >/dev/null   # warm up — get to task 1

# CRITICAL: StepRequest envelope — the /step body must be:
#   {"action": {"task_id": N, "payload": {...}}}
# Sending the action dict directly (without the "action" key) causes HTTP 422.
# HTTP mode is also stateless: each /step runs on a fresh env with _current_task_id=1,
# so task-1 payloads are always graded correctly without needing a prior /reset.
UPPER_PAYLOAD='{"action":{"task_id":1,"payload":{"product_id":"P-00192","name":"Wireless Mouse","price":29.99,"listed_date":"2024-03-15","quantity":142,"category":"ELECTRONICS","brand":"LOGITECH","rating":4.5,"discount_pct":10}}}'
STEP_OUT=$(curl -s -X POST -H "Content-Type: application/json" \
  -d "${UPPER_PAYLOAD}" "$PING_URL/step" --max-time 15 2>/dev/null || true)
SCORE_UP=$(printf '%s' "${STEP_OUT}" | grep -oP '(?<="cumulative_score":)[0-9.]+' 2>/dev/null || echo "0")

if python3 -c "import sys; sys.exit(0 if float('${SCORE_UP}') >= 0.99 else 1)" 2>/dev/null; then
  pass "Phase 1 RC-2/P0-1: live uppercase category/brand normalised (score=${SCORE_UP})"
else
  fail "Phase 1 RC-2/P0-1: uppercase not normalised on live Space (score=${SCORE_UP}, expected ≥ 0.99)"
  hint "Ensure server/tasks.py grade_task1 applies .strip().lower() for 'str (lowercase)' fields."
  hint "Also verify the local_test smoke test uses reset_to_task1 before the uppercase step."
fi

log "${BOLD}Step 8/8: Phase 3 — live smoke test for tasks 4-10 (HTTP 200)${NC} ..."

# Verify the live Space accepts Phase 3 action envelopes without crashing.
# HTTP /step always grades as Task 1 (stateless fresh env), so we only check
# for HTTP 200 (not the score). HTTP 500 = tasks_extended import failure;
# HTTP 422 = payload schema mismatch in models.py.

_live_step_check() {
  local label="$1" payload="$2"
  local out code
  out=$(curl -s -w "\n__CODE__%{http_code}" -X POST \
    -H "Content-Type: application/json" -d "${payload}" \
    "$PING_URL/step" --max-time 15 2>/dev/null)
  code=$(printf '%s' "${out}" | grep '__CODE__' | sed 's/__CODE__//')
  if [ "${code}" = "200" ]; then
    pass "${label} → HTTP 200"
  else
    fail "${label} → HTTP ${code} (expected 200)"
    hint "HTTP 500 = tasks_extended import error. HTTP 422 = payload schema mismatch."
  fi
}

_live_step_check "Phase 3 Task 4 currency-normalization" \
  '{"action":{"task_id":4,"payload":{"records":[{"product_id":"P001","name":"Laptop Pro","price_usd":1299.99,"currency":"USD"}]}}}'

_live_step_check "Phase 3 Task 5 address-standardization" \
  '{"action":{"task_id":5,"payload":{"records":[{"id":"A001","street":"350 Fifth Ave","city":"New York","state":"NY","zip":"10118","country":"US"}]}}}'

_live_step_check "Phase 3 Task 6 date-normalization" \
  '{"action":{"task_id":6,"payload":{"records":[{"id":"D001","normalized_date":"2024-01-15"}]}}}'

_live_step_check "Phase 3 Task 7 phone-normalization" \
  '{"action":{"task_id":7,"payload":{"records":[{"id":"PH001","e164":"+12125551234"}]}}}'

_live_step_check "Phase 3 Task 8 taxonomy-mapping" \
  '{"action":{"task_id":8,"payload":{"records":[{"product_id":"PR001","l1":"Electronics","l2":"Computers","l3":"Laptops"}]}}}'

_live_step_check "Phase 3 Task 9 null-imputation" \
  '{"action":{"task_id":9,"payload":{"records":[{"id":"N001","category":"Electronics","price":299.99,"rating":4.2,"stock":50}]}}}'

_live_step_check "Phase 3 Task 10 unit-conversion" \
  '{"action":{"task_id":10,"payload":{"records":[{"id":"U001","name":"USB-C Hub","weight_kg":0.32,"price":45.50}]}}}'

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All 8/8 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Phase 1 + Phase 2 + Phase 3 verified.${NC}\n"
printf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"
printf "${BOLD}========================================${NC}\n"
printf "\n"

exit 0
