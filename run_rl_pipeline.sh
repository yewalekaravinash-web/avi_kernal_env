#!/bin/bash

# run_rl_pipeline.sh — Master automation script for DataCleaner RL Environment
# Author: Antigravity AI
# Version: 1.0.0

# ── Configuration ─────────────────────────────────────────────────────────────
if (-not $env:HF_TOKEN) {
    $env:HF_TOKEN = Read-Host "Enter HF Token"
}
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-72B-Instruct}"
API_BASE_URL="${API_BASE_URL:-https://router.huggingface.co/v1}"
SERVER_PORT=8000
SERVER_URL="http://localhost:$SERVER_PORT"
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# ANSI Colors for premium console output
GREEN="\033[92m"
YELLOW="\033[93m"
RED="\033[91m"
CYAN="\033[96m"
BOLD="\033[1m"
RESET="\033[0m"

echo -e "${BOLD}${CYAN}======================================================================"
echo -e "   DATA CLEANER RL ENGINE — AUTOMATED PIPELINE"
echo -e "======================================================================${RESET}"

# ── Step 1: Dependency Check & Auto-Fix ───────────────────────────────────────
echo -e "\n${BOLD}[1/5] Checking Pre-requisites...${RESET}"

# check if uv is available (preferred tool)
if command -v uv >/dev/null 2>&1; then
    USE_UV=true
    PY_CMD="uv run python"
    echo -e "Using ${GREEN}uv${RESET} for environment management."
elif command -v uv.exe >/dev/null 2>&1; then
    USE_UV=true
    PY_CMD="uv.exe run python"
    echo -e "Using ${GREEN}uv.exe${RESET} for environment management."
else
    USE_UV=false
    if command -v python3 >/dev/null 2>&1; then PY_CMD="python3"; else PY_CMD="python"; fi
    if command -v pip3 >/dev/null 2>&1; then PIP_CMD="pip3"; else PIP_CMD="pip"; fi
fi

if [ "$USE_UV" = false ]; then
    echo -e "Installing dependencies from server/requirements.txt..."
    $PIP_CMD install -r server/requirements.txt --quiet --break-system-packages || {
        echo -e "${YELLOW}Warning: pip install failed. Retrying with --user...${RESET}"
        $PIP_CMD install -r server/requirements.txt --user --quiet --break-system-packages
    }
fi

# ── Step 2: Port Cleanup & Server Startup ─────────────────────────────────────
echo -e "\n${BOLD}[2/5] Initializing Server...${RESET}"

# Check if port is in use (robust check for both Linux/macOS and Windows/WSL)
if command -v lsof >/dev/null 2>&1; then
    PID=$(lsof -t -i:$SERVER_PORT)
    if [ ! -z "$PID" ]; then
        echo -e "${YELLOW}Port $SERVER_PORT is occupied by PID $PID. Terminating...${RESET}"
        kill -9 $PID
    fi
fi

echo -e "Starting FastAPI server on port $SERVER_PORT..."
export HF_TOKEN=$HF_TOKEN
export MODEL_NAME=$MODEL_NAME
export API_BASE_URL=$API_BASE_URL

nohup $PY_CMD -m server.app > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!

# Wait for server to be healthy
echo -ne "Waiting for server to be healthy..."
MAX_RETRIES=120
COUNT=0
while [ $COUNT -lt $MAX_RETRIES ]; do
    if curl -s "$SERVER_URL/health" | grep -q "healthy"; then
        echo -e " ${GREEN}READY${RESET}"
        break
    fi
    echo -ne "."
    sleep 1
    COUNT=$((COUNT+1))
done

if [ $COUNT -eq $MAX_RETRIES ]; then
    echo -e "\n${RED}Error: Server failed to start within 30s. Check $LOG_DIR/server.log${RESET}"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# ── Step 3: Execute RL Training ───────────────────────────────────────────────
echo -e "\n${BOLD}[3/5] Starting RL Engine Training (10 Tasks)...${RESET}"
echo -e "This will run 3 cycles across all 10 tasks with weight adjustment."

export OPENENV_SERVER_URL=$SERVER_URL
$PY_CMD rl_trainer.py

RL_STATUS=$?
if [ $RL_STATUS -ne 0 ]; then
    echo -e "${RED}RL training failed with exit code $RL_STATUS.${RESET}"
fi

# ── Step 4: Generate Performance Graph ────────────────────────────────────────
echo -e "\n${BOLD}[4/5] Fetching Prometheus Metrics & Rendering Graph...${RESET}"

if [ -f "./scratch/console_grapher.py" ]; then
    $PY_CMD ./scratch/console_grapher.py "$SERVER_URL/metrics"
else
    echo -e "${YELLOW}Warning: console_grapher.py not found. Skipping graph.${RESET}"
fi

# ── Step 5: Cleanup ───────────────────────────────────────────────────────────
echo -e "\n${BOLD}[5/5] Finalizing...${RESET}"
echo -e "Stopping server (PID $SERVER_PID)..."
kill $SERVER_PID 2>/dev/null

echo -e "\n${BOLD}${GREEN}======================================================================"
echo -e "   PIPELINE COMPLETE"
echo -e "======================================================================${RESET}"
echo -e "Detailed logs available in:"
echo -e "  - RL Training: rl_training.log"
echo -e "  - Server Output: $LOG_DIR/server.log"
