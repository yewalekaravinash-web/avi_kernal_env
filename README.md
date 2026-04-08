---
title: Avi Kernal Env
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---
# DataCleaner — OpenEnv Structured Data Extraction & Cleaning Environment

A fully deterministic, OpenEnv-compliant environment for benchmarking AI agents
on real-world enterprise data cleaning tasks.

## Overview

Data extraction and cleaning accounts for an estimated 60–80% of data scientists'
working time. This environment provides three graded tasks that mirror real ETL
and MDM (Master Data Management) workflows, with fully deterministic graders and
meaningful partial reward signals at every step.

## Action & Observation Spaces

**Action** (`DataCleanerAction`)
| Field | Type | Description |
|-------|------|-------------|
| `task_id` | int | Active task index (1/2/3) |
| `payload` | dict | Agent's cleaned/extracted answer |

**Observation** (`DataCleanerObservation`)
| Field | Type | Description |
|-------|------|-------------|
| `task_id` | int | Current task |
| `task_name` | str | Human-readable task name |
| `instruction` | str | What the agent must do |
| `input_data` | any | Raw data to process |
| `schema_hint` | dict | Target schema (Task 1 only) |
| `step_feedback` | str | Grader feedback from last step |
| `cumulative_score` | float | Score so far [0.0–1.0] |

## Tasks

### Task 1 — Field Extraction (Easy)
Extract and normalize a raw JSON product record into a typed Pydantic schema.
Requires: whitespace trimming, type casting, date normalization (→ YYYY-MM-DD),
string lowercasing.
**Grader**: Field-level exact match across 9 fields. Score = correct/total.

### Task 2 — Duplicate Detection & Merge (Medium)
Identify duplicate company records in a 10-row dataset and produce canonical
merged records. Tests fuzzy entity resolution across address/phone format variants.
**Grader**: Cluster set-match score (0.5 weight) + merged field accuracy (0.5 weight).

### Task 3 — Multi-Source Reconciliation (Hard)
Three source systems (CRM, Billing, Marketing) contain conflicting field values
for 3 entities. Apply a provided source-priority policy field-by-field to produce
a golden master record.
**Grader**: Per-field match vs. ground truth. Null-when-value-exists penalty: −0.05 per field.

## Reward Architecture

```
reward = field_accuracy × task_weight
Task 1: uniform field weights, max=1.0
Task 2: 0.5 × cluster_score + 0.5 × merge_score
Task 3: field_accuracy − null_penalty (capped at −0.20)
```

Partial rewards: each step returns the delta above the previous cumulative score,
enabling RLHF-style training signals throughout the episode.

## Setup & Usage

### Local (no Docker)
```bash
pip install openenv-core[core] openai pandas pydantic rapidfuzz uvicorn
uvicorn server.app:app --host 0.0.0.0 --port 8000
# in another terminal:
export HF_TOKEN=your_token
export OPENENV_SERVER_URL=http://localhost:8000
python inference.py
```

### Docker
```bash
docker build -t data-cleaner-env .
docker run --cpus="2" --memory="8g" -p 8000:8000 \
  -e HF_TOKEN=$HF_TOKEN \
  data-cleaner-env

# Run inference against running container:
export OPENENV_SERVER_URL=http://localhost:8000
python inference.py
```

### Validate
```bash
pip install openenv-core
openenv validate
```

## Baseline Performance Scores

Tested with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Router:

| Task | Name | Baseline Score |
|------|------|---------------|
| 1 | field-extraction | ~0.78 |
| 2 | dedup-merge | ~0.60 |
| 3 | multi-source-reconciliation | ~0.47 |
| **Avg** | | **~0.62** |

*Scores are reproducible given fixed temperature=0.2.*

## Deploy to Hugging Face Spaces

```bash
# 1. Login
huggingface-cli login   # paste HF_TOKEN when prompted

# 2. Create Space (Dockerfile type)
huggingface-cli repo create <YOUR_HF_USERNAME>/data-cleaner-env --type space --space_sdk docker

# 3. Push
openenv push --repo-id <YOUR_HF_USERNAME>/data-cleaner-env

# 4. Set secrets in Space Settings:
#    HF_TOKEN, API_BASE_URL, MODEL_NAME

# 5. Verify Space is in Running state before submission.
```

## Environment Variables

| Variable | Default | Required |
|----------|---------|----------|
| `HF_TOKEN` | — | Yes (API key) |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | Yes |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Yes |
| `LOCAL_IMAGE_NAME` | — | Only for docker mode |
| `OPENENV_SERVER_URL` | `http://localhost:8000` | Only for server mode |

## Tags
`openenv` `data-cleaning` `structured-extraction` `deduplication` `mdm`
