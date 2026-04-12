# 🐛 Bug Reporting Guide — Edge-Forge

> **Before filing a bug:** Read this guide completely. A well-formed report
> gets fixed in hours. An incomplete report sits in triage for weeks.

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-v1-blue)](https://github.com/meta-pytorch/OpenEnv)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-green)](./LICENSE)

Edge-Forge is a reinforcement learning environment built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) protocol, containerized with Docker, and deployed to Hugging Face Spaces. Its failure modes span multiple layers — the OpenEnv SDK client (`EdgeForgeEnv`), the FastAPI server (`server/app.py`), the stateful mock API (`mock_api.py`), Docker containerization, and the LLM-driven inference script (`inference.py`). This guide provides environment-specific diagnostic procedures, severity classification, and structured templates for each failure category so that every report contains the information maintainers need to reproduce and fix the issue.

---

## 1. Before You Report

Work through this checklist. If any step resolves your issue, no bug report is needed.

### 1.1 Verify Your Environment

```bash
# Python version — must be 3.10, 3.11, or 3.12
python --version

# openenv-core installed and compatible
pip show openenv-core        # confirm version ≥ 0.2.2

# OpenAI SDK installed
pip show openai              # confirm version ≥ 1.0.0

# Docker running (required for containerized deployment)
docker info                  # must succeed with no errors

# Required environment variables
echo $HF_TOKEN               # must return a non-empty value
echo $API_BASE_URL            # must return your LLM endpoint (default: https://router.huggingface.co/v1)
echo $MODEL_NAME              # must return your model identifier (default: Qwen/Qwen2.5-72B-Instruct)
```

### 1.2 Run a Quick Health Check

```bash
# Start the server locally
uvicorn server.app:app --host 0.0.0.0 --port 8000

# In a separate terminal — verify the health endpoint
curl -f http://localhost:8000/health

# Verify reset returns a valid observation
curl -s -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{}' | python -m json.tool
```

### 1.3 Check Known Issues

Review [Section 5: Known Issues & Limitations](#5-known-issues--limitations) below.
Your issue may already be documented with a workaround.

### 1.4 Check Existing Issues

Search GitHub Issues for your error message:

```
https://github.com/SoumyaWasule/Edge-Forge/issues?q=<error+keyword>
```

### 1.5 Run the Smoke Tests

```bash
# From the repository root
pytest tests/test_environment.py -v
```

All 6 tests must pass. If any fail, include the output in your bug report.

---

## 2. Severity Classification

Assign **ONE** severity level to your bug. Mislabeling severity delays triage.

| Severity | Label | Definition | Examples | SLA |
|----------|-------|------------|----------|-----|
| 🔴 Critical | `severity: critical` | Prevents all users from using the environment | Docker build fails, HF Space returns 500 on `/reset`, `inference.py` exits with non-zero, `server/app.py` crashes on import | Same day |
| 🟠 High | `severity: high` | Prevents a specific task or feature from working | `grade_easy` always returns `0.01`, `step()` hangs indefinitely, reward calculation returns `NaN`, WebSocket connection drops mid-episode | 48 hours |
| 🟡 Medium | `severity: medium` | Degrades functionality but a workaround exists | Intermittent `step()` timeout, `covered_branches` list incomplete, `_summarize_api_result` returns wrong summary string, LLM fallback always triggers | 1 week |
| 🟢 Low | `severity: low` | Cosmetic or minor issue with no functional impact | Typo in dashboard HTML, incorrect docstring, minor formatting issue in `[STEP]` log output | Next release |
| 💡 Enhancement | `severity: enhancement` | Not a bug — a feature request or improvement | New task suggestion, additional branch in `mock_api.py`, reward function tuning idea | Backlog |

### Severity Decision Flowchart

```
Is the environment completely unusable?
├── YES → 🔴 Critical
└── NO
    ├── Is one task or feature completely broken?
    │   ├── YES → 🟠 High
    │   └── NO
    │       ├── Does a workaround exist?
    │       │   ├── YES → 🟡 Medium
    │       │   └── NO → 🟠 High
    └── Is it cosmetic/minor?
        ├── YES → 🟢 Low
        └── NO → 🟡 Medium
```

---

## 3. Bug Report Templates

Select the template that matches your failure category. Each template is self-contained.

<details>
<summary>📋 Template 1: Environment / Deployment Failure — Click to expand</summary>

### Bug Report: Environment / Deployment Failure

Use this template when the environment server won't start, imports fail, or the FastAPI app crashes on initialization.

```
Title: [DEPLOYMENT] Brief description of what's broken
Severity: [🔴 Critical | 🟠 High | 🟡 Medium]
Environment: [Local Python | Local Docker | HF Space]
Date: YYYY-MM-DD
Reporter: @github-username
```

#### Environment Information

| Field | Value |
|-------|-------|
| Edge-Forge Version | `git rev-parse --short HEAD` output |
| Python Version | `python --version` output |
| openenv-core Version | `pip show openenv-core \| grep Version` output |
| openai Version | `pip show openai \| grep Version` output |
| fastapi Version | `pip show fastapi \| grep Version` output |
| uvicorn Version | `pip show uvicorn \| grep Version` output |
| pydantic Version | `pip show pydantic \| grep Version` output |
| OS / Platform | e.g., Ubuntu 22.04 / macOS 14 / Windows WSL2 |
| Deployment Target | Local / Hugging Face Space |

#### Diagnostic Output

```bash
# Run the server directly and capture startup output:
uvicorn server.app:app --host 0.0.0.0 --port 8000 2>&1 | head -50
```

```
[paste output here]
```

Full error traceback:

```python
[paste complete traceback here — not just the last line]
```

#### Reproduction Steps

**Starting state:** [Fresh clone / existing virtualenv / Docker container]

1. [Exact command or action, including all flags]
2. [Next exact step]
3. [Continue until the failure occurs]

**Frequency:** Always / ~X% of the time / Intermittent (describe pattern)

#### Expected vs Actual

**Expected:**
[What should have happened — e.g., "Server starts on port 8000 and `/health` returns HTTP 200"]

**Actual:**
[What actually happened — e.g., "ImportError: cannot import name 'create_app' from 'openenv.core.env_server.http_server'"]

#### Impact Assessment

- **Who is affected:** All users / Users on [specific platform]
- **Workaround available:** Yes — [describe] / No
- **Blocks submission validator:** Yes / No

#### Possible Root Cause (Optional)

[If you've diagnosed the likely cause, reference the specific file and line number — e.g., `server/app.py:17` imports `create_app` which may not exist in older openenv-core versions]

</details>

<details>
<summary>📋 Template 2: OpenEnv API Failure — Click to expand</summary>

### Bug Report: OpenEnv API Failure

Use this template for bugs in the `/reset`, `/step`, or `/state` endpoints — wrong responses, missing fields, Pydantic validation errors, or OpenEnv spec violations.

```
Title: [API] Brief description of what's broken
Severity: [🔴 Critical | 🟠 High | 🟡 Medium]
Environment: [Local Python | Local Docker | HF Space]
Date: YYYY-MM-DD
Reporter: @github-username
```

#### Environment Information

| Field | Value |
|-------|-------|
| Edge-Forge Version | `git rev-parse --short HEAD` output |
| Python Version | `python --version` output |
| openenv-core Version | `pip show openenv-core \| grep Version` output |
| Deployment Target | Local / Hugging Face Space |
| HF Space URL | `https://huggingface.co/spaces/<username>/<space-name>` |

#### Affected Endpoint

- [ ] `POST /reset`
- [ ] `POST /step`
- [ ] `GET /state`
- [ ] `POST /grade`
- [ ] `GET /tasks`

#### Diagnostic Output

```bash
# Test reset
curl -s -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{}' | python -m json.tool

# Test step with SET_FIELD
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "SET_FIELD", "field": "age", "value": 25}' | python -m json.tool

# Test step with SUBMIT
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "SUBMIT"}' | python -m json.tool

# Test state
curl -s http://localhost:8000/state | python -m json.tool
```

```
[paste the failing endpoint's output here]
```

#### Request Payload

```json
[exact JSON body you sent to the endpoint]
```

#### Response Received

```json
[exact JSON response, including HTTP status code]
```

#### Expected Response

Reference the `EdgeForgeObservation` model from `models.py`:
- `last_status`: int (0, 200, or 500)
- `covered_branches`: List[str]
- `current_input`: Dict[str, Any]
- `last_error`: Optional[str]
- `submit_outcomes`: List[str]

```json
[what the response should have been, per the spec]
```

#### Reproduction Steps

1. `curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}'`
2. [Next step]
3. [Continue until the bug is triggered]

**Frequency:** Always / ~X% of the time / Intermittent

#### Impact Assessment

- **Who is affected:** All users / Users running [specific task]
- **Workaround available:** Yes — [describe] / No
- **Blocks submission validator:** Yes / No

#### Possible Root Cause (Optional)

[Reference specific file — e.g., `server/edge_forge_env_environment.py:120` in the `step()` method, or `server/app.py:266` in `step_stateful()`]

</details>

<details>
<summary>📋 Template 3: Task / Grader Failure — Click to expand</summary>

### Bug Report: Task / Grader Failure

Use this template when a specific task produces wrong scores, a grader behaves unexpectedly, or reward values are outside [0.0, 1.0].

```
Title: [GRADER] Brief description of what's broken
Severity: [🟠 High | 🟡 Medium]
Environment: [Local Python | Local Docker | HF Space]
Date: YYYY-MM-DD
Reporter: @github-username
```

#### Affected Task

- [ ] `easy_task` — Trigger SSN Error (grader: `grade_easy`)
- [ ] `medium_task` — Maximize Coverage (grader: `grade_medium`)
- [ ] `hard_task` — Stateful Crash Trap (grader: `grade_hard`)

#### Grader Source

Which grader file is affected?

- [ ] `graders.py` (module-level, for validator discovery)
- [ ] `tasks.py` (module-level, used by `inference.py`)
- [ ] `server/app.py` (module-level, used by `/grade` endpoint)
- [ ] `server/edge_forge_env_environment.py` (instance-level, bound to environment)

#### Score Information

| Field | Value |
|-------|-------|
| Score Returned | [e.g., 0.01] |
| Score Expected | [e.g., 0.99] |
| `submit_outcomes` at grading time | [paste the list] |
| `covered_branches` at grading time | [paste the list] |

#### Diagnostic Output

```python
# Reproduce the grading issue directly
from server.edge_forge_env_environment import EdgeForgeEnvironment
from models import EdgeForgeAction

env = EdgeForgeEnvironment()
env.reset()

# [paste the exact action sequence that should trigger the grader condition]
# Example for easy_task:
env.step(EdgeForgeAction(action_type="SET_FIELD", field="action", value="open_account"))
env.step(EdgeForgeAction(action_type="SUBMIT"))
env.step(EdgeForgeAction(action_type="SET_FIELD", field="action", value="verify_identity"))
env.step(EdgeForgeAction(action_type="SET_FIELD", field="ssn", value="abc"))
env.step(EdgeForgeAction(action_type="SUBMIT"))

print("submit_outcomes:", env.state["submit_outcomes"])
print("grade_easy:", env.grade_easy())
```

```
[paste output here]
```

#### Expected vs Actual

**Expected:**
[e.g., "After triggering 'SSN must be numeric' in submit_outcomes, `grade_easy()` should return 0.99"]

**Actual:**
[e.g., "`grade_easy()` returns 0.01 despite 'SSN must be numeric' being in submit_outcomes"]

#### Impact Assessment

- **Grader consistency:** Do all four grader locations (graders.py, tasks.py, server/app.py, environment.py) return the same score for this input?
- **Inference impact:** Does `inference.py` report a different score than the `/grade` endpoint?
- **Blocks submission validator:** Yes / No

</details>

<details>
<summary>📋 Template 4: Inference Script Failure — Click to expand</summary>

### Bug Report: Inference Script Failure

Use this template for bugs in `inference.py` — wrong `[START]`/`[STEP]`/`[END]` format, LLM API errors, scoring aggregation bugs, or runtime exceeded.

```
Title: [INFERENCE] Brief description of what's broken
Severity: [🔴 Critical | 🟠 High | 🟡 Medium]
Environment: [Local Python | Local Docker]
Date: YYYY-MM-DD
Reporter: @github-username
```

#### Environment Information

| Field | Value |
|-------|-------|
| Edge-Forge Version | `git rev-parse --short HEAD` output |
| Python Version | `python --version` output |
| API_BASE_URL | [redact key but show format, e.g., `https://router.huggingface.co/v1`] |
| MODEL_NAME | [e.g., `Qwen/Qwen2.5-72B-Instruct`] |
| HF_TOKEN set | Yes / No |
| IMAGE_NAME | [if running via Docker, else empty] |
| ENV_BASE_URL | [if running against local server, e.g., `http://localhost:8000`] |

#### Failure Category

- [ ] Script exits with non-zero return code
- [ ] `[START]`/`[STEP]`/`[END]` log format is malformed
- [ ] LLM API call fails (timeout, auth error, rate limit)
- [ ] Fallback actions triggered unexpectedly
- [ ] Score is outside [0.0, 1.0]
- [ ] `parse_llm_response()` fails to extract valid JSON
- [ ] `coerce_field_value()` produces wrong type

#### Diagnostic Output

```bash
# Run inference and capture both stdout and stderr
API_BASE_URL=$API_BASE_URL \
  MODEL_NAME=$MODEL_NAME \
  HF_TOKEN=$HF_TOKEN \
  python inference.py 2>&1 | tee inference.log

# Check log format compliance
grep -E "^\[START\]|\[STEP\]|\[END\]" inference.log | head -20

# Check for DEBUG messages (stderr fallback indicators)
grep "\[DEBUG\]" inference.log
```

**stdout (structured logs):**
```
[paste [START]/[STEP]/[END] lines here]
```

**stderr (debug output):**
```
[paste [DEBUG] lines here — these indicate LLM failures or fallback usage]
```

#### Expected vs Actual

**Expected log format:**
```
[START] task=easy_task env=edge_forge_env model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1 action=SET_FIELD(action=open_account) reward=0.50 done=false error=null
[STEP]  step=2 action=SUBMIT reward=9.00 done=false error=null
[END]   success=true steps=5 score=0.990 rewards=0.50,9.00,...
```

**Actual:**
```
[paste what you actually got]
```

#### Impact Assessment

- **Tasks affected:** All / easy_task / medium_task / hard_task
- **LLM calls succeeding:** Yes / No / Partial (fallback triggered for N steps)
- **Blocks submission validator:** Yes / No

</details>

<details>
<summary>📋 Template 5: Docker / HF Space Deployment Failure — Click to expand</summary>

### Bug Report: Docker / HF Space Deployment Failure

Use this template for Docker build failures, container crashes, port conflicts, missing environment variables, or Hugging Face Space errors.

```
Title: [DOCKER] Brief description of what's broken
Severity: [🔴 Critical | 🟠 High | 🟡 Medium]
Environment: Docker / HF Space
Date: YYYY-MM-DD
Reporter: @github-username
```

#### Environment Information

| Field | Value |
|-------|-------|
| Docker Version | `docker --version` output |
| Base Image | `ghcr.io/meta-pytorch/openenv-base:latest` |
| OS / Platform | e.g., Ubuntu 22.04 / macOS 14 ARM / Windows WSL2 |
| HF Space URL | `https://huggingface.co/spaces/<username>/<space-name>` |

#### Failure Category

- [ ] Docker build fails (`docker build` exits non-zero)
- [ ] Container starts but crashes immediately
- [ ] Container starts but `/health` returns non-200
- [ ] Port 8000 not accessible from host
- [ ] HF Space build fails
- [ ] HF Space returns 500 / shows "Building" indefinitely
- [ ] `uv sync` fails inside container
- [ ] Environment variable not passed to container

#### Diagnostic Output

```bash
# Build with verbose output
docker build --no-cache --progress=plain -t edge-forge . 2>&1 | tee build.log

# Run with all env vars and port mapping
docker run -e API_BASE_URL=$API_BASE_URL \
  -e MODEL_NAME=$MODEL_NAME \
  -e HF_TOKEN=$HF_TOKEN \
  -p 8000:8000 \
  edge-forge

# Check container logs
docker logs <container-id> --tail 100

# Inspect container if it crashes
docker run --entrypoint /bin/bash -it edge-forge

# Verify health endpoint from inside the container
docker exec <container-id> curl -f http://localhost:8000/health
```

**Build log (last 50 lines):**
```
[paste build.log tail here]
```

**Container logs:**
```
[paste container stdout/stderr here]
```

#### Dockerfile Context

The current Dockerfile uses:
- **Base image:** `ghcr.io/meta-pytorch/openenv-base:latest`
- **Workdir:** `/app/env`
- **Entrypoint:** `uvicorn server.app:app --host 0.0.0.0 --port 8000`
- **Health check:** `curl -f http://localhost:8000/health`
- **PYTHONPATH:** `/app/env`
- **Package manager:** `uv sync`

#### Reproduction Steps

1. `git clone https://github.com/SoumyaWasule/Edge-Forge.git && cd Edge-Forge`
2. `docker build -t edge-forge .`
3. `docker run -p 8000:8000 edge-forge`
4. [Describe where it fails]

**Frequency:** Always / ~X% of the time

#### Expected vs Actual

**Expected:**
[e.g., "Container starts, uvicorn binds to 0.0.0.0:8000, `/health` returns 200"]

**Actual:**
[e.g., "uv sync fails with 'No solution found' for openenv-core dependency"]

#### Impact Assessment

- **Who is affected:** All Docker users / HF Space only / Specific architecture (ARM/x86)
- **Workaround available:** Yes — [describe] / No
- **Blocks submission validator:** Yes / No

</details>

---

## 4. How to Collect Diagnostic Information

### 4.1 System Health Check

```bash
# Python environment
python --version
pip show openenv-core openai fastapi uvicorn pydantic

# Docker (if applicable)
docker --version
docker info | grep -E "Server Version|Operating System|Total Memory"
```

### 4.2 Docker Diagnostic Commands

```bash
# Build with verbose output (captures all layer details)
docker build --no-cache --progress=plain -t edge-forge . 2>&1 | tee build.log

# Run with all env vars and port mapping
docker run \
  -e API_BASE_URL=$API_BASE_URL \
  -e MODEL_NAME=$MODEL_NAME \
  -e HF_TOKEN=$HF_TOKEN \
  -p 8000:8000 \
  edge-forge

# Check container logs after crash
docker logs $(docker ps -lq) --tail 100

# Get a shell inside the container for debugging
docker run --entrypoint /bin/bash -it edge-forge

# Verify installed packages inside container
docker exec $(docker ps -lq) pip show openenv-core openai fastapi uvicorn
```

### 4.3 OpenEnv API Diagnostic Commands

```bash
# Test reset() endpoint
curl -s -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{}' | python -m json.tool

# Test state() endpoint
curl -s http://localhost:8000/state | python -m json.tool

# Test step() with SET_FIELD
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "SET_FIELD", "field": "age", "value": 25}' | python -m json.tool

# Test step() with SUBMIT
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "SUBMIT"}' | python -m json.tool

# Test step() with RESET action
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "RESET"}' | python -m json.tool

# Test /tasks endpoint
curl -s http://localhost:8000/tasks | python -m json.tool

# Test /grade endpoint
curl -s -X POST http://localhost:8000/grade \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_task"}' | python -m json.tool

# Health check
curl -sf http://localhost:8000/health && echo "OK" || echo "FAILED"
```

### 4.4 Inference Script Diagnostic Commands

```bash
# Run inference with full output capture
API_BASE_URL=$API_BASE_URL \
  MODEL_NAME=$MODEL_NAME \
  HF_TOKEN=$HF_TOKEN \
  python inference.py > inference_stdout.log 2> inference_stderr.log

# Check log format compliance
grep -cE "^\[START\]" inference_stdout.log   # expect: 3 (one per task)
grep -cE "^\[END\]"   inference_stdout.log   # expect: 3 (one per task)

# Check for fallback usage (indicates LLM call failures)
grep -c "\[DEBUG\]" inference_stderr.log

# Verify all scores are in valid range
grep "^\[END\]" inference_stdout.log | while read line; do
  score=$(echo "$line" | grep -oP 'score=\K[0-9.]+')
  echo "Score: $score"
  if (( $(echo "$score < 0.0 || $score > 1.0" | bc -l) )); then
    echo "  ⚠️  OUT OF RANGE"
  fi
done

# Run a single task's grader in isolation
python -c "
from server.edge_forge_env_environment import EdgeForgeEnvironment
from models import EdgeForgeAction

env = EdgeForgeEnvironment()
env.reset()
env.step(EdgeForgeAction(action_type='SET_FIELD', field='action', value='open_account'))
env.step(EdgeForgeAction(action_type='SUBMIT'))
print('submit_outcomes:', env.state['submit_outcomes'])
print('grade_easy:', env.grade_easy())
print('grade_medium:', env.grade_medium())
print('grade_hard:', env.grade_hard())
"
```

---

## 5. Known Issues & Limitations

These are documented behaviors that are **not** bugs. Do not file reports for these.

| ID | Title | Severity | Status | Workaround |
|----|-------|----------|--------|------------|
| KI-001 | Single-session environment (no concurrent episodes) | 🟡 Medium | By Design | Serialize all API calls; do not run multiple agents simultaneously |
| KI-002 | Stochastic thresholds randomize on each `reset()` | 🟢 Low | By Design | Accept that `age_limit` (16–21) and `enterprise_days` (300–400) vary per episode |
| KI-003 | `MAX_STEPS` hard cap at 30 | 🟡 Medium | By Design | Structure exploration to stay under 30 steps; `medium_task` allows 30, others less |
| KI-004 | No session persistence across container restarts | 🟡 Medium | Won't Fix v1.0 | Call `/reset` after every container restart |
| KI-005 | Grader functions duplicated in four locations | 🟢 Low | Planned v1.1 | All four copies return identical scores — no action needed |
| KI-006 | `uv.lock` excluded from Docker builds by `.dockerignore` | 🟢 Low | By Design | Docker performs a fresh `uv sync` resolve; builds may be slower |

---

### KI-001: Single-Session Environment (No Concurrent Episodes)

**Status:** By Design  
**Affects:** All users running multiple agents or parallel inference

**Description:**  
The FastAPI server maintains a single `EdgeForgeEnvironment` instance (`_env_instance` in `server/app.py:244`) guarded by a `threading.Lock`. All `/reset`, `/step`, and `/state` calls operate on this shared instance. Concurrent requests from multiple agents will interleave their episodes, corrupting state.

**Workaround:**  
Serialize all API calls. Run only one agent at a time per server instance. For parallel evaluation, start multiple Docker containers on different ports.

**Root Cause:**  
`max_concurrent_envs=1` is passed to `create_app()` at `server/app.py:99`. The stateful HTTP routes replace the framework's stateless defaults to enable multi-step RL episodes, but only one episode context exists.

---

### KI-002: Stochastic Thresholds Randomize on Each `reset()`

**Status:** By Design  
**Affects:** Agents that assume deterministic environment behavior

**Description:**  
Each `reset()` call generates random thresholds — `age_limit` between 16 and 21, `enterprise_days` between 300 and 400 (`server/edge_forge_env_environment.py:104-107`). This means the same input sequence may trigger different branches across episodes.

**Workaround:**  
Design agents to adapt to varying thresholds rather than memorizing fixed input-to-branch mappings. The grading criteria (`submit_outcomes`) are deterministic given the same inputs within an episode.

---

### KI-003: `MAX_STEPS` Hard Cap at 30

**Status:** By Design  
**Affects:** Agents that need extended exploration for `medium_task`

**Description:**  
The environment terminates any episode at 30 steps (`server/edge_forge_env_environment.py:67`), regardless of task. The inference script further limits: `easy_task` to 10 steps, `medium_task` to 30, and `hard_task` to 15 (`inference.py:156-188`).

**Workaround:**  
Optimize action sequences. The fallback action sequences in `inference.py` demonstrate that `easy_task` can be solved in 5 steps and `hard_task` in 4 steps.

---

### KI-004: No Session Persistence Across Container Restarts

**Status:** Won't Fix in v1.0  
**Affects:** Users who restart the Docker container mid-episode

**Description:**  
The `EdgeForgeEnvironment.state` is stored in Python memory (`server/edge_forge_env_environment.py:70`). When the container restarts, all state is lost. The `/state` endpoint will return `{"episode_id": "", "step_count": 0}`.

**Workaround:**  
Call `/reset` after every container restart. Design agents to handle episode resets gracefully.

---

### KI-005: Grader Functions Duplicated in Four Locations

**Status:** Planned for v1.1  
**Affects:** Contributors modifying grading logic

**Description:**  
Identical grader functions (`grade_easy`, `grade_medium`, `grade_hard`) exist in:
1. `graders.py` — for OpenEnv validator discovery
2. `tasks.py` — for `inference.py` imports
3. `server/app.py` — for the `/grade` endpoint and validator discovery
4. `server/edge_forge_env_environment.py` — instance methods on `EdgeForgeEnvironment`

All four return identical scores for the same input. This duplication exists because the OpenEnv validator discovers graders at different import paths depending on the execution context.

**Workaround:**  
If modifying grading logic, update **all four locations** and run `pytest tests/test_environment.py -v` to verify consistency.

---

### KI-006: `uv.lock` Excluded from Docker Builds

**Status:** By Design  
**Affects:** Build speed

**Description:**  
The `.dockerignore` file excludes `uv.lock`, so Docker builds perform a fresh dependency resolve via `uv sync --no-install-project --no-editable`. This avoids cross-platform lockfile incompatibilities but makes builds slower.

**Workaround:**  
None needed. Accept slightly longer build times. If builds fail due to dependency resolution, this is likely a transient PyPI or GitHub issue — retry after a few minutes.

---

## 6. Bug Triage Process

### Lifecycle of a Bug Report

```
Filed → Labelled (24h) → Confirmed (48h) → Assigned → In Progress → Fixed → Verified → Closed
```

### Labels Applied During Triage

| Label | Meaning |
|-------|---------|
| `severity: critical` | See [Section 2](#2-severity-classification) |
| `severity: high` | See [Section 2](#2-severity-classification) |
| `severity: medium` | See [Section 2](#2-severity-classification) |
| `severity: low` | See [Section 2](#2-severity-classification) |
| `status: needs-info` | Report is incomplete — reporter must respond within 7 days |
| `status: confirmed` | Reproduced by maintainer |
| `status: in-progress` | Fix is being worked on |
| `status: fixed` | Fix merged, pending release |
| `type: deployment` | Docker / HF Space issue |
| `type: api` | `/step`, `/reset`, `/state` issue |
| `type: task` | Task / grader issue |
| `type: inference` | `inference.py` issue |
| `type: spec-compliance` | OpenEnv spec violation |
| `duplicate` | Same as existing issue |
| `wontfix` | By design or out of scope |
| `good-first-issue` | Suitable for new contributors |

### What Happens After You File

1. **Within 24 hours:** A maintainer applies severity and type labels.
2. **Within 48 hours:** Maintainer attempts to reproduce; adds `status: confirmed` or requests more info.
3. **If `status: needs-info`:** Reporter has 7 days to provide missing details before the issue is closed.
4. **When fixed:** The fix commit is linked to the issue; reporter is asked to verify.
5. **After verification:** Issue is closed with the version tag in which the fix ships.

---

## 7. Security Vulnerabilities

> **Do NOT file security vulnerabilities as public GitHub issues.**

If you discover a security vulnerability in Edge-Forge — including issues with sandbox isolation, unauthorized environment access, or token/credential exposure — report it privately:

- **GitHub:** Use the [Security Advisories](https://github.com/SoumyaWasule/Edge-Forge/security/advisories) tab
- **Include:** Full description, reproduction steps, potential impact, and your suggested fix if any

You will receive acknowledgement within 48 hours. We follow responsible disclosure practices and will credit you in the security advisory unless you prefer to remain anonymous.

### Security-Relevant Areas

| Area | File(s) | Concern |
|------|---------|---------|
| Docker container isolation | `Dockerfile` | Session boundary enforcement between agents |
| HF Token handling | `inference.py:46` | `HF_TOKEN` passed via env var — verify no log leakage |
| API key in OpenAI client | `inference.py:485` | `API_KEY` used in `OpenAI()` constructor |
| Input validation boundary | `server/edge_forge_env_environment.py:44-49` | `_validate_field()` enforces type constraints at the API boundary |
| Pydantic model validation | `models.py` | `EdgeForgeAction` and `EdgeForgeObservation` enforce schema |
| Request body parsing | `server/app.py:266-274` | `step_stateful()` accepts arbitrary dict — validate `action_type` |

---

## 8. Contributing a Fix

Found the bug? Want to fix it? Here's the workflow:

### Setup for Local Development

```bash
# Clone the repository
git clone https://github.com/SoumyaWasule/Edge-Forge.git
cd Edge-Forge

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify the environment starts
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Running Tests Before Your Fix

```bash
# Run the full test suite
pytest tests/test_environment.py -v

# Verify all graders return valid scores
python -c "
from server.edge_forge_env_environment import EdgeForgeEnvironment
env = EdgeForgeEnvironment()
env.reset()
for g in [env.grade_easy, env.grade_medium, env.grade_hard]:
    score = g()
    assert 0.0 <= score <= 1.0, f'Score out of range: {score}'
    print(f'{g.__name__}: {score}')
print('All graders OK')
"

# Run inference to verify baseline scores (requires LLM access)
API_BASE_URL=$API_BASE_URL \
  MODEL_NAME=$MODEL_NAME \
  HF_TOKEN=$HF_TOKEN \
  python inference.py
```

### Submitting a Pull Request

1. Create a branch: `git checkout -b fix/brief-description-of-bug`
2. Make your changes and test them
3. Run the full test suite — all tests must pass
4. Reference the bug report in your PR description: `Fixes #<issue-number>`
5. Wait for review — we aim to review PRs within 72 hours

### Pull Request Checklist

- [ ] Bug is reproducible on `main` branch before my change
- [ ] My change fixes the bug without breaking existing functionality
- [ ] I have run `pytest tests/test_environment.py -v` — all tests pass
- [ ] I have run `python inference.py` and baseline scores are not regressed
- [ ] The `[START]`/`[STEP]`/`[END]` log format is unchanged
- [ ] Docker build succeeds with my changes: `docker build -t edge-forge .`
- [ ] If I modified grading logic, I updated all four grader locations
- [ ] I have added a test case that would have caught this bug (if applicable)

---

## 9. Appendix — Quick Reference

### Appendix A: Most Common Errors & Fixes

| Error Message | Likely Cause | Quick Fix |
|---------------|--------------|-----------|
| `ImportError: cannot import name 'create_app' from 'openenv.core.env_server.http_server'` | `openenv-core` version too old | `pip install "openenv-core[core]>=0.2.2"` |
| `ImportError: openenv is required. Install with: uv sync` | OpenEnv not installed in current environment | `pip install -e .` or `uv sync` |
| `[DEBUG] WARNING: HF_TOKEN/API_KEY not set` | Missing API key — LLM calls will fail, fallbacks will be used | Set `HF_TOKEN` env var: `export HF_TOKEN=hf_...` |
| `[DEBUG] LLM request failed: ...` | LLM API unreachable, rate-limited, or auth failed | Verify `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` are correct |
| `Episode already terminated. Please reset.` | Calling `/step` after episode reached `MAX_STEPS` (30) or full coverage | Call `/reset` to start a new episode |
| `Age is required` / `Income is required` | Submitted payload without setting `age` or `income` fields | Use `SET_FIELD` actions to set required fields before `SUBMIT` |
| `SSN must be numeric` | Non-numeric string passed as `ssn` after `open_account` → `verify_identity` | This is a **valid edge case** to trigger for `easy_task` — not a bug |
| `SSN missing during pending verification` | `verify_identity` called with no `ssn` after `open_account` set status to `pending` | This is a **valid edge case** to trigger for `hard_task` — not a bug |
| `ConnectionRefusedError` on `http://localhost:8000` | Server not running or wrong port | Start server: `uvicorn server.app:app --host 0.0.0.0 --port 8000` |
| `docker build` fails at `uv sync` | Dependency resolution failure (transient PyPI/GitHub issue) | Retry build; if persistent, check if `openenv-core>=0.2.2` is available |

### Appendix B: OpenEnv Spec Compliance Checklist

Verify these before reporting an API issue:

- [ ] `POST /reset` returns valid `EdgeForgeObservation` with fields: `last_status`, `covered_branches`, `current_input`, `last_error`, `submit_outcomes`
- [ ] `POST /step` returns valid response with `observation`, `reward` (float), and `done` (bool)
- [ ] `GET /state` returns `{"episode_id": "...", "step_count": N}`
- [ ] All grader scores are in range `[0.0, 1.0]` — specifically `[0.01, 0.99]` by design
- [ ] All responses are valid JSON with `Content-Type: application/json`
- [ ] `GET /health` returns HTTP 200
- [ ] `GET /tasks` returns a list of 3 tasks, each with `id`, `name`, and `has_grader: true`
- [ ] HF Space URL returns HTTP 200 on the root path

### Appendix C: Log Format Reference

The exact `[START]`/`[STEP]`/`[END]` format as emitted by `inference.py`:

```
[START] task=<task_id> env=edge_forge_env model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

**Example — complete `easy_task` episode:**

```
[START] task=easy_task env=edge_forge_env model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1 action=SET_FIELD(action=open_account) reward=0.56 done=false error=null
[STEP]  step=2 action=SUBMIT reward=59.00 done=false error=null
[STEP]  step=3 action=SET_FIELD(action=verify_identity) reward=0.11 done=false error=null
[STEP]  step=4 action=SET_FIELD(ssn=abc) reward=0.11 done=false error=null
[STEP]  step=5 action=SUBMIT reward=84.00 done=false error=SSN must be numeric
[END]   success=true steps=5 score=0.990 rewards=0.56,59.00,0.11,0.11,84.00
```

**Format details:**
- `action` field: `SET_FIELD(<field>=<value>)` for SET_FIELD actions, `SUBMIT` or `RESET` for others
- `reward`: two decimal places (`%.2f`)
- `score`: three decimal places (`%.3f`)
- `rewards`: comma-separated, two decimal places each
- `done`/`success`: lowercase `true` or `false`
- `error`: verbatim error string or `null`

### Appendix D: Architecture Quick Reference

```
                ┌──────────────────────────────────────────────────┐
                │                  inference.py                     │
                │   LLM Agent → EdgeForgeEnv client → WebSocket    │
                └───────────────────┬──────────────────────────────┘
                                    │
                                    ▼
                ┌──────────────────────────────────────────────────┐
                │              server/app.py (FastAPI)              │
                │   /reset  /step  /state  /grade  /tasks  /health │
                └───────────────────┬──────────────────────────────┘
                                    │
                                    ▼
                ┌──────────────────────────────────────────────────┐
                │     server/edge_forge_env_environment.py          │
                │   EdgeForgeEnvironment.reset() / .step()         │
                └───────────────────┬──────────────────────────────┘
                                    │
                                    ▼
                ┌──────────────────────────────────────────────────┐
                │              mock_api.py                          │
                │   process_application() → 19 code paths          │
                └──────────────────────────────────────────────────┘
```

---

*Last updated: 2026-04-12 · Edge-Forge v0.1.0 · [Report an issue](https://github.com/SoumyaWasule/Edge-Forge/issues/new)*
