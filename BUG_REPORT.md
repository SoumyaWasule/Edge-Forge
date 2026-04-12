# 🐛 Bug Reporting Guide — Edge-Forge

> A well-formed report gets fixed in hours. An incomplete report sits in triage for weeks.

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-v1-blue)](https://github.com/meta-pytorch/OpenEnv)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-green)](./LICENSE)

Edge-Forge is an RL environment built on the OpenEnv protocol, containerized with Docker, and deployed to Hugging Face Spaces. Its failure modes span the OpenEnv SDK client, FastAPI server, stateful mock API, Docker layer, and LLM-driven inference script. This guide provides structured templates and diagnostic procedures for each failure category.

---

## 1. Before You Report

Work through this checklist first. If any step resolves your issue, no report is needed.

```bash
# 1. Verify Python (must be 3.10–3.12)
python --version

# 2. Verify core dependencies
pip show openenv-core openai fastapi uvicorn pydantic

# 3. Verify Docker (if applicable)
docker info

# 4. Verify environment variables
echo $HF_TOKEN          # non-empty
echo $API_BASE_URL       # LLM endpoint (default: https://router.huggingface.co/v1)
echo $MODEL_NAME         # model ID (default: Qwen/Qwen2.5-72B-Instruct)

# 5. Run smoke tests
pytest tests/test_environment.py -v

# 6. Quick health check
curl -sf http://localhost:8000/health && echo "OK" || echo "FAILED"
```

Also check [Section 5: Known Issues](#5-known-issues--limitations) and [existing GitHub Issues](https://github.com/SoumyaWasule/Edge-Forge/issues).

---

## 2. Severity Classification

| Severity | Label | Definition | SLA |
|----------|-------|------------|-----|
| 🔴 Critical | `severity: critical` | Environment completely unusable — Docker build fails, server crashes, `/reset` returns 500 | Same day |
| 🟠 High | `severity: high` | One task/feature broken — grader always returns `0.01`, `step()` hangs, reward is `NaN` | 48 hours |
| 🟡 Medium | `severity: medium` | Degraded but workaround exists — intermittent timeout, wrong branch label, LLM fallback always triggers | 1 week |
| 🟢 Low | `severity: low` | Cosmetic — typo, docstring error, minor log formatting | Next release |
| 💡 Enhancement | `severity: enhancement` | Feature request — new task, reward tuning, new API branch | Backlog |

```
Is the environment completely unusable?
├── YES → 🔴 Critical
└── NO → Is one task/feature completely broken?
    ├── YES → 🟠 High
    └── NO → Does a workaround exist?
        ├── YES → 🟡 Medium
        └── NO → 🟠 High
```

---

## 3. Bug Report Templates

Select the template matching your failure. Each is self-contained.

<details>
<summary>📋 Template 1: Environment / Deployment Failure</summary>

```
Title:     [DEPLOYMENT] <brief description>
Severity:  🔴 Critical | 🟠 High
Date:      YYYY-MM-DD
Reporter:  @github-username
```

**Environment:** Edge-Forge version (`git rev-parse --short HEAD`), Python version, openenv-core version, OS/platform, deployment target (Local / HF Space).

**Diagnostic — run and paste output:**

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 2>&1 | head -50
```

**Reproduction Steps:** Numbered list of exact commands from fresh clone to failure.

**Expected:** Server starts on port 8000, `/health` returns 200.
**Actual:** [Exact error with full traceback.]

**Impact:** Who is affected / workaround available / blocks validator?

</details>

<details>
<summary>📋 Template 2: OpenEnv API Failure (/reset, /step, /state)</summary>

```
Title:     [API] <brief description>
Severity:  🔴 Critical | 🟠 High | 🟡 Medium
Date:      YYYY-MM-DD
Reporter:  @github-username
```

**Affected Endpoint:** `/reset` / `/step` / `/state` / `/grade` / `/tasks`

**Diagnostic — run the failing call and paste output:**

```bash
# Reset
curl -s -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}' | python -m json.tool

# Step (SET_FIELD)
curl -s -X POST http://localhost:8000/step -H "Content-Type: application/json" \
  -d '{"action_type": "SET_FIELD", "field": "age", "value": 25}' | python -m json.tool

# Step (SUBMIT)
curl -s -X POST http://localhost:8000/step -H "Content-Type: application/json" \
  -d '{"action_type": "SUBMIT"}' | python -m json.tool

# State
curl -s http://localhost:8000/state | python -m json.tool
```

**Request payload sent:** [exact JSON]
**Response received:** [exact JSON + HTTP status]
**Expected response:** Per `EdgeForgeObservation` model — `last_status` (int), `covered_branches` (List[str]), `current_input` (Dict), `last_error` (Optional[str]), `submit_outcomes` (List[str]).

**Reproduction Steps:** Numbered list.
**Impact:** Who is affected / workaround / blocks validator?

</details>

<details>
<summary>📋 Template 3: Task / Grader Failure</summary>

```
Title:     [GRADER] <brief description>
Severity:  🟠 High | 🟡 Medium
Date:      YYYY-MM-DD
Reporter:  @github-username
```

**Affected Task:**
- `easy_task` — Trigger SSN Error (`grade_easy`: expects `"SSN must be numeric"` in `submit_outcomes`)
- `medium_task` — Maximize Coverage (`grade_medium`: `unique_outcomes / 19`)
- `hard_task` — Stateful Crash Trap (`grade_hard`: expects `"SSN missing during pending verification"`)

**Score returned:** [e.g., 0.01] · **Score expected:** [e.g., 0.99]
**`submit_outcomes` at grading time:** [paste list]

**Diagnostic — reproduce in isolation:**

```python
from server.edge_forge_env_environment import EdgeForgeEnvironment
from models import EdgeForgeAction

env = EdgeForgeEnvironment()
env.reset()
# [paste exact action sequence]
print("submit_outcomes:", env.state["submit_outcomes"])
print("grade:", env.grade_easy())  # or grade_medium / grade_hard
```

**Grader consistency:** Do all four locations (`graders.py`, `tasks.py`, `server/app.py`, `server/edge_forge_env_environment.py`) return the same score?

</details>

<details>
<summary>📋 Template 4: Inference Script Failure</summary>

```
Title:     [INFERENCE] <brief description>
Severity:  🔴 Critical | 🟠 High | 🟡 Medium
Date:      YYYY-MM-DD
Reporter:  @github-username
```

**Config:** `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` set (yes/no), `IMAGE_NAME` (if Docker).

**Failure type:** Script exits non-zero / log format malformed / LLM calls fail / scores out of range / fallback always triggers.

**Diagnostic:**

```bash
API_BASE_URL=$API_BASE_URL MODEL_NAME=$MODEL_NAME HF_TOKEN=$HF_TOKEN \
  python inference.py > stdout.log 2> stderr.log

grep -E "^\[START\]|^\[END\]" stdout.log    # expect 3 of each
grep "\[DEBUG\]" stderr.log                  # fallback/error indicators
```

**stdout ([START]/[STEP]/[END] lines):** [paste]
**stderr ([DEBUG] lines):** [paste]

**Expected log format:**
```
[START] task=easy_task env=edge_forge_env model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1 action=SET_FIELD(action=open_account) reward=0.56 done=false error=null
[END]   success=true steps=5 score=0.990 rewards=0.56,59.00,...
```

</details>

<details>
<summary>📋 Template 5: Docker / HF Space Deployment Failure</summary>

```
Title:     [DOCKER] <brief description>
Severity:  🔴 Critical | 🟠 High
Date:      YYYY-MM-DD
Reporter:  @github-username
```

**Docker version / OS / HF Space URL (if applicable).**

**Failure type:** Build fails / container crashes / `/health` non-200 / port 8000 inaccessible / `uv sync` fails / env var missing.

**Diagnostic:**

```bash
# Build
docker build --no-cache --progress=plain -t edge-forge . 2>&1 | tee build.log

# Run
docker run -e API_BASE_URL=$API_BASE_URL -e MODEL_NAME=$MODEL_NAME \
  -e HF_TOKEN=$HF_TOKEN -p 8000:8000 edge-forge

# Logs
docker logs $(docker ps -lq) --tail 100

# Shell into container
docker run --entrypoint /bin/bash -it edge-forge
```

**Dockerfile context:** Base `ghcr.io/meta-pytorch/openenv-base:latest`, entrypoint `uvicorn server.app:app --host 0.0.0.0 --port 8000`, health check on `/health`, package manager `uv sync`.

**Build log (last 50 lines):** [paste]
**Container logs:** [paste]

</details>

---

## 4. Diagnostic Quick Reference

### API Endpoints

```bash
curl -sf http://localhost:8000/health                                          # Health
curl -s -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}'  # Reset
curl -s http://localhost:8000/state                                             # State
curl -s http://localhost:8000/tasks                                             # Tasks
curl -s -X POST http://localhost:8000/grade -H "Content-Type: application/json" \
  -d '{"task_id": "easy_task"}'                                                # Grade
```

### Grader Isolation Test

```bash
python -c "
from server.edge_forge_env_environment import EdgeForgeEnvironment
env = EdgeForgeEnvironment()
env.reset()
for g in [env.grade_easy, env.grade_medium, env.grade_hard]:
    score = g()
    assert 0.0 <= score <= 1.0, f'Out of range: {score}'
    print(f'{g.__name__}: {score}')
"
```

---

## 5. Known Issues & Limitations

Do **not** file reports for these — they are documented behaviors.

| ID | Issue | Status | Workaround |
|----|-------|--------|------------|
| KI-001 | **Single-session server** — one `EdgeForgeEnvironment` instance shared across all requests (`server/app.py:244`). Concurrent agents corrupt state. | By Design | Serialize calls; use separate containers for parallel eval. |
| KI-002 | **Stochastic thresholds** — `age_limit` (16–21) and `enterprise_days` (300–400) randomize on each `reset()` (`edge_forge_env_environment.py:104-107`). | By Design | Design agents to adapt rather than memorize fixed mappings. |
| KI-003 | **MAX_STEPS = 30** — episodes terminate at 30 steps regardless of task. `easy_task` further limits to 10, `hard_task` to 15. | By Design | Optimize action sequences — easy solves in 5 steps, hard in 4. |
| KI-004 | **No session persistence** — environment state is in-memory (`edge_forge_env_environment.py:70`). Container restarts lose all state. | Won't Fix v1.0 | Call `/reset` after any restart. |
| KI-005 | **Graders duplicated in 4 files** — `graders.py`, `tasks.py`, `server/app.py`, `edge_forge_env_environment.py`. All return identical scores. | Planned v1.1 | Update all four if modifying grading logic. |
| KI-006 | **`uv.lock` excluded from Docker** — fresh `uv sync` resolve on every build (`.dockerignore`). Slower builds but avoids cross-platform lockfile issues. | By Design | Retry on transient resolution failures. |

---

## 6. Bug Triage Process

```
Filed → Labelled (24h) → Confirmed (48h) → Assigned → In Progress → Fixed → Verified → Closed
```

| Label | Meaning |
|-------|---------|
| `severity: critical/high/medium/low` | See [Section 2](#2-severity-classification) |
| `status: needs-info` | Incomplete report — 7 days to respond before auto-close |
| `status: confirmed` | Reproduced by maintainer |
| `status: in-progress` / `status: fixed` | Fix underway / merged |
| `type: deployment` / `type: api` / `type: task` / `type: inference` | Failure category |
| `type: spec-compliance` | OpenEnv spec violation |
| `duplicate` / `wontfix` / `good-first-issue` | Standard labels |

---

## 7. Security Vulnerabilities

> **Do NOT file security vulnerabilities as public GitHub issues.**

Report privately via [GitHub Security Advisories](https://github.com/SoumyaWasule/Edge-Forge/security/advisories). Include: description, reproduction steps, impact, and suggested fix. Acknowledgement within 48 hours.

**Security-relevant areas:** Docker isolation (`Dockerfile`), HF Token handling (`inference.py:46`), API key usage (`inference.py:485`), input validation boundary (`edge_forge_env_environment.py:44-49`), Pydantic schema enforcement (`models.py`), request parsing (`server/app.py:266-274`).

---

## 8. Contributing a Fix

```bash
git clone https://github.com/SoumyaWasule/Edge-Forge.git && cd Edge-Forge
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn server.app:app --host 0.0.0.0 --port 8000   # verify server starts
pytest tests/test_environment.py -v                    # verify tests pass
```

### PR Checklist

- [ ] Bug is reproducible on `main` before my change
- [ ] Fix doesn't break existing functionality
- [ ] `pytest tests/test_environment.py -v` — all pass
- [ ] `python inference.py` — baseline scores not regressed
- [ ] `[START]`/`[STEP]`/`[END]` log format unchanged
- [ ] `docker build -t edge-forge .` succeeds
- [ ] If grading logic changed, all four grader locations updated
- [ ] Test case added for the bug (if applicable)

---

## 9. Appendix

### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ImportError: cannot import name 'create_app'` | `openenv-core` too old | `pip install "openenv-core[core]>=0.2.2"` |
| `[DEBUG] WARNING: HF_TOKEN/API_KEY not set` | Missing API key | `export HF_TOKEN=hf_...` |
| `[DEBUG] LLM request failed: ...` | LLM unreachable or auth failed | Verify `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` |
| `Episode already terminated. Please reset.` | Step called after MAX_STEPS or full coverage | Call `/reset` |
| `Age is required` / `Income is required` | Submitted without required fields | `SET_FIELD` before `SUBMIT` |
| `SSN must be numeric` | Non-numeric SSN after open→verify sequence | **Expected** for `easy_task` — not a bug |
| `SSN missing during pending verification` | verify_identity without SSN after open_account | **Expected** for `hard_task` — not a bug |
| `ConnectionRefusedError` on localhost:8000 | Server not running | `uvicorn server.app:app --host 0.0.0.0 --port 8000` |

### Log Format Reference

```
[START] task=<task_id> env=edge_forge_env model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
```

### OpenEnv Compliance Checklist

- [ ] `POST /reset` returns `EdgeForgeObservation` with all fields
- [ ] `POST /step` returns `observation`, `reward` (float), `done` (bool)
- [ ] `GET /state` returns `{"episode_id": "...", "step_count": N}`
- [ ] Grader scores in `[0.01, 0.99]`
- [ ] `GET /health` returns HTTP 200
- [ ] `GET /tasks` returns 3 tasks with `has_grader: true`

---

*Last updated: 2026-04-12 · Edge-Forge v0.1.0 · [Report an issue](https://github.com/SoumyaWasule/Edge-Forge/issues/new)*
