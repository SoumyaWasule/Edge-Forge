---
title: Edge Forge Env
emoji: ⚡
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# Edge-Forge Lite
**Autonomous Synthetic Staging Engine with Stateful Bug Discovery**

![HF Space Status](https://img.shields.io/badge/HF%20Space-Deployed-green?logo=huggingface) ![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue) ![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-100%25-brightgreen) ![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker) ![License](https://img.shields.io/badge/License-BSD_3--Clause-blue)

**Edge-Forge Lite trains agents to autonomously fuzz a stateful loan-processing API, teaching them to chain multi-step payloads (like opening an account before bypassing identity verification) to discover critical production crashes that random fuzzing cannot reach.**

[**View Live on Hugging Face Spaces ↗**](https://huggingface.co/spaces/SoumyaW/edge_forge_env)

---

## 💥 The Problem (Why This Matters)

In the real world, developers cannot test against production data due to strict privacy laws like GDPR and SOC2, leaving edge cases dangerously undiscovered. Traditional fuzzing tools blast APIs with random inputs, but they fundamentally fail against **stateful architectures** where bugs only trigger after a specific sequence of valid API calls (e.g., initiating a loan -> entering pending state -> skipping an SSN verification step). When these undiscovered stateful bugs hit production, they cause silent data corruption, system crashes, and blocked user accounts, costing engineering teams thousands of hours in incident response. Edge-Forge Lite frames edge-case discovery as an RL problem, proving that intelligent agents can learn deep API lifecycles and navigate logic thresholds that defeat naive fuzzers.

---

## 🎯 Environment at a Glance

| Metric | Detail |
| :--- | :--- |
| **Environment Name** | `edge_forge_env` (v0.1.0) |
| **Tasks** | 3 (Easy / Medium / Hard) |
| **Action Space** | Sparse, 3 types: `SET_FIELD`, `RESET`, `SUBMIT` |
| **Observation Space** | Dict (status, input state, errors, outcomes, branch labels) |
| **Reward Range** | `0.01` → `0.99` (mapped from raw points) |
| **Partial Credit** | Yes (`medium_task` scores linearly by unique outcome count / 19) |
| **Stateful** | Yes (Multi-step HTTP sessions via `server.app` locking) |
| **Base Image** | `ghcr.io/meta-pytorch/openenv-base:latest` |
| **HF Space** | *Deployed* |

---

## 🏗️ Architecture Diagram

```text
                      +---------------------------------------+
                      |         Qwen2.5-72B LLM Agent         |
                      +---------------------------------------+
                                  |                 ^
      {"action_type": "SUBMIT"}   |                 |  Observation Models
                                  V                 |  (status, error, etc.)
  +========================================================================+
  |                              Docker Boundary                           |
  | +--------------------------------------------------------------------+ |
  | |                        OpenEnv API Layer                           | |
  | |   POST /step (Stateful) | POST /reset (Stateful) | GET /state      | |
  | +--------------------------------------------------------------------+ |
  |            |                                            |              |
  |            v                                            v              |
  | +----------------------+                        +--------------------+ |
  | | EdgeForgeEnvironment |                        |     Graders        | |
  | |  - Tracks Step #     |  <-- submit_outcomes --|  - grade_easy()    | |
  | |  - Calculates Reward |                        |  - grade_medium()  | |
  | |  - Validates Types   |                        |  - grade_hard()    | |
  | +----------------------+                        +--------------------+ |
  |            |                                                           |
  |            v                                                           |
  | +--------------------------------------------------------------------+ |
  | |                          Mock API System                           | |
  | |  - 19 Code Branches (6 Logic Layers)                               | |
  | |  - app_state: {status: pending, verification_attempts: 1}          | |
  | |  - thresholds: {enterprise_days: 350, age_limit: 18}               | |
  | +--------------------------------------------------------------------+ |
  +========================================================================+
```

*(Note: Edge-Forge Lite removes OpenEnv's default stateless routers and implements custom thread-safe logic to maintain an internal `_env_instance` for true RL episode pacing.)*

---

## 🪜 Task Ladder

| # | Task Name | Difficulty | What the Agent Must Do | Grader Logic | Max Score |
|---|---|---|---|---|---|
| 1 | `easy_task` | Easy | Submit a payload with `action: verify_identity` and `ssn: "abc"` to trigger string-validation error. | Checks if `"SSN must be numeric"` is identically matched in `submit_outcomes`. | 0.99 |
| 2 | `medium_task` | Medium | Maximize API branch coverage by submitting varied combinations of age, income, balance, and regions. | Counts unique actual API response strings divided by total possible (`19`), clamped to float bounds. | 0.99 |
| 3 | `hard_task` | Hard | Trigger a stateful crash trap. Must submit `action: open_account`, then submit `action: verify_identity` but omit the SSN. | Checks if `"SSN missing during pending verification"` is identically matched in `submit_outcomes`. | 0.99 |

**Progression Analysis:**
The difficulty scales from a simple field-typing validation (Easy), to a brute-force combination search space (Medium), to a strict multi-step state sequence (Hard). `hard_task` is exceedingly difficult for non-RL approaches because the crash *only* exists if the API's internal `app_state["status"]` is set to `"pending"` during a prior `SUBMIT` step. The agent must sequence interactions correctly and remember its embedded session state to hit the trap.

---

## 📈 Reward Function Deep Dive

Edge-Forge implements process supervision to guide the agent through its complex state space, combined with strict efficiency penalties:

* **0.0 (Failure / Penalties):** 
  * `-2.0` for typing invalid data (e.g. passing a string for `age`).
  * `-1.0` step penalty to force efficiency.
  * `-1.0` for passing unknown fields.
* **Partial Credit / Intermediate (> 0.0):** 
  * `+0.5 * completeness` given when setting a field, proportional to how many fields are correctly populated. 
  * `+1.0` exploration bonus when the agent correctly uses the `RESET` action after a `SUBMIT`.
  * `+10.0` for discovering any new unique branch.
* **1.0 / Major Milestones:** 
  * `+50.0` for triggering an error on a new branch.
  * `+25.0` bonus for hitting specific deep states (`deep_branch`, `stateful_crash`, `ssn_format_bug`, `enterprise_debt_recovery`, etc.).
  * `+100.0` massive completion bonus for discovering all 19 branches before the 30-step episode limit (`MAX_STEPS`).

*(Note: The environment's internal RL reward returns integers/floats > 1.0 during steps to guide RL policies. However, the OpenEnv compliance graders explicitly score final episodic outputs strictly inside `0.01 -> 0.99` bounds to adhere to standard validation limitations.)*

---

## 🔍 Observation & Action Space Reference

### Observation Space
```python
class EdgeForgeObservation(Observation):
    last_status: int           # HTTP-like status: 200 (success), 500 (error)
    covered_branches: List[str]# List of code branches discovered so far
    current_input: Dict        # Current input payload being built by agent
    last_error: Optional[str]  # Error message if submission failed
    submit_outcomes: List[str] # Observable API response strings
```
The agent observes the direct outcome of its most recent step, accumulated coverage, and its current built-up payload. The environment is *partially observable*; the agent *never* sees the internal `app_state` (e.g., `verification_attempts`) or the stochastic thresholds (e.g., `enterprise_days` limits), forcing it to probe blindly until it maps the boundary.

### Action Space
```python
class EdgeForgeAction(Action):
    action_type: str       # "SET_FIELD", "RESET", or "SUBMIT"
    field: Optional[str]   # "age", "income", "user_type", "action", "ssn", etc.
    value: Optional[Any]   # Value matching the field constraint
```
The constraint here turns standard dictionary manipulation into a multi-step MDP. Because the agent can only `SET_FIELD` one property per step, assembling a "perfect" user payload takes several actions before a `SUBMIT`. Any type hallucination incurs an immediate `-2.0` penalty.

---

## ⚡ Quickstart (Exact Commands)

### Option A: Run via Docker (Recommended)
```bash
# 1. Build the container natively
docker build -t edge_forge_env:latest .

# 2. Run the environment mapping strictly to port 8000
docker run -p 8000:8000 -e IMAGE_NAME=edge_forge_env:latest edge_forge_env:latest

# 3. Test functionality directly
curl -X POST http://localhost:8000/reset
```

### Option B: Run Locally
```bash
# 1. Install dependencies via uv
uv sync

# 2. Start the FastAPI server internally
uv run uvicorn edge_forge_env.server.app:app --host 0.0.0.0 --port 8000

# 3. In a separate terminal, execute the OpenEnv standard inference loop
export HF_TOKEN="your_hf_token"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export ENV_BASE_URL="http://localhost:8000"
uv run python inference.py
```

### Option C: Use the live HF Space
Visit the live API directly. It fully supports `POST /reset`, `POST /step`, and `GET /state` per OpenEnv requirements and will stream back standard HTTP observation outputs locally.

**Expected Baseline Output Log (from inference.py)**
```text
[START] task=medium_task env=edge_forge_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=SET_FIELD(age=25) reward=-0.94 done=false error=null
[STEP] step=2 action=SET_FIELD(income=50000) reward=-0.89 done=false error=null
[STEP] step=3 action=SUBMIT reward=8.00 done=false error=null
[END] success=true steps=3 score=0.158 rewards=-0.94,-0.89,8.00
```

---

## 📊 Baseline Results

| Task | Random Baseline | LLM Agent (Qwen 2.5 72B) | Delta |
|------|-----------------|--------------------------|-------|
| `easy_task` | ~0.01 | 0.99 | +0.98 |
| `medium_task` | ~0.15 | ~0.45* | +0.30 |
| `hard_task` | ~0.01 | 0.99 | +0.98 |

*\*The Medium task evaluates exploration coverage out of 19 total API outcomes. Random agents cannot strategically expand logic branches, whereas the Qwen agent demonstrates planning by retaining context and iterating across fields.*

*(Note: Maximum runtime consistently clocks under 4 minutes per episodic loop, firmly satisfying the 20-minute SLA on 2 vCPU / 8 GB RAM hackathon spec limits.)*

---

## 🧠 Technical Architecture & Novel Design Decisions

**1. Simulating Core Software Engineering Failures**  
Edge-Forge Lite is designed around the philosophy that good RL environments map to expensive real-world problems. Fuzzing API structures with complex types, mocked privacy restrictions, and state dependencies is a multi-million dollar QA issue. The schema requires process-supervised exploration, rewarding the agent not just for finishing, but for successfully compiling complex dictionary payloads step-by-step.

**2. Stateful Session Entanglement**  
Unlike typical OpenEnv configurations which run strictly as stateless request-response loops, Edge-Forge intentionally patches OpenEnv's stateless HTTP routes inside `app.py`. It uses a threading `_env_lock` natively over `_env_instance` to preserve genuine HTTP session persistence. This structurally allows agents to trigger the deeply embedded `stateful_crash`—a bug that ONLY manifests if the API receives an `open_account` submit, transitions its internal machine to `pending`, and *then* receives a `verify_identity` submit missing an SSN.

**3. Deterministic Validation Through Observable Outcomes**  
Many LLM-graded environments use "LLM-as-a-judge", which introduces unreliability, non-determinism, and execution latency. Edge-Forge instead relies on 100% deterministic logic. The environment traps output natively from the `mock_api.py` and stores raw API messages in `submit_outcomes`. The graders (`tasks.py`) strictly parse these observable output strings for exact matches, eliminating hallucination in the reward metric and ensuring high-speed continuous evaluation.

**4. OpenEnv Strict Spec Implementation Constraints**  
To comply perfectly with the standard SDK boundary, the environment leverages complete Pydantic typed validation (`EdgeForgeAction`, `EdgeForgeObservation`). In `inference.py`, specific coercion strategies (`coerce_field_value`) handle LLM string-int hallucination faults linearly before natively executing the SDK `env.step()`. This securely mirrors real-world production parsing layers defensively filtering agent outputs.

**5. Multi-Step Planning and Tool Use Deficits**  
The LLM cannot dump a completed JSON. It must use single-field tools incrementally (`SET_FIELD`), materially retaining memory of its previous actions to plan when to optimally emit the `SUBMIT` action, utilizing `RESET` to pivot its search space dynamically out of deep thresholds.

**6. The Missing Ecosystem Segment**  
Within the OpenEnv Hub, environments tilt heavily towards pure coding, logical puzzles, or 2D game proxies. Edge-Forge Lite introduces foundational **stateful black-box penetration testing**. It solidly serves as a benchmark for measuring if an LLM can functionally infer hidden system architecture and deliberately break it using chained payloads.

---

## ✅ OpenEnv Spec Compliance

- [x] ✅ **`openenv.yaml` with all required fields** — Located at root, defining `edge_forge_env`, `0.1.0`. 
- [x] ✅ **Typed Pydantic models for Action, Observation, State** — Defined strictly in `models.py` natively integrating OpenEnv core base types.
- [x] ✅ **`step()` endpoint implemented** — Custom stateful hook inside `server/app.py`.
- [x] ✅ **`reset()` endpoint implemented** — Custom stateful hook inside `server/app.py`.
- [x] ✅ **`state()` endpoint implemented** — Custom stateful hook returning embedded step counts inside `server/app.py`.
- [x] ✅ **Minimum 3 tasks with graders** — `easy_task`, `medium_task`, `hard_task` bound in `tasks.py`.
- [x] ✅ **Reward scores in 0.0–1.0 range** — All graders clamped structurally to `0.01`–`0.99`.
- [x] ✅ **`inference.py` in root directory** — Executes strict inference loop mapped against spec.
- [x] ✅ **`[START]/[STEP]/[END]` log format** — Exact format parsing compliant natively emitted into stdout.
- [x] ✅ **Dockerfile builds successfully** — Configured correctly exposing port 8000 traversing standard UV constraints.
- [x] ✅ **Deployed to Hugging Face Spaces** — Custom HTML 200 HTTP dashboard exposed natively on boot. 
- [x] ✅ **Runtime < 20 minutes on 2 vCPU / 8GB RAM** — Fully async inference loops operate under 4m structurally.
- [x] ✅ **API_BASE_URL, MODEL_NAME, HF_TOKEN** — Ingested natively via `os.getenv` into OpenAI client inside `inference.py`.
- [x] ✅ **OpenAI client used for all LLM calls** — Executed exactly as `client.chat.completions.create`.

---

## 🚀 Why Edge-Forge Lite Belongs in the OpenEnv Hub

Edge-Forge Lite functionally bridges the structural gap between basic toy logic puzzles and highly complex software QA testbeds. The current OpenEnv registry completely lacks environments dedicated to **stateful API exploration and synthetic payload sequencing**. By permanently including Edge-Forge, Meta researchers can efficiently benchmark model capabilities natively embedded across deep stateful memory retention, strict tool-chaining, and architectural inference, scaling RL agents into rigorous, self-driving QA automation engineers.

---

## 📁 Project Structure

```text
edge-forge-lite/
├── openenv.yaml          # Environment specification manifest
├── Dockerfile            # Container definition utilizing uv package management
├── inference.py          # Compliant OpenEnv standard LLM-agent inference loop
├── pyproject.toml        # Application dependency & module maps
├── client.py             # OpenEnv SDK client definition wrapper
├── models.py             # Precise Pydantic configurations for typing Actions/Observations
├── mock_api.py           # Core Application Logic: 19 branches, 6 layers, thresholds
├── tasks.py              # Task definitions and deterministic string-match grader logic
└── server/
    ├── app.py            # FastAPI implementation overriding standard stateless routers
    └── edge_forge_env_environment.py # Python RL MDP rules, reward shaping, and constraints
```

---

## 🤝 Contributing & License

**Testing Environment Statefulness Locally**
Utilize the built-in logic testing loops natively by running `uv run pytest` (requires `#pytest>=8.0.0` structured from `[dev]` dependencies inside `pyproject.toml`).

**Adding New Grader Tasks**
1. Modify `mock_api.py` to insert new logic branches or expected failure outputs into `TOTAL_BRANCHES`.
2. Update `TOTAL_OUTCOMES` explicitly in `tasks.py` and `app.py`.
3. Add a new explicit grader sequentially matching your expected string output evaluated against `submit_outcomes`.
4. Append task definition explicitly to `TASKS` structurally inside `server/app.py` and `tasks.py`. 

**License**
Licensed natively under the BSD-3-Clause License.

*For OpenEnv ecosystem telemetry, join the [OpenEnv Discord](https://discord.gg/) or review explicit tracker issues natively embedded onto the Meta PyTorch GitHub.*
