# ⚡ Edge-Forge — Quickstart Guide

Get a running RL loop in under 5 minutes.

| Path | Time |
|---|---|
| Live API (no install) | ~1 min |
| Python SDK client | ~3 min |
| Local Docker | ~5 min |
| Run inference baseline | ~7 min |

| Goal | Path |
|---|---|
| Poke the API with curl | [Path A](#path-a-live-api--zero-install) |
| Build an agent in Python | [Path B](#path-b-python-sdk-client) |
| Offline / high-throughput eval | [Path C](#path-c-local-docker) |
| Run the LLM baseline agent | [Path D](#path-d-run-the-baseline-inference-agent) |

---

## Path A: Live API — Zero Install

No pip. No Docker. Curl only.

**Step 1 — Confirm the Space is live:**
```bash
curl https://soumyaw-edge-forge-env.hf.space/health
```
✅ Expected: `{"status":"healthy"}`

**Step 2 — Start an episode:**
```bash
curl -X POST https://soumyaw-edge-forge-env.hf.space/reset \
  -H "Content-Type: application/json" -d '{}'
```
✅ Expected: JSON with `observation` containing empty `covered_branches`, `current_input: {}`, `submit_outcomes: []`

**Step 3 — Set fields and submit:**
```bash
curl -X POST https://soumyaw-edge-forge-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "SET_FIELD", "field": "action", "value": "verify_identity"}'

curl -X POST https://soumyaw-edge-forge-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "SET_FIELD", "field": "ssn", "value": "abc"}'

curl -X POST https://soumyaw-edge-forge-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "SUBMIT"}'
```
✅ Expected: `last_status: 500`, `last_error: "SSN must be numeric"`, reward includes branch discovery bonus

**Step 4 — Check state:**
```bash
curl https://soumyaw-edge-forge-env.hf.space/state
```
✅ Expected: `{"episode_id": "current", "step_count": 3}`

You've completed a full reset → set fields → submit → observe loop. The environment is working.

---

## Path B: Python SDK Client

**Step 1 — Install:**
```bash
pip install "openenv-core[core]>=0.2.2" "openai>=1.0.0"
```

**Step 2 — Run a minimal loop:**
```python
import asyncio
from openenv.core import EnvClient
from openenv.core.env_server.types import Action
from pydantic import Field
from typing import Optional, Any

class EdgeForgeAction(Action):
    action_type: str = Field(...)
    field: Optional[str] = Field(default=None)
    value: Optional[Any] = Field(default=None)

async def run():
    env = EnvClient(base_url="https://soumyaw-edge-forge-env.hf.space")
    try:
        result = await env.reset()
        print(f"Initial obs: {result}")
        actions = [
            {"action_type": "SET_FIELD", "field": "action", "value": "verify_identity"},
            {"action_type": "SET_FIELD", "field": "ssn", "value": "abc"},
            {"action_type": "SUBMIT"},
        ]
        for i, act in enumerate(actions, 1):
            result = await env.step(EdgeForgeAction(**act))
            print(f"Step {i} | reward={result.reward:.2f} | done={result.done}")
        print(f"\nSubmit outcomes: {result.observation.submit_outcomes}")
    finally:
        await env.close()

asyncio.run(run())
```
✅ Expected: Step log showing rewards, ending with `"SSN must be numeric"` in `submit_outcomes`

**Step 3 — Explore further:** Replace the action sequence with varied `age`, `income`, `balance`, `user_type`, `region` combinations. Each unique API response path adds to `covered_branches`. Target: 19 branches.

---

## Path C: Local Docker

For offline use, CI pipelines, or high-throughput parallel evaluation.

**Step 1 — Build:**
```bash
git clone https://github.com/SoumyaWasule/Edge-Forge.git && cd Edge-Forge
docker build -t edge_forge_env:latest .
```

**Step 2 — Run:**
```bash
docker run -d --name edge-forge -p 8000:8000 \
  -e IMAGE_NAME=edge_forge_env:latest --memory=512m --cpus=1 \
  edge_forge_env:latest
```

**Step 3 — Verify (~5s for Uvicorn startup):**
```bash
curl http://localhost:8000/health
```
✅ Expected: `{"status":"healthy"}`

**Step 4 — Use it:** Point any client at `http://localhost:8000` — identical API.

**Step 5 — Cleanup:**
```bash
docker stop edge-forge && docker rm edge-forge
```

---

## Path D: Run the Baseline Inference Agent

Runs `inference.py` — an LLM-powered agent using Qwen 2.5 72B.

**Step 1 — Clone and install:**
```bash
git clone https://github.com/SoumyaWasule/Edge-Forge.git && cd Edge-Forge
pip install -e ".[dev]"
```

**Step 2 — Set environment variables:**
```bash
export HF_TOKEN="your_hf_token_here"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export API_BASE_URL="https://router.huggingface.co/v1"
export ENV_BASE_URL="https://soumyaw-edge-forge-env.hf.space"
```

**Step 3 — Run:**
```bash
python inference.py
```
✅ Expected:
```
[START] task=easy_task env=edge_forge_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=SET_FIELD(action=verify_identity) reward=-0.50 done=false error=null
[STEP] step=2 action=SET_FIELD(ssn=abc) reward=-0.25 done=false error=null
[STEP] step=3 action=SUBMIT reward=50.00 done=false error=null
[END] success=true steps=3 score=0.990 rewards=-0.50,-0.25,50.00
```

**Baseline scores:**

| Task | Random | LLM Baseline | Challenge |
|---|---|---|---|
| `easy_task` | ~0.01 | **0.99** | Single SSN validation path |
| `medium_task` | ~0.15 | **~0.45** | 19-branch systematic exploration |
| `hard_task` | ~0.01 | **0.99** | Two-step stateful crash sequence |

---

## Explore Interactively

- **Swagger UI:** [soumyaw-edge-forge-env.hf.space/docs](https://soumyaw-edge-forge-env.hf.space/docs) — execute every endpoint from the browser
- **Dashboard:** [soumyaw-edge-forge-env.hf.space](https://soumyaw-edge-forge-env.hf.space/) — environment status, tasks, action space overview

---

## API Quick Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check |
| `/reset` | POST | Start new episode |
| `/step` | POST | Take one action (`action_type`, `field`, `value`) |
| `/state` | GET | Current episode state |
| `/tasks` | GET | List tasks with grader status |
| `/grade` | POST | Grade a task (`task_id`) |
| `/docs` | GET | Swagger UI |

**Actions:** `SET_FIELD` (set a payload field) · `SUBMIT` (send payload to mock API) · `RESET` (clear payload)

**Valid Fields:** `age` (int) · `income` (int) · `balance` (int) · `user_type` (string) · `region` (string) · `days_active` (int) · `credit_score` (int) · `action` (string) · `ssn` (string/int)

---

## Reward Signal & Task Scoring

**Step rewards (internal RL):**
- `+10.0` new branch · `+50.0` new error branch · `+25.0` deep/stateful branch · `-2.0` type violation · `-1.0` per step

**Grader scores (OpenEnv, 0.01–0.99):**

| Task | Objective | Scoring |
|---|---|---|
| `easy_task` | Trigger `"SSN must be numeric"` | Binary: 0.99 or 0.01 |
| `medium_task` | Maximize unique API outcomes | Linear: unique / 19 |
| `hard_task` | Trigger `"SSN missing during pending verification"` | Binary: 0.99 or 0.01 |

The hard task requires `open_account` → `verify_identity` (no SSN). Random agents almost never find it.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `curl: (7) Failed to connect` | Space sleeping — send any request to wake it, wait 10s, retry |
| `{"detail": "Method Not Allowed"}` | POST for `/reset` and `/step`, GET for `/state` and `/health` |
| Reward always negative | Must `SUBMIT` to trigger branch rewards — `SET_FIELD` alone won't score |
| Type penalty (`-2.0`) | Wrong type — `age` needs int `25`, not string `"25"` |
| `Episode already terminated` | Max 30 steps — call `/reset` for new episode |
| Docker port conflict | Use `-p 8001:8000`, update BASE_URL to `:8001` |
| Import errors | Run `pip install -e .` from project root |

---

## 📚 More Documentation

| Document | What it covers |
|---|---|
| [README.md](./README.md) | Architecture, task design, reward shaping, baseline analysis |
| [HF_SPACE_SETUP.md](./HF_SPACE_SETUP.md) | Deploying your own instance to HuggingFace Spaces |
| [DEPLOYMENT.md](./DEPLOYMENT.md) | Four deployment modes, hardware reqs, RL training integration |
| [inference.py](./inference.py) | Baseline LLM agent with fallback action sequences |
| [openenv.yaml](./openenv.yaml) | Environment specification manifest |
| [/docs](https://soumyaw-edge-forge-env.hf.space/docs) | Interactive API documentation (Swagger) |
