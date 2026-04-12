# 🚀 Deployment Guide — Edge-Forge

> **TL;DR for hackathon judges:** The live Hugging Face Space is at
> **[https://soumyaw-edge-forge-env.hf.space](https://soumyaw-edge-forge-env.hf.space)**
>
> To reproduce locally:
> ```bash
> docker build -t edge-forge:latest .
> docker run -p 8000:8000 edge-forge:latest
> curl http://localhost:8000/health
> ```

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)
![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-100%25-brightgreen)
![HF Space](https://img.shields.io/badge/HF%20Space-Deployed-green?logo=huggingface)

Edge-Forge supports four deployment modes: **Local Uvicorn** for rapid development with hot-reload, **Local Docker** for exact reproduction of the Hugging Face Space container, **`openenv push`** for one-command deployment to HF Spaces (recommended for first-time setup), and **Manual Git Push** for direct version control over every file in the Space repository. Choose the mode that fits your workflow — all four produce an identical OpenEnv-compatible environment serving on port `8000`.

---

## Table of Contents

1. [Prerequisites & Hardware Requirements](#1-prerequisites--hardware-requirements)
2. [Choosing Your Deployment Mode](#2-choosing-your-deployment-mode)
3. [Mode A: Local Uvicorn — Development](#3-mode-a-local-uvicorn--development)
4. [Mode B: Local Docker — Exact Reproduction](#4-mode-b-local-docker--exact-reproduction)
5. [Mode C: Hugging Face Spaces via `openenv push`](#5-mode-c-hugging-face-spaces-via-openenv-push-recommended)
6. [Mode D: Manual Git Push to HF Space](#6-mode-d-manual-git-push-full-control)
7. [Post-Deployment Verification](#7-post-deployment-verification)
8. [Using Edge-Forge for RL Training](#8-using-edge-forge-for-rl-training)
9. [Troubleshooting](#9-troubleshooting)
10. [Updating the Deployment](#10-updating-the-deployment)
11. [Environment Variables Reference](#11-environment-variables-reference)

---

## 1. Prerequisites & Hardware Requirements

### 1.1 Required Tools

| Tool | Version | Install | Verify |
|------|---------|---------|--------|
| Python | 3.10, 3.11, or 3.12 | [python.org](https://python.org) | `python --version` |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | `uv --version` |
| Docker | 24.0+ | [docs.docker.com](https://docs.docker.com/get-docker/) | `docker --version` |
| Git | 2.30+ | [git-scm.com](https://git-scm.com) | `git --version` |
| openenv-core | ≥0.2.2 | `pip install "openenv-core[core]>=0.2.2"` | `openenv --version` |
| Hugging Face CLI | latest | `pip install "huggingface_hub[cli]"` | `huggingface-cli --version` |

### 1.2 Required Credentials

| Credential | Purpose | Where to Get |
|------------|---------|--------------|
| `HF_TOKEN` | Authenticate with HF Hub & LLM inference APIs | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — create a token with **write** scope |
| `API_BASE_URL` | OpenAI-compatible LLM API endpoint for `inference.py` | Your API provider (default: `https://router.huggingface.co/v1`) |
| `MODEL_NAME` | Model identifier for the LLM client | e.g., `Qwen/Qwen2.5-72B-Instruct`, `meta-llama/Meta-Llama-3-8B-Instruct` |

### 1.3 Hardware Requirements

| Deployment Mode | CPU | RAM | Storage | GPU |
|----------------|-----|-----|---------|-----|
| Local Uvicorn (dev) | 1 core | 2 GB | 500 MB | Not required |
| Local Docker | 2+ cores | 4 GB | 2 GB (image) | Not required |
| HF Space (Free Tier) | 2 vCPU | 16 GB | Managed | Not available |
| HF Space (CPU Upgrade) | 8 vCPU | 32 GB | Managed | Not available |
| Hackathon validator | 2 vCPU | 8 GB | — | Not required |

> **Hackathon constraint:** `inference.py` must complete all 3 tasks in under 20 minutes on 2 vCPU / 8 GB RAM.
> Edge-Forge has been validated to complete in **under 4 minutes** on this hardware.

---

## 2. Choosing Your Deployment Mode

```
What do you need?
│
├─ Just run the environment locally to develop/test
│  └─ → MODE A: Local Uvicorn (fastest iteration)
│
├─ Reproduce the exact hackathon submission environment
│  └─ → MODE B: Local Docker (exact replica of HF Space)
│
├─ Deploy to Hugging Face for public access / judge review
│  ├─ First time deploying → MODE C: openenv push (recommended, one command)
│  └─ Manual git control preferred → MODE D: Git push to HF Space
│
└─ Use Edge-Forge in RL training with TRL/TorchForge
   └─ → Section 8: Training Integration
```

| Mode | Best For | Startup Time | Concurrent Sessions | Command |
|------|----------|-------------|---------------------|---------|
| A: Local Uvicorn | Development, debugging | ~3 seconds | 1–4 | `uvicorn server.app:app ...` |
| B: Local Docker | Reproduce HF Space exactly | ~30 seconds | Up to 128 | `docker run ...` |
| C: `openenv push` | First HF Space deployment | ~5 minutes (build) | 128 (free tier) | `openenv push ...` |
| D: Git push | Manual HF Space control | ~5 minutes (build) | 128 (free tier) | `git push` |

---

## 3. Mode A: Local Uvicorn — Development

**Use this when:** You are actively modifying environment code and need fast iteration.
Hot-reload is enabled — changes take effect without restarting.

### 3.1 Setup

```bash
# Clone the repository
git clone https://github.com/SoumyaWasule/Edge-Forge.git
cd Edge-Forge

# Install all dependencies using uv
uv sync

# OR using pip with a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

### 3.2 Start the Server

```bash
# Set required environment variables
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_your_token_here"

# Start the server with hot-reload
uvicorn server.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload \
  --log-level info
```

On Windows PowerShell:

```powershell
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
$env:HF_TOKEN = "hf_your_token_here"

uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload --log-level info
```

### 3.3 Verify the Server Is Running

```bash
# Health check
curl http://localhost:8000/health
# Expected: {"status":"healthy"} with HTTP 200

# Test reset() endpoint
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{}'
# Expected: {"last_status":0,"covered_branches":[],"current_input":{},"last_error":null,"submit_outcomes":[]}

# Test state() endpoint
curl http://localhost:8000/state
# Expected: {"episode_id":"current","step_count":0}

# List available tasks
curl http://localhost:8000/tasks
# Expected: [{"id":"easy_task","name":"Trigger SSN Error","has_grader":true},...]

# Interactive API docs
# Open http://localhost:8000/docs in your browser
```

### 3.4 Run Inference Against Local Server

```bash
# Point inference.py at your local server
export ENV_BASE_URL="http://localhost:8000"

python inference.py
```

Expected output:

```
[START] task=easy_task env=edge_forge_env model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1 action=SET_FIELD(action=open_account) reward=-0.44 done=false error=null
[STEP]  step=2 action=SUBMIT reward=9.00 done=false error=null
...
[END]   success=true steps=5 score=0.990 rewards=-0.44,9.00,...

[START] task=medium_task env=edge_forge_env model=Qwen/Qwen2.5-72B-Instruct
...
[END]   success=true steps=30 score=0.158 rewards=...

[START] task=hard_task env=edge_forge_env model=Qwen/Qwen2.5-72B-Instruct
...
[END]   success=true steps=4 score=0.990 rewards=...
```

### 3.5 Limitations of This Mode

- No Docker isolation — environment runs in your Python environment, not the container
- Hot-reload is not representative of deployed behavior
- Max concurrent sessions: 1 (single-threaded by default without `--workers`)
- `PYTHONPATH` must include the project root for imports to resolve
- Use **Mode B** to verify Docker-specific behavior before deploying

---

## 4. Mode B: Local Docker — Exact Reproduction

**Use this when:** You need to verify the deployment before pushing to HF Spaces,
or reproduce the exact environment the hackathon validator will run.

### 4.1 Build the Docker Image

```bash
# Standard build
docker build -t edge-forge:latest .

# With verbose output (recommended before first push)
docker build --no-cache --progress=plain -t edge-forge:latest . 2>&1 | tee build.log

# Expected build time: ~3–5 minutes on a standard machine (cold)
# Expected image size: ~1.5 GB (includes openenv-base + uv dependencies)
```

The Dockerfile uses a **multi-stage build**:
1. **Builder stage** — installs `git`, copies source, runs `uv sync` to resolve and install all dependencies
2. **Runtime stage** — copies only the virtual environment and source code from the builder

Base image: `ghcr.io/meta-pytorch/openenv-base:latest`

### 4.2 Run the Container

```bash
docker run \
  --name edge-forge \
  -e API_BASE_URL="$API_BASE_URL" \
  -e MODEL_NAME="$MODEL_NAME" \
  -e HF_TOKEN="$HF_TOKEN" \
  -p 8000:8000 \
  --rm \
  edge-forge:latest
```

For background execution:

```bash
docker run -d \
  --name edge-forge \
  -e API_BASE_URL="$API_BASE_URL" \
  -e MODEL_NAME="$MODEL_NAME" \
  -e HF_TOKEN="$HF_TOKEN" \
  -p 8000:8000 \
  edge-forge:latest

# Follow logs
docker logs -f edge-forge

# Stop
docker stop edge-forge
```

### 4.3 Verify the Container

```bash
# Wait for startup (container needs ~5 seconds to initialize)
sleep 5

# Health check
curl http://localhost:8000/health

# Test reset() — must return valid Observation
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{}' | python -m json.tool

# Test step() — execute a SET_FIELD action
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"SET_FIELD","field":"age","value":25}' | python -m json.tool

# Test state() — must return valid State
curl http://localhost:8000/state | python -m json.tool

# Run the full OpenEnv validator
openenv validate --url http://localhost:8000

# Expected output:
# ✓ Environment responds (HTTP 200)
# ✓ reset() returns valid Observation
# ✓ step() returns valid StepResult with reward in [0.0, 1.0]
# ✓ state() returns valid State
# ✓ openenv.yaml is valid
# ✓ All 3 tasks enumerated
# passed: true

# Run inference against local Docker
export ENV_BASE_URL="http://localhost:8000"
python inference.py
```

### 4.4 Pull the HF Space Image Directly (Reproduce Without Building)

Once deployed to HF Spaces, anyone can pull and run the exact image:

```bash
# Login to HF Docker registry
docker login registry.hf.space \
  -u SoumyaW \
  -p $HF_TOKEN

# Pull the deployed image
docker pull registry.hf.space/soumyaw-edge-forge-env:latest

# Run it
docker run -d \
  -p 8000:8000 \
  --platform linux/amd64 \
  registry.hf.space/soumyaw-edge-forge-env:latest
```

This is the definitive reproducibility path — the exact binary that runs on HF Spaces.

---

## 5. Mode C: Hugging Face Spaces via `openenv push` (Recommended)

**Use this when:** Deploying for the first time, or when you want the OpenEnv CLI
to handle Space configuration, frontmatter, and YAML validation automatically.

### 5.1 Authenticate

```bash
# Login to Hugging Face
huggingface-cli login
# Enter your HF_TOKEN when prompted (write scope required)
```

### 5.2 Validate Locally Before Pushing

```bash
# Run the full pre-submission validator first
openenv validate --verbose

# Expected: all checks pass before pushing to HF
```

### 5.3 Push to Hugging Face Spaces

```bash
# Push to your namespace (creates Space if it doesn't exist)
openenv push --repo-id SoumyaW/edge_forge_env

# Push to an organization
openenv push --repo-id your-org/edge-forge

# Push as private Space (useful during development)
openenv push --repo-id SoumyaW/edge_forge_env --private

# Push with a specific base image override
openenv push --repo-id SoumyaW/edge_forge_env \
  --base-image ghcr.io/meta-pytorch/openenv-base:latest
```

**What `openenv push` does automatically:**

1. Validates `openenv.yaml` completeness
2. Injects Hugging Face Space frontmatter into README
3. Uploads all project files to the Space repository
4. Triggers Docker build on HF infrastructure
5. Returns the live Space URL when ready

### 5.4 Monitor the Build

```bash
# Watch build logs via CLI
huggingface-cli repo info SoumyaW/edge_forge_env --repo-type space

# Or watch in browser:
# https://huggingface.co/spaces/SoumyaW/edge_forge_env/logs
```

Typical build timeline:
- Docker layer caching (cold): ~4–6 minutes
- Docker layer caching (warm): ~1–2 minutes
- Container startup: ~10–30 seconds

### 5.5 Verify the Live Deployment

```bash
# Set your Space URL
SPACE_URL="https://soumyaw-edge-forge-env.hf.space"

# Health check (must return HTTP 200)
curl -f $SPACE_URL/health
echo "Exit code: $?"  # Must be 0

# Test reset() — must return valid Observation
curl -X POST $SPACE_URL/reset \
  -H "Content-Type: application/json" \
  -d '{}' | python -m json.tool

# Test state() — must return valid State
curl $SPACE_URL/state | python -m json.tool

# Run full OpenEnv validator against live Space
openenv validate --url $SPACE_URL

# Run inference against live Space
API_BASE_URL="https://router.huggingface.co/v1" \
MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" \
HF_TOKEN="$HF_TOKEN" \
ENV_BASE_URL="$SPACE_URL" \
python inference.py
```

> **Hackathon automated gate:** The submission validator pings `$SPACE_URL/reset`
> and expects HTTP 200 with a valid Observation JSON. Verify this manually before submitting.

---

## 6. Mode D: Manual Git Push (Full Control)

**Use this when:** You want direct git control over every file in the Space,
or when `openenv push` doesn't fit your workflow.

### 6.1 Create the Space

Go to [huggingface.co/spaces/new](https://huggingface.co/new-space) and configure:

| Setting | Value |
|---------|-------|
| Owner | Your HF username or organization |
| Space name | `edge_forge_env` (must match your inference.py URL expectations) |
| License | BSD 3-Clause |
| SDK | **Docker** ← critical: must be Docker, not Gradio or Streamlit |
| Hardware | CPU Basic (free tier — sufficient for hackathon evaluation) |
| Visibility | **Public** ← required for hackathon automated gate |

### 6.2 Clone and Push

```bash
# Clone the empty Space
git clone https://huggingface.co/spaces/SoumyaW/edge_forge_env
cd edge_forge_env

# Copy all project files
cp -r /path/to/Edge-Forge/. .

# The Space README.md needs HF frontmatter — verify it starts with:
head -12 README.md
# Expected:
# ---
# title: Edge Forge Env
# emoji: ⚡
# colorFrom: blue
# colorTo: purple
# sdk: docker
# app_port: 8000
# pinned: false
# license: bsd-3-clause
# short_description: Stateful API fuzzing RL environment for edge-case discovery
# ---

# Verify Dockerfile is present (HF requires it for Docker SDK)
ls -la Dockerfile  # Must exist

# Commit and push
git add .
git commit -m "feat: deploy Edge-Forge RL environment v1.0"
git push
```

### 6.3 Configure Space Secrets

In the Space settings (**Settings → Repository secrets**), add:

| Secret Name | Value | Required |
|-------------|-------|----------|
| `HF_TOKEN` | Your Hugging Face token | Yes — for inference |
| `API_BASE_URL` | LLM API endpoint (e.g., `https://router.huggingface.co/v1`) | Yes — for inference |
| `MODEL_NAME` | Model identifier (e.g., `Qwen/Qwen2.5-72B-Instruct`) | Yes — for inference |

> **Security:** Never commit secrets to git. Always use Space secrets.
> HF secrets are injected as environment variables at container runtime.

### 6.4 Update the Deployment

```bash
# Make your changes, then:
git add .
git commit -m "fix: description of what changed"
git push

# HF automatically detects the push and triggers a rebuild
# Monitor at: https://huggingface.co/spaces/SoumyaW/edge_forge_env/logs
```

---

## 7. Post-Deployment Verification

Run this checklist after every deployment before submitting to the hackathon.

### 7.1 Automated Gate Requirements

The hackathon validator checks these automatically. Verify all manually first:

```bash
SPACE_URL="https://soumyaw-edge-forge-env.hf.space"

echo "=== Checking HTTP 200 on health endpoint ==="
curl -f -s -o /dev/null -w "%{http_code}" $SPACE_URL/health
# Must print: 200

echo ""
echo "=== Checking reset() returns valid Observation ==="
curl -s -X POST $SPACE_URL/reset \
  -H "Content-Type: application/json" \
  -d '{}' | python -c "
import sys, json
data = json.load(sys.stdin)
print('reset() response keys:', list(data.keys()))
assert 'covered_branches' in data, 'Missing covered_branches!'
assert 'current_input' in data, 'Missing current_input!'
assert 'submit_outcomes' in data, 'Missing submit_outcomes!'
print('✅ reset() OK')
"

echo ""
echo "=== Checking state() endpoint ==="
curl -s $SPACE_URL/state | python -c "
import sys, json
data = json.load(sys.stdin)
print('state() response keys:', list(data.keys()))
assert 'episode_id' in data, 'Missing episode_id!'
assert 'step_count' in data, 'Missing step_count!'
print('✅ state() OK')
"

echo ""
echo "=== Checking /tasks endpoint ==="
curl -s $SPACE_URL/tasks | python -c "
import sys, json
data = json.load(sys.stdin)
assert len(data) == 3, f'Expected 3 tasks, got {len(data)}'
ids = [t['id'] for t in data]
assert 'easy_task' in ids and 'medium_task' in ids and 'hard_task' in ids
print('✅ All 3 tasks registered:', ids)
"

echo ""
echo "=== Running full OpenEnv validator ==="
openenv validate --url $SPACE_URL --verbose

echo ""
echo "=== Running inference.py end-to-end ==="
time python inference.py 2>&1 | tail -20
# Must complete in under 20 minutes
# Must show [END] blocks with scores in [0.0, 1.0]
```

### 7.2 Manual Verification Checklist

- [ ] HF Space status shows **"Running"** (green dot) — not "Building" or "Error"
- [ ] `curl -f $SPACE_URL/health` returns exit code 0
- [ ] `curl -X POST $SPACE_URL/reset` returns JSON with `covered_branches`, `current_input`, `submit_outcomes`
- [ ] `curl $SPACE_URL/state` returns JSON with `episode_id` and `step_count`
- [ ] `curl $SPACE_URL/tasks` returns 3 tasks with `has_grader: true`
- [ ] `openenv validate --url $SPACE_URL` prints `passed: true`
- [ ] `python inference.py` runs to completion without error
- [ ] All 3 tasks produce `[END]` log entries with scores in `[0.0, 1.0]`
- [ ] Total inference runtime is under 20 minutes (validated at < 4 minutes)
- [ ] Space URL matches what is submitted on the hackathon dashboard

### 7.3 HF Space Sleep Mode Warning

> ⚠️ **Important:** HF Spaces on the free tier **sleep after ~48 hours of inactivity**.
> A sleeping Space returns **HTTP 503**, which will fail the automated gate.
>
> To prevent this:
> - Send a request within the last 30 minutes before the submission deadline
> - Consider upgrading to **CPU Upgrade** tier for persistent uptime during judging
> - The first request to a sleeping Space triggers a cold start (~30–60 seconds)

```bash
# Wake a sleeping Space
curl https://soumyaw-edge-forge-env.hf.space/health
# Wait 30–60 seconds for cold start, then re-run verification
sleep 60
curl -f https://soumyaw-edge-forge-env.hf.space/health
```

---

## 8. Using Edge-Forge for RL Training

Once deployed, Edge-Forge can be used as a training environment with
the major open-source RL frameworks. This is the real purpose of OpenEnv environments.

### 8.1 Connect via Python Client

```python
from client import EdgeForgeEnv
from models import EdgeForgeAction, EdgeForgeObservation

# Connect to HF Space
env = EdgeForgeEnv(base_url="https://soumyaw-edge-forge-env.hf.space")
result = await env.reset()
print(result.observation.covered_branches)  # []

# Execute an action
action = EdgeForgeAction(action_type="SET_FIELD", field="age", value=25)
result = await env.step(action)
print(result.observation.current_input)     # {"age": 25}
print(result.reward)                        # ~-0.44

# Submit and observe branch discovery
action = EdgeForgeAction(action_type="SUBMIT")
result = await env.step(action)
print(result.observation.covered_branches)  # ["missing_income"]
print(result.observation.submit_outcomes)   # ["Income is required"]

# Connect to local Docker for high-throughput training
env = EdgeForgeEnv(base_url="http://localhost:8000")
```

### 8.2 TRL Integration (GRPO Training)

```python
import asyncio
from trl import GRPOTrainer, GRPOConfig
from client import EdgeForgeEnv
from models import EdgeForgeAction
from tasks import grade_easy, grade_medium, grade_hard

TASK_GRADERS = {
    "easy_task": grade_easy,
    "medium_task": grade_medium,
    "hard_task": grade_hard,
}

async def rollout_episode(env, task_id, action_sequence):
    """Run a single episode and return the grader score."""
    result = await env.reset()
    
    for action_dict in action_sequence:
        action = EdgeForgeAction(**action_dict)
        result = await env.step(action)
        if result.done:
            break
    
    # Grade using observable outcomes
    final_obs = {
        "submit_outcomes": result.observation.submit_outcomes,
        "covered_branches": result.observation.covered_branches,
    }
    grader = TASK_GRADERS[task_id]
    score = grader(final_obs)
    return score


def rollout_func(prompts, trainer):
    """Custom rollout function using Edge-Forge as the reward environment."""
    env = EdgeForgeEnv(base_url="https://soumyaw-edge-forge-env.hf.space")
    
    results = {"input_ids": [], "attention_mask": [], "rewards": []}
    
    for prompt in prompts:
        # Parse the model's generated action sequence
        action_sequence = parse_actions_from_completion(prompt)
        
        # Run the episode and get the grader score
        score = asyncio.run(rollout_episode(env, "medium_task", action_sequence))
        results["rewards"].append(score)
    
    return results


# Configure trainer with OpenEnv environment
config = GRPOConfig(
    model_name="Qwen/Qwen2.5-72B-Instruct",
    max_completion_length=512,
)

trainer = GRPOTrainer(
    model=model,
    config=config,
    rollout_func=rollout_func,
)
trainer.train()
```

### 8.3 Performance Characteristics for Training

| Configuration | Max Concurrent Sessions | Notes |
|--------------|------------------------|-------|
| HF Space Free Tier (2 vCPU) | ~128 | Sufficient for single-GPU GRPO |
| Local Docker (8-core machine) | ~2048 | Best efficiency for training |
| Local Uvicorn (single worker) | 1–4 | Development only |

> **Concurrent sessions note:** The default `max_concurrent_envs` is set to `1` in the Edge-Forge
> server configuration. For parallel GRPO rollouts, modify `server/app.py` line 99:
> ```python
> app = create_app(
>     EdgeForgeEnvironment,
>     EdgeForgeAction,
>     EdgeForgeObservation,
>     env_name="edge_forge_env",
>     max_concurrent_envs=128,  # Increase for parallel training
> )
> ```

For large-scale training (>128 concurrent), run the Docker container locally
with multiple uvicorn workers:

```bash
docker run -d \
  -p 8000:8000 \
  edge-forge:latest \
  sh -c "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 8"
```

---

## 9. Troubleshooting

### 9.1 Failure Category Map

```
Environment not responding?
├── HF Space shows "Error" → Section 9.2
├── HF Space shows "Building" → Section 9.3
├── HF Space shows "Running" but curl fails → Section 9.4
├── Docker build fails locally → Section 9.5
├── Docker runs but endpoints fail → Section 9.6
├── OpenEnv validator fails → Section 9.7
└── inference.py fails → Section 9.8
```

### 9.2 HF Space Shows "Error"

```bash
# Get full build and runtime logs
# Browser: Space page → Logs tab → select "Build logs" and "Container logs"

# Common causes:
# 1. Dockerfile syntax error → check build logs for the exact line
# 2. Missing dependency in pyproject.toml → add it and repush
# 3. Port mismatch → Dockerfile must EXPOSE 8000, CMD must use --port 8000
# 4. Entrypoint crashes → check container logs for Python traceback
```

> **HF Spaces critical constraint:** The container must listen on port **8000**.
> The `app_port: 8000` frontmatter and the Dockerfile's `CMD` must agree.
> Verify: `grep "8000" Dockerfile` should show both the HEALTHCHECK and CMD lines.

### 9.3 HF Space Stuck on "Building"

```bash
# Normal build time: 3–8 minutes for first build, 1–2 minutes with cache
# If building for >15 minutes, the build has likely failed silently

# Check: are all files actually pushed?
git log --oneline -5  # Verify your commit is there
git push              # Re-push if needed

# Check: is the Dockerfile valid?
docker build --no-cache -t edge-forge-test . 2>&1 | head -50
# Fix any errors shown, then repush
```

### 9.4 Space Running but curl Fails

```bash
# Symptom: Space shows green "Running" but curl returns connection refused or 503

# Cause 1: Space is waking from sleep
curl https://soumyaw-edge-forge-env.hf.space/health
# Retry after 60 seconds — first request wakes the Space

# Cause 2: Wrong URL format
# Correct format: https://soumyaw-edge-forge-env.hf.space
# Note: hyphens replaced underscores; username and space-name joined with hyphen
echo $SPACE_URL  # Verify the format

# Cause 3: Port mismatch (most common Docker issue)
# Edge-Forge uses port 8000 — check your Dockerfile
grep -E "EXPOSE|8000|port" Dockerfile

# Cause 4: Application crashed after start
# Check container logs: Space page → Logs tab → Container logs
```

### 9.5 Docker Build Fails Locally

```bash
# Get detailed output
docker build --no-cache --progress=plain -t edge-forge . 2>&1 | tee build.log
grep -i "error\|failed\|cannot" build.log

# Common failures and fixes:

# "Could not find a version that satisfies the requirement openenv-core"
#   → The base image may already include it; check with:
#   → docker run --rm ghcr.io/meta-pytorch/openenv-base:latest pip show openenv-core

# "COPY failed: file not found"
#   → Verify all source paths exist: ls -la pyproject.toml server/ models.py

# "network error" during uv sync
#   → Add --no-cache to docker build; check internet connectivity

# uv.lock cross-platform issues
#   → The Dockerfile uses "uv sync --no-install-project --no-editable" (no frozen lock)
#   → uv.lock is in .dockerignore, so this should not cause issues
```

### 9.6 Docker Runs but Endpoints Return Errors

```bash
# Check container is actually running
docker ps | grep edge-forge

# Check container logs for startup errors
docker logs edge-forge 2>&1 | head -50

# Test each endpoint individually
curl -v http://localhost:8000/health
curl -v -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}'
curl -v http://localhost:8000/state

# Common: "422 Unprocessable Entity" on /step
# → The request body must include "action_type" at minimum
# → Correct: {"action_type": "SUBMIT"}
# → Correct: {"action_type": "SET_FIELD", "field": "age", "value": 25}

# Common: "500 Internal Server Error" on /step
# → Check if reset() was called first — step() auto-resets but may have import issues
# → Check container logs: docker logs edge-forge 2>&1 | tail -20
```

### 9.7 OpenEnv Validator Fails

```bash
openenv validate --url $SPACE_URL --verbose

# For each failing check:

# "✗ Environment does not respond"
# → curl $SPACE_URL/health first; if that fails, see Section 9.4

# "✗ reset() does not return valid Observation"
# → curl -X POST $SPACE_URL/reset -d '{}' and compare to models.py
# → Must return: last_status, covered_branches, current_input, last_error, submit_outcomes

# "✗ Scores out of range [0.0, 1.0]"
# → Graders return values in [0.01, 0.99]; verify with:
# → curl -X POST $SPACE_URL/grade -H "Content-Type: application/json" \
#       -d '{"task_id":"easy_task"}'

# "✗ openenv.yaml invalid"
# → Verify: cat openenv.yaml
# → Must have: spec_version, name, version, runtime, app, port, tasks, api
```

### 9.8 inference.py Fails

```bash
# Run with full output
python inference.py 2>&1 | tee inference_debug.log

# Common failures:

# "Connection refused" to environment
# → Is the environment running? Check ENV_BASE_URL (default: http://localhost:8000)
# → For HF Space: export ENV_BASE_URL="https://soumyaw-edge-forge-env.hf.space"

# "Authentication error" or "401 Unauthorized" on LLM calls
# → Check HF_TOKEN is set: echo $HF_TOKEN | head -c 10
# → Check API_BASE_URL is correct: echo $API_BASE_URL
# → Fallback actions will execute if LLM calls fail (scores will still be produced)

# "ModuleNotFoundError: No module named 'edge_forge_env'"
# → Run: pip install -e . OR ensure PYTHONPATH includes project root
# → The import falls back to direct imports (client, models, tasks) automatically

# Timeout / hangs forever
# → MAX_STEPS is 10/30/15 per task; tasks should complete in seconds
# → If LLM API is slow, fallback actions kick in automatically
# → Profile: time python inference.py

# "Runtime exceeded 20 minutes" (hackathon disqualification)
# → Identify which task is slow from [START]/[END] timestamps
# → Check if the LLM API is responding slowly (try a faster/smaller model)
# → Fallback actions guarantee completion within seconds per task
```

---

## 10. Updating the Deployment

### 10.1 Update via `openenv push` (Recommended)

```bash
# Make your code changes, then:
openenv validate --verbose    # Validate before pushing
openenv push --repo-id SoumyaW/edge_forge_env  # Deploy
```

### 10.2 Update via Git Push

```bash
git add .
git commit -m "fix: description of what changed"
git push hf main  # Push to the HF Space remote

# HF automatically detects the push and triggers a rebuild
# Monitor at: https://huggingface.co/spaces/SoumyaW/edge_forge_env/logs
```

### 10.3 Rollback to Previous Version

```bash
# Find the commit to roll back to
git log --oneline -10

# Roll back to specific commit
git revert <commit-hash>
git push hf main

# OR force reset (use with caution)
git reset --hard <commit-hash>
git push hf main --force
```

---

## 11. Environment Variables Reference

A complete reference of every environment variable the application accepts:

| Variable | Required | Default | Description | Set In |
|----------|----------|---------|-------------|--------|
| `HF_TOKEN` | ✅ Yes | — | Hugging Face token (write scope); used as API key for LLM calls | HF Space secret / `docker -e` / `export` |
| `API_BASE_URL` | ⬜ No | `https://router.huggingface.co/v1` | OpenAI-compatible LLM API endpoint | HF Space secret / `docker -e` |
| `MODEL_NAME` | ⬜ No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier for LLM calls | HF Space secret / `docker -e` |
| `IMAGE_NAME` | ⬜ No | `None` | Docker image name; when set, `inference.py` uses `from_docker_image()` | Hackathon validator sets this |
| `ENV_BASE_URL` | ⬜ No | `http://localhost:8000` | Environment server URL for `inference.py` when `IMAGE_NAME` is unset | `export` / `docker -e` |
| `API_KEY` | ⬜ No | Falls back to `HF_TOKEN` | Alternative API key variable (checked if `HF_TOKEN` is absent) | `export` / `docker -e` |

### Setting Variables Locally

```bash
# Export for the current shell session
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_..."
export ENV_BASE_URL="http://localhost:8000"

# Or use a .env file (never commit this file)
cat > .env << 'EOF'
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
HF_TOKEN=hf_...
ENV_BASE_URL=http://localhost:8000
EOF

# Load from .env file
source .env

# OR with docker:
docker run --env-file .env -p 8000:8000 edge-forge:latest
```

> **Security reminder:** Add `.env` to `.gitignore` immediately:
> ```bash
> echo ".env" >> .gitignore && git add .gitignore && git commit -m "chore: ignore .env file"
> ```
> `.env` is already in this project's `.gitignore`.

---

## Quick Reference Card

| What | Value |
|------|-------|
| **Live Space URL** | `https://soumyaw-edge-forge-env.hf.space` |
| **GitHub Repository** | `https://github.com/SoumyaWasule/Edge-Forge` |
| **HF Space Repository** | `https://huggingface.co/spaces/SoumyaW/edge_forge_env` |
| **Port** | `8000` |
| **App Entry Point** | `server.app:app` |
| **Base Docker Image** | `ghcr.io/meta-pytorch/openenv-base:latest` |
| **Health Endpoint** | `GET /health` |
| **Reset Endpoint** | `POST /reset` |
| **Step Endpoint** | `POST /step` |
| **State Endpoint** | `GET /state` |
| **Tasks Endpoint** | `GET /tasks` |
| **Grade Endpoint** | `POST /grade` |
| **API Docs** | `GET /docs` (Swagger UI) |
| **Dashboard** | `GET /` (HTML) |
| **Tasks** | `easy_task`, `medium_task`, `hard_task` |
| **Max Steps** | 10 (easy), 30 (medium), 15 (hard) |
| **Grader Score Range** | `[0.01, 0.99]` |
| **Validated Runtime** | < 4 minutes on 2 vCPU / 8 GB RAM |

---

*Built for the **Meta × PyTorch × Hugging Face OpenEnv Hackathon***
