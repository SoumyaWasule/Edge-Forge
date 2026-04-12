# 🚀 Deployment Guide — Edge-Forge

> **TL;DR:** Live Space → **[https://soumyaw-edge-forge-env.hf.space](https://soumyaw-edge-forge-env.hf.space)**
> Reproduce locally: `docker build -t edge-forge . && docker run -p 8000:8000 edge-forge`

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue) ![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker) ![OpenEnv](https://img.shields.io/badge/OpenEnv-100%25-brightgreen) ![HF Space](https://img.shields.io/badge/HF%20Space-Deployed-green?logo=huggingface)

Four deployment modes: **Local Uvicorn** (fast dev), **Local Docker** (exact HF replica), **`openenv push`** (one-command deploy), **Git Push** (manual control). All serve an identical OpenEnv environment on port `8000`.

---

## 1. Prerequisites

### 1.1 Tools

| Tool | Version | Verify |
|------|---------|--------|
| Python | 3.10–3.12 | `python --version` |
| uv | latest | `uv --version` |
| Docker | 24.0+ | `docker --version` |
| Git | 2.30+ | `git --version` |
| openenv-core | ≥0.2.2 | `openenv --version` |
| HF CLI | latest | `huggingface-cli --version` |

### 1.2 Credentials

| Variable | Purpose | Default |
|----------|---------|---------|
| `HF_TOKEN` | HF Hub auth & LLM API key | — (required) |
| `API_BASE_URL` | OpenAI-compatible LLM endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | LLM model identifier | `Qwen/Qwen2.5-72B-Instruct` |

Get `HF_TOKEN` at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — create with **write** scope.

### 1.3 Hardware

| Mode | CPU | RAM | GPU |
|------|-----|-----|-----|
| Local Uvicorn | 1 core | 2 GB | No |
| Local Docker | 2+ cores | 4 GB | No |
| HF Space (Free) | 2 vCPU | 16 GB | No |
| Hackathon validator | 2 vCPU | 8 GB | No |

> **Hackathon constraint:** `inference.py` must complete in under 20 minutes on 2 vCPU / 8 GB.
> Edge-Forge validates in **under 4 minutes**.

---

## 2. Deployment Mode Decision Matrix

```
What do you need?
├─ Develop/test locally         → Mode A (Uvicorn)
├─ Reproduce exact HF container → Mode B (Docker)
├─ Deploy to HF Spaces          → Mode C (openenv push) or Mode D (git push)
└─ RL training integration      → Section 8
```

| Mode | Best For | Startup | Command |
|------|----------|---------|---------|
| A: Uvicorn | Dev + hot-reload | ~3s | `uvicorn server.app:app ...` |
| B: Docker | Exact HF replica | ~30s | `docker run -p 8000:8000 ...` |
| C: `openenv push` | First deploy | ~5min | `openenv push --repo-id ...` |
| D: Git push | Manual control | ~5min | `git push` |

---

## 3. Mode A: Local Uvicorn

```bash
git clone https://github.com/SoumyaWasule/Edge-Forge.git && cd Edge-Forge
uv sync  # OR: python -m venv .venv && source .venv/bin/activate && pip install -e .

export HF_TOKEN="hf_..." API_BASE_URL="https://router.huggingface.co/v1" MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

Windows PowerShell:
```powershell
$env:HF_TOKEN="hf_..."; $env:API_BASE_URL="https://router.huggingface.co/v1"; $env:MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

Verify:
```bash
curl http://localhost:8000/health                    # → {"status":"healthy"}
curl -X POST http://localhost:8000/reset -d '{}'     # → Observation JSON
curl http://localhost:8000/state                     # → {"episode_id":"current","step_count":0}
curl http://localhost:8000/tasks                     # → 3 tasks with graders
```

Run inference:
```bash
export ENV_BASE_URL="http://localhost:8000"
python inference.py
```

> **Limitations:** No Docker isolation, single-threaded, max 1–4 concurrent sessions. Use Mode B before deploying.

---

## 4. Mode B: Local Docker

### Build & Run

```bash
docker build -t edge-forge:latest .
# Verbose: docker build --no-cache --progress=plain -t edge-forge:latest . 2>&1 | tee build.log

docker run --name edge-forge -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" -e API_BASE_URL="$API_BASE_URL" -e MODEL_NAME="$MODEL_NAME" \
  --rm edge-forge:latest

# Background: add -d, then: docker logs -f edge-forge | docker stop edge-forge
```

The Dockerfile uses a **multi-stage build** on `ghcr.io/meta-pytorch/openenv-base:latest`: builder installs deps via `uv sync`, runtime copies only `.venv` + source.

### Verify

```bash
sleep 5 && curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset -d '{}' | python -m json.tool
curl -X POST http://localhost:8000/step -H "Content-Type: application/json" \
  -d '{"action_type":"SET_FIELD","field":"age","value":25}' | python -m json.tool
openenv validate --url http://localhost:8000
```

### Pull from HF Registry (skip building)

```bash
docker login registry.hf.space -u SoumyaW -p $HF_TOKEN
docker pull registry.hf.space/soumyaw-edge-forge-env:latest
docker run -d -p 8000:8000 --platform linux/amd64 registry.hf.space/soumyaw-edge-forge-env:latest
```

---

## 5. Mode C: `openenv push` (Recommended)

```bash
huggingface-cli login                            # Enter HF_TOKEN with write scope
openenv validate --verbose                       # Pre-flight check
openenv push --repo-id SoumyaW/edge_forge_env    # Deploy (creates Space if needed)

# Private: openenv push --repo-id SoumyaW/edge_forge_env --private
# Custom base: openenv push --repo-id SoumyaW/edge_forge_env --base-image ghcr.io/meta-pytorch/openenv-base:latest
```

`openenv push` validates YAML, injects HF frontmatter, uploads files, triggers Docker build, returns the live URL.

Monitor: `https://huggingface.co/spaces/SoumyaW/edge_forge_env/logs` — cold build ~4–6 min, warm ~1–2 min.

### Verify Live

```bash
SPACE_URL="https://soumyaw-edge-forge-env.hf.space"
curl -f $SPACE_URL/health && echo "✅ Health OK"
curl -X POST $SPACE_URL/reset -d '{}' | python -m json.tool
openenv validate --url $SPACE_URL
ENV_BASE_URL="$SPACE_URL" python inference.py
```

> **Hackathon gate:** The validator pings `/reset` and expects HTTP 200 with valid Observation JSON.

---

## 6. Mode D: Manual Git Push

### Create Space

At [huggingface.co/new-space](https://huggingface.co/new-space): SDK = **Docker**, Hardware = CPU Basic, Visibility = **Public**.

### Push

```bash
git clone https://huggingface.co/spaces/SoumyaW/edge_forge_env && cd edge_forge_env
cp -r /path/to/Edge-Forge/. .

# Verify README frontmatter starts with:
# ---
# title: Edge Forge Env
# sdk: docker
# app_port: 8000
# ---

git add . && git commit -m "feat: deploy Edge-Forge v1.0" && git push
```

### Space Secrets (Settings → Repository secrets)

| Secret | Value | Required |
|--------|-------|----------|
| `HF_TOKEN` | Your HF token | Yes |
| `API_BASE_URL` | LLM endpoint | Yes |
| `MODEL_NAME` | Model ID | Yes |

> **Never commit secrets to git.** HF injects secrets as env vars at runtime.

### Update

```bash
git add . && git commit -m "fix: what changed" && git push
# Rebuild auto-triggers. Logs: https://huggingface.co/spaces/SoumyaW/edge_forge_env/logs
```

---

## 7. Post-Deployment Verification

### Automated Gate Script

```bash
SPACE_URL="https://soumyaw-edge-forge-env.hf.space"

curl -f -s -o /dev/null -w "Health: %{http_code}\n" $SPACE_URL/health

curl -s -X POST $SPACE_URL/reset -d '{}' | python -c "
import sys, json; d = json.load(sys.stdin)
assert all(k in d for k in ['covered_branches','current_input','submit_outcomes'])
print('✅ reset() OK:', list(d.keys()))"

curl -s $SPACE_URL/state | python -c "
import sys, json; d = json.load(sys.stdin)
assert 'episode_id' in d and 'step_count' in d
print('✅ state() OK')"

curl -s $SPACE_URL/tasks | python -c "
import sys, json; d = json.load(sys.stdin)
assert len(d) == 3; print('✅ Tasks:', [t['id'] for t in d])"

openenv validate --url $SPACE_URL --verbose
time python inference.py 2>&1 | tail -10
```

### Checklist

- [ ] HF Space shows **"Running"** (green dot)
- [ ] `curl -f $SPACE_URL/health` exits 0
- [ ] `/reset` returns valid Observation JSON
- [ ] `/tasks` returns 3 tasks with `has_grader: true`
- [ ] `openenv validate` prints `passed: true`
- [ ] `inference.py` completes with `[END]` blocks, scores in `[0.01, 0.99]`
- [ ] Runtime under 20 minutes (validated < 4 min)
- [ ] Space URL matches hackathon dashboard submission

### Sleep Mode Warning

> ⚠️ **HF Spaces on the free tier sleep after ~48h of inactivity.** A sleeping Space returns HTTP 503.
> Wake it before the deadline: `curl $SPACE_URL/health && sleep 60 && curl -f $SPACE_URL/health`

---

## 8. RL Training Integration

### Python Client

```python
from client import EdgeForgeEnv
from models import EdgeForgeAction

env = EdgeForgeEnv(base_url="https://soumyaw-edge-forge-env.hf.space")
result = await env.reset()

action = EdgeForgeAction(action_type="SET_FIELD", field="age", value=25)
result = await env.step(action)
print(result.observation.covered_branches, result.reward)

result = await env.step(EdgeForgeAction(action_type="SUBMIT"))
print(result.observation.submit_outcomes)  # ["Income is required"]
```

### TRL / GRPO Integration

```python
import asyncio
from client import EdgeForgeEnv
from models import EdgeForgeAction
from tasks import grade_easy, grade_medium, grade_hard

GRADERS = {"easy_task": grade_easy, "medium_task": grade_medium, "hard_task": grade_hard}

async def rollout(env, task_id, actions):
    result = await env.reset()
    for a in actions:
        result = await env.step(EdgeForgeAction(**a))
        if result.done: break
    obs = {"submit_outcomes": result.observation.submit_outcomes,
           "covered_branches": result.observation.covered_branches}
    return GRADERS[task_id](obs)

def rollout_func(prompts, trainer):
    env = EdgeForgeEnv(base_url="https://soumyaw-edge-forge-env.hf.space")
    rewards = [asyncio.run(rollout(env, "medium_task", parse_actions(p))) for p in prompts]
    return {"rewards": rewards}
```

### Scaling

| Config | Concurrent Sessions |
|--------|-------------------|
| HF Space Free (2 vCPU) | ~128 |
| Local Docker (8-core) | ~2048 |
| Local Uvicorn (1 worker) | 1–4 |

For parallel rollouts, increase `max_concurrent_envs` in `server/app.py` line 99, or run with multiple workers:

```bash
docker run -d -p 8000:8000 edge-forge:latest \
  sh -c "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 8"
```

---

## 9. Troubleshooting

### Failure Map

```
Environment not responding?
├── HF Space "Error"           → 9.1
├── HF Space "Building" >15m   → 9.2
├── Space "Running", curl fails → 9.3
├── Docker build fails          → 9.4
├── Endpoints return errors     → 9.5
└── inference.py fails          → 9.6
```

### 9.1 HF Space Error

Check **Logs tab → Build logs / Container logs**. Common causes: Dockerfile syntax error, missing dep in `pyproject.toml`, port mismatch (must be `8000`), entrypoint crash.

### 9.2 Build Stuck

Normal: 3–8 min cold, 1–2 min warm. If >15 min: verify commit pushed (`git log --oneline -5`), test locally (`docker build --no-cache .`).

### 9.3 Running but Unreachable

- **Sleeping Space:** First request wakes it, retry after 60s
- **Wrong URL:** Must be `https://soumyaw-edge-forge-env.hf.space` (hyphens, not underscores)
- **Port mismatch:** `grep 8000 Dockerfile` must show HEALTHCHECK + CMD
- **Crashed:** Check container logs in Logs tab

### 9.4 Docker Build Fails

```bash
docker build --no-cache --progress=plain . 2>&1 | tee build.log
grep -i "error\|failed" build.log
```

Common: missing `openenv-core` (base image may include it), `COPY` path not found (verify `pyproject.toml`, `server/` exist), network error during `uv sync`.

### 9.5 Endpoint Errors

- **422 on `/step`:** Body must include `action_type`. Example: `{"action_type":"SUBMIT"}`
- **500 on `/step`:** Call `/reset` first. Check logs: `docker logs edge-forge | tail -20`
- **Empty `/state`:** Normal before first `/reset` — returns `{"episode_id":"","step_count":0}`

### 9.6 inference.py Fails

- **Connection refused:** Set `ENV_BASE_URL` correctly (default `http://localhost:8000`)
- **401 Unauthorized:** Check `HF_TOKEN` is set; fallbacks execute if LLM fails
- **ModuleNotFoundError:** Run `pip install -e .` — imports auto-fallback to direct modules
- **Slow/timeout:** Fallbacks guarantee completion in seconds; check LLM API latency

---

## 10. Updating & Rollback

```bash
# Via openenv
openenv validate --verbose && openenv push --repo-id SoumyaW/edge_forge_env

# Via git
git add . && git commit -m "fix: what changed" && git push hf main

# Rollback
git revert <hash> && git push hf main
# OR: git reset --hard <hash> && git push hf main --force
```

---

## 11. Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | ✅ | — | HF auth + LLM API key |
| `API_BASE_URL` | ⬜ | `https://router.huggingface.co/v1` | LLM endpoint |
| `MODEL_NAME` | ⬜ | `Qwen/Qwen2.5-72B-Instruct` | LLM model |
| `IMAGE_NAME` | ⬜ | `None` | Docker image (set by validator) |
| `ENV_BASE_URL` | ⬜ | `http://localhost:8000` | Env server URL for inference |
| `API_KEY` | ⬜ | Falls back to `HF_TOKEN` | Alternative API key |

```bash
# .env file (never commit — already in .gitignore)
cat > .env << 'EOF'
HF_TOKEN=hf_...
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
ENV_BASE_URL=http://localhost:8000
EOF
source .env                                          # Shell
docker run --env-file .env -p 8000:8000 edge-forge   # Docker
```

---

## Quick Reference

| Key | Value |
|-----|-------|
| **Live URL** | `https://soumyaw-edge-forge-env.hf.space` |
| **GitHub** | `https://github.com/SoumyaWasule/Edge-Forge` |
| **HF Space** | `https://huggingface.co/spaces/SoumyaW/edge_forge_env` |
| **Port** | `8000` |
| **Entry Point** | `server.app:app` |
| **Base Image** | `ghcr.io/meta-pytorch/openenv-base:latest` |
| **Endpoints** | `/health` `/reset` `/step` `/state` `/tasks` `/grade` `/docs` `/` |
| **Tasks** | `easy_task` (10 steps), `medium_task` (30), `hard_task` (15) |
| **Score Range** | `[0.01, 0.99]` |
| **Runtime** | < 4 min on 2 vCPU / 8 GB |

---

*Built for the **Meta × PyTorch × Hugging Face OpenEnv Hackathon***
