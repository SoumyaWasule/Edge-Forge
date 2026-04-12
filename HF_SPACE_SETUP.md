# 🔧 Edge-Forge Lite — HuggingFace Space Setup Guide

Operational deployment runbook for getting Edge-Forge live on HuggingFace Spaces.

| Resource | URL |
|----------|-----|
| HF Space | `https://huggingface.co/spaces/<HF_USERNAME>/edge_forge_env` |
| Live API Docs | `https://<HF_USERNAME>-edge-forge-env.hf.space/docs` |
| Health Check | `https://<HF_USERNAME>-edge-forge-env.hf.space/health` |
| OpenEnv Validator | `openenv validate --url https://<HF_USERNAME>-edge-forge-env.hf.space` |

---

## 1. Prerequisites

```bash
python --version          # 3.10+
git --version             # 2.30+
git lfs version           # must be installed
huggingface-cli --version # pip install huggingface_hub
docker --version          # 24.0+
openenv --version         # pip install openenv-core (≥0.2.2)
```

Authenticate with HuggingFace (write-scope token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)):

```bash
huggingface-cli login
```

---

## 2. Create the HuggingFace Space

**Method A — Web UI:** Go to [huggingface.co/new-space](https://huggingface.co/new-space) → Name: `edge_forge_env` → SDK: **Docker** → Hardware: **CPU Basic** → Visibility: **Public** → Create.

**Method B — CLI:**
```bash
huggingface-cli repo create edge_forge_env --type space --space-sdk docker
```

Live URL after creation: `https://<HF_USERNAME>-edge-forge-env.hf.space`

> 💡 HF converts underscores to hyphens in the subdomain. `edge_forge_env` → `edge-forge-env`.

---

## 3. Clone and Push

```bash
git clone https://huggingface.co/spaces/<HF_USERNAME>/edge_forge_env
cd edge_forge_env
cp -r /path/to/Edge-Forge/* .
cp /path/to/Edge-Forge/.dockerignore /path/to/Edge-Forge/.gitignore .
```

**Required file tree** (verify before pushing):

```
├── README.md                           ← YAML frontmatter required (Section 4)
├── openenv.yaml                        ← OpenEnv manifest (3 tasks)
├── pyproject.toml                      ← Dependencies + package config
├── Dockerfile                          ← Multi-stage build on openenv-base
├── __init__.py                         ← Package exports
├── client.py                           ← EnvClient (WebSocket)
├── models.py                           ← EdgeForgeAction + EdgeForgeObservation
├── mock_api.py                         ← 19-branch stateful API simulation
├── tasks.py / graders.py               ← Task definitions + grader functions
├── inference.py                        ← Baseline LLM agent
├── server/
│   ├── app.py                          ← FastAPI entrypoint (stateful routes)
│   └── edge_forge_env_environment.py   ← Core RL environment
└── tests/
    └── test_environment.py
```

```bash
git add . && git commit -m "feat: initial Edge-Forge deployment" && git push
```

> 💡 Cold build: **3–7 min**. Warm rebuilds (unchanged layers): **1–2 min**. Monitor in HF dashboard → Logs tab.

---

## 4. README.md Frontmatter (REQUIRED)

HF will **not** render the Space without valid YAML frontmatter as the first content in `README.md`:

```yaml
---
title: Edge Forge Env
emoji: ⚡
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
license: bsd-3-clause
short_description: Stateful API fuzzing RL environment for edge-case discovery
tags:
  - openenv
  - reinforcement-learning
  - meta-pytorch-hackathon
---
```

> ⚠️ `sdk: docker` and `app_port: 8000` are **non-negotiable**. Wrong SDK = Python runtime instead of Docker. Wrong port = every request returns 502.

---

## 5. Environment Variables & Secrets

Set via **Space Settings → Variables and secrets**.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | For inference | — | HF access token for LLM API calls |
| `API_BASE_URL` | For inference | `https://router.huggingface.co/v1` | LLM endpoint |
| `MODEL_NAME` | For inference | `Qwen/Qwen2.5-72B-Instruct` | LLM model ID |
| `IMAGE_NAME` | No | — | Docker image (auto-set by validator) |
| `ENV_BASE_URL` | For inference | `http://localhost:8000` | Environment server URL |

> ⚠️ **Always set `HF_TOKEN` as a Secret, not a Variable.** Secrets are encrypted and never exposed in logs.

The server itself runs with **zero** env vars — all variables above are only for `inference.py`.

---

## 6. Verify the Deployment

After the build completes ("Application startup complete" in Logs):

```bash
BASE=https://<HF_USERNAME>-edge-forge-env.hf.space

# 1. Health
curl $BASE/health
# → {"status":"healthy","environment":"edge_forge_env","version":"0.1.0"}

# 2. Reset (start episode)
curl -X POST $BASE/reset -H "Content-Type: application/json" -d '{}'
# → {"last_status":0,"covered_branches":[],"current_input":{},"last_error":null,"submit_outcomes":[]}

# 3. Step (SET_FIELD)
curl -X POST $BASE/step -H "Content-Type: application/json" \
  -d '{"action_type":"SET_FIELD","field":"age","value":25}'
# → {"current_input":{"age":25},"reward":-0.444...,"done":false,...}

# 4. Step (SUBMIT)
curl -X POST $BASE/step -H "Content-Type: application/json" \
  -d '{"action_type":"SUBMIT"}'
# → {"last_status":500,"covered_branches":["missing_income"],"last_error":"Income is required",...}

# 5. Tasks
curl $BASE/tasks
# → [{"id":"easy_task","name":"Trigger SSN Error","has_grader":true},...]

# 6. Swagger UI → open $BASE/docs in browser
# 7. Dashboard  → open $BASE/ in browser
```

---

## 7. OpenEnv Validation

**Official hackathon gate.** Must pass before submission.

```bash
openenv validate --url https://<HF_USERNAME>-edge-forge-env.hf.space
```

Expected:
```
✓ Health endpoint reachable
✓ Metadata schema valid
✓ Reset/Step endpoints functional
✓ Reward in range [0.0, 1.0]
✓ openenv.yaml manifest valid
✓ All 3 tasks validated: easy_task, medium_task, hard_task
All checks passed. Environment is OpenEnv compliant.
```

> ⚠️ Run this before every submission. A single failing check disqualifies the environment.

---

## 8. Three Access Modes

**Mode 1 — Live HTTP API** (no install):
```bash
curl https://<HF_USERNAME>-edge-forge-env.hf.space/health
curl -X POST .../reset -d '{}'
curl -X POST .../step -H "Content-Type: application/json" -d '{"action_type":"SUBMIT"}'
```

**Mode 2 — Python SDK**:
```bash
pip install git+https://huggingface.co/spaces/<HF_USERNAME>/edge_forge_env
```
```python
from client import EdgeForgeEnv
from models import EdgeForgeAction
env = EdgeForgeEnv(base_url="https://<HF_USERNAME>-edge-forge-env.hf.space")
result = await env.reset()
result = await env.step(EdgeForgeAction(action_type="SET_FIELD", field="age", value=25))
```

**Mode 3 — Docker** (local high-throughput):
```bash
docker pull registry.hf.space/<HF_USERNAME>-edge-forge-env:latest
docker run -p 8000:8000 registry.hf.space/<HF_USERNAME>-edge-forge-env:latest
```

---

## 9. Local Testing Before Push

```bash
# Build
docker build -t edge-forge:local .

# Run (simulate HF CPU Basic constraints)
docker run -p 8000:8000 --memory=512m --cpus=1 edge-forge:local

# Verify
curl http://localhost:8000/health
openenv validate --url http://localhost:8000

# Run inference locally
ENV_BASE_URL=http://localhost:8000 HF_TOKEN=hf_xxx python inference.py
```

> 💡 Always validate locally first. HF build logs are harder to debug than local Docker output.

---

## 10. Updating the Space

```bash
git add . && git commit -m "fix: <describe change>" && git push
# Space rebuilds automatically. HF caches Docker layers — unchanged layers build instantly.
```

Rollback: `git revert HEAD && git push` or `git reset --hard <hash> && git push --force`

---

## 11. Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| Space stuck "Building" >15 min | Dockerfile error | Check Build logs. Test locally: `docker build --no-cache .` |
| `/health` returns 502 | Container not started | Wait 60s after build, retry |
| `/health` returns 503 | Space sleeping | Send any request to wake (~10s cold start) |
| `/step` returns 422 | Missing `action_type` | Body needs `{"action_type":"SUBMIT"}` |
| `/step` returns 500 | No active episode | Call `/reset` first |
| `openenv validate` fails | Task name mismatch | Verify IDs in `openenv.yaml`: `easy_task`, `medium_task`, `hard_task` |
| Git push rejected | LFS not configured | `git lfs install` then retry |
| Import error in logs | Missing dependency | Add to `pyproject.toml`, push again |
| `inference.py` 401 | `HF_TOKEN` not set | Set as Secret in Space Settings. Fallbacks run regardless |
| **Space sleeps during judging** | **Default sleep policy** | **Settings → Sleep time → "Never"** |

---

## 12. Space Settings Checklist

✅ Visibility: **Public**
✅ SDK: **Docker**
✅ App Port: **8000**
✅ Hardware: **CPU Basic**
✅ Sleep time: **Never** ← set this before hackathon submission
✅ Persistent storage: Not required
✅ Linked dataset: Not required

---

## 13. Monitoring & Logs

**HF Dashboard:** Space page → **Logs** tab → toggle Build/Container logs. Filter for `ERROR`, `WARNING`.

Healthy logs look like:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     "POST /reset HTTP/1.1" 200 OK
```

---

## 14. Performance

| Tier | vCPUs | RAM | Concurrent Sessions |
|------|-------|-----|-------------------|
| CPU Basic (Free) | 2 | 16 GB | ~128 |
| CPU Upgrade | 8 | 32 GB | ~512 |

CPU Basic is sufficient for hackathon evaluation. Edge-Forge completes full inference in **<4 minutes** on 2 vCPU / 8 GB.

---

## 15. Support & Resources

| Resource | Link |
|----------|------|
| HuggingFace Spaces Docs | https://huggingface.co/docs/hub/spaces |
| HuggingFace Docker Spaces | https://huggingface.co/docs/hub/spaces-sdks-docker |
| OpenEnv GitHub | https://github.com/meta-pytorch/OpenEnv |
| OpenEnv Hackathon | https://scaler.com/school-of-technology/meta-pytorch-hackathon |
| OpenEnv Course | https://deepwiki.com/raun/openenv-course |
| Edge-Forge Repo | https://github.com/SoumyaWasule/Edge-Forge |
| Edge-Forge Issues | https://github.com/SoumyaWasule/Edge-Forge/issues |

---

*Built for the **Meta × PyTorch × Hugging Face OpenEnv Hackathon***
