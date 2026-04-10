# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Edge Forge Env Environment.

Architecture:
  - Uses OpenEnv's create_app for schema, health, WS/MCP endpoints.
  - REMOVES the framework's stateless /reset and /step HTTP routes.
  - Replaces them with stateful versions that share a persistent
    EdgeForgeEnvironment instance, enabling multi-step RL episodes.
"""

import threading

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.serialization import serialize_observation
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import EdgeForgeAction, EdgeForgeObservation
    from .edge_forge_env_environment import EdgeForgeEnvironment
except ImportError:
    from models import EdgeForgeAction, EdgeForgeObservation
    from server.edge_forge_env_environment import EdgeForgeEnvironment


# ================================================================
# GRADER FUNCTIONS (module-level, for validator discovery)
# Graders verify actual API response messages (observable outcomes)
# rather than internally-tracked branch labels.
# ================================================================
TOTAL_OUTCOMES = 19


def _get_submit_outcomes(state):
    """Extract submit_outcomes from state (dict or object)."""
    if isinstance(state, dict):
        return state.get("submit_outcomes", [])
    return getattr(state, "submit_outcomes", [])


def grade_easy(observation):
    """Grade easy task: verify the agent triggered a real SSN format validation error."""
    try:
        outcomes = _get_submit_outcomes(observation)
        return 0.99 if "SSN must be numeric" in outcomes else 0.01
    except Exception:
        return 0.01


def grade_medium(observation):
    """Grade medium task: score based on diversity of actual API outcomes triggered."""
    try:
        outcomes = _get_submit_outcomes(observation)
        unique = len(set(outcomes))
        raw = unique / TOTAL_OUTCOMES
        return max(0.01, min(raw, 0.99))
    except Exception:
        return 0.01


def grade_hard(observation):
    """Grade hard task: verify the agent triggered the real SSN-missing crash."""
    try:
        outcomes = _get_submit_outcomes(observation)
        return 0.99 if "SSN missing during pending verification" in outcomes else 0.01
    except Exception:
        return 0.01


# ================================================================
# TASKS — module-level list for validator discovery
# ================================================================
TASKS = [
    {"id": "easy_task", "name": "Trigger SSN Error", "grader": grade_easy},
    {"id": "medium_task", "name": "Maximize Coverage", "grader": grade_medium},
    {"id": "hard_task", "name": "Stateful Crash Trap", "grader": grade_hard},
]


def get_tasks():
    """Return task definitions with bound graders."""
    return TASKS


# ================================================================
# Create base app (registers schema, health, WS, MCP, etc.)
# ================================================================
app = create_app(
    EdgeForgeEnvironment,
    EdgeForgeAction,
    EdgeForgeObservation,
    env_name="edge_forge_env",
    max_concurrent_envs=1,
)


# Remove framework's stateless /reset, /step, and /state routes
# We replace them with stateful versions that share _env_instance
_routes_to_remove = {"/reset", "/step", "/state"}
app.routes[:] = [
    route for route in app.routes
    if not (hasattr(route, "path") and route.path in _routes_to_remove)
]


# ================================================================
# Root dashboard — shows environment info when visiting the Space
# ================================================================
from fastapi.responses import HTMLResponse


@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
async def root_dashboard():
    """Serve the Edge Forge environment dashboard."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Edge Forge Environment</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #e0e0e0; min-height: 100vh; padding: 2rem;
        }
        .container { max-width: 900px; margin: 0 auto; }
        h1 {
            font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #00d2ff, #7b2ff7);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .subtitle { color: #aaa; font-size: 1.1rem; margin-bottom: 2rem; }
        .card {
            background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;
            backdrop-filter: blur(10px);
        }
        .card h2 {
            font-size: 1.3rem; color: #7b2ff7; margin-bottom: 1rem;
            border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 0.5rem;
        }
        .task {
            display: flex; justify-content: space-between; align-items: center;
            padding: 0.75rem 1rem; margin: 0.5rem 0; border-radius: 8px;
            background: rgba(255,255,255,0.04);
        }
        .task-name { font-weight: 600; }
        .badge {
            padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;
        }
        .badge-easy { background: #1b5e20; color: #81c784; }
        .badge-medium { background: #e65100; color: #ffb74d; }
        .badge-hard { background: #b71c1c; color: #ef9a9a; }
        .endpoint {
            font-family: 'Consolas', monospace; padding: 0.4rem 0;
            display: flex; gap: 1rem; align-items: center;
        }
        .method {
            padding: 2px 8px; border-radius: 4px; font-size: 0.75rem;
            font-weight: 700; min-width: 50px; text-align: center;
        }
        .method-get { background: #1b5e20; color: #81c784; }
        .method-post { background: #0d47a1; color: #64b5f6; }
        .status {
            display: inline-flex; align-items: center; gap: 8px;
            padding: 8px 16px; border-radius: 20px;
            background: rgba(76, 175, 80, 0.15); color: #81c784;
            font-weight: 600;
        }
        .status-dot {
            width: 10px; height: 10px; border-radius: 50%;
            background: #4caf50; animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
        }
        a { color: #64b5f6; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡ Edge Forge</h1>
        <p class="subtitle">Synthetic Test Data Generation &amp; Edge-Case Discovery Environment</p>

        <div class="card">
            <h2>Status</h2>
            <div class="status"><div class="status-dot"></div> Environment Running</div>
            <p style="margin-top:1rem;color:#aaa;">
                OpenEnv-compatible RL environment for discovering API validation bugs
                through intelligent exploration.
            </p>
        </div>

        <div class="card">
            <h2>Tasks</h2>
            <div class="task">
                <span class="task-name">🔍 easy_task — Trigger SSN Error</span>
                <span class="badge badge-easy">Easy</span>
            </div>
            <div class="task">
                <span class="task-name">📊 medium_task — Maximize Coverage</span>
                <span class="badge badge-medium">Medium</span>
            </div>
            <div class="task">
                <span class="task-name">💥 hard_task — Stateful Crash Trap</span>
                <span class="badge badge-hard">Hard</span>
            </div>
        </div>

        <div class="card">
            <h2>API Endpoints</h2>
            <div class="endpoint"><span class="method method-post">POST</span> <a href="/docs#/Environment%20Control/reset_stateful_reset_post">/reset</a> — Reset environment</div>
            <div class="endpoint"><span class="method method-post">POST</span> <a href="/docs#/Environment%20Control/step_stateful_step_post">/step</a> — Execute action</div>
            <div class="endpoint"><span class="method method-get">GET</span> <a href="/state">/state</a> — Current state</div>
            <div class="endpoint"><span class="method method-get">GET</span> <a href="/tasks">/tasks</a> — List tasks</div>
            <div class="endpoint"><span class="method method-post">POST</span> <a href="/docs#/Tasks/grade_task_grade_post">/grade</a> — Grade task</div>
            <div class="endpoint"><span class="method method-get">GET</span> <a href="/health">/health</a> — Health check</div>
            <div class="endpoint"><span class="method method-get">GET</span> <a href="/docs">/docs</a> — Swagger UI</div>
        </div>

        <div class="card">
            <h2>Action Space</h2>
            <p style="color:#aaa;margin-bottom:0.75rem;">Agents interact via three action types:</p>
            <div class="endpoint">SET_FIELD — Set a field value: <code style="color:#7b2ff7">{"action_type": "SET_FIELD", "field": "age", "value": 25}</code></div>
            <div class="endpoint">SUBMIT — Submit current payload: <code style="color:#7b2ff7">{"action_type": "SUBMIT"}</code></div>
            <div class="endpoint">RESET — Clear input fields: <code style="color:#7b2ff7">{"action_type": "RESET"}</code></div>
        </div>
    </div>
</body>
</html>"""


# Persistent environment for stateful HTTP episodes
_env_lock = threading.Lock()
_env_instance: EdgeForgeEnvironment = EdgeForgeEnvironment()


@app.post(
    "/reset",
    tags=["Environment Control"],
    summary="Reset the environment",
    description="Reset the environment to its initial state and return the initial observation.",
)
async def reset_stateful(request: dict = {}):
    """Reset the persistent environment and return initial observation."""
    with _env_lock:
        observation = _env_instance.reset()
    return serialize_observation(observation)


@app.post(
    "/step",
    tags=["Environment Control"],
    summary="Execute an action",
    description="Execute an action in the environment. State persists across calls within an episode.",
)
async def step_stateful(request: dict = {}):
    """Execute an action in the persistent environment."""
    if "action" in request:
        action_data = request["action"]
    elif "action_type" in request:
        action_data = request
    else:
        action_data = request.get("action", {})
    action = EdgeForgeAction(**action_data)

    with _env_lock:
        observation = _env_instance.step(action)
    return serialize_observation(observation)


@app.get(
    "/tasks",
    tags=["Tasks"],
    summary="List available tasks with graders",
)
async def list_tasks_endpoint():
    """Return task definitions with grader status."""
    return [
        {"id": t["id"], "name": t["name"], "has_grader": True}
        for t in TASKS
    ]


@app.post(
    "/grade",
    tags=["Tasks"],
    summary="Grade a specific task",
)
async def grade_task(request: dict = {}):
    """Grade a task using actual API outcomes from the current episode."""
    task_id = request.get("task_id", "")
    with _env_lock:
        obs = {
            "submit_outcomes": list(_env_instance.state.get("submit_outcomes", []))
            if _env_instance.state else [],
            "covered_branches": list(_env_instance.state.get("covered_branches", []))
            if _env_instance.state else [],
        }
    for t in TASKS:
        if t["id"] == task_id:
            return {"task_id": task_id, "score": t["grader"](obs)}
    return {"error": f"Unknown task_id: {task_id}", "score": 0.01}


@app.get(
    "/state",
    tags=["State Management"],
    summary="Get current environment state",
    description="Returns the current episode state including episode_id and step_count.",
)
async def get_state():
    """Return current environment state using the persistent env instance."""
    with _env_lock:
        if _env_instance.state is None:
            return {"episode_id": "", "step_count": 0}
        return {
            "episode_id": "current",
            "step_count": _env_instance.state.get("steps", 0),
        }


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution.

    Usage:
        uv run --project . server
        python -m edge_forge_env.server.app
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
