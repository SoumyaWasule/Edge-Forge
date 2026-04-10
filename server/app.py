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
