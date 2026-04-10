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
# ================================================================
def grade_easy(observation):
    """Grade easy task: discover SSN format bug."""
    try:
        if isinstance(observation, dict):
            branches = observation.get("covered_branches", [])
        else:
            branches = getattr(observation, "covered_branches", [])
        return 0.99 if "ssn_format_bug" in branches else 0.01
    except Exception:
        return 0.01


def grade_medium(observation):
    """Grade medium task: branch coverage ratio."""
    try:
        if isinstance(observation, dict):
            branches = observation.get("covered_branches", [])
        else:
            branches = getattr(observation, "covered_branches", [])
        raw = len(branches) / 19.0
        return max(0.01, min(raw, 0.99))
    except Exception:
        return 0.01


def grade_hard(observation):
    """Grade hard task: trigger stateful crash."""
    try:
        if isinstance(observation, dict):
            branches = observation.get("covered_branches", [])
        else:
            branches = getattr(observation, "covered_branches", [])
        return 0.99 if "stateful_crash" in branches else 0.01
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


# Remove framework's stateless /reset and /step routes
_routes_to_remove = {"/reset", "/step"}
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
    """Grade a task using the current environment state."""
    task_id = request.get("task_id", "")
    with _env_lock:
        obs = {
            "covered_branches": list(_env_instance.state.get("covered_branches", []))
            if _env_instance.state else []
        }
    for t in TASKS:
        if t["id"] == task_id:
            return {"task_id": task_id, "score": t["grader"](obs)}
    return {"error": f"Unknown task_id: {task_id}", "score": 0.01}


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
