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


# ── Create base app (registers schema, health, WS, MCP, etc.) ──────
app = create_app(
    EdgeForgeEnvironment,
    EdgeForgeAction,
    EdgeForgeObservation,
    env_name="edge_forge_env",
    max_concurrent_envs=1,
)


# ── Remove framework's stateless /reset and /step routes ────────────
# The OpenEnv framework registers these as stateless (fresh env per request).
# We need to replace them with stateful versions for multi-step episodes.
_routes_to_remove = {"/reset", "/step"}
app.routes[:] = [
    route for route in app.routes
    if not (hasattr(route, "path") and route.path in _routes_to_remove)
]


# ── Persistent environment for stateful HTTP episodes ───────────────
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
    action_data = request.get("action", {})
    action = EdgeForgeAction(**action_data)

    with _env_lock:
        observation = _env_instance.step(action)
    return serialize_observation(observation)


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
