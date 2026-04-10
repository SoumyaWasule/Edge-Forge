# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Edge Forge Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import EdgeForgeAction, EdgeForgeObservation


class EdgeForgeEnv(
    EnvClient[EdgeForgeAction, EdgeForgeObservation, State]
):
    """
    Client for the Edge Forge Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with EdgeForgeEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.covered_branches)
        ...
        ...     action = EdgeForgeAction(action_type="SET_FIELD", field="age", value=25)
        ...     result = client.step(action)
        ...     print(result.observation.current_input)

    Example with Docker:
        >>> client = EdgeForgeEnv.from_docker_image("edge_forge_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(EdgeForgeAction(action_type="SUBMIT"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: EdgeForgeAction) -> Dict:
        """
        Convert EdgeForgeAction to JSON payload for step message.

        Args:
            action: EdgeForgeAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        payload = {"action_type": action.action_type}
        if action.field is not None:
            payload["field"] = action.field
        if action.value is not None:
            payload["value"] = action.value
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[EdgeForgeObservation]:
        """
        Parse server response into StepResult[EdgeForgeObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with EdgeForgeObservation
        """
        obs_data = payload.get("observation", {})
        observation = EdgeForgeObservation(
            last_status=obs_data.get("last_status", 0),
            covered_branches=obs_data.get("covered_branches", []),
            current_input=obs_data.get("current_input", {}),
            last_error=obs_data.get("last_error"),
            submit_outcomes=obs_data.get("submit_outcomes", []),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
