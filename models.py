# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Edge Forge Environment.
This environment simulates a code-aware testing system where an agent mutates inputs
to explore branches and trigger edge-case failures.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Optional, Any, Dict, List


# =========================
# ACTION
# =========================
class EdgeForgeAction(Action):
    """
    Action that the agent can take.

    Types:
    - SET_FIELD → modify input data
    - RESET → clear current input
    - SUBMIT → send data to API
    """

    action_type: str = Field(
        ...,
        description="Type of action: SET_FIELD, RESET, SUBMIT"
    )

    field: Optional[str] = Field(
        default=None,
        description="Field to modify (age, income, user_type, balance, days_active, credit_score, region, action, ssn)"
    )

    value: Optional[Any] = Field(
        default=None,
        description="Value to assign to the field"
    )


# =========================
# OBSERVATION
# =========================
class EdgeForgeObservation(Observation):
    """
    Observation returned after each step.

    Includes:
    - API response status
    - Which branches have been covered
    - Current input state
    - Any error message
    """

    last_status: int = Field(
        default=0,
        description="HTTP-like status: 200 (success), 500 (error)"
    )

    covered_branches: List[str] = Field(
        default_factory=list,
        description="List of code branches discovered so far"
    )

    current_input: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current input payload being constructed by the agent"
    )

    last_error: Optional[str] = Field(
        default=None,
        description="Error message if last action caused a failure"
    )

    submit_outcomes: List[str] = Field(
        default_factory=list,
        description="Accumulated API response summaries from each submission, used by graders to verify actual API behavior"
    )