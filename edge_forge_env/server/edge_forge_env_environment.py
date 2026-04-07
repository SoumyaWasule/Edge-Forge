"""
Edge Forge Environment — core RL environment implementation.

This environment simulates an application under test. The agent builds
input payloads (SET_FIELD), submits them (SUBMIT), and receives feedback
on which code branches were triggered and whether errors occurred.

Key design decisions:
  - SUBMIT is NOT terminal — the agent can submit multiple times per
    episode to discover branches on different code paths.
  - RESET clears the current input AND gives a small exploration bonus
    when switching to a new payload after a submission.
  - Intermediate rewards are provided for input completeness (process
    supervision), not just at submission time.
  - Input validation prevents type-confusion crash rewards.
  - Application state is reset each episode, requiring agents to learn
    multi-step call sequences to trigger stateful bugs.
"""

import random
from typing import Tuple, Dict, Any
from edge_forge_env.models import EdgeForgeAction, EdgeForgeObservation
from edge_forge_env.mock_api import process_application, TOTAL_BRANCHES
from openenv.core.env_server.types import EnvironmentMetadata


# ── Input validation ────────────────────────────────────────────────
VALID_FIELDS = {
    "age": (int, float),
    "income": (int, float),
    "user_type": str,
    "balance": (int, float),
    "days_active": (int, float),
    "credit_score": (int, float),
    "region": str,
    "action": str,           # stateful API: "open_account", "verify_identity"
    "ssn": (str, int),       # for identity verification
}


def _validate_field(field: str, value: Any) -> bool:
    """Return True if value has an acceptable type for the given field."""
    expected = VALID_FIELDS.get(field)
    if expected is None:
        return False
    return isinstance(value, expected)


class EdgeForgeEnvironment:

    MAX_STEPS = 30

    def __init__(self):
        self.state = None

    def get_metadata(self) -> EnvironmentMetadata:
        """Return environment metadata for OpenEnv validation."""
        return EnvironmentMetadata(
            name="Edge-Forge",
            description="Autonomous Synthetic Staging Engine with Stateful Bugs"
        )

    # ────────────────────────────────────────────────────────────────
    # RESET
    # ────────────────────────────────────────────────────────────────
    def reset(self) -> EdgeForgeObservation:
        self.state = {
            "current_input": {},
            "covered_branches": set(),
            "steps": 0,
            "submits": 0,
            "done": False,
            "app_state": {"status": None, "verification_attempts": 0},
            "thresholds": {
                "age_limit": random.randint(16, 21),
                "enterprise_days": random.randint(300, 400),
            },
        }

        return EdgeForgeObservation(
            last_status=0,
            covered_branches=[],
            current_input={},
        )

    # ────────────────────────────────────────────────────────────────
    # STEP
    # ────────────────────────────────────────────────────────────────
    def step(self, action: EdgeForgeAction) -> EdgeForgeObservation:

        # Guard: reject actions after episode termination
        if self.state is not None and self.state.get("done", False):
            return EdgeForgeObservation(
                last_status=0,
                covered_branches=list(self.state["covered_branches"]),
                current_input=self.state["current_input"],
                last_error="Episode already terminated. Please reset.",
                reward=0.0,
                done=True,
                metadata={},
            )

        reward = 0.0
        done = False
        info = {}
        error_msg = None
        status = 0

        # Safety: auto-reset if step called without reset
        if self.state is None:
            self.reset()

        self.state["steps"] += 1

        # ── ACTION: SET_FIELD ───────────────────────────────────────
        if action.action_type == "SET_FIELD":
            if action.field in VALID_FIELDS:
                if action.value is not None and _validate_field(action.field, action.value):
                    self.state["current_input"][action.field] = action.value
                    # Intermediate reward — input completeness
                    completeness = len(self.state["current_input"]) / len(VALID_FIELDS)
                    reward += completeness * 0.5
                else:
                    # Penalize invalid type
                    reward -= 2.0
            else:
                reward -= 1.0  # unknown field

        # ── ACTION: RESET ───────────────────────────────────────────
        elif action.action_type == "RESET":
            if self.state["submits"] > 0 and len(self.state["current_input"]) > 0:
                reward += 1.0  # exploration diversity bonus
            self.state["current_input"] = {}

        # ── ACTION: SUBMIT ──────────────────────────────────────────
        elif action.action_type == "SUBMIT":
            self.state["submits"] += 1
            old_coverage = set(self.state["covered_branches"])

            result, covered = process_application(
                self.state["current_input"],
                self.state["app_state"],
                self.state["thresholds"],
            )

            # Detect errors
            if result.get("status") == "error":
                status = 500
                error_msg = result.get("error", "Unknown error")
                if covered - old_coverage:
                    reward += 50.0
            else:
                status = 200

            # Coverage reward — only for NEW branches
            new_branches = covered - old_coverage
            self.state["covered_branches"].update(covered)
            reward += len(new_branches) * 10.0

            # Bonus for deep/nested/stateful branches
            hard_branches = {
                "deep_branch", "enterprise_debt_recovery",
                "restricted_region_override", "enterprise_premium",
                "stateful_crash", "ssn_format_bug", "account_verified",
            }
            hard_new = new_branches & hard_branches
            reward += len(hard_new) * 25.0

            # Clear input after submission
            self.state["current_input"] = {}

            info["submit_result"] = result
            info["new_branches"] = list(new_branches)

        # ── Step penalty (always applied) ───────────────────────────
        reward -= 1.0

        # ── Termination ─────────────────────────────────────────────
        if self.state["steps"] >= self.MAX_STEPS:
            done = True

        # ── Coverage milestone bonus ────────────────────────────────
        coverage_ratio = len(self.state["covered_branches"]) / TOTAL_BRANCHES
        if coverage_ratio >= 1.0 and not done:
            reward += 100.0
            done = True

        # ── Track done state ────────────────────────────────────────
        if done:
            self.state["done"] = True

        observation = EdgeForgeObservation(
            last_status=status,
            covered_branches=list(self.state["covered_branches"]),
            current_input=self.state["current_input"],
            last_error=error_msg,
            reward=reward,
            done=done,
            metadata=info,
        )

        return observation

    # ────────────────────────────────────────────────────────────────
    # ASYNC WRAPPERS (required by OpenEnv)
    # ────────────────────────────────────────────────────────────────
    async def reset_async(self):
        return self.reset()

    async def step_async(self, action):
        return self.step(action)

    def close(self):
        pass