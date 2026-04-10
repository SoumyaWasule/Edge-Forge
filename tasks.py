"""
Tasks and graders for Edge Forge Environment.

Each grader returns a float in (0, 1) exclusive — strictly between 0.01 and 0.99.
Graders verify actual API response messages (observable outcomes)
rather than internally-tracked branch labels.
"""

from typing import Any


# Total number of unique API response outcomes across all code paths
TOTAL_OUTCOMES = 19


def _get_submit_outcomes(state: Any) -> list:
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


TASKS = [
    {"id": "easy_task", "name": "Trigger SSN Error", "grader": grade_easy},
    {"id": "medium_task", "name": "Maximize Coverage", "grader": grade_medium},
    {"id": "hard_task", "name": "Stateful Crash Trap", "grader": grade_hard},
]


def get_tasks():
    """Return task definitions with bound graders."""
    return TASKS