"""
Graders for Edge Forge Environment.

Each grader returns a float in [0, 1].
The validator discovers graders by importing this file.

Graders verify actual API response messages (observable outcomes)
rather than internally-tracked branch labels. This ensures grading
is based on real API behavior the agent demonstrably triggered.
"""

from typing import Any


# Total number of unique API response outcomes across all code paths
TOTAL_OUTCOMES = 19


def _get_submit_outcomes(state: Any) -> list:
    """Extract submit_outcomes from state (dict or object)."""
    if isinstance(state, dict):
        return state.get("submit_outcomes", [])
    return getattr(state, "submit_outcomes", [])


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def grade_easy(state: Any) -> float:
    """Grade easy task: verify the agent triggered a real SSN format validation error.

    Checks that the actual API returned "SSN must be numeric" — the real error
    message from the mock API's SSN format validation path.
    """
    try:
        outcomes = _get_submit_outcomes(state)
        return 0.99 if "SSN must be numeric" in outcomes else 0.01
    except Exception:
        return 0.01


def grade_medium(state: Any) -> float:
    """Grade medium task: score based on diversity of actual API outcomes triggered.

    Measures how many distinct API response messages the agent elicited,
    divided by the total number of unique API code paths (19).
    """
    try:
        outcomes = _get_submit_outcomes(state)
        unique = len(set(outcomes))
        raw = unique / TOTAL_OUTCOMES
        return clamp01(raw) if raw > 0 else 0.01
    except Exception:
        return 0.01


def grade_hard(state: Any) -> float:
    """Grade hard task: verify the agent triggered the real SSN-missing crash.

    Checks that the actual API returned "SSN missing during pending verification"
    — the real error message from the stateful crash path.
    """
    try:
        outcomes = _get_submit_outcomes(state)
        return 0.99 if "SSN missing during pending verification" in outcomes else 0.01
    except Exception:
        return 0.01
