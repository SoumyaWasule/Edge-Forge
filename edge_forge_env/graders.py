"""
Graders for Edge Forge Environment.

Each grader returns a float in [0, 1].
The validator discovers graders by importing this file.
"""

from typing import Any


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def grade_easy(state: Any) -> float:
    """Grade easy task: discover SSN format bug."""
    try:
        if isinstance(state, dict):
            branches = state.get("covered_branches", [])
        else:
            branches = getattr(state, "covered_branches", [])
        return 0.99 if "ssn_format_bug" in branches else 0.01
    except Exception:
        return 0.01


def grade_medium(state: Any) -> float:
    """Grade medium task: branch coverage ratio."""
    try:
        if isinstance(state, dict):
            branches = state.get("covered_branches", [])
        else:
            branches = getattr(state, "covered_branches", [])
        raw = len(branches) / 19.0
        return clamp01(raw) if raw > 0 else 0.01
    except Exception:
        return 0.01


def grade_hard(state: Any) -> float:
    """Grade hard task: trigger stateful crash."""
    try:
        if isinstance(state, dict):
            branches = state.get("covered_branches", [])
        else:
            branches = getattr(state, "covered_branches", [])
        return 0.99 if "stateful_crash" in branches else 0.01
    except Exception:
        return 0.01
