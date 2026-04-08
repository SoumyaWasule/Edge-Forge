"""
Tasks and graders for Edge Forge Environment.

Each grader returns a float in (0, 1) exclusive — strictly between 0.01 and 0.99.
"""


TOTAL_BRANCHES = 19


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
        raw = len(branches) / TOTAL_BRANCHES
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


TASKS = [
    {"id": "easy_task", "name": "Trigger SSN Error", "grader": grade_easy},
    {"id": "medium_task", "name": "Maximize Coverage", "grader": grade_medium},
    {"id": "hard_task", "name": "Stateful Crash Trap", "grader": grade_hard},
]


def get_tasks():
    """Return task definitions with bound graders."""
    return TASKS