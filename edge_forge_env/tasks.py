"""
Tasks and graders for Edge Forge Environment.

Each grader returns a float in (0, 1) exclusive — strictly between 0.01 and 0.99.
OpenEnv Phase 2 validator uses STRICT inequality: 0 < score < 1.
"""

from edge_forge_env.mock_api import TOTAL_BRANCHES


# =========================
# EASY TASK — TRIGGER SSN BUG
# =========================
def grade_easy(observation):
    """
    Score 0.99 if the agent discovers the SSN format validation bug.
    Requires sequential API calls: open_account -> verify_identity with invalid SSN.
    """
    try:
        branches = observation.get("covered_branches", []) if isinstance(observation, dict) else getattr(observation, "covered_branches", [])
        return 0.99 if "ssn_format_bug" in branches else 0.01
    except Exception:
        return 0.01


# =========================
# MEDIUM TASK — BRANCH COVERAGE
# =========================
def grade_medium(observation):
    """
    Score = proportion of unique branches discovered.
    Clamped to [0.01, 0.99] for strict validator compliance.
    """
    try:
        branches = observation.get("covered_branches", []) if isinstance(observation, dict) else getattr(observation, "covered_branches", [])
        raw_score = len(branches) / TOTAL_BRANCHES
        return max(0.01, min(raw_score, 0.99))
    except Exception:
        return 0.01


# =========================
# HARD TASK — STATEFUL CRASH
# =========================
def grade_hard(observation):
    """
    Score 0.99 if the stateful crash is triggered via account lifecycle sequence.
    Requires: open_account -> verify_identity (without SSN) while account is pending.
    """
    try:
        branches = observation.get("covered_branches", []) if isinstance(observation, dict) else getattr(observation, "covered_branches", [])
        return 0.99 if "stateful_crash" in branches else 0.01
    except Exception:
        return 0.01