"""
Tasks and graders for Edge Forge Environment.

Each grader returns a deterministic float in [0.0, 1.0].
"""

from edge_forge_env.mock_api import TOTAL_BRANCHES


# =========================
# EASY TASK — TRIGGER AN ERROR
# =========================
def grade_easy(observation):
    """
    Score 1.0 if the agent discovers the SSN format validation bug.
    Requires sequential API calls: open_account → verify_identity with invalid SSN.
    """
    if "ssn_format_bug" in observation.covered_branches:
        return 1.0
    return 0.0


# =========================
# MEDIUM TASK — BRANCH COVERAGE
# =========================
def grade_medium(observation):
    """
    Score = proportion of unique branches discovered.
    Clamped to [0.0, 1.0] for safety.
    """
    covered = len(observation.covered_branches)
    return min(covered / TOTAL_BRANCHES, 1.0)


# =========================
# HARD TASK — DEEP BRANCH
# =========================
def grade_hard(observation):
    """
    Score 1.0 if the stateful crash is triggered via account lifecycle sequence.
    Requires: open_account → verify_identity (without SSN) while account is pending.
    """
    if "stateful_crash" in observation.covered_branches:
        return 1.0
    return 0.0