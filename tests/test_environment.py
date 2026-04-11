"""Smoke tests for Edge Forge environment."""
import pytest
from server.edge_forge_env_environment import EdgeForgeEnvironment
from models import EdgeForgeAction


class TestEnvironment:
    def setup_method(self):
        self.env = EdgeForgeEnvironment()

    def test_reset_returns_observation(self):
        obs = self.env.reset()
        assert obs is not None
        assert obs.last_status == 0          # No API call made yet
        assert obs.covered_branches == []
        assert obs.submit_outcomes == []

    def test_step_set_field(self):
        self.env.reset()
        action = EdgeForgeAction(action_type="SET_FIELD", field="age", value=25)
        obs = self.env.step(action)
        assert obs is not None
        assert obs.current_input.get("age") == 25

    def test_step_submit(self):
        self.env.reset()
        action = EdgeForgeAction(action_type="SET_FIELD", field="action", value="open_account")
        self.env.step(action)
        submit = EdgeForgeAction(action_type="SUBMIT")
        obs = self.env.step(submit)
        assert len(obs.submit_outcomes) > 0

    def test_graders_return_valid_range(self):
        self.env.reset()
        for grader in [self.env.grade_easy, self.env.grade_medium, self.env.grade_hard]:
            score = grader()
            assert 0.0 <= score <= 1.0

    def test_easy_grader_scores_on_ssn_bug(self):
        """Instance grader must match module-level grader logic."""
        self.env.reset()
        # Step 1: open_account sets app_state["status"] = "pending"
        self.env.step(EdgeForgeAction(action_type="SET_FIELD", field="action", value="open_account"))
        self.env.step(EdgeForgeAction(action_type="SUBMIT"))
        # Step 2: verify_identity with non-numeric SSN triggers "SSN must be numeric"
        self.env.step(EdgeForgeAction(action_type="SET_FIELD", field="action", value="verify_identity"))
        self.env.step(EdgeForgeAction(action_type="SET_FIELD", field="ssn", value="abc"))
        self.env.step(EdgeForgeAction(action_type="SUBMIT"))
        assert self.env.grade_easy() == 0.99

    def test_hard_grader_scores_on_stateful_crash(self):
        """Instance grader must match module-level grader logic."""
        self.env.reset()
        # open_account → SUBMIT → verify_identity (no SSN) → SUBMIT
        self.env.step(EdgeForgeAction(action_type="SET_FIELD", field="action", value="open_account"))
        self.env.step(EdgeForgeAction(action_type="SUBMIT"))
        self.env.step(EdgeForgeAction(action_type="SET_FIELD", field="action", value="verify_identity"))
        self.env.step(EdgeForgeAction(action_type="SUBMIT"))
        assert self.env.grade_hard() == 0.99
