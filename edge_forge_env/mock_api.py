"""
Mock Application API — simulates a complex loan/user processing system.

This module represents the "system under test" — the application whose
edge cases the RL agent must discover through synthetic input generation.

Design principles:
  - Returns errors as result dicts instead of raising exceptions,
    so branch coverage is never lost.
  - Branches are layered with nested conditions to require multi-step
    reasoning (not just single-field guessing).
  - Stateful bugs require SEQUENTIAL API calls (e.g., open_account THEN
    verify_identity) — random agents can't reliably discover these.
  - 19 distinct branches provide a meaningful exploration space.
"""

TOTAL_BRANCHES = 19



def process_application(data: dict, app_state: dict, thresholds: dict) -> tuple[dict, set]:
    """
    Process a loan/user application and return (result, covered_branches).

    Args:
        data: Input payload from the agent.
        app_state: Mutable application state (persists within episode).
        thresholds: Randomized thresholds for stochastic evaluation.

    Every code path adds its branch label to `covered` before returning,
    ensuring the caller always receives complete coverage information.
    """
    covered = set()

    action = data.get("action")
    age = data.get("age")
    income = data.get("income")
    user_type = data.get("user_type")
    balance = data.get("balance", 0)
    days_active = data.get("days_active", 0)
    credit_score = data.get("credit_score", 0)
    region = data.get("region")

    # ── Stateful API: Account lifecycle ─────────────────────────────
    # These branches REQUIRE sequential calls — random agents can't
    # reliably discover them because they need prior state.

    if action == "open_account":
        covered.add("account_opened")
        app_state["status"] = "pending"
        return {"status": "ok", "account": "pending"}, covered

    if action == "verify_identity":
        covered.add("verify_attempt")
        app_state["verification_attempts"] = app_state.get("verification_attempts", 0) + 1

        if app_state.get("status") == "pending":
            # Bug: SSN required during pending verification
            if data.get("ssn") is None:
                covered.add("stateful_crash")
                return {"status": "error", "error": "SSN missing during pending verification"}, covered

            # Bug: SSN with invalid format after pending
            if not str(data.get("ssn", "")).isdigit():
                covered.add("ssn_format_bug")
                return {"status": "error", "error": "SSN must be numeric"}, covered

            app_state["status"] = "active"
            covered.add("account_verified")
            return {"status": "ok", "account": "active"}, covered

        return {"status": "ok", "note": "no pending account"}, covered

    # ── Layer 1: Required field validation ──────────────────────────
    if age is None:
        covered.add("missing_age")
        return {"status": "error", "error": "Age is required"}, covered

    if income is None:
        covered.add("missing_income")
        return {"status": "error", "error": "Income is required"}, covered

    # ── Layer 2: Basic eligibility ──────────────────────────────────
    if age < thresholds.get("age_limit", 18):
        covered.add("underage")
        return {"status": "rejected", "reason": "underage"}, covered

    if income < 0:
        covered.add("negative_income")
        return {"status": "error", "error": "Invalid income value"}, covered

    # ── Layer 3: Financial risk assessment ──────────────────────────
    if balance < -1000:
        covered.add("extreme_debt")
        # Nested: extreme-debt enterprise recovery
        if user_type == "enterprise" and income > 50000:
            covered.add("enterprise_debt_recovery")
            return {"status": "recovery_program", "risk": "high"}, covered
        return {"status": "rejected", "reason": "extreme_debt"}, covered

    if credit_score > 0 and credit_score < 300:
        covered.add("terrible_credit")
        return {"status": "rejected", "reason": "credit_too_low"}, covered

    # ── Layer 4: User-type specific paths ───────────────────────────
    if user_type == "enterprise":
        covered.add("enterprise_path")

        # Deep nested: enterprise + deficit + long tenure
        if balance < 0 and days_active > thresholds.get("enterprise_days", 365):
            covered.add("deep_branch")
            return {"status": "special_case", "tier": "enterprise_veteran"}, covered

        if income > 100000:
            covered.add("enterprise_premium")
            return {"status": "approved", "tier": "premium"}, covered

        return {"status": "approved", "tier": "standard_enterprise"}, covered

    # ── Layer 5: Region-specific compliance ─────────────────────────
    if region == "restricted":
        covered.add("restricted_region")
        if income > 75000 and days_active > 180:
            covered.add("restricted_region_override")
            return {"status": "approved", "note": "compliance_override"}, covered
        return {"status": "pending_review", "reason": "region_restriction"}, covered

    # ── Layer 6: New user handling ──────────────────────────────────
    if days_active < 10:
        covered.add("new_user")
        return {"status": "limited", "reason": "new_account"}, covered

    # ── Default: approved ───────────────────────────────────────────
    covered.add("approved")
    return {"status": "approved"}, covered