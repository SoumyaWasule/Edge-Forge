"""
Baseline inference script for Edge Forge.

Demonstrates a random agent that builds input payloads and submits
them to discover branches and stateful bugs. Shows how the environment
rewards sequential decision-making over random guessing.
"""

import random
import requests
import time

BASE_URL = "http://localhost:8000"

# ── Color codes for terminal output ─────────────────────────────────
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"

# ── All valid fields and their value generators ─────────────────────
FIELD_GENERATORS = {
    "age": lambda: random.randint(10, 60),
    "income": lambda: random.choice([-500, 0, 5000, 30000, 80000, 120000]),
    "user_type": lambda: random.choice(["normal", "enterprise"]),
    "balance": lambda: random.choice([-2000, -500, -1, 0, 100, 3000]),
    "days_active": lambda: random.choice([0, 5, 50, 200, 400]),
    "credit_score": lambda: random.choice([0, 200, 500, 750]),
    "region": lambda: random.choice([None, "us", "restricted"]),
    "action": lambda: random.choice(["open_account", "verify_identity", None]),
    "ssn": lambda: random.choice([None, "123456789", "invalid", 12345]),
}

FIELDS = list(FIELD_GENERATORS.keys())


def random_action():
    """Generate a random action weighted toward productive choices."""
    r = random.random()

    if r < 0.50:
        # SET_FIELD — build payload
        field = random.choice(FIELDS)
        value = FIELD_GENERATORS[field]()
        return {"action_type": "SET_FIELD", "field": field, "value": value}
    elif r < 0.85:
        # SUBMIT — try the current payload
        return {"action_type": "SUBMIT"}
    else:
        # RESET — clear and try different combo
        return {"action_type": "RESET"}


def reset_env():
    response = requests.post(f"{BASE_URL}/reset", json={})
    return response.json()


def step_env(action):
    response = requests.post(f"{BASE_URL}/step", json={"action": action})
    try:
        return response.json()
    except Exception:
        print(f"{RED}❌ RAW RESPONSE: {response.text}{RESET}")
        raise


def format_action(action):
    """Format action for display."""
    a_type = action["action_type"]
    if a_type == "SET_FIELD":
        return f"SET_FIELD({action.get('field')}={action.get('value')})"
    return a_type


def run_episode(episode_num, verbose=True):
    reset_env()
    done = False
    total_reward = 0.0
    step_count = 0
    all_branches = set()
    crashes = 0

    if verbose:
        print(f"\n{BOLD}{CYAN}{'═' * 60}{RESET}")
        print(f"{BOLD}{CYAN}  EPISODE {episode_num}{RESET}")
        print(f"{BOLD}{CYAN}{'═' * 60}{RESET}")

    while not done:
        action = random_action()
        result = step_env(action)

        if "observation" not in result:
            print(f"{RED}❌ Invalid response: {result}{RESET}")
            raise Exception("Step response missing observation")

        obs = result["observation"]
        reward = result.get("reward", 0)
        done = result["done"]
        total_reward += reward
        step_count += 1

        new_branches = set(obs["covered_branches"]) - all_branches
        all_branches = set(obs["covered_branches"])

        if verbose:
            # Build step output
            action_str = format_action(action)

            if new_branches:
                branch_str = f" {GREEN}🟢 NEW: {', '.join(new_branches)}{RESET}"
            else:
                branch_str = ""

            if obs["last_status"] == 500:
                crashes += 1
                status_str = f" {RED}🚨 CRASH: {obs.get('last_error', 'unknown')}{RESET}"
            elif obs["last_status"] == 200:
                status_str = f" {DIM}✓ OK{RESET}"
            else:
                status_str = ""

            if reward > 5:
                reward_str = f" {YELLOW}💰 +{reward:.0f}{RESET}"
            elif reward < -1:
                reward_str = f" {DIM}{reward:.1f}{RESET}"
            else:
                reward_str = ""

            print(f"  {DIM}[{step_count:2d}]{RESET} {action_str}{status_str}{branch_str}{reward_str}")

    if verbose:
        print(f"\n  {BOLD}Results:{RESET}")
        print(f"  ├── Steps:    {step_count}")
        print(f"  ├── Reward:   {total_reward:.1f}")
        print(f"  ├── Branches: {len(all_branches)}/19 {_coverage_bar(len(all_branches), 19)}")
        print(f"  ├── Crashes:  {crashes}")
        print(f"  └── Branches: {sorted(all_branches)}")

    return {
        "branches": all_branches,
        "total_reward": total_reward,
        "steps": step_count,
        "crashes": crashes,
        "last_obs": obs,
    }


def _coverage_bar(covered, total, width=20):
    """Generate a visual progress bar."""
    filled = int(width * covered / total)
    bar = f"{'█' * filled}{'░' * (width - filled)}"
    pct = covered / total * 100
    if pct >= 80:
        return f"{GREEN}[{bar}] {pct:.0f}%{RESET}"
    elif pct >= 40:
        return f"{YELLOW}[{bar}] {pct:.0f}%{RESET}"
    else:
        return f"{RED}[{bar}] {pct:.0f}%{RESET}"


def main():
    print(f"\n{BOLD}{MAGENTA}🔥 EDGE-FORGE — Baseline Random Agent{RESET}")
    print(f"{DIM}   Autonomous Synthetic Staging Engine{RESET}")
    print(f"{DIM}   Testing random policy against 19-branch application...{RESET}")

    best_coverage = 0
    best_result = None
    all_discovered = set()

    start = time.time()

    for ep in range(5):
        result = run_episode(ep + 1)
        all_discovered.update(result["branches"])

        if len(result["branches"]) > best_coverage:
            best_coverage = len(result["branches"])
            best_result = result

    elapsed = time.time() - start

    # ── Final summary ───────────────────────────────────────────────
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}  FINAL RESULTS (5 episodes, {elapsed:.1f}s){RESET}")
    print(f"{'═' * 60}")

    print(f"\n  {BOLD}Total unique branches discovered: {len(all_discovered)}/19{RESET}")
    print(f"  {_coverage_bar(len(all_discovered), 19)}")
    print(f"\n  All discovered: {sorted(all_discovered)}")

    # Score using best episode's final observation
    best_obs = best_result["last_obs"]

    easy = 1.0 if "ssn_format_bug" in all_discovered else 0.0
    medium = min(len(all_discovered) / 19, 1.0)
    hard = 1.0 if "stateful_crash" in all_discovered else 0.0

    print(f"\n  {BOLD}Scores:{RESET}")
    print(f"  ├── Easy   (ssn_bug):   {_score_badge(easy)}")
    print(f"  ├── Medium (coverage):  {_score_badge(medium)} ({len(all_discovered)}/19)")
    print(f"  └── Hard   (stateful):  {_score_badge(hard)}")

    has_stateful = "stateful_crash" in all_discovered
    has_ssn_bug = "ssn_format_bug" in all_discovered

    print(f"\n  {BOLD}RL Insight:{RESET}")
    if not has_ssn_bug:
        print(f"  {YELLOW}⚠️  Random agent FAILED to discover SSN format bug{RESET}")
        print(f"  {DIM}   (Requires: open_account → verify_identity + invalid SSN){RESET}")
    else:
        print(f"  {GREEN}✓ Random agent found SSN format bug (lucky!){RESET}")

    if not has_stateful:
        print(f"  {YELLOW}⚠️  Random agent FAILED to discover stateful crash{RESET}")
        print(f"  {DIM}   (Requires: open_account → verify_identity without SSN){RESET}")
    else:
        print(f"  {GREEN}✓ Random agent found stateful crash (lucky!){RESET}")

    print(f"\n  {DIM}→ An RL agent would learn these sequences, a random agent cannot.{RESET}")
    print()


def _score_badge(score):
    """Format a score as a colored badge."""
    if score >= 1.0:
        return f"{GREEN}{BOLD}1.00 ✓{RESET}"
    elif score >= 0.5:
        return f"{YELLOW}{score:.2f}{RESET}"
    elif score > 0.0:
        return f"{RED}{score:.2f}{RESET}"
    else:
        return f"{RED}{BOLD}0.00 ✗{RESET}"


if __name__ == "__main__":
    main()