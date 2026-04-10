"""
Baseline inference script for Edge Forge.

Implements the mandatory OpenEnv inference pattern:
  - Uses OpenAI client for LLM-driven action decisions
  - Runs each task as a separate [START]/[STEP]s/[END] episode
  - Uses the OpenEnv SDK client (EdgeForgeEnv)
  - Emits structured stdout logs per the spec

Environment Variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    IMAGE_NAME     Docker image name for the environment (optional for local dev).

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import re
import sys
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

# Import from the edge_forge_env package (installed) or directly (root execution)
try:
    from edge_forge_env import EdgeForgeEnv, EdgeForgeAction, EdgeForgeObservation
    from edge_forge_env import grade_easy, grade_medium, grade_hard
except ImportError:
    from client import EdgeForgeEnv
    from models import EdgeForgeAction, EdgeForgeObservation
    from tasks import grade_easy, grade_medium, grade_hard


# ================================================================
# Environment Variables (mandatory spec)
# ================================================================
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = "edge_forge_env"

# LLM parameters
TEMPERATURE = 0.7
MAX_TOKENS = 256
SUCCESS_THRESHOLD = 0.5

# Field type mapping for LLM response coercion
NUMERIC_FIELDS = {"age", "income", "balance", "days_active", "credit_score"}
STRING_FIELDS = {"user_type", "region", "action", "ssn"}


def validate_env() -> None:
    """Validate mandatory environment variables before running inference."""
    missing = []
    if not os.getenv("API_BASE_URL"):
        missing.append("API_BASE_URL")
    if not os.getenv("MODEL_NAME"):
        missing.append("MODEL_NAME")
    if not os.getenv("HF_TOKEN"):
        missing.append("HF_TOKEN")
    if missing:
        print(
            f"[FATAL] Missing mandatory environment variables: {', '.join(missing)}",
            file=sys.stderr,
            flush=True,
        )
        print(
            "[FATAL] Required: API_BASE_URL, MODEL_NAME, HF_TOKEN",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)


# ================================================================
# Task Definitions
# ================================================================
EASY_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an agent interacting with a loan/user processing API test environment.
    Your goal is to discover edge cases by submitting different input combinations.

    The API processes user data and may behave unexpectedly under certain conditions.
    Explore the "action" field (try "open_account" and "verify_identity") and the
    "ssn" field with various value types. Some combinations produce interesting results.

    AVAILABLE ACTIONS (respond with exactly one JSON object per turn):
    - {"action_type": "SET_FIELD", "field": "<name>", "value": <val>}
    - {"action_type": "SUBMIT"}
    - {"action_type": "RESET"}

    VALID FIELDS & TYPES:
    - age, income, balance, days_active, credit_score: integer
    - user_type: string ("normal" or "enterprise")
    - region: string ("us" or "restricted")
    - action: string ("open_account" or "verify_identity")
    - ssn: string or integer

    Try different field values and sequences. Observe the errors and status codes
    returned to guide your next action.

    RESPOND WITH ONLY A JSON OBJECT. No explanation, no markdown, just JSON.""")

MEDIUM_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an agent exploring a loan/user processing API test environment.
    Your goal is to maximize branch coverage — discover as many distinct code
    paths as possible by submitting varied input combinations.

    The API has many branches triggered by different field values and combinations.
    Explore systematically: vary one field at a time, observe what changes in
    covered_branches, then try something new. Use RESET between attempts to
    start fresh.

    AVAILABLE ACTIONS (respond with exactly one JSON object per turn):
    - {"action_type": "SET_FIELD", "field": "<name>", "value": <val>}
    - {"action_type": "SUBMIT"}
    - {"action_type": "RESET"}

    VALID FIELDS & TYPES:
    - age: integer
    - income: integer (can be negative)
    - balance: integer (can be negative)
    - days_active: integer
    - credit_score: integer
    - user_type: string ("normal" or "enterprise")
    - region: string ("us" or "restricted")
    - action: string ("open_account" or "verify_identity")
    - ssn: string or integer

    After each SUBMIT, check which new branches appeared in covered_branches.
    Prioritize trying combinations you haven't explored yet.

    RESPOND WITH ONLY A JSON OBJECT. No explanation, no markdown, just JSON.""")

HARD_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an agent interacting with a stateful loan/user processing API.
    Your goal is to discover a crash condition in the API.

    The API maintains state between submissions within a session. Certain sequences
    of actions can leave the API in an inconsistent state that causes crashes on
    subsequent calls. Think about what happens when you initiate a process but
    don't complete it properly before making a follow-up request.

    AVAILABLE ACTIONS (respond with exactly one JSON object per turn):
    - {"action_type": "SET_FIELD", "field": "<name>", "value": <val>}
    - {"action_type": "SUBMIT"}
    - {"action_type": "RESET"}

    VALID FIELDS & TYPES:
    - age, income, balance, days_active, credit_score: integer
    - user_type: string ("normal" or "enterprise")
    - region: string ("us" or "restricted")
    - action: string ("open_account" or "verify_identity")
    - ssn: string or integer

    Focus on the stateful "action" field. Explore multi-step workflows and observe
    how the API state changes. What happens if a step in the workflow is skipped
    or a required field is omitted?

    RESPOND WITH ONLY A JSON OBJECT. No explanation, no markdown, just JSON.""")

TASKS = [
    {
        "id": "easy_task",
        "grader": grade_easy,
        "max_steps": 10,
        "system_prompt": EASY_SYSTEM_PROMPT,
        "fallback_actions": [
            {"action_type": "SET_FIELD", "field": "action", "value": "open_account"},
            {"action_type": "SUBMIT"},
            {"action_type": "SET_FIELD", "field": "action", "value": "verify_identity"},
            {"action_type": "SET_FIELD", "field": "ssn", "value": "abc"},
            {"action_type": "SUBMIT"},
        ],
    },
    {
        "id": "medium_task",
        "grader": grade_medium,
        "max_steps": 30,
        "system_prompt": MEDIUM_SYSTEM_PROMPT,
        "fallback_actions": [
            {"action_type": "SET_FIELD", "field": "age", "value": 25},
            {"action_type": "SET_FIELD", "field": "income", "value": 50000},
            {"action_type": "SUBMIT"},
        ],
    },
    {
        "id": "hard_task",
        "grader": grade_hard,
        "max_steps": 15,
        "system_prompt": HARD_SYSTEM_PROMPT,
        "fallback_actions": [
            {"action_type": "SET_FIELD", "field": "action", "value": "open_account"},
            {"action_type": "SUBMIT"},
            {"action_type": "SET_FIELD", "field": "action", "value": "verify_identity"},
            {"action_type": "SUBMIT"},
        ],
    },
]



# ================================================================
# Structured Logging (exact format per spec)
# ================================================================
def log_start(task: str, env: str, model: str) -> None:
    """Emit [START] line at episode begin."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    """Emit [STEP] line after each env.step() returns."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit [END] line after env.close(), always emitted even on exception."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ================================================================
# Action formatting for [STEP] log
# ================================================================
def format_action(action_dict: Dict[str, Any]) -> str:
    """Format an action dict for the [STEP] action= field."""
    if action_dict.get("action_type") == "SET_FIELD":
        return f"SET_FIELD({action_dict.get('field')}={action_dict.get('value')})"
    return action_dict.get("action_type", "UNKNOWN")


# ================================================================
# Type coercion for LLM responses
# ================================================================
def coerce_field_value(field: str, value: Any) -> Any:
    """Coerce a field value to the correct Python type.

    The environment's _validate_field checks isinstance(), so if the LLM
    returns "25" (string) for age, it gets rejected with a -2.0 penalty.
    This function ensures correct types.
    """
    if field in NUMERIC_FIELDS:
        if value is None:
            return 0
        try:
            if isinstance(value, (int, float)):
                return int(value)
            text = str(value)
            return int(text) if "." not in text else int(float(text))
        except (ValueError, TypeError):
            return 0
    if field in STRING_FIELDS:
        if value is None:
            return None
        return str(value)
    return value


# ================================================================
# LLM Response Parsing
# ================================================================
def parse_llm_response(text: str) -> Optional[Dict[str, Any]]:
    """Parse LLM response text into an action dict.

    Handles: raw JSON, markdown-fenced JSON, JSON embedded in prose.
    """
    text = text.strip()

    # Remove markdown code fences if present
    if "```" in text:
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = text.rstrip("`").strip()

    # Try direct JSON parse
    try:
        action = json.loads(text)
        if isinstance(action, dict) and "action_type" in action:
            return action
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object in the text
    match = re.search(r"\{[^{}]+\}", text)
    if match:
        try:
            action = json.loads(match.group())
            if isinstance(action, dict) and "action_type" in action:
                return action
        except json.JSONDecodeError:
            pass

    return None


# ================================================================
# LLM Action Decision
# ================================================================
def build_user_prompt(
    step: int,
    covered_branches: List[str],
    current_input: Dict[str, Any],
    last_error: Optional[str],
    last_reward: float,
    history: List[str],
) -> str:
    """Build the user prompt with current observation state."""
    history_block = "\n".join(history[-5:]) if history else "None"
    return textwrap.dedent(f"""\
        Step: {step}
        Covered branches so far: {covered_branches}
        Current input payload: {json.dumps(current_input)}
        Last error: {last_error or "None"}
        Last reward: {last_reward:.2f}
        Recent history:
        {history_block}

        Decide your next action. Respond with a single JSON object.""")


def get_llm_action(
    client: OpenAI,
    task_config: Dict,
    step: int,
    covered_branches: List[str],
    current_input: Dict[str, Any],
    last_error: Optional[str],
    last_reward: float,
    history: List[str],
    fallback_idx: int,
) -> Dict[str, Any]:
    """Get the next action from the LLM, with deterministic fallback.

    If the LLM call fails or returns unparseable output, falls back to
    the pre-defined action sequence for the current task.
    """
    user_prompt = build_user_prompt(
        step, covered_branches, current_input, last_error, last_reward, history
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": task_config["system_prompt"]},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        action = parse_llm_response(text)

        if action:
            # Coerce field values to correct Python types
            if action.get("action_type") == "SET_FIELD" and action.get("field"):
                action["value"] = coerce_field_value(
                    action["field"], action.get("value")
                )
            return action
        else:
            print(f"[DEBUG] Could not parse LLM response: {text!r}", file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", file=sys.stderr, flush=True)

    # Fallback: use pre-defined action sequence for this task
    fallbacks = task_config["fallback_actions"]
    idx = fallback_idx % len(fallbacks)
    return fallbacks[idx].copy()


# ================================================================
# Run a Single Task Episode
# ================================================================
async def run_task(env, client: OpenAI, task_config: Dict) -> None:
    """Run one task episode with [START]/[STEP]s/[END] logging.

    Each task gets its own reset, step loop, grading, and log block.
    The grader scores the final observation to produce a score in [0, 1].
    """
    task_id = task_config["id"]
    max_steps = task_config["max_steps"]
    grader = task_config["grader"]

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    covered_branches: List[str] = []
    submit_outcomes: List[str] = []

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment — no task parameter (env doesn't support it)
        result = await env.reset()

        covered_branches = result.observation.covered_branches
        current_input = result.observation.current_input
        last_error = result.observation.last_error
        submit_outcomes = result.observation.submit_outcomes
        last_reward = 0.0
        history: List[str] = []

        for step in range(1, max_steps + 1):
            if result.done:
                break

            # Get action from LLM (with fallback)
            action_dict = get_llm_action(
                client=client,
                task_config=task_config,
                step=step,
                covered_branches=covered_branches,
                current_input=current_input,
                last_error=last_error,
                last_reward=last_reward,
                history=history,
                fallback_idx=step - 1,
            )

            # Build typed EdgeForgeAction
            action = EdgeForgeAction(
                action_type=action_dict.get("action_type", "SUBMIT"),
                field=action_dict.get("field"),
                value=action_dict.get("value"),
            )

            # Execute step
            result = await env.step(action)

            # Extract data from StepResult (not from observation)
            reward = result.reward or 0.0
            done = result.done
            error = result.observation.last_error

            rewards.append(reward)
            steps_taken = step
            last_reward = reward
            covered_branches = result.observation.covered_branches
            current_input = result.observation.current_input
            submit_outcomes = result.observation.submit_outcomes
            last_error = error

            # Emit [STEP] log
            log_step(
                step=step,
                action=format_action(action_dict),
                reward=reward,
                done=done,
                error=error,
            )

            history.append(
                f"Step {step}: {format_action(action_dict)} -> "
                f"reward={reward:+.2f} branches={len(covered_branches)}"
            )

            if done:
                break

        # Grade using actual API outcomes (not self-reported branch labels)
        final_obs = {
            "submit_outcomes": submit_outcomes,
            "covered_branches": covered_branches,
        }
        score = grader(final_obs)
        score = max(0.0, min(score, 1.0))
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} error: {exc}", file=sys.stderr, flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ================================================================
# Main Entry Point
# ================================================================
async def main() -> None:
    """Run inference across all 3 tasks."""
    validate_env()
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Connect to environment: Docker image (validator) or local server (dev)
    if IMAGE_NAME:
        env = await EdgeForgeEnv.from_docker_image(IMAGE_NAME)
    else:
        env_url = os.getenv("ENV_BASE_URL", "http://localhost:8000")
        env = EdgeForgeEnv(base_url=env_url)

    try:
        for task_config in TASKS:
            await run_task(env, client, task_config)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    asyncio.run(main())