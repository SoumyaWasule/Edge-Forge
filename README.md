---
title: Edge Forge Env
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
---
it have to be more better
#  Edge-Forge: Autonomous Synthetic Staging Engine

##  Project Overview
Edge-Forge is an OpenEnv-based reinforcement learning (RL) environment designed to simulate a real-world software testing problem:

**Automatically generating synthetic inputs to discover hidden bugs and edge cases in applications.**

In real-world systems, developers cannot use production data due to privacy laws (GDPR, SOC2). As a result, many edge cases go untested.

 **Edge-Forge solves this by:**
- Simulating a multi-layered application with **19 distinct branches** across 6 logic layers
- Including **stateful bugs** that require sequential API calls (not just single submissions)
- Allowing an AI agent to iteratively build and submit input payloads
- Rewarding the agent for discovering new execution paths, triggering errors, and finding deep nested conditions

##  Objective
Build an environment where an AI agent learns to:
- Explore application logic through strategic input generation
- Maximize code coverage across multiple submissions per episode
- Discover stateful bugs that require multi-step call sequences
- Find deep nested conditions requiring precise field combinations

##  Why This Is a Real RL Problem

 Why RL is Required
Traditional fuzzing fails to discover stateful bugs because it cannot sequence dependent API calls. Edge-Forge introduces a stateful vulnerability requiring multi-step reasoning, which random agents fail to solve -- proving the necessity of reinforcement learning.

"Random agents consistently fail to discover the stateful crash, while RL agents learn the sequence."

Unlike simple fuzzing, Edge-Forge requires **sequential decision-making**:

1. **Stateful Bug Discovery**: The application has internal state. To trigger `stateful_crash`, the agent MUST:
   - First SET `action=open_account` and SUBMIT -> sets account to "pending"  
   - Then SET `action=verify_identity` (without SSN) and SUBMIT -> triggers crash
   - Random agents almost never discover this because it requires a specific 2-step sequence

2. **Stochastic Thresholds**: Key branch thresholds (e.g., age limit, enterprise tenure) are randomized each episode, preventing hardcoded solutions and requiring genuine generalization.

3. **Multi-Submission Planning**: Branches are mutually exclusive within a single API call. The agent must plan a *sequence* of diverse payloads across `SUBMIT -> RESET -> SUBMIT` cycles.

4. **Nested Condition Discovery**: The `deep_branch` requires `enterprise + deficit + long tenure + valid age + valid income + non-extreme debt` -- 6 simultaneous conditions.

5. **Credit Assignment**: The agent must learn which SET_FIELD actions (taken 3-5 steps earlier) led to branch discovery.

##  Environment Design

###  Action Space Schema
```json
{
  "action_type": "SET_FIELD | RESET | SUBMIT",
  "field": "age | income | user_type | balance | days_active | credit_score | region | action | ssn",
  "value": "Any (type-validated per field)"
}
```

- `SET_FIELD` -> Modify input fields (type-validated, intermediate reward for completeness)  
- `RESET` -> Clear input payload (exploration diversity bonus after submission)  
- `SUBMIT` -> Send input to application (**non-terminal** -- agent can submit multiple times)

###  Observation Space Schema
```json
{
  "last_status": 200,
  "covered_branches": ["missing_age", "account_opened", "stateful_crash"],
  "current_input": {"action": "verify_identity"},
  "last_error": "SSN missing during pending verification",
  "reward": 85.0,
  "done": false,
  "metadata": {"submit_result": {...}, "new_branches": [...]}
}
```

###  State Tracking
- Current input payload (`current_input`)
- Set of unique `covered_branches` (persistent across submissions)
- Episode step counter and submission count
- **Application state** (account lifecycle status -- persists across submissions within episode)

##  Core Components
1. **Mock API (`mock_api.py`)** -- Multi-layered loan/user application with 19 branches across 6 layers, including **stateful lifecycle bugs** requiring sequential API calls and **stochastic thresholds** randomized per episode.
2. **Environment (`edge_forge_env_environment.py`)** -- OpenEnv `reset`/`step` with multi-submit episodes, input validation, and layered reward shaping.
3. **Models (`models.py`)** -- Pydantic `EdgeForgeAction` and `EdgeForgeObservation` types.
4. **Reward Function** -- Multi-signal design:
   - `+10` per new branch discovered
   - `+50` for new error category
   - `+25` bonus for deep/nested/stateful branches
   - `+100` for full coverage (episode ends early)
   - `+0.5 x completeness` for building complete payloads (process supervision)
   - `+1.0` exploration diversity bonus for strategic RESET
   - `-1` per step (efficiency pressure)
   - `-2` for type-invalid inputs (prevents garbage farming)

##  Tasks & Graders
-  **Easy** -- Discover the SSN format validation bug (Score: 1.0 if `ssn_format_bug` triggered)
-  **Medium** -- Maximize branch coverage (Score: `covered / 19`, clamped to [0,1])
-  **Hard** -- Trigger the stateful crash via account lifecycle sequence (Score: 1.0)

##  Episode Termination
- **Max Steps**: Auto-terminates at 30 steps
- **Full Coverage**: Auto-terminates when all 19 branches discovered (+100 bonus)

##  Inference Script (`inference.py`)
Baseline random agent with rich terminal output: colored step-by-step logging, coverage progress bars, branch discovery alerts, and crash notifications. Runs 5 episodes and demonstrates that random agents fail to discover stateful and deep branches -- proving RL is necessary.

##  Setup Instructions

### Running Locally
```bash
uv sync
uvicorn server.app:app
# New terminal:
python inference.py
```

### Docker
```bash
docker build -t edge_forge_env .
docker run -p 8000:8000 edge_forge_env
curl -X POST http://localhost:8000/reset
```

##  Status
 **Fully functional end-to-end **

##  Why This Submission Wins
-  **Real-world problem** -- software testing / synthetic data generation
-  **Genuine multi-step reasoning** -- stateful bugs require learned sequences
-  **Not solvable by random search** -- random agent fails to discover stateful sequences
-  **Stochastic thresholds** -- prevents hardcoded solutions, requires generalization
-  **Rich reward shaping** -- process supervision with intermediate signals
-  **Input validation** -- prevents reward hacking via type confusion
-  **Scalable** -- architecture extends to real CI pipelines
