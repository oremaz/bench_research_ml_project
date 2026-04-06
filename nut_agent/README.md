# NutriCoach & Recipe Lab

Two Streamlit apps sharing a common foundation: **NutriCoach** (stateful LLM nutritionist agent) and **Recipe Lab** (stateless ML recipe analyzer).

## Project Structure

```
nut_agent/
  shared/           Shared modules used by both apps
    config.py         Constants (BMR, macros, activity multipliers, prompts)
    utils.py          BMI calculation, nutrition target validation
    auth.py           Argon2 password hashing, user registration/login
    schemas.py        Pydantic models (UserProfile, DailyLog, NutritionTargets, etc.)
    memory.py         MemoryManager — per-user structured storage with bounded context
  nutricoach/       Stateful LLM-powered nutritionist agent
    agent.py          LangGraph ReAct agent with tool calling
    tools.py          Agent tools (calculate targets, log intake, progress, profile update)
    intent.py         LLM-based intent classification via structured output
    app.py            Streamlit UI (chat, dashboard, daily tracking)
  recipe_lab/       Stateless ML recipe analyzer
    predictor.py      Embedding-based prediction (difficulty, meal type, time class)
    app.py            Streamlit UI (analyze and compare recipes)
  tests/            86 unit tests
  secrets/          Per-user data (excluded from version control)
```

## Quickstart

```bash
cd nut_agent

# NutriCoach (requires GOOGLE_API_KEY in .env or secrets/)
streamlit run nutricoach/app.py

# Recipe Lab (requires trained ML models in ../food_preds/results/)
streamlit run recipe_lab/app.py

# Run tests
source ../.venv/bin/activate && python -m pytest tests/ -v
```

## NutriCoach

LangGraph agent powered by Google Gemini with proper tool calling:

- **Intent classification** — LLM-based (`llm.with_structured_output`) replaces keyword routing
- **Tool calling** — `bind_tools`/`ToolNode` ReAct loop: calculate targets, log intake, get progress, update profile
- **Structured memory** — per-user JSON files (profile, nutrition targets, daily logs, weekly summaries) with bounded context assembly (~1000 tokens)
- **Auth** — argon2 hashing with transparent SHA-256 migration for legacy users
- **UI** — `st.chat_message()` for safe rendering, dashboard reading from real daily logs, quick actions, daily tracking

### Agent Graph

```
START -> classify_intent -> agent -> (tool_node -> agent)* -> END
```

The agent node injects memory context as a system message and delegates calculations to tools rather than performing math itself.

### Using Programmatically

```python
from nutricoach.agent import build_nutricoach_graph, create_initial_state
from langchain_core.messages import HumanMessage, AIMessage

graph = build_nutricoach_graph("your_google_api_key", "username")
state = create_initial_state(profile_complete=True)
state["messages"] = [HumanMessage(content="Calculate my nutrition targets")]

result = graph.invoke(state, {"recursion_limit": 20})
for msg in reversed(result["messages"]):
    if isinstance(msg, AIMessage):
        print(msg.content)
        break
```

## Recipe Lab

Standalone recipe analyzer — no login, no conversation state:

- **Analyze** a recipe: ML predictions (difficulty, meal type, time class) + optional LLM interpretation
- **Compare** two recipes side-by-side
- Works with trained models from `../food_preds/results/`; degrades gracefully without them

## Configuration

Nutrition constants are in `shared/config.py`:
- `BMR_CONSTANTS` — Mifflin-St Jeor equation parameters
- `ACTIVITY_MULTIPLIERS` — sedentary through very active
- `MACRO_RATIOS` — protein/carb/fat distribution
- `WEIGHT_GOAL_ADJUSTMENTS` — caloric surplus/deficit for goals

ML models are loaded from `../food_preds/results/` (difficulty, meal type, time class).

## Testing

```bash
source ../.venv/bin/activate && python -m pytest tests/ -v
```

86 tests across 6 files covering utils, auth, memory, predictor, intent, and agent. All tests run without API keys or ML models (mocked).
