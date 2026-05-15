# NutriCoach & Recipe Lab

Two Streamlit apps sharing a common foundation: **NutriCoach** (stateful LLM nutritionist agent with food image analysis) and **Recipe Lab** (stateless ML recipe analyzer).

## Project Structure

```
nut_agent/
  shared/              Shared modules used by both apps
    config.py            Constants (BMR, macros, activity multipliers, prompts)
    utils.py             BMI calculation, nutrition target validation
    auth.py              Argon2 password hashing, user registration/login
    schemas.py           Pydantic models (UserProfile, DailyLog, NutritionTargets, etc.)
    memory.py            MemoryManager: per-user structured storage with bounded context
  nutricoach/          Stateful LLM-powered nutritionist agent
    agent.py             LangGraph v2 agent with SqliteSaver checkpointer
    tools.py             Agent tools (targets, intake, progress, profile, food image analysis)
    intent.py            LLM-based intent classification via structured output
    app.py               Streamlit UI (chat, food analysis, dashboard, daily tracking)
    food_vision/         Multi-method food image analysis module
      base.py              Abstract base class, FoodItem/FoodAnalysisResult data models
      nutrition_db.py      Local nutrition DB (130+ foods, USDA/CIQUAL values, fuzzy matching)
      rf_detr_analyzer.py  Method 1: RF-DETR object detection + fine-tuning pipeline
      vlm_analyzer.py      Method 2: Pure vLLM (Claude Opus via OpenRouter, 3-step chain)
      clip_analyzer.py     Method 3: CLIP zero-shot + LLM ensemble
      rag_vlm_analyzer.py  Method 4: RAG-enhanced VLM (DietAI24-inspired)
      compare.py           Comparison framework: benchmark all methods side-by-side
      README.md            Detailed docs for food vision module
  recipe_lab/          Stateless ML recipe analyzer
    predictor.py         Embedding-based prediction (difficulty, meal type, time class)
    app.py               Streamlit UI (analyze and compare recipes)
  tests/               86 unit tests
  secrets/             Per-user data (excluded from version control)
```

## Quickstart

```bash
cd nut_agent

# Install dependencies
pip install langgraph-checkpoint-sqlite openai rfdetr supervision

# NutriCoach (requires GOOGLE_API_KEY + optionally OPENROUTER_API_KEY for food vision)
streamlit run nutricoach/app.py

# Recipe Lab (requires trained ML models in ../food_preds/results/)
streamlit run recipe_lab/app.py

# Run tests
source ../.venv/bin/activate && python -m pytest tests/ -v

# Compare food analysis methods on an image
python -m nutricoach.food_vision.compare --image plate.jpg
```

## NutriCoach

### Agent Architecture (v2)

LangGraph agent powered by Google Gemini with 5 tools:

```
START -> agent -> (tool_node -> agent)* -> END
```

**v2 changes** (based on [feedback analysis](../todo/)):
- **Dropped intent classification node**: was an extra LLM call that never actually routed; now the agent handles intent naturally
- **SqliteSaver checkpointer**: replaces ~80 lines of manual JSON serialization for conversation persistence
- **Removed ghost state fields**: `user_profile`, `nutrition_targets`, `todays_log` were declared but never read; state now contains only `messages`
- **Added food image analysis**: new `analyze_food_image` tool with 4 analysis methods

### Tools

| Tool | Description |
|------|-------------|
| `calculate_personalized_nutrition_targets` | BMR/TDEE/macros from profile |
| `log_daily_intake` | Log meals, weight, energy level |
| `get_progress_summary` | Weekly trends and statistics |
| `update_user_profile` | Modify profile fields |
| `analyze_food_image` | Analyze food photo → ingredients + portions + calories |

### Food Image Analysis

Take a photo of your plate → get estimated ingredients, quantities, and calories.

4 methods available (see [food_vision/README.md](nutricoach/food_vision/README.md)):

1. **RF-DETR**: Object detection (needs fine-tuning, runs offline)
2. **Pure vLLM**: Claude Opus 3-step prompt chain via OpenRouter
3. **CLIP + LLM**: Zero-shot classification + LLM refinement
4. **RAG VLM**: Database-grounded VLM estimation (recommended)

### Persistence

Conversation history is now managed by LangGraph's `SqliteSaver`:
- Each conversation gets a `thread_id`
- State is automatically saved/restored via the checkpointer
- No manual JSON serialization needed
- DB stored at `secrets/{username}_checkpoints.db`

### Using Programmatically

```python
from nutricoach.agent import build_nutricoach_graph, create_initial_state
from langchain_core.messages import HumanMessage, AIMessage

graph = build_nutricoach_graph("your_google_api_key", "username")

# Conversations are thread-based
config = {"configurable": {"thread_id": "session-1"}, "recursion_limit": 20}
result = graph.invoke(
    {"messages": [HumanMessage(content="Calculate my nutrition targets")]},
    config=config,
)

for msg in reversed(result["messages"]):
    if isinstance(msg, AIMessage):
        print(msg.content)
        break
```

## Recipe Lab

Standalone recipe analyzer, with no login or conversation state:

- **Analyze** a recipe: ML predictions (difficulty, meal type, time class) + optional LLM interpretation
- **Compare** two recipes side-by-side
- Works with trained models from `../food_preds/results/`; degrades gracefully without them

## Configuration

Nutrition constants are in `shared/config.py`:
- `BMR_CONSTANTS`: Mifflin-St Jeor equation parameters
- `ACTIVITY_MULTIPLIERS`: sedentary through very active
- `MACRO_RATIOS`: protein/carb/fat distribution
- `WEIGHT_GOAL_ADJUSTMENTS`: caloric surplus/deficit for goals

Environment variables:
- `GOOGLE_API_KEY`: Required for Gemini LLM
- `OPENROUTER_API_KEY`: Required for food image analysis (Methods 2-4)
- `ROBOFLOW_API_KEY`: Optional, for downloading food detection datasets

## Testing

```bash
source ../.venv/bin/activate && python -m pytest tests/ -v
```

86 tests across 6 files covering utils, auth, memory, predictor, intent, and agent. All tests run without API keys or ML models (mocked).
