# NutriCoach & Recipe Lab

Two Streamlit apps sharing a common foundation: **NutriCoach** (stateful LLM nutritionist agent with food image analysis) and **Recipe Lab** (stateless ML recipe analyzer).

## Project Structure

```
nut_agent/
  SESSION_REPORT.md    Detailed report of the 2026-07 repair/training/feature session
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
                         plus per-serving nutrient estimation (fine-tuned bge head)
    app.py               Streamlit UI (analyze and compare recipes)
  tests/               Unit and integration test suite (see Testing)
  secrets/             Per-user data (excluded from version control)
```

## Quickstart

```bash
# From the repository root (uses the repo uv environment)

# NutriCoach (needs GOOGLE_API_KEY or OPENROUTER_API_KEY; OpenRouter free tier works)
PYTHONPATH=nut_agent uv run streamlit run nut_agent/nutricoach/app.py

# Recipe Lab (needs trained models in ../ml_pipeline/results/; embeddings run locally)
PYTHONPATH=nut_agent uv run streamlit run nut_agent/recipe_lab/app.py

# Train the Recipe Lab models (LightGBM/XGBoost/MLP on local embeddings, GPU)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run python ml_pipeline/train_recipe_models.py

# Run tests
PYTHONPATH=nut_agent uv run python -m pytest nut_agent/tests/ -v

# Compare food analysis methods on an image
cd nut_agent && python -m nutricoach.food_vision.compare --image plate.jpg
```

## NutriCoach

### Agent Architecture (v2)

LangGraph agent powered by Google Gemini, or by an OpenRouter model when only
OPENROUTER_API_KEY is available (defaults to a free tool-capable model with
server-side fallbacks; override with OPENROUTER_AGENT_MODEL). 5 tools:

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
| `log_daily_intake` | Log meals with the agent's calorie/macro estimates; daily totals recomputed |
| `get_progress_summary` | Weekly trends and statistics |
| `update_user_profile` | Modify profile fields |
| `analyze_food_image` | Analyze food photo → ingredients + portions + calories |
| `log_water_intake` | Add water to today's log, reports remaining vs target |
| `lookup_food_nutrition` | Per-portion calories/macros from the local USDA/CIQUAL DB (no API cost) |
| `get_remaining_daily_budget` | "What can I still eat today?" — targets minus logged intake |
| `save_meal_plan` / `get_meal_plan` | Persist the weekly meal plan; powers compliance checks and grocery lists |
| `generate_weekly_summary` | Aggregate the last 7 daily logs into a stored weekly review |

### Food Image Analysis

Take a photo of your plate → get estimated ingredients, quantities, and calories.

4 methods available (see [food_vision/README.md](nutricoach/food_vision/README.md)):

1. **RF-DETR**: Object detection (needs fine-tuning, runs offline)
2. **Pure vLLM**: Claude Opus 3-step prompt chain via OpenRouter
3. **CLIP + LLM**: Zero-shot classification + LLM refinement (CLIP ViT-B/32
   default; `backend="jina"` switches to jina-v5-omni-small — benchmarked on
   Food101 in the food_vision README, CLIP wins 0.796 vs 0.743 top-1)
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

# Pass a Google key, an OpenRouter key (sk-or-...), or None to use
# GOOGLE_API_KEY / OPENROUTER_API_KEY from the environment.
graph = build_nutricoach_graph(None, "username")

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
- Works with trained models from `../ml_pipeline/results/`; degrades gracefully without them
- Text embeddings are computed locally with sentence-transformers (`BAAI/bge-base-en-v1.5`,
  GPU when available); the same encoder is used at training and inference time, and
  `ml_pipeline/results/recipe_models_meta.json` records the embedding backend, label
  order, and test metrics
- Labels: difficulty `Easy` / `More effort` (2-class, `A challenge` merged at training),
  meal type `Breakfast` / `Lunch/Dinner` (binary), time class `<15` / `15-30` / `30-60` / `>60 min`
- Nutrient estimation: per-serving `kcal`, `fat`, `saturates`, `carbs`, `sugars`,
  `fibre`, `protein`, `salt` via multi-target regression on the same embeddings

### Method Benchmarks (2026-07)

Run `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run python ml_pipeline/bench_recipe_methods.py`
from the repo root; full per-task results land in
`ml_pipeline/results/bench_recipe_methods.json`. All numbers below are on the
held-out test splits (`recipes_df_test_bis.csv` for difficulty/time/nutrients,
the out-of-domain `recipes_df_test.csv` for meal type) with the
`BAAI/bge-base-en-v1.5` embedding backend. **Bold** = deployed model.

Classification, test accuracy / macro F1:

| Method | Difficulty | Meal binary | Time class |
|--------|-----------|-------------|------------|
| logreg probe | 0.601 / 0.578 | 0.831 / 0.785 | 0.603 / 0.565 |
| kNN (cosine) | 0.805 / 0.510 | 0.836 / 0.792 | 0.555 / 0.512 |
| LightGBM | 0.801 / 0.647 | **0.806 / 0.739** | **0.656 / 0.616** |
| XGBoost | 0.806 / 0.678 | 0.801 / 0.730 | 0.638 / 0.578 |
| CatBoost (new) | 0.798 / 0.709 | 0.806 / 0.739 | 0.634 / 0.596 |
| Stacking ensemble (new) | **0.813 / 0.717** | 0.806 / 0.739 | 0.635 / 0.593 |
| MLP | 0.798 / 0.444 | 0.801 / 0.730 | 0.588 / 0.485 |
| bge-base fine-tuned | 0.826 / 0.752 | 0.846 / 0.807 | 0.694 / 0.658 |

Fine-tuning bge-base end-to-end wins every task and is the accuracy ceiling,
but the deployed classifiers stay on frozen embeddings (shared encoder pass,
tiny checkpoints, one registry convention); the gap is 1-4 points. The
stacking ensemble is the best deployable model on difficulty; CatBoost is the
strongest single GBM on difficulty macro F1 (0.709).

Nutrient regression (per-serving), kcal MAE / mean R2 over the 8 targets:

| Method | kcal MAE | mean R2 |
|--------|----------|---------|
| predict-train-mean baseline | 168.3 | -0.63 |
| Ridge | 154.1 | -0.01 |
| kNN retrieval | 161.7 | -0.06 |
| LightGBM | 177.7 | -0.18 |
| CatBoost (new) | 165.4 | -0.03 |
| Stacking ensemble (new) | 174.2 | -0.07 |
| **bge-base fine-tuned regression head** | **133.2** | **+0.20** |

Frozen-embedding regressors barely beat the mean baseline: sentence
embeddings do not encode ingredient quantities well. The fine-tuned
regression head reads the raw text and is the only method with real signal
(protein MAE 3.7 g vs 12.5 g baseline, fat 9.1 g, saturates 4.7 g, sugars
11.9 g, salt 0.29 g). It is deployed as the primary nutrients model
(`ml_pipeline/results/nutrients_bge_regressor/`), with CatBoost on
embeddings as the fallback when the checkpoint or transformers is missing.

**Coherence validation (2026-07)**. Per-recipe check on the 182 held-out
test recipes: median kcal error 31%, 49% of recipes within +/-30% and 66%
within +/-50% of the true per-serving calories (Spearman 0.43); protein and
fat have larger relative errors on low-absolute-value recipes. Directional
sanity checks all rank correctly (burger > salad kcal, cake > chicken
sugars, chicken > cake protein, ...), but magnitudes compress toward the
dataset mean, underestimating rich dishes. On three published recipes not
in any split (BBC Good Food ME): chicken tikka masala predicted 339 kcal /
29.6 g protein vs actual 345 / 31 (excellent); chocolate chip banana bread
466 kcal vs 306 (overestimate); miso salmon traybake 394 kcal / 32.7 g
protein vs 610 / 42 (underestimate, right profile). Verdict: usable as a
per-serving ballpark and for comparing/ranking recipes; not tracking-grade,
and the app labels it accordingly. NutriCoach meal logging continues to use
the local nutrition DB + LLM portions, not this model.

### Embedding Backend: bge-base vs jina-v5-omni-small

`jinaai/jina-embeddings-v5-omni-small` (1.74B, 1024-d, CC BY-NC 4.0) was
benchmarked against `BAAI/bge-base-en-v1.5` (109M, 768-d) as the recipe
encoder. bge-base wins on every task despite being 16x smaller, so it stays
the default (`RECIPE_EMBEDDING_MODEL` overrides; the jina path is supported
by `LocalEmbedder`). Best method per task and backend:

| Task | bge-base | jina-v5-omni-small |
|------|----------|--------------------|
| Difficulty (acc / f1) | 0.813 / 0.717 | 0.809 / 0.675 |
| Meal binary (acc / f1) | 0.836 / 0.792 | 0.821 / 0.767 |
| Time class (acc / f1) | 0.656 / 0.616 | 0.619 / 0.581 |
| Nutrients (kcal MAE) | 154.1 | 159.4 |

### Meal Type: Binary vs 3-Class

The former 3-class head (Breakfast / Dinner / Lunch) topped out at 0.637
test accuracy because Lunch and Dinner recipes are nearly indistinguishable
from text; Breakfast-vs-rest separability is unchanged whether trained
binary or 3-class (a 3-class model collapsed to binary scores up to 0.846).
Training directly on the binary task lifts the deployed label's accuracy
from ~0.64 to 0.81-0.85, so Recipe Lab now ships the binary
`Breakfast` / `Lunch/Dinner` classifier.

## Configuration

Nutrition constants are in `shared/config.py`:
- `BMR_CONSTANTS`: Mifflin-St Jeor equation parameters
- `ACTIVITY_MULTIPLIERS`: sedentary through very active
- `MACRO_RATIOS`: protein/carb/fat distribution
- `WEIGHT_GOAL_ADJUSTMENTS`: caloric surplus/deficit for goals

Environment variables:
- `GOOGLE_API_KEY`: Optional, preferred LLM (Gemini) when set
- `OPENROUTER_API_KEY`: LLM fallback for the agent and required for food image analysis (Methods 2-4)
- `OPENROUTER_AGENT_MODEL` / `OPENROUTER_VISION_MODEL`: Optional model overrides (defaults are free-tier models)
- `RECIPE_EMBEDDING_MODEL`: Optional Recipe Lab embedding backend override for training
  (default `BAAI/bge-base-en-v1.5`; `jinaai/jina-embeddings-v5-omni-small` is supported)
- `ROBOFLOW_API_KEY`: Optional, for downloading food detection datasets

## Testing

```bash
# CPU suite (no keys or models needed)
PYTHONPATH=nut_agent uv run python -m pytest nut_agent/tests/ -v \
    --ignore=nut_agent/tests/test_gpu_pipeline.py --ignore=nut_agent/tests/test_openrouter_live.py

# GPU integration suite (trained models, embedder, CLIP; cuda:1)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=nut_agent uv run python -m pytest nut_agent/tests/test_gpu_pipeline.py -v

# Live OpenRouter tests (skipped without OPENROUTER_API_KEY; tolerate free-tier 429s)
PYTHONPATH=nut_agent uv run python -m pytest nut_agent/tests/test_openrouter_live.py -v
```

106 CPU tests across 7 files (utils, auth, memory, predictor, intent, agent, tools),
plus a GPU integration suite (embedder, trained recipe models, nutrients head,
CLIP and Jina zero-shot backends, RF-DETR) and live agent tests. CPU tests run
without API keys or ML models.
