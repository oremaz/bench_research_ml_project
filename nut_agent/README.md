# NutriCoach & Recipe Lab

Two Streamlit applications exploring how LLM agents, local classifiers, and
vision models can support nutrition tracking and recipe analysis. NutriCoach
keeps a user profile and journal; Recipe Lab analyzes recipes without an account.
Both are experimental applications, and their nutrition outputs are estimates.

## NutriCoach

After registering and entering a profile, users can chat about their nutrition
goals, ask for meal plans, and record meals and water intake. The app calculates
calorie and macro targets from profile inputs, tracks intake against those
targets, and displays logged nutrition and weight trends. Daily tracking allows
meals and weight measurements to be recorded on separate dates.

The assistant uses an OpenRouter LLM in a LangGraph tool-calling loop. Tools
handle calculations, food lookups, profile updates, logging, and saved meal plans.
Structured profiles and journals are stored as per-user JSON files, while
LangGraph's SQLite checkpointer persists conversation state. Each LLM call uses
structured user context and a bounded recent conversation history.

Users can also upload a food photo to estimate foods, portions, calories, and
macros. NutriCoach uses only **Single-shot VLM** (`vlm_single`): one OpenRouter
image call supplies food names, estimated grams, and nutrients. Users can edit
the dish name and quantities, add or remove ingredients, and review recalculated
totals before explicitly saving the meal. New or renamed ingredients need a
database match or manually entered per-100g values. The default model is
`dots-studio/dots-3-note-preview:free`.

This choice follows the Nutrition5k pilot runs, which did not establish a clear
accuracy gain from the extra pipeline steps. The comparison scripts, alternative
methods, and recorded results remain available for research in
[Food Vision](nutricoach/food_vision/README.md). Its revised benchmark compares
single-shot VLM, RGB regression, specialized Food-R1 and a geometry/database
hybrid on a reproducible 50-dish official-test subset, scoring calories, macros
and total mass. The Dots baseline is complete; pending methods and the prepared
Jean Zay job are documented separately from measured results.

## Recipe Lab

Enter a recipe description to analyze it, or compare two recipes side by side.
Recipe Lab is available as a standalone app and as a tab inside NutriCoach.
It does not maintain a nutrition journal or conversation history.

Classification runs locally using frozen
`LiquidAI/LFM2.5-Embedding-350M` text embeddings and LightGBM classifiers:

- Difficulty: `Easy` or `More effort`, with `A challenge` merged into the latter.
- Meal type: breakfast or lunch/dinner.
- Total preparation and cooking time: `<15`, `15-30`, `30-60`, or `>60 min`.

On first analysis, the app loads the embedding model, embeds
`ml_pipeline/recipes_df.csv`, and trains the classifiers. This can take several
minutes and requires the dataset and an initial model download. Embeddings and
classifiers are cached under `nut_agent/secrets/recipe_lab_cache/`, keyed by
dataset content and the pinned model revision. Compatible existing BGE or Jina
checkpoints can serve as a fallback if the LFM2.5 path is unavailable.

With an OpenRouter key, the app also extracts recipe structure, estimates
per-serving calories and seven nutrients in zero shot, and explains the results.
The LLM may infer ingredients, quantities, or steps omitted from the description,
so these assumptions affect both classification and nutrition estimates.
Without a key, local classification remains available, with basic text parsing.

Recipe benchmark code lives in
[`ml_pipeline/bench_recipe_methods.py`](../ml_pipeline/bench_recipe_methods.py).
Historical BGE results do not validate the current LFM2.5 classifiers, and recipe
classification scores do not measure nutrition estimation accuracy.

## Running the apps

Use the repository's `uv` environment and root dependency files
(`requirements.txt`, or `requirements-mac.txt` on macOS). On macOS, LightGBM
also requires the OpenMP runtime: `brew install libomp`.

Run from the repository root:

```bash
# NutriCoach, including the Recipe Lab tab
PYTHONPATH=. uv run streamlit run nut_agent/nutricoach/app.py

# Standalone Recipe Lab
PYTHONPATH=. uv run streamlit run nut_agent/recipe_lab/app.py
```

Set `OPENROUTER_API_KEY` or enter it in the app for chat and API-based analysis.
`OPENROUTER_MODEL_ID` sets the initial chat/recipe model; it is also editable in
the UI. Food Analysis has a separate vision model field: choose a model that
accepts images. Availability and cost depend on the selected provider and model.
The sidebar's **Reasoning effort** defaults to `high` and applies to chat,
recipes, and API-based Food Vision. Available levels are `low`, `medium`,
`high`, `xhigh`, and `max`; choose one supported by the selected model.
API calls have an 8,192-token combined reasoning/output budget.
Recipe text, chat context, and photos used in API calls are sent to OpenRouter.
Local user data and model caches live under `nut_agent/secrets/`, which is
excluded from version control.

## Code and tests

`nutricoach/` contains the agent, tools, UI, and food vision methods;
`recipe_lab/` contains the recipe UI and prediction pipeline. `shared/` holds
configuration, authentication, Pydantic schemas, and structured memory.
Recipe classifiers reuse the ML utilities in `ml_pipeline/`.

Run the tests that do not require live APIs or GPU integration:

```bash
PYTHONPATH=. uv run python -m pytest nut_agent/tests/ -q \
    --ignore=nut_agent/tests/test_gpu_pipeline.py \
    --ignore=nut_agent/tests/test_openrouter_live.py
```

The separate GPU and live OpenRouter test modules exercise model inference and
API integration when their required resources are available.
