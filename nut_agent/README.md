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
single-shot VLM, RGB regression and specialized Food-R1
on a reproducible 500-dish official-test subset, scoring calories, macros
and total mass. It also runs the public FoodLMM author checkpoint and a separately
labeled third-party DPF reproduction. FoodLMM has separate 16-bit and 4-bit
decoder benchmark rows, with CUDA/MPS device selection in its isolated runner.
The original 16-bit row is retained alongside a cached 16-bit control for matched
precision comparisons.
MPS requires validation on Apple hardware. The four older RF-DETR, CLIP, RAG and
standalone database-grounded pipelines and the geometry hybrid have been removed. Current measurements,
H100 jobs and unavailable paper systems are documented in Food Vision; the
50-dish reports remain historical pilot results.

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

The sidebar's **Model provider** selects **OpenRouter**, **Groq**, or
**Local (Qwen)** for NutriCoach chat and food photos. OpenRouter is the default.
Set `OPENROUTER_API_KEY` or `GROQ_API_KEY`, or enter the selected provider's key
in the sidebar. Keys are separate. Local inference needs no API key and runs
`Qwen/Qwen3.8-27B` through Transformers in the Streamlit process.
Chat and vision model IDs remain separately editable. `OPENROUTER_MODEL_ID`,
`GROQ_MODEL_ID`, and `LOCAL_MODEL_ID` override provider defaults.
Choose image-capable models for photos. Provider switches reset both model IDs
and reasoning settings to compatible defaults and rebuild the chat agent.
Reasoning defaults to `high` for hosted providers and `none` locally;
the menu exposes each backend's supported levels. Local generation is greedy.
Responses have an 8,192-token budget. Hosted chat/photos go to the selected
provider; local inputs remain on the machine running Streamlit.
Local **BF16** needs roughly 58 GB just for weights plus working memory;
**4-bit (CUDA)** needs `bitsandbytes` from the root GPU dependencies and reduces
GPU memory use. BF16 is the default and the precision used for the H100 benchmark;
4-bit is an optional memory-saving setting whose accuracy needs separate evaluation.
This does not establish compatibility with a 32 GB CPU-only
machine. The H100 smoke test measured 18.52 GiB peak reserved GPU memory in
4-bit mode, but 52.12 GiB peak host RSS during loading.
`LOCAL_LLM_DEVICE` can select an explicit local device.
See [Food Vision setup and benchmark commands](nutricoach/food_vision/README.md).
Recipe Lab retains its existing OpenRouter configuration.
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
On 2026-10-12 the complete NutriCoach suite passed on H100 with **192 passed and
12 skipped**, including real local Qwen photo and chat tool inference; this
historical suite also included the subsequently removed geometry tests.
The separate 4-bit memory test also passed. See
[local Qwen validation](nutricoach/food_vision/qwen_local_validation.md) for resource
measurements and skipped checks, and
[Qwen benchmark results](nutricoach/food_vision/nutrition5k_500_qwen_results.md)
for the completed 500-dish one-shot baseline and five remaining methods.
After geometry removal, the current NutriCoach suite passed with **171 passed
and 7 skipped**, excluding live hosted and GPU integration modules. Streamlit
checks passed; real GPU inference was not rerun for this deletion.
