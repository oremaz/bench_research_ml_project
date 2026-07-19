# NutriCoach / Recipe Lab — Session Report (2026-07-16)

Goal of the session: (1) make everything in `nut_agent/` actually work in practice,
train the missing models on GPU (cuda:1 only), and cover it with a test suite;
(2) once the customer has a functional nutritionist agent, add the features that
were missing from the day-to-day routine.

## Starting state (audit findings)

The codebase looked complete but was not functional for a real customer:

1. **Recipe Lab was dead code.** `recipe_lab/predictor.py` imported from a
   `food_preds` package that no longer exists (renamed to `ml_pipeline/`), called
   `load_model()` with an outdated signature, and pointed at `food_preds/results/`
   which is absent. `google.api_core` (a hard import) was not even installed.
2. **No trained models existed.** `results/` was empty; the three Recipe Lab
   classifiers (difficulty, meal type, time class) had never been trained in this
   checkout, and no RF-DETR food-detection checkpoint existed.
3. **Label/checkpoint mismatches vs the actual training recipe** (bench-food.ipynb):
   the predictor declared 3 difficulty classes (actual: 2 after "A challenge" is
   merged), 5 meal classes (actual: 3 — Breakfast/Dinner/Lunch), and looked for
   checkpoints under `total_time_class_train` (actual: `total_time_train`).
4. **Train/inference embedding inconsistency.** Training embeddings were Gemini
   `text-embedding-004` (768-d); the predictor requested `gemini-embedding-2-preview`
   (3072-d) and fell back to a 3072-d zero vector. Any prediction would have been
   garbage even with models present.
5. **No usable LLM credentials path.** Everything required `GOOGLE_API_KEY`, which
   does not exist on this machine; only `OPENROUTER_API_KEY` (free tier) is
   available. The agent, recipe enhancement, and interpretation were all dead.
6. **Food vision defaulted to `anthropic/claude-opus-4-6`** — unusable on a free
   tier.
7. **RF-DETR analyzer counted non-food as food.** With COCO weights, supervision
   exposes `class_name` for every detection, which bypassed the food filter:
   "dining table" and "cell phone" received calorie estimates (reproduced on a
   real photo).
8. **`VLMAnalyzerSingleShot`'s prompt was never `.format()`ed**, so the model
   received literal `{{ }}` braces.
9. **Dashboard plotted `DailyLog.total_calories`, which nothing ever computed.**
10. **`WeeklySummary` schema and storage existed but nothing ever wrote one.**

## Step 1 — Fixes (everything working in practice)

### Recipe Lab (`recipe_lab/predictor.py`, rewritten)
- Imports fixed to `ml_pipeline` modules; checkpoints loaded with
  `load_model_by_name(...)` against the repo's `index.jsonl` convention.
- Embeddings now computed **locally** with sentence-transformers
  `BAAI/bge-base-en-v1.5` (768-d, normalized, GPU when available). The **same
  encoder object is imported by the training script**, so train and inference
  cannot drift; `ml_pipeline/results/recipe_models_meta.json` (written at training
  time) records the backend, label order, and test metrics, and the predictor
  reads it as the source of truth.
- Labels corrected: difficulty `[Easy, More effort]`, meal
  `[Breakfast, Dinner, Lunch]`, time `[<15, 15-30, 30-60, >60 min]`.
- LLM enhancement/interpretation: Gemini when `GOOGLE_API_KEY` is set, otherwise
  OpenRouter free tier, otherwise graceful fallback (predictions still work with
  no key at all).

### NutriCoach agent (`nutricoach/agent.py`)
- Provider auto-selection: Google key → Gemini; otherwise OpenRouter via
  `langchain-openai` (`ChatOpenAI` with OpenRouter base URL). Keys starting with
  `sk-or-` are recognized when typed into the app.
- Default OpenRouter model: `google/gemma-4-31b-it:free` (tool-capable) with
  **server-side fallbacks** (`extra_body={"models": [...]}`) to other free
  tool-capable models, because free upstreams are frequently rate-limited.
- Agent node retries transient failures the SDK does not retry (HTTP-200 error
  bodies with 500/429, `free-models-per-min` caps, empty completions), with
  window-sized backoff.
- System prompt hardened for small free models: never claim a tool ran without
  calling it; read USER CONTEXT before asking for data the app already has.

### Food vision (`nutricoach/food_vision/`)
- Default model for methods 2-4 is now `OPENROUTER_VISION_MODEL`
  (free-tier default, overridable per env) with server-side fallbacks on every
  API call; Claude Opus remains a config choice for paid tiers.
- CLIP analyzer now runs on GPU (`device` parameter, auto-cuda).
- RF-DETR: food whitelist enforced for COCO weights (fixes "cell phone: 525 kcal");
  fine-tuned checkpoints keep their own class names.
- Single-shot VLM prompt braces fixed.

### Apps
- NutriCoach app accepts a Google **or** OpenRouter key (prefilled from env);
  Recipe Lab works without any key (local embeddings).
- `shared/config.py`: `food_preds` path replaced by `ml_pipeline`; OpenRouter
  constants added. `langchain-openai` added to `requirements*.in`.

## Step 1 — Models trained this session (all on cuda:1)

### Recipe Lab models — `ml_pipeline/train_recipe_models.py`
Mirrors the three classification tasks of `bench-food.ipynb` (same
BenchmarkRunner conventions, seed 42, `use_class_weights`) but on local GPU
embeddings of `recipe_text` (1465 train recipes). Per task, LightGBM, XGBoost,
and an MLP (GPU) were trained and checkpointed under `ml_pipeline/results/`
(`difficulty_train`, `meal_train`, `total_time_train`, 9 checkpoints total);
the predictor serves the LightGBM ones.

Held-out test metrics (test CSVs from the original benchmark; the meal test set
is strongly out-of-distribution — categories like "High protein", "Keto"):

| Task | Model | Accuracy | Macro-F1 |
|------|-------|---------:|---------:|
| Difficulty (2-cls) | LightGBM | 0.836 | 0.713 |
| Difficulty (2-cls) | XGBoost | 0.803 | 0.697 |
| Meal type (3-cls) | LightGBM | 0.552 | 0.545 |
| Meal type (3-cls) | XGBoost | 0.567 | 0.551 |
| Time class (4-cls) | LightGBM | 0.664 | 0.605 |
| Time class (4-cls) | XGBoost | 0.616 | 0.569 |

Sanity checks on real prompts: "porridge with banana" → Breakfast, <15 min;
"beef wellington" → Dinner, >60 min.

### RF-DETR food detector — `ml_pipeline/prepare_foodseg103_coco.py` + `train_rf_detr_food.py`
FoodSeg103 (4983 train / 500 val images, 103 ingredient classes) downloaded from
HF and converted from segmentation masks to COCO bounding boxes (connected
components per class, small-component filtering). RF-DETR base fine-tuned on
cuda:1 (batch 8, grad-accum 2, lr 1e-4).

Results after 8 epochs (~45 min on the RTX 6000 Ada), evaluated on the 500-image
held-out validation split (103 classes):

| Metric | Value |
|--------|------:|
| EMA mAP@50:95 | **0.381** |
| EMA mAP@50 | 0.433 |
| mAP@75 | 0.381 |

Best per-class AP@50:95: grape 0.96, pumpkin 0.87, lemon 0.81, broccoli 0.78,
carrot 0.77, banana 0.75, corn 0.76. Rare/ambiguous classes (bamboo shoots,
wonton dumplings, salad) remain near 0 — more epochs would help.

Checkpoints: `ml_pipeline/results/rf_detr_food/checkpoint_best_{ema,regular,total}.pth`.
Use from the analyzer with the class count of the fine-tuned head:

```python
RFDETRAnalyzer(model_path="ml_pipeline/results/rf_detr_food/checkpoint_best_total.pth",
               num_classes=103)
```

On the beignets test photo the fine-tuned model detects food-specific classes
("biscuit", "cake") instead of the COCO model's "dining table"/"cell phone".

## Step 1 — Verification done

- **Full unit suite**: 98 passed (was 83; `test_predictor.py` rewritten for the
  new predictor, `test_tools.py` added).
- **GPU suite** (`tests/test_gpu_pipeline.py`, new): local embedder on cuda
  (dim/normalization/batch-vs-single), all 3 trained checkpoints load and
  predict with correct probability widths and sensible labels, metadata/predictor
  consistency, CLIP zero-shot + full pipeline on cuda, RF-DETR non-food filtering
  (slow-marked), portion-map monotonicity. 10 passed, 2 slow-skipped by default.
- **Live OpenRouter suite** (`tests/test_openrouter_live.py`, new; skips without
  key or when rate-limited): agent answers and executes
  `calculate_personalized_nutrition_targets` (BMR/TDEE verified against
  Mifflin-St Jeor by hand), and targets persist to per-user memory.
- **End-to-end runs**: agent computed targets for a real profile via the free
  tier (correct 1780 BMR / 2759 TDEE / -500 deficit); RAG-VLM correctly analyzed
  a food101 beignets photo ("fried dough balls coated in sugar", 450 kcal);
  CLIP-ensemble identified "donut" on the same photo with DB-grounded macros;
  water/lookup tools verified live.
- **Streamlit smoke tests**: both apps render through `streamlit.testing.v1.AppTest`
  with zero exceptions.

## Step 2 — New day-to-day features

What a customer actually does between nutritionist visits: drink water, ask
"can I eat this?", ask "what's left today?", follow a weekly plan, shop for it,
and review the week. None of that existed. Six tools added (11 total), all
covered by unit tests and wired into the system prompt and Quick Actions UI:

| Feature | Tool | Notes |
|---------|------|-------|
| Water tracking | `log_water_intake` | Accumulates into `DailyLog.water_intake_ml` (field existed, nothing wrote it); reports remaining vs the 35 ml/kg target |
| "Can I eat this?" | `lookup_food_nutrition` | Fuzzy lookup in the local 130-food USDA/CIQUAL DB; zero API cost, instant |
| "What's left today?" | `get_remaining_daily_budget` | Targets minus today's logged calories/macros/water |
| Weekly meal plan | `save_meal_plan` / `get_meal_plan` | Persisted per ISO week; injected (bounded) into agent context; grocery lists are built from the stored plan |
| Weekly review | `generate_weekly_summary` | Aggregates the last 7 daily logs into the previously-dead `WeeklySummary` storage |
| Real daily totals | `log_daily_intake` (extended) | Agent now passes per-meal calorie/macro estimates; totals recomputed from meals, so the dashboard and budget tool finally have data (photo meals included) |

UI: three new Quick Action buttons (remaining budget, grocery list from plan,
weekly summary).

## How to run

```bash
# NutriCoach (OpenRouter free tier works; Google key preferred if available)
PYTHONPATH=nut_agent uv run streamlit run nut_agent/nutricoach/app.py

# Recipe Lab (no key needed for predictions)
PYTHONPATH=nut_agent uv run streamlit run nut_agent/recipe_lab/app.py

# Tests
PYTHONPATH=nut_agent uv run python -m pytest nut_agent/tests/ -v                     # CPU suite
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=nut_agent uv run python -m pytest \
    nut_agent/tests/test_gpu_pipeline.py -v                                          # GPU suite
RUN_SLOW_GPU_TESTS=1 CUDA_VISIBLE_DEVICES=1 PYTHONPATH=nut_agent uv run python -m \
    pytest nut_agent/tests/test_gpu_pipeline.py -v                                   # + RF-DETR

# Retrain
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run python ml_pipeline/train_recipe_models.py
PYTHONPATH=. uv run python ml_pipeline/prepare_foodseg103_coco.py
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run python ml_pipeline/train_rf_detr_food.py
```

## Limitations

- **Free-tier LLM quality and quota.** The infrastructure (tools, memory,
  retries, fallbacks) is verified, but free models occasionally ignore part of a
  multi-step instruction or hallucinate; `free-models-per-min` caps add latency
  (the agent now backs off and retries). With `GOOGLE_API_KEY` or a paid
  OpenRouter model (`OPENROUTER_AGENT_MODEL`/`OPENROUTER_VISION_MODEL`) quality
  improves with zero code changes.
- **Meal-type model is weak out of distribution** (0.55 acc on a test set drawn
  from different recipe categories); in-distribution behavior is sensible.
- **Recipe embeddings changed backend** (Gemini text-embedding-004 → local
  bge-base-en-v1.5) so predictions are not comparable with the original
  notebook's numbers, but train/inference are now provably consistent and the
  product works offline.
- The original 3-step `vlm_claude` chain was exercised live only via the RAG
  variant's shared code path (each full chain costs 3 free-tier requests); its
  single steps use identical client/parse code.

---

# Phase 2 (2026-07-17): Embedding/method benchmarks, binary meal, nutrients

## What was asked

1. Benchmark a stronger embedding model (jinaai/jina-embeddings-v5-omni-small)
   against BAAI/bge-base-en-v1.5 for Recipe Lab.
2. Make meal type a binary classifier (Breakfast vs Lunch/Dinner).
3. Benchmark jina-v5 against CLIP for zero-shot food image classification
   (Method 3) and report both food-analysis benchmarks in the READMEs.
4. Find better methods for difficulty / meal / time, including (follow-up
   request) a CatBoost pipeline and a stacking ensemble, and a reliable
   method to estimate nutrients (kcal, fat, ...).

## Benchmarks run (all on cuda:1, seed 42)

- `ml_pipeline/bench_recipe_methods.py`: 2 embedding backends x 8 methods x
  4 classification tasks + 6 regressors + fine-tuned bge heads; results in
  `ml_pipeline/results/bench_recipe_methods.json` (incrementally saved,
  rerun-safe: reruns only compute missing combos).
- `ml_pipeline/bench_food_image_zeroshot.py`: CLIP ViT-B/32 vs jina-v5 on
  1000 Food101 validation images.

## Findings

- bge-base beats jina-v5-omni-small on every recipe task despite 16x fewer
  parameters; bge stays the default encoder (jina remains supported in
  `LocalEmbedder` and via RECIPE_EMBEDDING_MODEL).
- CLIP beats jina-v5 for zero-shot food classification: top-1 0.796 vs
  0.743, top-5 0.954 vs 0.918, 237 vs 687 ms/image. CLIP stays Method 3's
  default; `CLIPFoodAnalyzer(backend="jina")` is available.
- Binary meal type lifts deployed accuracy from ~0.64 (3-class) to
  0.81-0.85; Lunch vs Dinner is essentially not learnable from text.
- New methods: CatBoost is the strongest single GBM on difficulty macro F1
  (0.709); the stacking ensemble is the best deployable difficulty model
  (0.813 acc / 0.717 f1). Fine-tuned bge-base wins every classification
  task outright (0.826 / 0.846 / 0.694 accuracy).
- Nutrients: frozen-embedding regression barely beats the mean baseline
  (embeddings do not encode quantities); a fine-tuned bge-base regression
  head on raw text is the only reliable method (kcal MAE 133 vs 168
  baseline, protein MAE 3.7 g vs 12.5 g, mean R2 +0.20) and is deployed as
  the primary nutrients model with CatBoost-on-embeddings fallback.

## Deployment (ml_pipeline/results/, meta recipe_models_meta.json)

- difficulty: stacking ensemble; meal_type (binary): LightGBM;
  time_class: LightGBM; nutrients: fine-tuned bge regression head
  (`nutrients_bge_regressor/`) + CatBoost fallback checkpoint.
- Predictor resolves per-task model families from the meta file; Recipe Lab
  app shows an "Estimated Nutrition (per serving)" panel.
- New registry entries: catboost_classifier/regressor,
  stacking_classifier/regressor (bounded n_jobs=8 by default; n_jobs=-1 on
  this 128-core host caused loky worker crashes).

## Test state

- CPU suite: 106 passed (16 predictor incl. new model-family and nutrient
  tests).
- GPU suite: extended with nutrients-head prediction and jina zero-shot
  backend tests (final counts in the README Testing section).

## Limitations

- Nutrient estimates are per serving and coarse (kcal MAE ~130); the test
  split is out-of-domain vs training recipes, so R2 values are depressed
  for all methods.
- jina-v5-omni-small is CC BY-NC 4.0 (non-commercial); it is benchmarked
  and supported but not deployed.
- Free-tier OpenRouter limits unchanged from Phase 1.

## Nutrients coherence validation (follow-up)

Three checks of the deployed fine-tuned nutrients head:

1. Per-recipe on the 182 held-out test recipes: median kcal error 31%,
   49% within +/-30%, 66% within +/-50%, Spearman rank correlation 0.43
   (protein 0.43, fat 0.56).
2. Directional sanity (synthetic app-style inputs): 6/7 checks pass;
   all relative rankings correct (burger > salad kcal, cake > chicken
   sugars, chicken > cake protein, burger > rice fat). Failure mode is
   magnitude compression toward the ~330 kcal dataset mean (a double
   cheeseburger with fries predicted at only 242 kcal).
3. Web ground truth, three recipes verified absent from all splits
   (bbcgoodfoodme.com): tikka masala 339 pred vs 345 actual kcal and
   29.6 vs 31 g protein (excellent); chocolate chip banana bread 466 vs
   306 kcal (over); miso salmon traybake 394 vs 610 kcal, 32.7 vs 42 g
   protein (under, correct macro profile).

Verdict: usable for the customer as a per-serving ballpark and for
comparing recipes; not accurate enough for calorie tracking. The Recipe
Lab panel now carries a caption saying exactly that, and NutriCoach meal
logging is unaffected (it uses the nutrition DB + LLM portions).
