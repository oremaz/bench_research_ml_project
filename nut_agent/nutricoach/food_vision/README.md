# Food Vision

NutriCoach uses **Single-shot VLM** with `dots-studio/dots-3-note-preview:free`
by default. The Food Analysis tab lets users edit the meal name, change grams,
and add or delete ingredients before saving the reviewed meal. Nutrients update
without another model call. Renamed or new ingredients require an explicit local
database entry or manually entered per-100g values. Saving is explicit and the
review freezes after saving to prevent a second submission of the same draft.
The LangGraph photo tool also uses single-shot only; journal logging is opt-in.

The research benchmark compares four families on one RGB image per dish:

| Method | Implementation | Nutrition and quantity prediction |
|---|---|---|
| `vlm_single` | `vlm_analyzer.py` | One Dots image call predicts foods, grams and nutrients |
| `geometry_vlm_db` | `geometry_analyzer.py` | Dots recognition/retrieval, SAM masks, predicted depth, volume, bulk density, local nutrition values |
| `rgb_regression` | `supervised_analyzers.py`, `train_rgb_regression.py` | Frozen DINOv2 ViT-S/14 plus supervised ridge outputs for all five targets |
| `food_r1` | `supervised_analyzers.py` | Published specialized Qwen3-VL-8B checkpoint, whole-dish nutrition prediction |

These are whole-pipeline comparisons. Dots is shared by the two API pipelines;
the specialized model and RGB backbone necessarily have different weights.
The RGB baseline trains a lightweight regression head, not the entire backbone.
No local implementation is presented as an exact reproduction of a paper score.

## Why a nutrition database alone is insufficient

For every nutrient j, the calculation is `total_j = sum(grams_i * per_100g_ij / 100)`.
A database improves nutrient composition lookup but does not determine edible
mass, preparation, or invisible oil and sugar. The earlier RF-DETR pipeline used
serving priors; whole-image CLIP candidates were not ingredient detections; the
older RAG pipeline adjusted portions against typical ranges. Those choices do
not implement a physical portion estimator. Additional segmentation can also
accumulate quantity errors rather than improve meal totals.

The [Nutrola technical blog](https://nutrola.app/fr/blog/how-ai-estimates-portion-sizes-from-photos-technical-deep-dive)
describes segmentation, depth, scale, volume and food density. Our hybrid
implements that chain, with explicit assumptions and fallback reporting.
The article does not release the proprietary models or a reproducible evaluation.
The [BFM Cal AI article](https://www.bfmtv.com/tech/actualites/cal-ai-l-app-qui-promet-de-calculer-les-calories-a-partir-d-une-photo_AN-202503170555.html)
likewise does not establish a comparable Nutrition5k accuracy result.

## Geometry hybrid

Two image calls identify nonoverlapping food components and select among exact,
token and fuzzy local database candidates. Python computes nutrients from
per-100g values; it never asks the model to redo the arithmetic. An unmatched
food can use explicit model-estimated nutrient densities, identified in its trace.

SAM ViT-B segments the food boxes. Depth Anything V2 Metric Indoor Small predicts
depth from the RGB image. Back-projection and a robust exposed-plate plane fit
provide heights; integration of height times projected plate area estimates
volume. Overlapping columns are counted once. Volume in ml times bulk density
in g/ml gives grams, then scales every nutrient. Model revisions are pinned in
`geometry_analyzer.py`. Models load lazily and are shared between benchmark workers.

The RGB-only benchmark assumes a 26 cm plate diameter (or rectangular plate
width) and a 60 degree horizontal field of view. These are scale priors, not
measurements. Volume varies cubically with the assumed reference dimension.
The class accepts an explicit plate dimension for separate calibrated experiments;
this information is not supplied in the RGB-only benchmark.
Six illustrative density values come from the blog; other densities are VLM
assumptions. This is not a validated bulk-density database.

Bowls, absent reference surfaces, unreliable plane fits and zero visible volumes
retain the VLM gram estimate with an explicit geometry fallback. The JSON trace
records each volume, density source, scale assumption and actual geometry use.
This 2.5D approximation cannot reconstruct hidden layers or oil. The method must
be judged with its fallback rate; no accuracy improvement has yet been measured.

## Literature and protocol

Full PDFs read for the implementation and protocol:

- [Nutrition5k](https://arxiv.org/pdf/2103.03375), including incremental scans,
  official train/test isolation, direct regression and five nutrient targets.
- [Food-R1](https://arxiv.org/pdf/2606.04986), including the supplement's whole-dish
  nutrition format and supervised/RL training details.
- [Food Portion Estimation: From Pixels to Calories](https://arxiv.org/pdf/2602.05078),
  including scale ambiguity, occlusion and the volume-to-mass density gap.

Food-R1 inference uses the published
[checkpoint](https://huggingface.co/zy12123/Food-R1), pinned to revision
`c70e0d6585b1e81923432df46014d6ce32855e3f`. It predicts whole-dish mass,
calories, fat, carbohydrate and protein in an answer-only format consistent with
the training task. It receives neither ingredient labels nor measured mass.
Greedy generation uses 512 new tokens, with the image bounded to 1024 pixels
before processor resizing. This adapter is not a reproduction of SFT/GRPO training
or every paper evaluation setting. The approximately 17.5 GB checkpoint needs
substantial GPU memory; the supplied job targets a GPU with at least 40 GB.
The paper reports using official dataset splits; we cannot independently audit
all training data of published or API models.

`train_rgb_regression.py` uses the existing shared vision registry for DINOv2.
It selects overhead images only from official RGB training IDs intersected with
depth training IDs. Missing public images returning HTTP 404 are explicitly
excluded and recorded. Validation holds out 20 percent of UTC acquisition days,
keeping scans of a plate together. Feature/target scaling fits training data only.
The split groups by acquisition day as a conservative proxy for related scans;
no independent plate-sequence grouping file is supplied to this trainer.
Ridge strength is chosen by the mean validation MAE normalized with training-set
target means. The test subset is never used for selection or fitting.
The checkpoint includes encoder weights, preprocessing, target ordering, scaling,
head parameters, train/validation IDs and source/image hashes. The evaluator
rejects a regression checkpoint whose fitted IDs violate official train/test separation.

`evaluate_nutrition5k.py` freezes 50 dishes with seed 42 from the official RGB/depth
test-ID intersection and downloads one overhead RGB image each. No real depth,
reference mass or ingredient metadata is provided to an analyzer. The manifest
records source and image SHA-256 hashes. Training data is separate.

For all five targets, reports include MAE, signed error, normalized MAE,
RMSE, R2 and 2,000 dish-bootstrap MAE intervals. Normalized MAE is MAE divided
by mean ground truth, not the mean of per-dish percentage errors. Paired MAE
differences use common successful dishes for each method pair and target.
Failures and missing dishes are retained, with no fabricated predictions.
Bootstrap intervals describe sampling variability in this pilot, not calibrated
per-photo prediction uncertainty or repeated-generation variability.

A post-hoc oracle-mass diagnostic rescales predicted nutrients by
`reference_mass / predicted_mass` on positive-mass dishes. This helps investigate
portion versus composition errors. It uses ground truth only during scoring and
must never be reported as deployable RGB-only performance.

## Current measured results

See [the generated Dots report](nutrition5k_dots_results.md) and
[recomputable snapshot](nutrition5k_dots_results.json).
Dots single-shot completed all 50 images with high reasoning, temperature 0.1,
and an 8,192-token output budget: **97.1 kcal MAE**, **73.8 g mass MAE**,
**6.5 g fat**, **8.0 g carbohydrate**, **8.2 g protein**.

The geometry API run is pending because the free OpenRouter daily quota was
exhausted. Regression training/evaluation and Food-R1 evaluation are prepared
for Jean Zay but have not completed. Pending cells are not measured scores.
This 50-dish controlled overhead-image pilot is insufficient to establish
smartphone-photo generalization. Published paper scores are not inserted into
our local comparison; datasets, splits and available inputs differ.

Historical [low](nutrition5k_test50_results.json) and
[high](nutrition5k_test50_high_results.json) snapshots are retained with their
original model metadata. They are not Dots runs and do not enter this comparison.

## Run the benchmark

All commands start at the repository root in its `uv` environment.

```bash
# Prepare the frozen test images only
PYTHONPATH=. uv run python nut_agent/nutricoach/food_vision/evaluate_nutrition5k.py --prepare-only

# Train the supervised RGB baseline, using all available official training overheads
PYTHONPATH=. uv run python nut_agent/nutricoach/food_vision/train_rgb_regression.py --device cuda

# Local models require no OpenRouter key
PYTHONPATH=. uv run python nut_agent/nutricoach/food_vision/evaluate_nutrition5k.py \
    --methods rgb_regression food_r1 --device cuda \
    --output-dir nut_agent/nutricoach/food_vision/results/nutrition5k_50_local

# Run geometry after free-model quota reset, with OPENROUTER_API_KEY set
PYTHONPATH=. uv run python nut_agent/nutricoach/food_vision/evaluate_nutrition5k.py \
    --methods geometry_vlm_db --model dots-studio/dots-3-note-preview:free \
    --reasoning-effort high --workers 2 --request-interval 3.2 \
    --output-dir nut_agent/nutricoach/food_vision/results/nutrition5k_50_dots_geometry

# Regenerate the report as methods finish; select exactly one run per method
PYTHONPATH=. uv run python nut_agent/nutricoach/food_vision/report_nutrition5k.py \
    --run-dir nut_agent/nutricoach/food_vision/results/nutrition5k_50_dots_high_api \
              nut_agent/nutricoach/food_vision/results/nutrition5k_50_dots_geometry \
              nut_agent/nutricoach/food_vision/results/nutrition5k_50_local
```

The evaluator defaults to the four current methods. It saves every dish atomically
and resumes completed predictions; `--retry-errors` retries failures. Each run
records configuration, source/checkpoint hashes, API usage and resolved model IDs.
Use a new output directory after code/configuration changes. API calls share a
request limiter; persistent configuration failures and daily quota exhaustion stop
the run. API throttling/retries are included in elapsed time, so recorded latency
is not a controlled model-speed comparison. Local inference is serialized.
Data, full responses and checkpoints live in gitignored `data/` and `results/`.
The compact current snapshot is checked in alongside this README.

## Jean Zay

The job is prepared, not submitted from this workspace. Set up the repository's
GPU `uv` environment on Jean Zay using the root GPU dependencies and your site's
module configuration. It must already contain a CUDA-compatible torch, timm,
scikit-learn, Pillow and Transformers with Qwen3-VL support.

On a network-enabled preparation node, with the same shared cache/data paths as
the compute job, run:

```bash
PYTHONPATH=. uv run python nut_agent/nutricoach/food_vision/prepare_benchmark_assets.py --food-r1
PYTHONPATH=. uv run python nut_agent/nutricoach/food_vision/train_rgb_regression.py --prepare-only --device cpu
```

Transfer the existing Dots result directory and its frozen manifest to preserve
the baseline. Cache placement can use `HF_HOME` on shared project storage.
The job sets Hugging Face offline mode and uses `uv run --offline`.
From the repository root, choose an allocated account and GPU constraint:

```bash
sbatch --account=YOURPROJECT@a100 --constraint=a100 \
    nut_agent/nutricoach/food_vision/jobs/nutrition5k_jean_zay.sbatch all
# Replace all by rgb or food_r1 to run only one family.
```

Account, partition and module setup depend on your allocation; see the
[IDRIS guide](https://www.idris.fr/media/eng/ia/guide_nouvel_utilisateur_ia-eng.pdf).
The job uses one GPU, eight CPU cores and a four-hour limit. API evaluations run
separately on a network-enabled host. Copy resulting method directories back and
regenerate the report; it rejects mismatched test manifests and API model IDs.

## Validation in this checkout

On 2026-10-08, all **190 local tests passed**, excluding the live OpenRouter and
GPU-integration modules. This includes Streamlit AppTest coverage of photo
editing, add/delete, recalculation, user isolation and single journal submission.
The macOS run preloaded LightGBM/Torch and limited native threads and joblib to
one process after a native subprocess crash in an existing stacking test.
The exact invocation was:

```bash
LOKY_MAX_CPU_COUNT=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. uv run python -c 'import lightgbm; import torch; torch.set_num_threads(1); import pytest; raise SystemExit(pytest.main(["nut_agent/tests/", "-q", "-p", "no:cacheprovider", "--ignore=nut_agent/tests/test_gpu_pipeline.py", "--ignore=nut_agent/tests/test_openrouter_live.py"]))'
```

Real smoke checks covered DINOv2 head fitting on 16 official-training dishes,
checkpoint reload and inference, repeated offline training, the actual Food-R1
image processor, and pretrained SAM/depth inference. The tiny training run is
only an execution check and is excluded from the benchmark results.
Slurm shell syntax and `git diff --check` passed. Full Food-R1 generation and
full regression training/evaluation have not run here.

## Legacy research utilities

RF-DETR, CLIP plus VLM, serving-range RAG and the database-grounded VLM remain
available through `compare.py` and explicit evaluator method names. They are
excluded from the primary comparison. The local nutrition table is a small
130+ food reference table, not Nutrola's proprietary database or full USDA/FNDDS.

```bash
PYTHONPATH=.:nut_agent uv run python -m nutricoach.food_vision.compare \
    --image plate.jpg --methods vlm_single,geometry_vlm_db

# Existing RF-DETR FoodSeg103 training pipeline
PYTHONPATH=. uv run python nut_agent/nutricoach/food_vision/prepare_foodseg103_coco.py
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=nut_agent uv run python \
    -m nut_agent.nutricoach.food_vision.train_rf_detr_food
```

RF-DETR needs its food checkpoint and matching `food_classes.json`; generic COCO
fallback is disabled. Custom weights, labels, nutrition-name mappings and optional
measured portions remain supported by the comparison CLI. Supplied measurements
belong to a separate calibrated experiment.
`bench_food_image_zeroshot.py` compares CLIP/Jina Food101 classification only,
not portion or meal-nutrition accuracy.
