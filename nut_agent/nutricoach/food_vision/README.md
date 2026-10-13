# Food Vision

NutriCoach uses **Single-shot VLM** with OpenRouter and
`dots-studio/dots-3-note-preview:free` by default. The sidebar's **Model provider**
selects OpenRouter, Groq, or local `Qwen/Qwen3.8-27B` for photos and NutriCoach chat.
Groq defaults to `qwen/qwen3.8-27b` and uses a separate `GROQ_API_KEY`.
Local inference needs no API key. The Food Analysis tab lets users edit the meal name, change grams,
and add or delete ingredients before saving the reviewed meal. Nutrients update
without another model call. Renamed or new ingredients require an explicit local
database entry or manually entered per-100g values. Saving is explicit and the
review freezes after saving to prevent a second submission of the same draft.
The LangGraph photo tool follows the same provider selection; journal logging is opt-in.

## Local Qwen and hosted providers

`shared/llm_provider.py` centralizes provider endpoints, credentials and reasoning
parameters. Prompts and JSON parsing are shared across providers. Groq uses its
native `reasoning_effort` and hidden reasoning format; OpenRouter uses its
`reasoning` object. `none` is available in the provider menu and benchmark CLI.
See [Groq vision](https://console.groq.com/docs/vision) and
[Groq reasoning](https://console.groq.com/docs/reasoning) for compatible models.

The [Qwen author model card](https://huggingface.co/Qwen/Qwen3.8-27B) identifies
a native vision-language model with the Qwen3.5 architecture. Local loading is
pinned to revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` and requires
Transformers with `Qwen3_5ForConditionalGeneration` support, plus Accelerate.
The process caches one client, releases earlier weights when switching local
precision/model, serializes inference, bounds images to 1024 pixels,
and generates greedily. Local reasoning defaults to disabled; enabled levels
are `low`, `medium`, and `xhigh`, matching the released chat template.
The adapter decodes the author's XML tool-call format for the NutriCoach chat.
Malformed, incomplete, unknown or missing-argument tool calls are rejected.
Photos are base64-decoded locally and are never sent to a hosted provider.

BF16 weights occupy roughly 58 GB before activations and cache. A H100 80 GB is
the intended full-precision target. Optional 4-bit NF4 loading uses `bitsandbytes`
on CUDA; 32 GB CPU RAM alone is not enough for the BF16 model. Model downloads
and caches should be placed on project storage using `HF_HOME` and `UV_CACHE_DIR`.

```bash
export HF_HOME="$PWD/nut_agent/nutricoach/food_vision/data/huggingface"
export UV_CACHE_DIR="$PWD/nut_agent/nutricoach/food_vision/data/uv_cache"
PYTHONPATH=. uv run python nut_agent/nutricoach/food_vision/prepare_benchmark_assets.py --local-qwen

# Same frozen 500-dish manifest, single-shot VLM
PYTHONPATH=. uv run python nut_agent/nutricoach/food_vision/evaluate_nutrition5k.py \
    --provider local --model Qwen/Qwen3.8-27B --device cuda \
    --methods vlm_single --reasoning-effort none --workers 1 \
    --output-dir nut_agent/nutricoach/food_vision/results/nutrition5k_500_qwen_single

# Groq requires GROQ_API_KEY; no OpenRouter credentials are used
PYTHONPATH=. uv run python nut_agent/nutricoach/food_vision/evaluate_nutrition5k.py \
    --provider groq --model qwen/qwen3.8-27b --methods vlm_single \
    --reasoning-effort none --output-dir nut_agent/nutricoach/food_vision/results/nutrition5k_500_groq

# Run the complete NutriCoach suite and real local Qwen integration tests on H100
sbatch --account=mbm@h100 --partition=gpu_p6 --constraint=h100 \
    --qos=qos_gpu_h100-dev --cpus-per-task=24 \
    --output=nut_agent/nutricoach/food_vision/results/slurm/nutricoach-qwen-%j.log \
    nut_agent/nutricoach/food_vision/jobs/qwen_local_jean_zay.sbatch all

# Separate 4-bit image test with a 32 GiB CUDA allocation ceiling
sbatch --account=mbm@h100 --partition=gpu_p6 --constraint=h100 \
    --qos=qos_gpu_h100-dev --time=00:30:00 --cpus-per-task=24 \
    --output=nut_agent/nutricoach/food_vision/results/slurm/nutricoach-qwen-%j.log \
    nut_agent/nutricoach/food_vision/jobs/qwen_local_jean_zay.sbatch quantized
```

Use `--local-quantization 4bit` to evaluate the quantized variant in a separate
output directory. Provider, revision, quantization, preprocessing and code hashes
are stored in benchmark configuration; a changed configuration requires a new run.
Provider changes and removal of geometry change recorded source hashes. Earlier
Qwen and Dots runs remain measured snapshots; use a new output directory for
new evaluations. Their saved configurations and original hashes are preserved.

Benchmark mode generates `nutrition5k_500_qwen_results.json` and its Markdown
comparison automatically, using the four completed local research runs and the
Qwen single-shot baseline. Dots reports remain separate. For longer evaluations, submit
benchmark mode with the allocation's regular GPU QoS and an appropriate wall time.

The research benchmark compares five systems, including FoodLMM precision and
generation variants, on one RGB image per dish:

| Method | Implementation | Nutrition and quantity prediction |
|---|---|---|
| `vlm_single` | `vlm_analyzer.py` | One selected VLM image call predicts foods, grams and nutrients |
| `rgb_regression` | `supervised_analyzers.py`, `train_rgb_regression.py` | Frozen DINOv2 ViT-S/14 plus supervised ridge outputs for all five targets |
| `food_r1` | `supervised_analyzers.py` | Published specialized Qwen3-VL-8B checkpoint, whole-dish nutrition prediction |
| `foodlmm` | Saved original author-run records | Public FoodLMM-Chat 16-bit checkpoint with ten trained nutrition heads |
| `foodlmm_cached` | `evaluate_paper_system.py --method foodlmm` | Same 16-bit checkpoint with cached generation and a full-prefix nutrition pass |
| `foodlmm_4bit` | `evaluate_paper_system.py --method foodlmm --quantization 4bit` | Same checkpoint with 4-bit decoder weights and floating-point nutrition heads |
| `dpf_reproduction` | `evaluate_paper_system.py` | Third-party Food2K/ResNet101 fusion checkpoint with predicted RGB-only depth |

These are whole-pipeline comparisons. The selected VLM configures the one-shot
baseline; the specialized models and RGB backbone use their own weights.
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

## Literature and protocol

Full PDFs read for the implementation, release audit and protocol:

- [Nutrition5k](https://arxiv.org/pdf/2103.03375), including incremental scans,
  official train/test isolation, direct regression and five nutrient targets.
- [Food-R1](https://arxiv.org/pdf/2606.04986), including the supplement's whole-dish
  nutrition format and supervised/RL training details.
- [Food Portion Estimation: From Pixels to Calories](https://arxiv.org/pdf/2602.05078),
  including scale ambiguity, occlusion and the volume-to-mass density gap.
- [DPF-Nutrition](https://arxiv.org/pdf/2310.11702),
  [FoodLMM with supplement](https://arxiv.org/pdf/2312.14991),
  [CalorieLLaVA journal paper](https://www.mdpi.com/2072-6643/17/7/1128), and
  [DietAI24](https://www.nature.com/articles/s43856-025-01159-0.pdf).

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

`evaluate_nutrition5k.py` defaults to 500 dishes with seed 42 from the official RGB/depth
test-ID intersection and downloads one overhead RGB image each. No real depth,
reference mass or ingredient metadata is provided to an analyzer. The manifest
records source and image SHA-256 hashes. Training data is separate.

For all five targets, reports include MAE, signed error, normalized MAE,
RMSE, R2 and 2,000 dish-bootstrap MAE intervals. Normalized MAE is MAE divided
by mean ground truth, not the mean of per-dish percentage errors. Paired MAE
differences use common successful dishes for each method pair and target.
Failures and missing dishes are retained, with no fabricated predictions.
Bootstrap intervals describe sampling variability in this subset, not calibrated
per-photo prediction uncertainty or repeated-generation variability.

A post-hoc oracle-mass diagnostic rescales predicted nutrients by
`reference_mass / predicted_mass` on positive-mass dishes. This helps investigate
portion versus composition errors. It uses ground truth only during scoring and
must never be reported as deployable RGB-only performance.

## Current 500-dish evaluation

See [the 500-dish comparison](nutrition5k_500_results.md) and its
[recomputable snapshot](nutrition5k_500_results.json). Reports identify partial
runs by their success counts; errors are retained and metrics use successful
predictions. All five systems share the same manifest and image hashes.

H100 development job `814127` completed RGB training and the RGB/Food-R1
evaluations on 2026-10-09, exit `0:0`, in 13 minutes 27 seconds. Each local
method completed 500/500 dishes. Job `814967` completed all 500 DPF attempts:
489 valid predictions and 11 failures with negative nutritional outputs.
FoodLMM initially stopped before inference on a Python 3.10 hashing
compatibility error; the corrected job `815020` completed in 13 minutes
32 seconds with exit `0:0`, with 495 valid predictions and 5 failures out of
500 attempts. Completed calorie MAEs are 75.8 kcal for RGB, 77.8 for Food-R1,
49.0 for FoodLMM-Chat and 47.0 for the provisional DPF reproduction, each on
its successful dishes. These rows are not comparisons on identical success subsets.
The historical Dots single-shot evaluation used only the explicitly authorized
free model. On 2026-10-09 OpenRouter returned HTTP 429 `free-models-per-day`,
with a daily limit of 50 and zero remaining requests. Single-shot saved 49
attempts (41 valid, 8 failures). No paid fallback or daily resumptions are used.
The Qwen single-shot baseline subsequently completed all 500 dishes; see
[its five-method comparison](nutrition5k_500_qwen_results.md).

### Additional paper release coverage

| Paper system | Release checked | Local status or blocker |
|---|---|---|
| FoodLMM | [Author code](https://github.com/YuehaoYin/FoodLMM), [author weights](https://huggingface.co/Yueha0/FoodLMM-Chat) | Evaluated on all 500 dishes in isolated uv Python 3.10 / Transformers 4.31 |
| DPF-Nutrition | [Third-party reproduction](https://github.com/T0MYYY/Nutrition5k), [weights](https://huggingface.co/T0MYYY/dpf-nutrition) | Evaluated; explicitly not the author model or an exact reproduction |
| CalorieLLaVA | Full 2025 journal PDF and author publication listings | No released nutrition fine-tuned checkpoint located; generic LLaVA is not substituted |
| DietAI24 | [Author code](https://github.com/Runz96/DietAI24), full paper | Released config has empty Azure endpoints/keys and requires GPT-4 Turbo/Vision plus text-embedding-3-large; those paid deployments are unavailable |

FoodLMM uses the published **Chat** checkpoint, not the paper's S1 or
task-fine-tuned score. The released model is built on LISA/LLaVA 7B with
CLIP ViT-L/14 and trained nutrition heads; it does not use Qwen.
Its prompt requests the entire dish; numerical predictions
come from trained nutrition heads rather than parsing generated numeric guesses.
Generation is greedy with 256 new tokens and author CLIP preprocessing.
The nutrition runner omits the unused SAM segmentation branch after the nutrition
heads. CUDA uses BF16 and MPS uses FP16 for floating-point weights.
Generation uses a KV cache and an explicit attention mask. One full-prefix
forward pass then supplies the trained heads with the author's hidden-state
alignment. This avoids recomputing CLIP and the full decoder at every token.
The original 16-bit score stays in the report. Precision comparisons use the
additional cached 16-bit control, so both variants share the generation policy.

The 500-dish H100 precision job `1006022` completed with exit code `0:0` on
2026-10-13. The 4-bit run returned valid outputs for 474 dishes; the cached 16-bit
control succeeded on 496. On 472 common successful dishes, quantization increased
calorie MAE by 2.00 kcal (dish-bootstrap 95% interval 0.46 to 3.54 kcal).
All other paired target differences and failures are in the generated reports.
The separate success-only calorie rows are 47.9 and 49.0 kcal respectively and
use different dish subsets.

The measured inference peaks were 4.97 GiB allocated / 5.71 GiB reserved for
4-bit and 13.72 / 13.80 GiB for cached 16-bit. Mean inference was 1.18 and
0.75 seconds per dish respectively. Quantization saves about 64% of allocated
inference memory, with slightly worse paired accuracy, slower H100 inference and
more format failures (26 versus 4). The 4-bit packed file is approximately
4.3 GiB; its CUDA loading peak is 7.91 GiB allocated and peak host RSS is
7.42 GiB. These are separate CUDA/host measurements, not a Mac unified-memory
benchmark. The original 16-bit row remains 495/500 successes and 49.0 kcal MAE.
Cached 16-bit preserved all 495 original valid predictions exactly and recovered
the previously missing calorie token on one additional dish.
All required nutritional tokens and checkpoint parameters must exist.
The external release commit and model revisions are pinned in the runner.

The DPF reproduction replaces the paper's supervised DPT depth module with
Depth Anything V2 Small, and modifies attention/fusion. We use its released
336 x 448 RGB/depth transforms and per-image 16-bit normalized predicted depth.
Its code selects checkpoints on a validation subset of official training IDs.
The published weight file contains no fitted dish IDs, so its actual training
membership cannot be independently audited. The same limitation applies to
the training membership of published FoodLMM/Food-R1 and API models.
No measured depth, measured mass or test labels enter inference.
The DPF publisher's [initial model card](https://huggingface.co/T0MYYY/dpf-nutrition/blob/ea15de307a014ca48922899ab2275578aea6c8ea/README.md)
also describes an earlier hash-split leakage problem and states it was fixed
before this release. Without fitted dish IDs we cannot verify that repair in
the weight file. Its comparison remains provisional, with this provenance
limitation, rather than an independently validated paper reproduction.

## Historical 50-dish pilot

See [the generated comparison report](nutrition5k_dots_results.md) and
[recomputable snapshot](nutrition5k_dots_results.json).
Dots single-shot completed all 50 images with high reasoning, temperature 0.1,
and an 8,192-token output budget: **97.1 kcal MAE**, **73.8 g mass MAE**,
**6.5 g fat**, **8.0 g carbohydrate**, **8.2 g protein**.

H100 development job `798727` completed successfully on 2026-10-08 in 3 minutes
30 seconds. RGB regression and Food-R1 each completed all 50 images without
failures. RGB regression measured **101.4 kcal MAE**, **60.2 g mass MAE**,
**6.9 g fat**, **7.8 g carbohydrate**, **9.5 g protein**. Food-R1 measured
**109.2 kcal MAE**, **96.7 g mass MAE**, **6.7 g fat**, **7.0 g carbohydrate**,
**7.1 g protein**. The report records job details, training provenance and timings.

The removed geometry row in this historical report has no measured score.
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
    --output-dir nut_agent/nutricoach/food_vision/results/nutrition5k_500_local

# Run the one-shot baseline, with OPENROUTER_API_KEY set
PYTHONPATH=. uv run python nut_agent/nutricoach/food_vision/evaluate_nutrition5k.py \
    --methods vlm_single --model dots-studio/dots-3-note-preview:free \
    --reasoning-effort high --workers 2 --request-interval 3.2 \
    --output-dir nut_agent/nutricoach/food_vision/results/nutrition5k_500_dots_single

# Regenerate the report as methods finish; select exactly one run per method
PYTHONPATH=. uv run python nut_agent/nutricoach/food_vision/report_nutrition5k.py \
    --run-dir nut_agent/nutricoach/food_vision/results/nutrition5k_500_dots_single \
              nut_agent/nutricoach/food_vision/results/nutrition5k_500_rgb \
              nut_agent/nutricoach/food_vision/results/nutrition5k_500_food_r1 \
              nut_agent/nutricoach/food_vision/results/nutrition5k_500_dpf_reproduction \
              nut_agent/nutricoach/food_vision/results/nutrition5k_500_foodlmm
```

The evaluator defaults to one-shot VLM, RGB regression and Food-R1; the paper
runner evaluates FoodLMM and the DPF reproduction separately. It saves every dish atomically
and resumes completed predictions; `--retry-errors` retries failures. Each run
records configuration, source/checkpoint hashes, API usage and resolved model IDs.
Use a new output directory after code/configuration changes. API calls share a
request limiter; persistent configuration failures and daily quota exhaustion stop
the run. API throttling/retries are included in elapsed time, so recorded latency
is not a controlled model-speed comparison. Local inference is serialized.
Data, full responses and checkpoints live in gitignored `data/` and `results/`.
The compact current snapshot is checked in alongside this README.

## Jean Zay

Set up the repository's GPU `uv` environment on Jean Zay using the root GPU
dependencies and your site's
module configuration. It must already contain a CUDA-compatible torch, timm,
scikit-learn, Pillow and Transformers with Qwen3-VL support.

On a network-enabled preparation node, with the same shared cache/data paths as
the compute job, run:

```bash
PYTHONPATH=. uv run python nut_agent/nutricoach/food_vision/prepare_benchmark_assets.py --food-r1
PYTHONPATH=. uv run python nut_agent/nutricoach/food_vision/train_rgb_regression.py --prepare-only --device cpu
```

For the two additional paper systems, cache their pinned releases and install
FoodLMM's older dependencies in an isolated uv environment. DPF uses the main
repo environment. The Python install directory below keeps assets on project storage.

```bash
PYTHONPATH=. uv run python nut_agent/nutricoach/food_vision/prepare_benchmark_assets.py --papers
export UV_PYTHON_INSTALL_DIR="$PWD/nut_agent/nutricoach/food_vision/data/python"
uv python install 3.10
uv venv --python 3.10 nut_agent/nutricoach/food_vision/data/foodlmm_env
uv pip install --python nut_agent/nutricoach/food_vision/data/foodlmm_env/bin/python \
    'torch==2.6.0' 'torchvision==0.21.0' --index-url https://download.pytorch.org/whl/cu124
uv pip install --python nut_agent/nutricoach/food_vision/data/foodlmm_env/bin/python \
    -r requirements_foodlmm.in
```

### FoodLMM 4-bit and Apple Silicon

The runner accepts `--device auto`, `cuda`, `cuda:INDEX`, `mps`, or `cpu`.
`auto` chooses CUDA, then MPS, then CPU. The 4-bit variant uses
[Quanto](https://github.com/huggingface/optimum-quanto), which supports CUDA and
MPS. Only decoder linear weights are quantized; CLIP, embeddings, language output
and trained nutrition heads stay in floating point. No test images or labels are
used to quantize the weights.

Export the packed checkpoint on a machine with enough memory for the original
weights, then load that export directly on the target machine:

```bash
food_vision_dir=nut_agent/nutricoach/food_vision
uv run --python "$food_vision_dir/data/foodlmm_env/bin/python" python "$food_vision_dir/evaluate_paper_system.py" \
    --method foodlmm --device cuda --quantization 4bit --prepare-only \
    --export-quantized "$food_vision_dir/data/foodlmm_4bit" \
    --output-dir "$food_vision_dir/results/foodlmm_4bit_prepare"
uv run --python "$food_vision_dir/data/foodlmm_env/bin/python" python "$food_vision_dir/evaluate_paper_system.py" \
    --method foodlmm --device mps --quantization 4bit \
    --quantized-checkpoint "$food_vision_dir/data/foodlmm_4bit" \
    --output-dir "$food_vision_dir/results/nutrition5k_500_foodlmm_4bit_mps"
```

On macOS, install `requirements_foodlmm.in` in the isolated Python 3.10 environment
without the CUDA wheel index. Xcode command line tools are needed to compile
Quanto's MPS kernels. Copy the packed checkpoint and pinned FoodLMM code;
the tokenizer, config and CLIP weights must also be available through Hugging Face
or its cache. The benchmark additionally needs the frozen manifest and images.
Restoration builds the decoder on the meta device and assigns packed tensors, so
it does not allocate the full decoder in floating point first.
MPS compatibility has not been validated on physical Apple hardware; CUDA memory
measurements do not establish total unified-memory use or latency on a 16 GB Mac.

Submit the matching H100 comparison with the existing scheduler command and mode
`foodlmm_precision`. It exports the checkpoint, evaluates all 500 dishes in a
fresh 4-bit process, then profiles 16-bit FoodLMM and DPF. `resources.json` records
loading and inference GPU peaks, process peak host RSS, and the number of profiled
dishes. Device synchronization makes the recorded inference latency meaningful.
MPS records current allocated and driver memory, not a measured peak.

Rerun Dots on the new 500-dish manifest; the 50-dish baseline does not cover it.
Cache placement can use `HF_HOME` on shared project storage.
The job sets Hugging Face offline mode and uses `uv run --offline`.
From the repository root, choose an allocated account and GPU constraint:

```bash
sbatch --account=YOURPROJECT@a100 --constraint=a100 \
    nut_agent/nutricoach/food_vision/jobs/nutrition5k_jean_zay.sbatch all
# Replace all by rgb, food_r1, foodlmm, dpf_reproduction or papers.
```

Account, partition and module setup depend on your allocation; see the
[IDRIS guide](https://www.idris.fr/media/eng/ia/guide_nouvel_utilisateur_ia-eng.pdf).
The job defaults to one GPU, eight CPU cores and a four-hour limit. The verified
H100 development submission overrides this to 24 CPU cores and two hours:

```bash
# Export these same cache paths before preparing assets and submitting.
export HF_HOME="$PWD/nut_agent/nutricoach/food_vision/data/huggingface"
export UV_CACHE_DIR="$PWD/nut_agent/nutricoach/food_vision/data/uv_cache"
mkdir -p nut_agent/nutricoach/food_vision/results/slurm
PYTHONUNBUFFERED=1 sbatch --account=mbm@h100 --partition=gpu_p6 --constraint=h100 \
    --qos=qos_gpu_h100-dev --time=02:00:00 --cpus-per-task=24 \
    --output=nut_agent/nutricoach/food_vision/results/slurm/nutricoach-n5k-%j.log \
    nut_agent/nutricoach/food_vision/jobs/nutrition5k_jean_zay.sbatch all
```

Job `798727` was submitted on 2026-10-08 at 20:38 Paris time. Its initial queue
reason was `QOSGrpGRES`, indicating the development QoS GPU capacity was occupied.
At 20:40 it was running on `jzxh001`, with CUDA feature encoding confirmed in the log.
It completed at 20:43:17 Paris time with exit code `0:0`; all 50 dishes succeeded
for each local method. Total elapsed time was 3 minutes 30 seconds.
Preparation cached 50 test images, exactly matching the checked-in Dots manifest,
2,755 training/validation images (three HTTP 404 exclusions), DINOv2 weights and
the pinned Food-R1 checkpoint. Logs use `results/slurm/nutricoach-n5k-798727.log`;
method outputs use `results/nutrition5k_50_rgb` and `results/nutrition5k_50_food_r1`.
Inspect the completed job with `sacct -j 798727`.

API evaluations run separately on a network-enabled host.
Copy resulting method directories back and
regenerate the report; it rejects mismatched test manifests and API model IDs.

## Validation in this checkout

For the FoodLMM precision changes, **17 focused tests passed**, followed by
**176 passed and 4 skipped** across NutriCoach with GPU integration modules
excluded. The skipped tests require a live OpenRouter key. This includes both
precision rows, paired comparison signs, resource reporting, device selection and
Streamlit checks. Small CPU fixtures verified exact BF16 quantized restoration
and finite FP16 restoration while leaving the output head in floating point.
These fixtures do not validate actual MPS execution.
H100 cache checks reproduced all five saved predictions exactly on three dishes
for each precision. Full 500-dish profiling is tracked in the generated report.
DPF resource job `1005765` completed on 2026-10-13: all 500 predictions matched
the previous run exactly, with 0.69 GiB peak allocated GPU memory, 0.82 GiB peak
reserved memory, 1.34 GiB peak host RSS and 0.10 seconds mean inference latency.

After geometry removal on 2026-10-12, **50 focused tests passed**, followed by
**171 passed and 7 skipped** across NutriCoach with live hosted and GPU integration
modules excluded. This includes Streamlit checks and rejection of geometry by
the remaining method registry. Both current 500-dish JSON/Markdown reports were
regenerated and verified to contain exactly five methods without geometry.
Slurm shell syntax and `git diff --check` passed. The earlier H100 counts below
describe validation before geometry removal; GPU inference was not rerun for
this deletion.

H100 job `819312` completed on 2026-10-09 with exit `0:0` in 2 minutes 35 seconds:
**191 tests passed and 11 skipped** across the complete NutriCoach suite,
including real Qwen single-shot photo inference, geometry inference, chat tool
execution and the existing recipe GPU embedder tests. Skips cover absent recipe
checkpoints/model registry resources and live hosted tests without credentials.
Accelerate 1.12.0 and bitsandbytes 0.50.2 were installed in the existing VM `uv`
environment. The root Transformers pin and required tokenizer/safetensors pins
were synchronized with the supported local backend.

On 2026-10-12, all **12 provider tests passed**, including Streamlit provider
selection, local tool parsing, credential isolation and releasing earlier weights
when changing local precision. Job `963878` completed the separate 4-bit/32 GiB
CUDA memory check on 2026-10-12 with **1 test passed** in 5 minutes 32 seconds.
Real photo inference used a peak of **18.52 GiB reserved GPU memory**, but peak
host RSS during loading was **52.12 GiB**. This validates a CUDA allocation below
32 GiB on H100, not a machine with only 32 GB system RAM. Loader/inference elapsed
time was 305.6 seconds, including initialization; it is not steady-state latency.
Job `964118` completed the full H100 suite after this memory fix: **192 tests
passed, 12 skipped**, exit `0:0`, 2 minutes 7 seconds overall. The additional
skip is the separately validated opt-in quantized memory test.
Job `964397` reached its eight-hour limit after completing Qwen one-shot on
**500/500 dishes: 94.7 kcal MAE**. RGB regression and Food-R1 score 75.8 and
77.8 kcal respectively on the same 500 dishes, beating this baseline. FoodLMM
scores 49.0 kcal on 495 successful dishes; the provisional third-party DPF
reproduction scores 47.0 kcal on 489 successful dishes.
The geometry implementation, preparation option, benchmark registration and
its dedicated tests were removed on 2026-10-12 at the user's request. Resume
job `1002790` was cancelled. Current reports exclude geometry; raw predictions
and earlier validation logs remain historical records.
See [Qwen benchmark results](nutrition5k_500_qwen_results.md) and
[local backend validation](qwen_local_validation.md) for measured tests and memory.

On 2026-10-09, 66 focused tests passed across food vision, nutrition matching,
geometry, official split/evaluation, supervised adapters and agent integration.
This includes 500-dish selection, paper-report checks and rejection of removed methods.
All 14 photo-review tests passed, including Streamlit AppTest, after warming
imports to avoid a three-second startup timeout on the shared filesystem.
FoodLMM's actual author model class and nutritional tokenizer imported in its
isolated environment; all DPF checkpoint keys matched the pinned release.
The full 500-dish RGB, Food-R1, FoodLMM and DPF runs exercised real H100 inference.
Repeated paper-asset preparation succeeded completely offline.

Jean Zay preflight on 2026-10-08 passed all eight tests in
`nut_agent/tests/test_supervised_food_vision.py`, an offline Food-R1 processor and
Qwen3-VL model-class import check, Slurm shell syntax and scheduler submission
validation. A development allocation confirmed an NVIDIA H100 80 GB.
Repeated training-data preparation passed offline, and job `798727` began CUDA
feature encoding successfully.
The environment uses torch `2.6.0+cu124`, torchvision `0.21.0+cu124`, timm `1.0.9`
and Transformers `5.17.0`. The full GPU job completed RGB regression training
and both local 50-dish evaluations successfully; results are recorded above.
On 2026-10-09, report validation confirmed matching manifests, the checkpoint
hash and official split isolation, and recomputed metrics matched both saved
run summaries. All three targeted snapshot/report/comparison tests passed.

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
full regression training/evaluation subsequently completed in H100 job `798727`.

## Removed pipelines

RF-DETR plus serving priors, CLIP candidates plus VLM, the serving-range RAG
pipeline and the standalone database-grounded VLM have been deleted with their
dedicated training/preparation utilities and dependencies. Nutrition matching is
retained for user corrections in NutriCoach. The geometry hybrid and its tests
have also been deleted and removed from the active benchmark. Historical raw
predictions and earlier reports remain archival records.
The local nutrition table remains a small 130+ food reference table, not full FNDDS.
