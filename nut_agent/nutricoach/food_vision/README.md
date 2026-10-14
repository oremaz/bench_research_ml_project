# Food Vision

This benchmark compares five approaches for estimating whole-dish calories,
mass, fat, carbohydrate and protein from one RGB photo. It measures performance
on Nutrition5k; accuracy on web or phone photos requires a separate evaluation.

Every method is scored on the same **five targets**: total energy in kcal,
food mass in grams, and grams of fat, carbohydrate and protein. There are five
outputs because Nutrition5k provides these five reference quantities and the
benchmark evaluates each separately.

## Methods in practice

### Single-shot VLM (`vlm_single`)

1. **Provide the photo and instructions.** The prompt asks a vision-language
   model (VLM) to identify foods and estimate their portions and nutrients.
2. **Generate food-level estimates.** The response is a JSON list containing
   names, grams, calories and macronutrients for each proposed food. The model
   infers these quantities from the image and its learned nutritional knowledge.
3. **Aggregate the meal.** The adapter sums the food-level quantities to obtain
   whole-dish predictions comparable to Nutrition5k's labels. An omitted food
   also contributes nothing to those totals.

The entire prediction uses one generation request. The same pipeline supports
hosted OpenRouter/Groq models and local Qwen.

**Training:** this pipeline uses existing pretrained checkpoints as supplied.
Implementation: [vlm_analyzer.py](vlm_analyzer.py).

### RGB regression (`rgb_regression`)

1. **Extract visual features.** Frozen DINOv2 ViT-S/14 encodes the photo into a
   feature vector representing the dish's appearance.
2. **Predict nutrition.** A ridge regression head trained on these features
   predicts the five whole-dish targets. Features and targets are standardized
   during fitting; predictions are converted back to kcal and grams.

**Training:** DINOv2 has self-supervised visual pretraining on
142 million images; its weights remain frozen here. Only the regression head
is fitted on **Nutrition5k official training images**. Within this training set,
images captured on 20% of the acquisition days form the validation set. Grouping
by day helps keep related scans in the same split. Normalization parameters are
estimated on the remaining training images; the ridge regularization strength
is chosen on the validation set. The official test set is reserved for evaluation.
Sources: [DINOv2](https://github.com/facebookresearch/dinov2),
[local trainer](train_rgb_regression.py).

### Food-R1 (`food_r1`)

1. **Generate nutrition estimates.** The photo and a whole-dish instruction
   prompt the specialized Qwen3-VL-8B model to generate the five totals as text.
2. **Parse the answer.** The adapter extracts the benchmark fields from a fixed
   answer template and rejects missing, ambiguous or invalid values.

**Training sequence: SFT, then GRPO.** The paper describes supervised training
on **CalorieBench-80K, Food-101, VireoFood-172, Recipe1M, Nutrition5k and
FoodDialogues**. CalorieBench-80K is curated from MM-Food-100K; a 5K training
subset includes GPT-4o-generated reasoning examples.

**SFT: learn reference responses through next-token prediction.** Each example
contains an image, a task prompt and a target response. Qwen3-VL-8B is fine-tuned
with token-level cross-entropy on the response, conditioned on the image, prompt
and preceding reference tokens. The published SFT script updates the language
model, vision encoder and multimodal aligner.

For an illustrative meal labelled 250 g and 400 kcal, the response contains
these values as text. At the prefix `...provides about`, training supervises the
tokens encoding `400`, followed by `kcal`, and similarly supervises the mass and
macronutrient fields. This teaches the model to generate nutritional answers
from image-and-instruction context. Source: [SFT script](https://github.com/hustvl/Food-R1/blob/main/scripts/train_sft_megatron.sh).

**GRPO: optimize sampled answers against task rewards.** Starting from the SFT
checkpoint, the model samples **eight completions per image/prompt**. Applicable
reward components are combined with equal weights. GRPO uses each completion's
reward relative to the group to favour higher-scoring responses and suppress
lower-scoring ones. The paper also includes a KL penalty relative to a frozen
SFT reference model. Open-ended recipe, advice and dialogue tasks are excluded
from this stage.

For CalorieBench-80K, the [published reward code](https://github.com/hustvl/Food-R1/blob/main/foodr1/rewards.py)
implements:

| Component | Reward rule |
|---|---|
| Ingredients | Jaccard overlap between predicted and reference ingredient sets. Missing or extra names reduce overlap. |
| Quantities | For matched ingredient names and units, average `2 / (1 + exp(abs(q_pred - q_ref)))`. Exact quantities score 1; larger discrepancies reduce the score. |
| Total calories | `TotalKcalExactMatchReward`: 1 when parsed kcal equals reference kcal exactly; otherwise 0. A 1 kcal error receives 0 for this component. |
| Format | Requires a nonempty `<think>` before `<answer>` and the expected answer structure. The score also measures agreement between the declared and listed ingredient counts. |

We use the released model for inference. Sources: [full paper and supplement,
Sections 4.3–5.1 and Table S1](https://arxiv.org/pdf/2606.04986),
[GRPO script](https://github.com/hustvl/Food-R1/blob/main/scripts/train_grpo_caloriebench80k.sh),
[checkpoint](https://huggingface.co/zy12123/Food-R1),
[local adapter](supervised_analyzers.py).

### FoodLMM (`foodlmm`, `foodlmm_cached`, `foodlmm_4bit`)

1. **Encode the photo with CLIP.** Its visual features supply image information
   to the LISA/LLaVA 7B language model, alongside the whole-dish instruction.
2. **Generate markers for the nutritional outputs.** The vocabulary includes
   dedicated tokens such as `[MASS_TOTAL]` and `[CAL_TOTAL]`. When generating
   `[MASS_TOTAL]`, the decoder produces a hidden vector conditioned on the photo,
   instruction and preceding text. The marker identifies this generation step
   as the one from which to read the mass prediction.
3. **Read the numerical prediction from a regression head.** The adapter takes
   that hidden vector, applies a learned projection and passes it to the mass
   head, which outputs a value in grams. The other markers select the calorie,
   fat, carbohydrate and protein heads in the same way.

For example, immediately before emitting `[MASS_TOTAL]`, the language model's
Transformer decoder computes a hidden state `h` from the image, prompt and
preceding tokens. The language head uses `h` to predict `[MASS_TOTAL]`.
The adapter retrieves this same `h`, applies the learned projection, and feeds
the resulting vector to the mass regression head, which computes a continuous
value in grams, such as **250 g**. A different photo changes `h` and therefore
the predicted mass. During training, reference
responses supervise the marker tokens and measured nutritional labels supervise
the regression heads. A 250 g training dish therefore supplies both a token
target and a numerical target. The author's alignment uses the hidden state
that predicts the marker, immediately before the marker in the token sequence.
See [the author model's token masks and heads](https://github.com/YuehaoYin/FoodLMM/blob/b1c52d8ea94ef19e4801dbb5a13673732bafce27/model/LISA.py).

Our runner uses the authors' **FoodLMM-Chat** checkpoint and omits its segmentation
branch to run the whole-dish nutrition task.

**Declared training:** supervised adaptation with **LoRA** and trainable task
heads. Stage 1 combines
**VIREO Food-172, Recipe1M, Nutrition5k, FoodSeg103 and UECFoodPixComplete**.
Stage 2 adds **FoodDialogues and FoodReasonSeg**, generated using GPT-4 from
Nutrition5k and FoodSeg103, alongside the task datasets. Text prediction,
segmentation and nutrition losses are jointly optimized; the nutrition heads
learn from absolute errors plus a small squared-error term against dish- and
ingredient-level labels.
Sources: [author code and training configs](https://github.com/YuehaoYin/FoodLMM),
[Chat checkpoint](https://huggingface.co/Yueha0/FoodLMM-Chat).

The three report rows use the same learned checkpoint. `foodlmm` retains the
original 16-bit run; `foodlmm_cached` generates the response using a KV cache,
then runs an additional forward pass over the complete sequence to obtain the
hidden states used by the nutrition heads.
`foodlmm_4bit` follows the latter
procedure with Quanto qint4 decoder weights; the visual encoder and nutrition
heads remain in floating point. Implementation: [evaluate_paper_system.py](evaluate_paper_system.py).

### DPF reproduction (`dpf_reproduction`)

1. **Estimate relief with Depth Anything V2.** This frozen model predicts a
   relative-depth map from the RGB photo. The map adds a cue about visible shape
   and relief. Its values describe relative depth within the image.
2. **Combine appearance and relief.** Two network streams encode the RGB image
   and predicted depth. Their fusion lets nutritional prediction use both
   food-related appearance and geometric cues.
3. **Regress the five meal totals.** Heads map the fused representation directly
   to mass, calories and the three macronutrient quantities, using relationships
   learned from labelled dishes. Each quantity is a supervised prediction.

DPF learns associations between visual characteristics and nutritional values.
Its depth input is derived from the same photo and inherits its scale ambiguity.

**Declared training:** a **Food2K-pretrained ResNet-101** backbone is adapted
for supervised nutrition regression on **Nutrition5k**. The reproduction
combines the five absolute-error losses through their geometric mean, training
predictions against known mass and nutrient totals. We load its released weights.
The complete checkpoint is loaded with `strict=True`, including both backbones
and the nutrition heads. Separate Food2K initialization is unnecessary for
inference. The adapter skips the constructor's initialization-only fallback
message and confirms successful checkpoint loading instead; missing checkpoint
keys cause loading to fail rather than leave randomly initialized parameters.
This **third-party reproduction** adapts DPF with a different depth predictor
and modified fusion.
Sources: [reproduction and training description](https://github.com/T0MYYY/Nutrition5k/tree/ed37fbefd5582e7963c08c2815761ab4566cd838),
[checkpoint](https://huggingface.co/T0MYYY/dpf-nutrition),
[adapter](evaluate_paper_system.py).
Because this method is not trained on a super big food dataset, a lack of generalisation over web or manual images is observed in practice, despite the good results on Nutrition5k test set.

## Evaluation and interpretation

The frozen subset contains **500 dishes**, selected with seed 42 from the
intersection of Nutrition5k's official RGB and depth test IDs. Each method
receives one overhead RGB image; measured depth, true mass and ingredient
annotations are withheld. Source and image hashes identify the shared inputs.

| Report field | Meaning |
|---|---|
| MAE | Mean absolute error; kcal for energy, grams for mass and macros |
| Normalized MAE | MAE divided by the mean reference value for that target |
| RMSE / R² | Additional measures of error magnitude and explained variance |
| Bootstrap interval | Sampling variability across dishes, not uncertainty for an individual photo |

Errors use successful predictions only, so rows with different success counts
can describe different dish subsets. Paired comparisons use common successes.
The optional oracle-mass diagnostic rescales nutrients using true mass **during
scoring only**; it is not deployable RGB-only performance.

The locally fitted RGB head has recorded training/validation IDs. External model
training is described by its publishers: we cannot independently verify the
complete training membership of the released weights or API models. In
particular, the DPF publisher reports correcting an earlier split leakage issue,
but its checkpoint does not contain the fitted dish IDs needed to audit that fix.
Training on Nutrition5k and evaluating its held-out test split is legitimate;
using the same dataset name alone does not imply leakage.

A single photo leaves scale, hidden ingredients and added oil ambiguous.
Results on controlled cafeteria images should not be treated as evidence of
reliable portions on web photos. These adapters and inputs also differ from
some paper settings; published paper scores are not substituted for our runs.

## Results

[The 500-dish Qwen comparison](nutrition5k_500_qwen_results.md) includes the local
Qwen single-shot baseline, RGB regression, Food-R1, FoodLMM precision variants
and DPF. The report generator writes `nutrition5k_500_qwen_results.json` locally
with predictions, failures and configurations.

## Run the benchmark

From the repository root, use the main `uv` environment for RGB, VLM, Food-R1
and DPF. FoodLMM requires its separate environment below. Models, releases and
datasets need internet access on first download; `HF_HOME` selects the model cache.

```bash
PYTHONPATH=. uv run python nut_agent/nutricoach/food_vision/evaluate_nutrition5k.py --prepare-only
PYTHONPATH=. uv run python nut_agent/nutricoach/food_vision/train_rgb_regression.py --device cuda
PYTHONPATH=. uv run python nut_agent/nutricoach/food_vision/evaluate_nutrition5k.py \
    --methods rgb_regression food_r1 --device cuda --output-dir /path/to/local-run
PYTHONPATH=. uv run python nut_agent/nutricoach/food_vision/evaluate_nutrition5k.py \
    --methods vlm_single --provider local --model Qwen/Qwen3.8-27B \
    --reasoning-effort none --workers 1 --device cuda --output-dir /path/to/qwen-run
PYTHONPATH=. uv run python nut_agent/nutricoach/food_vision/evaluate_paper_system.py \
    --method dpf_reproduction --device auto --output-dir /path/to/dpf-run
PYTHONPATH=. uv run python nut_agent/nutricoach/food_vision/report_nutrition5k.py \
    --model Qwen/Qwen3.8-27B \
    --run-dir /path/to/qwen-run /path/to/local-run /path/to/dpf-run /path/to/foodlmm-run
```

For hosted VLM inference, choose `--provider openrouter` or `groq`, an
image-capable `--model`, and configure the corresponding API key. Local Qwen
BF16 needs roughly 58 GB for weights alone; `--local-quantization 4bit` uses
bitsandbytes on CUDA. Food-R1's checkpoint is roughly 17.5 GB before runtime
memory. These memory requirements are separate from FoodLMM's quantization.

Runs save predictions incrementally and resume completed dishes; `--retry-errors`
retries failures. Use a new output directory when code or configuration changes.
For offline GPU jobs, prepare assets with [prepare_benchmark_assets.py](prepare_benchmark_assets.py)
and submit [nutrition5k_jean_zay.sbatch](jobs/nutrition5k_jean_zay.sbatch) using
an account and GPU allocation appropriate to your site.

## FoodLMM in NutriCoach

NutriCoach's Food Analysis tab offers FoodLMM 4-bit (default) and DPF reproduction,
independently of the chat provider. DPF uses the existing RGB and predicted-depth
adapter in the main environment, downloads its assets on first use, and reuses
its loaded model. Switching methods clears the previous result.
The adapters return whole-dish totals; users can review,
correct and explicitly save them to the nutrition journal.

FoodLMM runs in an isolated Python 3.10 environment because its Transformers
version conflicts with the main app. Its worker adds the environment’s executable
directory to PATH so PyTorch can find Ninja for native extensions. From the repository root:

```bash
uv venv --python 3.10 nut_agent/nutricoach/food_vision/data/foodlmm_env
uv pip install --python nut_agent/nutricoach/food_vision/data/foodlmm_env/bin/python \
    -r requirements_foodlmm.in
```

On CUDA, install matching CUDA Torch/Torchvision wheels in that environment.
The app finds it automatically; `FOODLMM_PYTHON` selects another interpreter
and `FOODLMM_DEVICE` selects `auto`, `cuda`, `mps` or `cpu`. Auto prefers CUDA,
then MPS, then CPU. Git and Hugging Face access are needed for initial assets.

### FoodLMM 4-bit and Apple Silicon

The app loads an existing packed export from `data/foodlmm_4bit`, or from
`FOODLMM_QUANTIZED_CHECKPOINT`. Without one, it loads and quantizes the original
weights, requiring more loading memory, then automatically saves a packed export
to `data/foodlmm_4bit` for future starts. The terminal reports loading, export,
and model readiness. Both export files must be present to reuse the checkpoint.
FoodLMM prediction uses `torch.no_grad()` to preserve the tensor version counters
required by Quanto. On MPS, the runner bypasses Quanto 0.2.7's native unpacker
using tensor operations with the same packed value ordering. The native unpacker
recompiles Metal shaders and synchronizes the GPU on every call. A small
quantized MPS linear layer produced matching outputs with the replacement;
full-model MPS latency remains unvalidated. Generation stops after all five
whole-dish nutrition tokens appear, or at EOS/the 256-token limit. The full-prefix
nutrition-head pass is retained; causal states preceding the required tokens do
not depend on discarded trailing text. Generation and nutrition-head timings
are reported separately in the terminal. Existing benchmark reports predate this
early stopping change and have not been rerun. Prepare a packed export on a machine
with sufficient memory and copy it to the target machine:

```bash
PYTHONPATH=. nut_agent/nutricoach/food_vision/data/foodlmm_env/bin/python \
    nut_agent/nutricoach/food_vision/evaluate_paper_system.py \
    --method foodlmm --device cuda --quantization 4bit --prepare-only \
    --export-quantized nut_agent/nutricoach/food_vision/data/foodlmm_4bit \
    --output-dir /path/to/foodlmm-prepare
PYTHONPATH=. nut_agent/nutricoach/food_vision/data/foodlmm_env/bin/python \
    nut_agent/nutricoach/food_vision/evaluate_paper_system.py \
    --method foodlmm --device auto --quantization 4bit \
    --quantized-checkpoint nut_agent/nutricoach/food_vision/data/foodlmm_4bit \
    --output-dir /path/to/foodlmm-run
```

The benchmark runner also accepts `--quantization none` for the 16-bit control.
The app prepares its CLIP assets automatically; the benchmark needs them cached
first with `prepare_benchmark_assets.py --papers`. On macOS, Quanto's MPS kernels
require Xcode command line tools. CUDA memory/latency results do not establish
Apple Silicon performance; real FoodLMM MPS inference remains unvalidated here.
