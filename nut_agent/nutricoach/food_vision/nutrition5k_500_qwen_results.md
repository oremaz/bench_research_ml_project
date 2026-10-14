# Nutrition5k: current RGB-only benchmark

All methods use the same frozen 500-dish official-test RGB subset. Pending methods have no measured score.

| Method | Calories MAE (kcal) | Mass MAE (g) | Fat MAE (g) | Carbs MAE (g) | Protein MAE (g) |
|---|---:|---:|---:|---:|---:|
| vlm_single | 94.7 | 87.7 | 6.7 | 10.6 | 8.4 |
| rgb_regression | 75.8 | 50.1 | 5.6 | 7.3 | 7.6 |
| food_r1 | 77.8 | 79.6 | 5.0 | 7.3 | 5.6 |
| FoodLMM (16-bit) | 49.0 | 32.0 | 3.7 | 5.3 | 4.7 |
| FoodLMM (16-bit, cached) | 49.0 | 31.9 | 3.7 | 5.3 | 4.7 |
| FoodLMM (4-bit decoder) | 47.9 | 31.2 | 3.6 | 5.3 | 4.5 |
| dpf_reproduction | 47.0 | 31.2 | 3.6 | 5.4 | 4.6 |

MAE is computed on successful predictions; consult JSON for failures, missing dishes, signed errors,
normalized MAE, RMSE, R2, dish-bootstrap confidence intervals and paired comparisons for all five targets.
Oracle-mass errors are post-hoc diagnostics using reference masses, never deployable RGB-only scores.

Configurations and original model IDs are preserved in the JSON. Published paper scores are not local measurements.

The FoodLMM 4-bit variant quantizes decoder linear weights with Quanto qint4;
vision, embeddings, language output and nutrition heads remain at 16-bit precision on GPU.
The original 16-bit result is retained. The cached 16-bit control and 4-bit variant
use the same cached generation and full-prefix nutrition-head pass.

## Measured compute resources

GPU peaks include model weights. Loading and inference peaks are recorded separately;
host RSS is the process lifetime peak. Latency uses device synchronization and excludes model loading.

| Method | Device | Mean inference (s) | Inference peak allocated (GiB) | Inference peak reserved (GiB) | Loading peak allocated (GiB) | Peak host RSS (GiB) |
|---|---|---:|---:|---:|---:|---:|
| foodlmm_cached | cuda | 0.75 | 13.72 | 13.80 | 13.19 | 15.14 |
| foodlmm_4bit | cuda | 1.18 | 4.97 | 5.71 | 7.91 | 7.42 |
| dpf_reproduction | cuda | 0.10 | 0.69 | 0.82 | 0.44 | 1.34 |

VLM provider: `local`; model: `Qwen/Qwen3.8-27B`; reasoning: `none`.
Local revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; quantization: `none`.

FoodLMM uses the author's Chat checkpoint; DPF uses a third-party reproduction with
Depth Anything V2 and modified fusion. See [release coverage and limitations](README.md#methods-in-practice).
