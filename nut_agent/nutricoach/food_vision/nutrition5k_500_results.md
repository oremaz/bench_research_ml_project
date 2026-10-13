# Nutrition5k: current RGB-only benchmark

All methods use the same frozen 500-dish official-test RGB subset. Pending methods have no measured score.

| Method | Successes / 500 | Calories MAE (kcal) | Mass MAE (g) | Fat MAE (g) | Carbs MAE (g) | Protein MAE (g) |
|---|---:|---:|---:|---:|---:|---:|
| vlm_single | 41 / 500 | 97.6 | 61.2 | 6.3 | 10.0 | 6.7 |
| rgb_regression | 500 / 500 | 75.8 | 50.1 | 5.6 | 7.3 | 7.6 |
| food_r1 | 500 / 500 | 77.8 | 79.6 | 5.0 | 7.3 | 5.6 |
| FoodLMM (16-bit) | 495 / 500 | 49.0 | 32.0 | 3.7 | 5.3 | 4.7 |
| FoodLMM (16-bit, cached) | 496 / 500 | 49.0 | 31.9 | 3.7 | 5.3 | 4.7 |
| FoodLMM (4-bit decoder) | 474 / 500 | 47.9 | 31.2 | 3.6 | 5.3 | 4.5 |
| dpf_reproduction | 489 / 500 | 47.0 | 31.2 | 3.6 | 5.4 | 4.6 |

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

VLM provider: `openrouter`; model: `dots-studio/dots-3-note-preview:free`; reasoning: `high`.

## Run coverage

- `vlm_single`: 49 attempted, 8 failed, 451 not attempted, 0 fallback predictions; stopped at free API daily quota.
- `rgb_regression`: 500 attempted, 0 failed, 0 not attempted, 0 fallback predictions; all dishes attempted.
- `food_r1`: 500 attempted, 0 failed, 0 not attempted, 0 fallback predictions; all dishes attempted.
- `foodlmm`: 500 attempted, 5 failed, 0 not attempted, 0 fallback predictions; all dishes attempted.
- `foodlmm_cached`: 500 attempted, 4 failed, 0 not attempted, 0 fallback predictions; all dishes attempted.
- `foodlmm_4bit`: 500 attempted, 26 failed, 0 not attempted, 0 fallback predictions; all dishes attempted.
- `dpf_reproduction`: 500 attempted, 11 failed, 0 not attempted, 0 fallback predictions; all dishes attempted.

Partial runs must be compared on common successful dishes. An incomplete one-shot baseline
cannot establish a winner for the full 500-dish evaluation.

FoodLMM uses the author's Chat checkpoint; DPF uses a third-party reproduction with
Depth Anything V2 and modified fusion. See [release coverage and limitations](README.md#additional-paper-release-coverage).

## Paired calorie comparisons against one-shot

Differences are method MAE minus one-shot MAE on the same successful dishes; negative is better.
Comparisons with missing or failed predictions apply only to the common successful subset.

| Method | Common dishes | MAE difference (kcal) | Dish-bootstrap 95% interval (kcal) |
|---|---:|---:|---:|
| rgb_regression | 41 | -10.0 | [-49.2, 33.5] |
| food_r1 | 41 | -13.8 | [-55.5, 30.0] |
| foodlmm | 40 | -41.6 | [-78.2, -6.2] |
| foodlmm_cached | 40 | -41.6 | [-79.2, -6.1] |
| foodlmm_4bit | 38 | -20.3 | [-53.7, 12.6] |
| dpf_reproduction | 40 | -49.0 | [-84.9, -13.1] |

## FoodLMM precision comparison

Differences are 4-bit MAE minus `foodlmm_cached` MAE on common successful dishes; positive is worse.

| Target | Common dishes | MAE difference | Dish-bootstrap 95% interval |
|---|---:|---:|---:|
| calories | 472 | 2.00 | [0.41, 3.51] |
| mass_g | 472 | 0.76 | [-0.24, 1.72] |
| fat_g | 472 | 0.12 | [0.04, 0.20] |
| carbs_g | 472 | 0.35 | [0.22, 0.49] |
| protein_g | 472 | 0.14 | [0.01, 0.28] |

The paired calorie MAE change is +2.00 kcal. The 4-bit run has 26/500 failures; its 16-bit control has 4/500.
Method rows use different successful subsets; the precision comparison uses matched dishes.
