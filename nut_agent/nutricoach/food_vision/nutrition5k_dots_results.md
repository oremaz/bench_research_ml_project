# Nutrition5k: current RGB-only benchmark

All methods use the same frozen 50-dish official-test RGB subset. Pending methods have no measured score.

| Method | Successes / 50 | Calories MAE (kcal) | Mass MAE (g) | Fat MAE (g) | Carbs MAE (g) | Protein MAE (g) |
|---|---:|---:|---:|---:|---:|---:|
| vlm_single | 50 / 50 | 97.1 | 73.8 | 6.5 | 8.0 | 8.2 |
| geometry_vlm_db | 0 / 50 | pending | pending | pending | pending | pending |
| rgb_regression | 50 / 50 | 101.4 | 60.2 | 6.9 | 7.8 | 9.5 |
| food_r1 | 50 / 50 | 109.2 | 96.7 | 6.7 | 7.0 | 7.1 |

MAE is computed on successful predictions; consult JSON for failures, missing dishes, signed errors,
normalized MAE, RMSE, R2, dish-bootstrap confidence intervals and paired comparisons for all five targets.
Oracle-mass errors are post-hoc diagnostics using reference masses, never deployable RGB-only scores.

Configurations and original model IDs are preserved in the JSON. Published paper scores are not local measurements.

## Completed H100 development run

Job `798727` completed successfully on 2026-10-08, from 20:39:47 to 20:43:17
Paris time (3 minutes 30 seconds), with Slurm state `COMPLETED` and exit code `0:0`.
It used one NVIDIA H100 80 GB on `jzxh001`, 24 CPU cores and `qos_gpu_h100-dev`.
Both local methods completed all 50 frozen test dishes with zero failures,
zero fallback predictions and no API requests. The test manifest matches the Dots baseline.

The frozen DINOv2 RGB baseline fitted its ridge head on 2,250 training dishes
and selected `alpha=1000` using 505 validation dishes grouped by UTC acquisition
day, with seed 42. Three unavailable official-training images were excluded
because the public source returned HTTP 404. No test dishes entered fitting or selection.
Food-R1 used pinned revision `c70e0d6585b1e81923432df46014d6ce32855e3f`,
greedy decoding, a 512-new-token limit and an image-side limit of 1024 pixels.

Mean recorded inference time was 0.026 seconds per dish for RGB regression and
1.229 seconds for Food-R1. Recorded initialization took 5.785 and 53.363 seconds,
respectively. These timings exclude preparation and are not a controlled speed benchmark.

Relative to the Dots baseline, RGB regression had lower mass MAE, while Food-R1
had lower carbohydrate and protein MAE. Dots retained the lowest calorie and fat
MAE. These are observed errors on a 50-dish overhead-image pilot, not evidence of
smartphone-photo generalization; paired dish-bootstrap intervals are in the JSON.
The geometry/API evaluation remains pending.

Run artifacts:

- [Slurm log](results/slurm/nutricoach-n5k-798727.log)
- [RGB regression summary](results/nutrition5k_50_rgb/summary.json)
- [Food-R1 summary](results/nutrition5k_50_food_r1/summary.json)
- [RGB training provenance](results/rgb_regression/checkpoint.json)
- [Combined recomputable snapshot](nutrition5k_dots_results.json)

Full run artifacts live in ignored `results/`; the combined Markdown and compact
JSON snapshot preserve the measured comparison in the repository.
