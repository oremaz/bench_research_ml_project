# Nutrition5k: current RGB-only benchmark

All methods use the same frozen 50-dish official-test RGB subset. Pending methods have no measured score.

| Method | Successes / 50 | Calories MAE (kcal) | Mass MAE (g) | Fat MAE (g) | Carbs MAE (g) | Protein MAE (g) |
|---|---:|---:|---:|---:|---:|---:|
| vlm_single | 50 / 50 | 97.1 | 73.8 | 6.5 | 8.0 | 8.2 |
| geometry_vlm_db | 0 / 50 | pending | pending | pending | pending | pending |
| rgb_regression | 0 / 50 | pending | pending | pending | pending | pending |
| food_r1 | 0 / 50 | pending | pending | pending | pending | pending |

MAE is computed on successful predictions; consult JSON for failures, missing dishes, signed errors,
normalized MAE, RMSE, R2, dish-bootstrap confidence intervals and paired comparisons for all five targets.
Oracle-mass errors are post-hoc diagnostics using reference masses, never deployable RGB-only scores.

Configurations and original model IDs are preserved in the JSON. Published paper scores are not local measurements.
