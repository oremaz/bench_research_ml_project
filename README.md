# ML Research Portfolio

The projects in this former combined workspace now have three independent repositories:

| Repository | Focus |
|---|---|
| [ML Playground](https://github.com/oremaz/ml-playground) | Reusable `ml_pipeline` Python package, model benchmarking, augmentation, checkpointing, and tabular experiments |
| [NutriCoach](https://github.com/oremaz/nutricoach) | Nutrition agent, Recipe Lab, recipe training, and food-vision experiments |
| [AI Content Robustness](https://github.com/oremaz/ai-content-robustness) | Text/image detection, detector training notebooks, adversarial evasion, and robustness evaluation |

NutriCoach and AI Content Robustness install `ml_pipeline` from a pinned ML Playground Git commit. Each project has its own dependency specifications, documentation, and tests. Their relevant histories were preserved, including the earlier `food_preds` framework paths. The split includes working changes present on 2026-10-14.

This repository is now a small index. Its earlier combined source remains in Git history. On the research VM, the extracted repositories are in `split-repos/`; the original working folders, datasets, checkpoints, and private user data remain locally available and are ignored by this index.

Validated during the split: 78 ML framework tests, 204 NutriCoach tests, and 173 detector tests passed. Five tests skipped, and the GPU recipe, live OpenRouter, and live local-Qwen test modules were excluded. Framework installation from the published Git commit and application startup smoke checks passed. GPU training and full real-model inference were not rerun. See each project's README for commands and validation details.
