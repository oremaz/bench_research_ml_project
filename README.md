# Modular ML Pipeline for Multi-Modal Classification (AI-Generated Content Detection)

Modular platform for ML research across multiple domains: food prediction (recipe difficulty, meal type, nutrients, time) and AI-generated content detection (deepfakes, image authenticity). Integrates advanced augmentation, unified pipelines, automated benchmarking, third-party research code (FatFormer, DiffusionFake, TabR, GRANDE, TabM), two standalone apps (a conversational nutritionist agent and a recipe analysis lab), and a research framework for AI-content detection and adversarial evasion across text and image.

## Project Structure

- **ml_pipeline/**: Core ML experimentation suite (augmentation, pipelines, benchmarking, reproducibility, results). See `ml_pipeline/README.md`
- **nut_agent/**: Two Streamlit apps: **NutriCoach** (LLM-powered nutritionist agent) and **Recipe Lab** (ML-powered recipe analyzer). See `nut_agent/README.md`
- **ai_content_detector/**: Research framework for AI-content detection and adversarial evasion across text and image. Provides an ensemble of zero-shot and supervised detectors, RL-based evasion trainers (GRPO, MultiSPIN, DDPO), a multi-round attacker/defender arms race, and a Streamlit scoring app. Reuses `ml_pipeline/` checkpoints and wrappers. See `ai_content_detector/README.md`

## Quickstart

1. Install dependencies: `uv pip install -r requirements.txt` for the standard environment, or `uv pip install -r requirements_gpu.in` on the CUDA 12.4 GPU VM used for notebook training.
2. Interactive Benchmark UI: `cd ml_pipeline && streamlit run benchmark_app.py` (with OpenMP configured)
3. Launch NutriCoach: `cd nut_agent && streamlit run nutricoach/app.py`
4. Launch Recipe Lab: `cd nut_agent && streamlit run recipe_lab/app.py`
5. Launch AI Content Detector: `streamlit run ai_content_detector/app.py`

## Benchmark App (Streamlit)

Interactive UI for model benchmarking with registry-based configuration:
- **Launch**: `cd ml_pipeline && streamlit run benchmark_app.py`
- **Features**: Upload CSV datasets, select models/augmentations from registries, configure training parameters, visualize results
- **Location**: `ml_pipeline/benchmark_app.py`

**macOS Setup** (for XGBoost): Install OpenMP via MacPorts (`sudo port install libomp`) and set `DYLD_LIBRARY_PATH` permanently in your shell config.

Example (bash):
```
echo 'export DYLD_LIBRARY_PATH="/opt/local/lib/libomp:$DYLD_LIBRARY_PATH"' >> ~/.bash_profile
source ~/.bash_profile
```

## Testing

```bash
# ml_pipeline (73 tests: scoring, wrappers, pipelines, checkpoints, benchmarks)
cd ml_pipeline && python -m pytest tests/ -v

# nut_agent (86 tests: utils, auth, memory, predictor, intent, agent)
cd nut_agent && python -m pytest tests/ -v

# ai_content_detector (146 tests: detectors, rewards, GRPO/MultiSPIN math,
# arms-race equilibrium, WaRPAD, ensemble — pure-Python, CPU-only)
PYTHONPATH=. python -m pytest ai_content_detector/tests/ -v
```

## Notes

- User data and chat logs: `nut_agent/secrets/` (excluded from version control)
- ML models and results: `ml_pipeline/results/`
- Registry-based design for easy addition of models, augmentations, and metrics

For full details, consult the README in each subfolder.
