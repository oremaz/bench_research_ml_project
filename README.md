# Modular ML Pipeline for Multi-Modal Classification (AI-Generated Content Detection)

Modular platform for ML research across multiple domains: food prediction (recipe difficulty, meal type, nutrients, time) and AI-generated content detection (deepfakes, image authenticity). Integrates advanced augmentation, unified pipelines, automated benchmarking, third-party research code (FatFormer, DiffusionFake, TabR, GRANDE, TabM), and a conversational nutritionist agent.

## Project Structure

- **ml_pipeline/**: Core ML experimentation suite (augmentation, pipelines, benchmarking, reproducibility, results). See `ml_pipeline/README.md`
- **nut_agent/**: Streamlit-based conversational nutritionist agent (LLM + ML validation, user management). See `nut_agent/README.md`

## Quickstart

1. Install dependencies: `pip install -r requirements.txt`
2. Interactive Benchmark UI: `cd ml_pipeline && streamlit run benchmark_app.py` (with OpenMP configured)
3. Launch nutritionist agent: `cd nut_agent && streamlit run streamlit_app.py`

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

## Notes

- User data and chat logs: `nut_agent/secrets/` (excluded from version control)
- ML models and results: `ml_pipeline/results/`
- Registry-based design for easy addition of models, augmentations, and metrics

For full details, consult the README in each subfolder.
