# Food ML Project

This repository is a modular, extensible platform for food-related machine learning and AI research. It brings together advanced data augmentation, unified model pipelines, benchmarking, and a conversational nutritionist agent with a modern UI.

## Project Structure

- **food_preds/**: Core ML experimentation suite (augmentation, pipelines, benchmarking, reproducibility, results)
- **nut_agent/**: Streamlit-based conversational nutritionist agent (LLM + ML validation, user registration, session management)

## Getting Started

- See `food_preds/README.md` for details on data augmentation, model training, benchmarking, and reproducibility.
- See `food_preds/pipelines_torch/README.md` for pipeline and model registry usage.
- See `nut_agent/README.md` for the conversational agent, Streamlit UI, and user management features.

## Quickstart

1. Install dependencies (see setup instructions in each submodule)
2. Train or load models in `food_preds`
3. Launch the nutritionist agent UI:
   ```bash
   cd nut_agent
   streamlit run streamlit_app.py
   ```

## Notes

- All user data and chat logs are stored in `nut_agent/secrets/` (excluded from version control)
- ML models are stored in `food_preds/results/`
- Modular design: add new models, augmentations, or agent tools with minimal friction

For full details, consult the README in each subfolder.
