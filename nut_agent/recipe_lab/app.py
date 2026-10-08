"""
Smart Recipe Lab - Standalone Streamlit app for recipe analysis.
Stateless, no login required, no LangGraph.
"""

import sys
import os
import json
from pathlib import Path

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from shared.config import OPENROUTER_MODEL_ID, OPENROUTER_REASONING_EFFORT, OPENROUTER_REASONING_EFFORTS

def get_predictor():
    """Initialize or retrieve the FoodModelPredictor."""
    api_key = st.session_state.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")
    model_id = st.session_state.get("model_id") or OPENROUTER_MODEL_ID
    reasoning_effort = st.session_state.get("reasoning_effort", OPENROUTER_REASONING_EFFORT)
    predictor = st.session_state.get("food_predictor")
    if predictor is None or predictor.api_key != api_key or predictor.model_id != model_id:
        from recipe_lab.predictor import FoodModelPredictor
        with st.spinner("Preparing local embeddings and LightGBM models (first run may take several minutes)..."):
            try:
                st.session_state["food_predictor"] = FoodModelPredictor(api_key=api_key, model_id=model_id,
                                                                      reasoning_effort=reasoning_effort)
            except Exception as exc:
                st.error(f"Could not prepare recipe models: {exc}")
                return None
    st.session_state["food_predictor"].reasoning_effort = reasoning_effort
    return st.session_state["food_predictor"]


def display_analysis_results(analysis: dict):
    """Display analysis results for a single recipe."""
    if "error" in analysis:
        st.error(f"Analysis failed: {analysis['error']}")
        return

    # Enhanced recipe info
    enhanced_recipe = analysis.get("enhanced_recipe", {})
    if isinstance(enhanced_recipe, str):
        try:
            enhanced_recipe = json.loads(enhanced_recipe)
        except Exception:
            enhanced_recipe = {}

    if enhanced_recipe:
        recipe_name = enhanced_recipe.get("name", "N/A")
        ingredients = enhanced_recipe.get("ingredients", [])
        steps = enhanced_recipe.get("steps", [])

        if isinstance(ingredients, str):
            try:
                parsed = json.loads(ingredients)
                ingredients = parsed if isinstance(parsed, list) else [ingredients]
            except Exception:
                ingredients = [ingredients]

        if isinstance(steps, str):
            try:
                parsed = json.loads(steps)
                steps = parsed if isinstance(parsed, list) else [steps]
            except Exception:
                steps = [steps]

        st.markdown(f"### {recipe_name}")
        if ingredients:
            st.markdown("**Ingredients:**")
            for item in ingredients:
                st.markdown(f"- {item}")
        if steps:
            st.markdown("**Instructions:**")
            for i, step in enumerate(steps, 1):
                st.markdown(f"**{i}.** {step}")
        st.markdown("---")

    st.markdown("### LightGBM predictions")
    col1, col2, col3 = st.columns(3)

    with col1:
        difficulty = analysis.get("difficulty", {})
        st.markdown("**Difficulty**")
        if "all_probabilities" in difficulty:
            for label, prob in difficulty["all_probabilities"].items():
                st.write(f"{label}: {prob:.1%}")
        else:
            st.write(difficulty.get('prediction', 'Unknown'))

    with col2:
        meal_type = analysis.get("meal_type", {})
        st.markdown("**Meal Type**")
        if "all_probabilities" in meal_type:
            for label, prob in meal_type["all_probabilities"].items():
                st.write(f"{label.title()}: {prob:.1%}")
        else:
            st.write(meal_type.get('prediction', 'Unknown'))

    with col3:
        time_class = analysis.get("time_class", {})
        st.markdown("**Time Class**")
        if "all_probabilities" in time_class:
            for label, prob in time_class["all_probabilities"].items():
                st.write(f"{label}: {prob:.1%}")
        else:
            st.write(time_class.get('prediction', 'Unknown'))

    nutrients = analysis.get("nutrients", {})
    per_serving = nutrients.get("per_serving")
    if per_serving:
        st.markdown("### LLM Nutrition Estimate (per serving)")
        cols = st.columns(4)
        units = {"kcal": "kcal"}
        for i, (target, value) in enumerate(per_serving.items()):
            unit = units.get(target, "g")
            cols[i % 4].metric(target.capitalize(), f"{value:g} {unit}")
        st.caption("Zero-shot OpenRouter estimate. Use as a rough guide, not for precise tracking.")
    elif nutrients.get("error"):
        st.info(f"Nutrition estimate unavailable: {nutrients['error']}")


def main(embedded: bool = False):
    if embedded:
        st.subheader("Smart Recipe Lab")
    else:
        st.title("Smart Recipe Lab")
    st.markdown("Analyze recipes with local classifiers and zero-shot OpenRouter nutrition estimates.")

    if not embedded:
        with st.sidebar:
            st.header("Settings")
            st.selectbox("Reasoning effort", OPENROUTER_REASONING_EFFORTS,
                         index=OPENROUTER_REASONING_EFFORTS.index(OPENROUTER_REASONING_EFFORT), key="reasoning_effort",
                         help="Choose a level supported by your OpenRouter model.")
            api_key = st.text_input("OpenRouter API Key", type="password", value=os.environ.get("OPENROUTER_API_KEY", ""))
            model_id = st.text_input("OpenRouter model ID", value=os.environ.get("OPENROUTER_MODEL_ID", OPENROUTER_MODEL_ID))
            st.caption("Default model: [dots-studio/dots-3-note-preview:free](https://openrouter.ai/dots-studio/dots-3-note-preview:free)")
            st.session_state["api_key"] = api_key or None
            st.session_state["model_id"] = model_id.strip() or OPENROUTER_MODEL_ID
    else:
        st.session_state["model_id"] = st.session_state.get("model_id") or os.environ.get("OPENROUTER_MODEL_ID", OPENROUTER_MODEL_ID)

    # Main tabs
    tab_single, tab_compare = st.tabs(["Analyze Recipe", "Compare Recipes"])

    with tab_single:
        recipe_text = st.text_area(
            "Describe your recipe:",
            placeholder="e.g., Grilled salmon with lemon butter sauce, served with roasted asparagus",
            height=120,
        )
        analyze_button = st.button("Analyze Recipe", type="primary")

        if analyze_button and recipe_text.strip():
            predictor = get_predictor()
            if predictor is None:
                return

            with st.spinner("Analyzing recipe..."):
                analysis = predictor.analyze_recipe(recipe_text.strip())
            display_analysis_results(analysis)

            # LLM interpretation
            if "error" not in analysis:
                if predictor.client:
                    with st.spinner("Generating interpretation..."):
                        interpretation = predictor.generate_llm_interpretation(analysis)
                    st.markdown("### AI Interpretation")
                    st.markdown(interpretation)

        elif analyze_button:
            st.warning("Please enter a recipe description.")

    with tab_compare:
        st.markdown("Compare two recipes side by side.")
        col_a, col_b = st.columns(2)
        with col_a:
            recipe_a = st.text_area("Recipe A:", placeholder="Describe recipe A...", height=100, key="recipe_a")
        with col_b:
            recipe_b = st.text_area("Recipe B:", placeholder="Describe recipe B...", height=100, key="recipe_b")

        compare_button = st.button("Compare", type="primary")

        if compare_button and recipe_a.strip() and recipe_b.strip():
            predictor = get_predictor()
            if predictor is None:
                return

            col_res_a, col_res_b = st.columns(2)
            with col_res_a:
                st.markdown("## Recipe A")
                with st.spinner("Analyzing Recipe A..."):
                    analysis_a = predictor.analyze_recipe(recipe_a.strip())
                display_analysis_results(analysis_a)
            with col_res_b:
                st.markdown("## Recipe B")
                with st.spinner("Analyzing Recipe B..."):
                    analysis_b = predictor.analyze_recipe(recipe_b.strip())
                display_analysis_results(analysis_b)

        elif compare_button:
            st.warning("Please enter both recipe descriptions.")

    # Footer
    st.markdown("---")
    st.caption("Smart Recipe Lab - Powered by local Hugging Face embeddings, LightGBM, and OpenRouter")


if __name__ == "__main__":
    st.set_page_config(page_title="Smart Recipe Lab", page_icon="🧪", layout="wide")
    main()
