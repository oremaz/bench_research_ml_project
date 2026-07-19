"""
Smart Recipe Lab - Standalone Streamlit app for ML-powered recipe analysis.
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

st.set_page_config(
    page_title="Smart Recipe Lab",
    page_icon="🧪",
    layout="wide",
)


def get_predictor():
    """Initialize or retrieve the FoodModelPredictor."""
    if "food_predictor" not in st.session_state or st.session_state["food_predictor"] is None:
        api_key = st.session_state.get("api_key") or os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            return None
        from recipe_lab.predictor import FoodModelPredictor
        st.session_state["food_predictor"] = FoodModelPredictor(api_key=api_key)
    return st.session_state["food_predictor"]


def display_analysis_results(analysis: dict):
    """Display ML analysis results for a single recipe."""
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

    # ML Predictions
    st.markdown("### ML Model Predictions")
    col1, col2, col3 = st.columns(3)

    with col1:
        difficulty = analysis.get("difficulty", {})
        st.markdown("**Difficulty**")
        if "all_probabilities" in difficulty:
            for label, prob in difficulty["all_probabilities"].items():
                st.write(f"{label}: {prob:.1%}")
        else:
            st.write(f"{difficulty.get('prediction', 'Unknown')}: {difficulty.get('confidence', 0):.1%}")

    with col2:
        meal_type = analysis.get("meal_type", {})
        st.markdown("**Meal Type**")
        if "all_probabilities" in meal_type:
            for label, prob in meal_type["all_probabilities"].items():
                st.write(f"{label.title()}: {prob:.1%}")
        else:
            st.write(f"{meal_type.get('prediction', 'Unknown')}: {meal_type.get('confidence', 0):.1%}")

    with col3:
        time_class = analysis.get("time_class", {})
        st.markdown("**Time Class**")
        if "all_probabilities" in time_class:
            for label, prob in time_class["all_probabilities"].items():
                st.write(f"{label}: {prob:.1%}")
        else:
            st.write(f"{time_class.get('prediction', 'Unknown')}: {time_class.get('confidence', 0):.1%}")

    nutrients = analysis.get("nutrients", {})
    per_serving = nutrients.get("per_serving")
    if per_serving:
        st.markdown("### Estimated Nutrition (per serving)")
        cols = st.columns(4)
        units = {"kcal": "kcal"}
        for i, (target, value) in enumerate(per_serving.items()):
            unit = units.get(target, "g")
            cols[i % 4].metric(target.capitalize(), f"{value:g} {unit}")
        st.caption(
            "Ballpark estimate from the recipe text (typical calorie error "
            "30-50% per serving). Good for comparing recipes, not for "
            "precise tracking."
        )


def main():
    st.title("Smart Recipe Lab")
    st.markdown("Analyze recipes using trained ML models. Get predictions for difficulty, meal type, and cooking time.")

    # Sidebar: API key
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("Google API Key", type="password", value=os.environ.get("GOOGLE_API_KEY", ""))
        if api_key:
            st.session_state["api_key"] = api_key

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
                st.error("Please enter your Google API key in the sidebar.")
                return

            with st.spinner("Analyzing recipe..."):
                analysis = predictor.analyze_recipe(recipe_text.strip())
            display_analysis_results(analysis)

            # LLM interpretation
            if "error" not in analysis:
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
                st.error("Please enter your Google API key in the sidebar.")
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
    st.caption("Smart Recipe Lab - Powered by LightGBM + Google Gemini text embeddings")


if __name__ == "__main__":
    main()
