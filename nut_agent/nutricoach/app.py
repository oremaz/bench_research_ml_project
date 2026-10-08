"""
NutriCoach — Stateful AI nutritionist Streamlit app (v2).

Changes from v1:
- Uses LangGraph SqliteSaver for conversation persistence (replaced manual JSON)
- Simplified agent state (no ghost fields)
- Thread-based conversations via checkpointer config
- Added food image analysis tab
"""

import sys
import os
import json
import logging
import uuid
import tempfile
import hashlib
from pathlib import Path
from datetime import date, datetime

# On macOS, LightGBM must load before the agent's PyTorch dependencies.
if sys.platform == "darwin":
    try:
        import lightgbm
    except (ImportError, OSError):
        pass

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from langchain_core.messages import HumanMessage, AIMessage

from shared.config import OPENROUTER_MODEL_ID, OPENROUTER_REASONING_EFFORT, OPENROUTER_REASONING_EFFORTS, SECRETS_DIR, STREAMLIT_CONFIG
from shared.auth import (
    authenticate_user,
    register_user,
    load_user_info,
)
from shared.memory import MemoryManager
from shared.schemas import DailyLog, MealEntry
from nutricoach.agent import build_nutricoach_graph, create_initial_state
from nutricoach.tools import analyze_food_image, save_photo_analysis
from nutricoach.food_vision.nutrition_db import NutritionDB
from nutricoach.food_vision.review import NUTRIENT_FIELDS, photo_edit_rows, apply_photo_edits

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="NutriCoach",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

USERS_FILE = SECRETS_DIR / "users.json"


# --- Session State ---


def initialize_session_state():
    defaults = {
        "username": None,
        "password": None,
        "initialized": False,
        "is_new_user": True,
        "show_register": False,
        "agent_graph": None,
        "thread_id": None,
        "chat_history": [],
        "vision_model_id": OPENROUTER_MODEL_ID,
        "reasoning_effort": OPENROUTER_REASONING_EFFORT,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


initialize_session_state()


# --- Agent Invocation ---


def _thread_config() -> dict:
    """LangGraph config with thread_id for checkpointer."""
    return {"configurable": {
        "thread_id": st.session_state.get("thread_id", "default"),
        "username": st.session_state.get("username"),
        "openrouter_api_key": st.session_state.get("api_key"),
        "vision_model_id": st.session_state.get("vision_model_id", OPENROUTER_MODEL_ID),
        "reasoning_effort": st.session_state.get("reasoning_effort", OPENROUTER_REASONING_EFFORT),
    }}


def invoke_agent(user_message: str) -> str:
    """
    Send a message to the agent and return the response.
    The checkpointer handles state persistence automatically.
    """
    agent_graph = st.session_state["agent_graph"]
    if agent_graph is None:
        return "Agent not initialized. Please enter your API key."

    try:
        result = agent_graph.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config={
                **_thread_config(),
                "recursion_limit": 20,
            },
        )

        # Extract last AI message
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage):
                return msg.content

        return "I processed your message. How can I help?"
    except Exception as e:
        logger.error("Agent invocation error: %s", e)
        return f"I encountered an error: {str(e)}. Please try again."


def send_message(user_message: str):
    """Send a message through the agent and update chat history."""
    st.session_state["chat_history"].append(("user", user_message))
    with st.spinner("NutriCoach is thinking..."):
        response = invoke_agent(user_message)
    st.session_state["chat_history"].append(("assistant", response))
    st.rerun()


# --- Session Management (simplified with checkpointer) ---


def start_new_chat_session():
    """Start a new chat thread."""
    st.session_state["thread_id"] = str(uuid.uuid4())[:8]
    st.session_state["chat_history"] = []
    username = st.session_state.get("username")
    if username:
        _ensure_memory_migrated(username)


def _ensure_memory_migrated(username):
    """Ensure the user's profile is migrated to structured memory."""
    memory = MemoryManager(username, SECRETS_DIR)
    if memory.load_user_profile() is None:
        user_info = load_user_info(username, USERS_FILE)
        if user_info:
            memory.migrate_from_legacy(user_info)


# --- Init Agent ---


def init_agent():
    reasoning_effort = st.session_state.get("reasoning_effort", OPENROUTER_REASONING_EFFORT)
    if (st.session_state["agent_graph"] is not None
            and st.session_state.get("agent_reasoning_effort") == reasoning_effort):
        return True
    api_key = st.text_input("OpenRouter API Key", type="password", value=st.session_state.get("api_key") or os.environ.get("OPENROUTER_API_KEY", ""))
    model_id = st.text_input("OpenRouter model ID", value=st.session_state.get("model_id") or os.environ.get("OPENROUTER_MODEL_ID", OPENROUTER_MODEL_ID))
    st.caption("Default model: [dots-studio/dots-3-note-preview:free](https://openrouter.ai/dots-studio/dots-3-note-preview:free)")
    if api_key:
        st.session_state["api_key"] = api_key
        st.session_state["model_id"] = model_id.strip() or OPENROUTER_MODEL_ID
        username = st.session_state.get("username", "anonymous")
        try:
            st.session_state["agent_graph"] = build_nutricoach_graph(api_key, username, model_id=model_id.strip() or OPENROUTER_MODEL_ID,
                                                                       reasoning_effort=reasoning_effort)
            st.session_state["agent_reasoning_effort"] = reasoning_effort
            st.success("Agent initialized!")
            return True
        except Exception as e:
            st.error(f"Failed to initialize agent: {e}")
            return False
    else:
        st.warning("Please enter your OpenRouter API key.")
        return False


# --- Dashboard ---


def display_nutrition_dashboard():
    """Display a real dashboard from the user's daily logs."""
    username = st.session_state.get("username")
    if not username:
        st.info("Log in to see your nutrition dashboard.")
        return

    memory = MemoryManager(username, SECRETS_DIR)
    recent_logs = memory.load_recent_daily_logs(n=7)
    targets = memory.load_nutrition_targets()

    if not recent_logs:
        st.info("No daily logs yet. Start tracking your meals to see trends here!")
        return

    rows = []
    for log in reversed(recent_logs):
        rows.append({
            "Date": log.date,
            "Calories": log.total_calories or 0,
            "Protein (g)": log.total_protein_g or 0,
            "Carbs (g)": log.total_carbs_g or 0,
            "Fat (g)": log.total_fat_g or 0,
            "Weight (kg)": log.weight_kg,
            "Compliance": (log.compliance_score or 0) * 100,
        })
    df = pd.DataFrame(rows)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Daily Calories**")
        fig = px.line(df, x="Date", y="Calories", markers=True)
        if targets:
            fig.add_hline(y=targets.target_calories, line_dash="dash", annotation_text="Target")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if any(df["Weight (kg)"] > 0):
            st.markdown("**Weight Trend**")
            weight_df = df[df["Weight (kg)"] > 0]
            fig = px.line(weight_df, x="Date", y="Weight (kg)", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            latest = rows[-1] if rows else {}
            if latest.get("Protein (g)") or latest.get("Carbs (g)") or latest.get("Fat (g)"):
                st.markdown("**Today's Macros**")
                fig = go.Figure(data=[go.Pie(
                    labels=["Protein", "Carbs", "Fat"],
                    values=[latest["Protein (g)"], latest["Carbs (g)"], latest["Fat (g)"]],
                )])
                st.plotly_chart(fig, use_container_width=True)


# --- Chat Interface ---


def display_chat_interface():
    st.subheader("Chat with NutriCoach")

    chat_container = st.container()
    with chat_container:
        for role, message in st.session_state["chat_history"]:
            if role == "user":
                with st.chat_message("user"):
                    st.markdown(message)
            else:
                with st.chat_message("assistant"):
                    st.markdown(message)

    user_input = st.chat_input("Type your message here...")
    if user_input and st.session_state["agent_graph"]:
        send_message(user_input)


# --- Food Image Analysis ---


def _save_upload(uploaded) -> Path:
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded.name).suffix.lower()
    with tempfile.NamedTemporaryFile("wb", dir=SECRETS_DIR, suffix=suffix, delete=False) as f:
        f.write(uploaded.getvalue())
        return Path(f.name)


def display_food_analysis():
    """Tab for analyzing food photos."""
    st.subheader("Analyze Food Photo")
    st.markdown("Upload a photo of your meal to estimate ingredients, portions, and calories.")
    st.caption("Food photos are analyzed with Single-shot VLM.")
    st.text_input("Vision model ID", key="vision_model_id", help="Choose an OpenRouter model that accepts images.")

    uploaded = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png", "webp"])
    context = (st.session_state.get("username"), hashlib.sha256(uploaded.getvalue()).hexdigest()) if uploaded else None
    if st.session_state.get("photo_context") != context:
        st.session_state["photo_context"] = context
        st.session_state.pop("photo_result", None)
        st.session_state["photo_logged"] = False
    if not uploaded:
        return
    st.image(uploaded, caption="Your meal", use_container_width=True)
    if st.button("Analyze photo"):
        st.session_state.pop("photo_result", None)
        tmp_path = _save_upload(uploaded)
        try:
            with st.spinner("Analyzing your meal..."):
                result = analyze_food_image.invoke({"image_path": str(tmp_path)}, config=_thread_config())
            if result.get("error"):
                st.error(result["error"])
            else:
                st.session_state["photo_result"] = result
                st.session_state["photo_review_id"] = uuid.uuid4().hex[:8]
                st.session_state["photo_logged"] = False
        finally:
            tmp_path.unlink(missing_ok=True)
    if st.session_state.get("photo_result"):
        display_photo_review()


def display_photo_review():
    original = st.session_state["photo_result"]
    review_id = st.session_state["photo_review_id"]
    logged = st.session_state.get("photo_logged", False)
    st.markdown("**Review your meal**")
    meal_name = st.text_input("Meal name", value="Photo meal", key=f"photo_name_{review_id}", disabled=logged)
    st.caption("Edit grams, add rows, or select rows to delete ingredients. When changing a food, choose its nutrition source. "
               "Manual nutrient columns are per 100g; selecting a database food overrides those columns.")
    try:
        rows = st.session_state["photo_saved_rows"] if logged else photo_edit_rows(original)
    except (ValueError, TypeError, KeyError) as exc:
        st.error(str(exc))
        return
    sources = ["Photo estimate", "Manual", *[f"Database: {name}" for name in sorted(NutritionDB().db)]]
    columns = {
        "original_index": None,
        "name": st.column_config.TextColumn("Ingredient", required=True),
        "quantity_grams": st.column_config.NumberColumn("Grams", min_value=0.01, required=True),
        "nutrition_source": st.column_config.SelectboxColumn("Nutrition source", options=sources, required=True),
        **{field: st.column_config.NumberColumn(f"{field.replace('_g', '')} / 100g", min_value=0.0)
           for field in NUTRIENT_FIELDS},
    }
    edited = st.data_editor(pd.DataFrame(rows), num_rows="dynamic", hide_index=True,
                            column_config=columns, disabled=True if logged else ["original_index"],
                            key=f"photo_editor_{review_id}", use_container_width=True)
    try:
        corrected = apply_photo_edits(original, edited.to_dict("records"))
    except (ValueError, TypeError, KeyError) as exc:
        st.info(f"Complete your corrections: {exc}")
        return
    st.write(f"**{corrected.total_calories:.0f} kcal** · Protein: {corrected.total_protein_g:.1f}g · "
             f"Carbs: {corrected.total_carbs_g:.1f}g · Fat: {corrected.total_fat_g:.1f}g")
    if logged:
        st.success("This meal has been saved to today's journal.")
    elif st.button("Save reviewed meal to today's journal", key=f"photo_save_{review_id}"):
        try:
            save_photo_analysis(corrected, _thread_config(), meal_name=meal_name)
            st.session_state["photo_saved_rows"] = edited.to_dict("records")
            st.session_state["photo_logged"] = True
            st.rerun()
        except (ValueError, OSError) as exc:
            st.error(f"Could not save meal: {exc}")


# --- Quick Actions ---


def display_quick_actions():
    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Plan Today's Meals", use_container_width=True):
            if st.session_state["agent_graph"]:
                send_message("Please create a meal plan for today based on my profile and any constraints I mentioned.")
        if st.button("What Can I Still Eat Today?", use_container_width=True):
            if st.session_state["agent_graph"]:
                send_message("What do I have left in my nutrition budget today? Suggest a meal or snack that fits.")

    with col2:
        if st.button("Calculate Nutrition Targets", use_container_width=True):
            if st.session_state["agent_graph"]:
                send_message("Please calculate my daily nutrition targets based on my profile.")
        if st.button("Grocery List From My Plan", use_container_width=True):
            if st.session_state["agent_graph"]:
                send_message("Build a grocery list from my meal plan for this week, grouped by store section.")

    with col3:
        if st.button("Check My Progress", use_container_width=True):
            if st.session_state["agent_graph"]:
                send_message("How have I been doing this week? Please analyze my recent progress.")
        if st.button("Weekly Summary", use_container_width=True):
            if st.session_state["agent_graph"]:
                send_message("Generate my weekly summary and tell me what to improve next week.")


# --- Daily Results Tracker ---


def display_daily_results():
    with st.expander("Daily Results Tracking", expanded=False):
        date_tracked = st.date_input("Meals date:", max_value=date.today())
        record_weight = st.checkbox("I measured my weight")
        current_weight = None
        weight_date = None
        if record_weight:
            st.caption("Use the morning measurement, before breakfast. Date it on the morning it was measured, even when reviewing yesterday's meals.")
            col1, col2 = st.columns(2)
            with col1:
                current_weight = st.number_input("Measured weight (kg):", min_value=30.0, max_value=200.0, value=None, step=0.1)
            with col2:
                weight_date = st.date_input("Weight measurement date:", max_value=date.today())

        st.subheader("Meal Compliance")
        meals = ["Breakfast", "Lunch", "Dinner", "Snacks"]
        meal_data = {}
        for meal in meals:
            st.write(f"**{meal}:**")
            col1, col2 = st.columns([1, 2])
            with col1:
                followed = st.checkbox(f"Followed {meal.lower()} plan exactly", key=f"exact_{meal.lower()}")
            with col2:
                if not followed:
                    diff = st.text_area(f"Differences for {meal.lower()}:", key=f"diff_{meal.lower()}", height=60)
                else:
                    diff = "Followed plan exactly"
            meal_data[meal] = {"followed_exactly": followed, "differences": diff if not followed else "Followed plan exactly"}

        additional_info = st.text_area("Additional information:", placeholder="Energy levels, cravings, exercise, water intake, sleep...", height=80)

        if st.button("Submit Daily Results"):
            if st.session_state["agent_graph"]:
                if record_weight and current_weight is None:
                    st.error("Enter the measured weight or uncheck 'I measured my weight'.")
                    return
                if record_weight:
                    memory = MemoryManager(st.session_state["username"], SECRETS_DIR)
                    weight_day = weight_date.isoformat()
                    log = memory.load_daily_log(weight_day) or DailyLog(date=weight_day)
                    log.weight_kg = current_weight
                    memory.save_daily_log(log)
                meal_lines = []
                for meal, data in meal_data.items():
                    if data["followed_exactly"]:
                        meal_lines.append(f"- {meal}: Followed plan exactly")
                    else:
                        meal_lines.append(f"- {meal}: {data['differences']}")

                message = f"""Please analyze my daily results:

**Daily Tracking Results for {date_tracked}**
**Morning weight:** {f'{current_weight} kg on {weight_date}' if record_weight else 'Not measured'}
**Meal Compliance:**
{chr(10).join(meal_lines)}
**Additional Information:** {additional_info or "None provided"}

Please provide analysis of my progress and any recommendations."""

                send_message(message)


# --- User Profile Display ---


def display_user_profile():
    username = st.session_state.get("username")
    if not username:
        return
    user_info = load_user_info(username, USERS_FILE)
    if not user_info:
        return
    with st.expander("Your Profile", expanded=False):
        st.write(f"**Age:** {user_info.get('age')}")
        st.write(f"**Weight:** {user_info.get('weight')} kg")
        st.write(f"**Height:** {user_info.get('height')} cm")
        st.write(f"**Goal:** {user_info.get('primary_goal')}")
        st.write(f"**Activity:** {user_info.get('activity_level')}")
        prefs = user_info.get("dietary_preferences", [])
        if prefs:
            st.write(f"**Diet:** {', '.join(prefs)}")


# --- Registration Form ---


def registration_form():
    st.title("Register New Account")
    username = st.session_state.get("username", "")
    password = st.session_state.get("password", "")
    st.write(f"Username: **{username}**")

    st.markdown("---")
    st.subheader("Basic Information")

    col1, col2 = st.columns(2)
    with col1:
        weight = st.number_input("Weight (kg)", min_value=30, max_value=300, value=70)
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        age = st.number_input("Age", min_value=5, max_value=120, value=30)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
        activity_level = st.selectbox("Activity Level", [
            "Sedentary (desk job, no exercise)",
            "Lightly active (light exercise 1-3 days/week)",
            "Moderately active (moderate exercise 3-5 days/week)",
            "Very active (hard exercise 6-7 days/week)",
            "Extremely active (very hard exercise, physical job)",
        ])

    st.markdown("---")
    st.subheader("Goals & Preferences")

    col3, col4 = st.columns(2)
    with col3:
        primary_goal = st.selectbox("Primary Goal", [
            "Lose weight", "Gain weight", "Maintain weight", "Build muscle",
            "Improve health", "Increase energy", "Better sleep",
        ])
        dietary_preferences = st.multiselect("Dietary Preferences/Restrictions", [
            "None", "Vegetarian", "Vegan", "Pescatarian", "Keto", "Paleo",
            "Low-carb", "Low-fat", "Gluten-free", "Dairy-free", "Halal",
            "Kosher", "Nut allergies", "Shellfish allergies",
        ])
    with col4:
        cooking_experience = st.selectbox("Cooking Experience", [
            "Beginner (prefer simple recipes)",
            "Intermediate (comfortable with most recipes)",
            "Advanced (enjoy complex cooking)",
        ])
        budget_range = st.selectbox("Weekly Food Budget", [
            "$50-100", "$100-150", "$150-200", "$200-300", "$300+", "No specific budget",
        ])

    st.markdown("---")
    st.subheader("Meal Schedule")

    col5a, col5b = st.columns(2)
    with col5a:
        st.markdown("**Breakfast**")
        breakfast_enabled = st.checkbox("I eat breakfast", value=True)
        breakfast_location = st.selectbox("Breakfast location", ["At home", "On the go", "At work/office", "Varies"], key="breakfast_location")
        breakfast_time = st.selectbox("Breakfast cooking time", ["0 minutes", "5-10 minutes", "15-20 minutes", "30+ minutes"], key="breakfast_time")

        st.markdown("**Lunch**")
        lunch_enabled = st.checkbox("I eat lunch", value=True)
        lunch_location = st.selectbox("Lunch location", ["At work/office", "At home", "Restaurant/cafeteria", "Varies"], key="lunch_location")
        lunch_time = st.selectbox("Lunch cooking time", ["0 minutes", "5-10 minutes", "15-20 minutes", "30+ minutes", "Meal prep"], key="lunch_time")

    with col5b:
        st.markdown("**Dinner**")
        dinner_enabled = st.checkbox("I eat dinner", value=True)
        dinner_location = st.selectbox("Dinner location", ["At home", "Restaurant", "At work", "Varies"], key="dinner_location")
        dinner_time = st.selectbox("Dinner cooking time", ["0 minutes", "15-30 minutes", "30-60 minutes", "1+ hours", "Varies"], key="dinner_time")

        st.markdown("**Snacks**")
        snacks_enabled = st.checkbox("I eat snacks", value=True)
        snacks_frequency = st.selectbox("Snack frequency", ["No snacks", "1/day", "2/day", "3+/day", "As needed"], key="snacks_frequency")
        snacks_type = st.selectbox("Preferred snack type", ["Quick/ready-made", "Light prep", "Homemade", "Mixed", "Healthy only"], key="snacks_type")

    st.markdown("---")
    st.subheader("Health & Food Preferences")

    health_conditions = st.multiselect("Health Conditions (optional)", [
        "None", "Diabetes", "High blood pressure", "High cholesterol",
        "Heart disease", "PCOS", "Thyroid issues", "IBS/Digestive issues",
        "Food allergies", "Other",
    ])
    water_intake_goal = st.selectbox("Daily Water Intake Goal", ["1-2 liters", "2-3 liters", "3-4 liters", "4+ liters", "Not sure"])
    favorite_cuisines = st.multiselect("Favorite Cuisines", [
        "American", "Italian", "Mexican", "Asian", "Indian",
        "Mediterranean", "Middle Eastern", "French", "African", "Latin American", "No preference",
    ])
    foods_to_avoid = st.text_area("Foods to avoid (optional)", height=60)
    additional_notes = st.text_area("Additional notes (optional)", height=80)

    if st.button("Complete Registration", type="primary"):
        if not username or not password:
            st.error("Username and password are required.")
            return

        profile = {
            "weight": weight, "height": height, "age": age, "gender": gender,
            "activity_level": activity_level, "primary_goal": primary_goal,
            "dietary_preferences": dietary_preferences, "cooking_experience": cooking_experience,
            "budget_range": budget_range,
            "meal_schedule": {
                "breakfast": {"enabled": breakfast_enabled, "location": breakfast_location, "cooking_time": breakfast_time},
                "lunch": {"enabled": lunch_enabled, "location": lunch_location, "cooking_time": lunch_time},
                "dinner": {"enabled": dinner_enabled, "location": dinner_location, "cooking_time": dinner_time},
                "snacks": {"enabled": snacks_enabled, "frequency": snacks_frequency, "type": snacks_type},
            },
            "health_conditions": health_conditions, "water_intake_goal": water_intake_goal,
            "favorite_cuisines": favorite_cuisines, "foods_to_avoid": foods_to_avoid,
            "additional_notes": additional_notes, "registration_date": datetime.now().isoformat(),
        }

        success = register_user(username, password, profile, USERS_FILE)
        if success:
            st.session_state["is_new_user"] = False
            st.session_state["initialized"] = True
            st.session_state["show_register"] = False

            memory = MemoryManager(username, SECRETS_DIR)
            memory.migrate_from_legacy(profile)

            st.success("Registration complete! Welcome to NutriCoach!")
            st.rerun()
        else:
            st.error("Username already exists. Please choose another.")


# --- Landing Page ---


def landing_page():
    st.title("Welcome to NutriCoach")
    username = st.text_input("Username", key="landing_username")
    password = st.text_input("Password", type="password", key="landing_password")

    col1, col2 = st.columns(2)
    with col1:
        login_btn = st.button("Login")
    with col2:
        register_btn = st.button("Register")

    if login_btn and username and password:
        user_info = authenticate_user(username, password, USERS_FILE)
        if user_info is not None:
            st.session_state["username"] = username
            st.session_state["is_new_user"] = False
            st.session_state["initialized"] = True
            start_new_chat_session()
            st.success(f"Login successful! Thread: {st.session_state['thread_id']}")
            st.rerun()
        else:
            st.error("Invalid credentials. Please try again or register.")

    elif register_btn and username and password:
        st.session_state["username"] = username
        st.session_state["password"] = password
        st.session_state["show_register"] = True
        st.rerun()


# --- Main App ---


def main_app():
    if not st.session_state.get("thread_id"):
        start_new_chat_session()

    st.title("NutriCoach")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.title("Control Panel")
        st.selectbox("Reasoning effort", OPENROUTER_REASONING_EFFORTS, key="reasoning_effort",
                     help="Applies to chat, recipes, and Food Vision. Choose a level supported by your model.")
        if not init_agent():
            st.stop()
        st.markdown("---")
        display_user_profile()

        thread_id = st.session_state.get("thread_id")
        if thread_id:
            st.info(f"Thread: {thread_id}")

        st.markdown("---")
        if st.button("New Conversation"):
            start_new_chat_session()
            st.rerun()

    # Main content tabs
    tab_chat, tab_food, tab_recipe, tab_dashboard, tab_actions, tab_track = st.tabs([
        "Chat", "Food Analysis", "Recipe Lab", "Dashboard", "Quick Actions", "Daily Tracking",
    ])

    with tab_chat:
        display_chat_interface()

    with tab_food:
        display_food_analysis()

    with tab_recipe:
        from recipe_lab.app import main as display_recipe_lab
        display_recipe_lab(embedded=True)

    with tab_dashboard:
        display_nutrition_dashboard()

    with tab_actions:
        display_quick_actions()

    with tab_track:
        display_daily_results()

    st.markdown("---")
    st.caption("NutriCoach v2: LangGraph, SqliteSaver, OpenRouter, and Food Vision")


# --- Entry Point ---


def run():
    initialize_session_state()
    if not st.session_state.get("initialized", False):
        if st.session_state["show_register"]:
            registration_form()
        else:
            landing_page()
    else:
        main_app()


if __name__ == "__main__":
    run()
