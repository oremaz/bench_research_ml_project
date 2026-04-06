"""
NutriCoach tool definitions for LangGraph agent.
These tools are called by the LLM via bind_tools/ToolNode.
"""

import json
from datetime import date
from typing import Dict, Any, List, Optional
from pathlib import Path

from langchain_core.tools import tool

from shared.config import (
    BMR_CONSTANTS,
    ACTIVITY_MULTIPLIERS,
    WEIGHT_GOAL_ADJUSTMENTS,
    MACRO_RATIOS,
    WATER_ML_PER_KG,
    SECRETS_DIR,
)
from shared.utils import calculate_bmi, validate_nutrition_targets
from shared.memory import MemoryManager
from shared.schemas import (
    NutritionTargets,
    DailyLog,
    MealEntry,
)


# Module-level reference set by agent.py at graph build time
_current_username: Optional[str] = None


def set_current_user(username: str):
    """Set the current user for tool context."""
    global _current_username
    _current_username = username


def _get_memory() -> Optional[MemoryManager]:
    if _current_username:
        return MemoryManager(_current_username, SECRETS_DIR)
    return None


@tool
def calculate_personalized_nutrition_targets(
    weight_kg: float,
    height_cm: float,
    age: int,
    gender: str,
    activity_level: str,
    weight_goal: str,
) -> Dict[str, Any]:
    """
    Calculate personalized daily nutrition targets based on user profile and goals.

    Args:
        weight_kg: Current weight in kg
        height_cm: Height in cm
        age: Age in years
        gender: 'male' or 'female'
        activity_level: 'sedentary', 'light', 'moderate', 'active', 'very_active'
        weight_goal: 'lose', 'maintain', 'gain'

    Returns:
        Dictionary with personalized daily nutrition targets
    """
    try:
        # BMR using Mifflin-St Jeor equation
        bmr_const = BMR_CONSTANTS.get(gender.lower(), BMR_CONSTANTS["male"])
        bmr = (
            bmr_const["base"] * weight_kg
            + bmr_const["weight"] * height_cm
            - bmr_const["height"] * age
            + bmr_const["age"]
        )

        # TDEE
        activity_mult = ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)
        tdee = bmr * activity_mult

        # Adjust for weight goal
        goal_adjustment = WEIGHT_GOAL_ADJUSTMENTS.get(weight_goal, 0)
        target_calories = tdee + goal_adjustment

        # Macronutrients
        protein_g = weight_kg * MACRO_RATIOS["protein_per_kg"]
        fat_g = target_calories * MACRO_RATIOS["fat_percentage"] / MACRO_RATIOS["fat_calories_per_g"]
        remaining_calories = target_calories - (
            protein_g * MACRO_RATIOS["protein_calories_per_g"]
        ) - (fat_g * MACRO_RATIOS["fat_calories_per_g"])
        carbs_g = remaining_calories / MACRO_RATIOS["carb_calories_per_g"]

        # BMI
        bmi_info = calculate_bmi(weight_kg, height_cm)

        # Water
        water_ml = weight_kg * WATER_ML_PER_KG

        targets = {
            "target_calories": round(target_calories),
            "target_protein_g": round(protein_g),
            "target_carbs_g": round(carbs_g),
            "target_fat_g": round(fat_g),
            "target_water_ml": round(water_ml),
            "bmr": round(bmr),
            "tdee": round(tdee),
            "bmi": bmi_info["bmi"],
            "bmi_classification": bmi_info["classification"],
            "healthy_weight_range": bmi_info["healthy_weight_range"],
        }

        validation = validate_nutrition_targets(targets)

        # Persist targets to memory
        memory = _get_memory()
        if memory:
            memory.save_nutrition_targets(NutritionTargets(**targets))

        return {
            "targets": validation["targets"],
            "warnings": validation["warnings"],
            "is_valid": validation["is_valid"],
            "calculation_details": {
                "bmr": round(bmr),
                "tdee": round(tdee),
                "activity_multiplier": activity_mult,
                "goal_adjustment": goal_adjustment,
            },
        }

    except Exception as e:
        return {"error": f"Failed to calculate targets: {str(e)}"}


@tool
def log_daily_intake(
    meals_description: str,
    weight_kg: Optional[float] = None,
    energy_level: Optional[str] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Log the user's daily food intake, weight, and notes.

    Args:
        meals_description: Description of meals eaten (will be parsed into structured data)
        weight_kg: Optional current weight in kg
        energy_level: Optional energy level ('low', 'moderate', 'high')
        notes: Optional additional notes about the day

    Returns:
        Confirmation of logged data
    """
    memory = _get_memory()
    if not memory:
        return {"error": "No user context available for logging"}

    try:
        today = date.today().isoformat()
        existing_log = memory.load_daily_log(today)

        if existing_log:
            log = existing_log
        else:
            log = DailyLog(date=today)

        # Add a meal entry from the description
        log.meals.append(MealEntry(
            meal_type="logged",
            description=meals_description[:500],
        ))

        if weight_kg is not None:
            log.weight_kg = weight_kg
        if energy_level is not None:
            log.energy_level = energy_level
        if notes is not None:
            log.notes = notes

        memory.save_daily_log(log)

        return {
            "logged": True,
            "date": today,
            "meals_count": len(log.meals),
            "weight_kg": log.weight_kg,
            "message": f"Successfully logged intake for {today}",
        }

    except Exception as e:
        return {"error": f"Failed to log intake: {str(e)}"}


@tool
def get_progress_summary(days: int = 7) -> Dict[str, Any]:
    """
    Get a summary of the user's recent nutrition progress.

    Args:
        days: Number of recent days to include (default 7)

    Returns:
        Progress summary with trends and statistics
    """
    memory = _get_memory()
    if not memory:
        return {"error": "No user context available"}

    try:
        recent_logs = memory.load_recent_daily_logs(n=days)
        weekly_summary = memory.load_current_week_summary()
        targets = memory.load_nutrition_targets()

        summary = {
            "days_logged": len(recent_logs),
            "requested_days": days,
        }

        if recent_logs:
            weights = [l.weight_kg for l in recent_logs if l.weight_kg is not None]
            calories = [l.total_calories for l in recent_logs if l.total_calories is not None]
            compliance = [l.compliance_score for l in recent_logs if l.compliance_score is not None]

            if weights:
                summary["weight_trend"] = {
                    "latest": weights[0],
                    "oldest": weights[-1],
                    "change": round(weights[0] - weights[-1], 1) if len(weights) > 1 else 0,
                }
            if calories:
                summary["avg_daily_calories"] = round(sum(calories) / len(calories))
            if compliance:
                summary["avg_compliance"] = round(sum(compliance) / len(compliance), 2)

            # Daily breakdown
            summary["daily_logs"] = [
                {
                    "date": l.date,
                    "calories": l.total_calories,
                    "weight": l.weight_kg,
                    "meals": len(l.meals),
                    "compliance": l.compliance_score,
                }
                for l in recent_logs
            ]

        if targets:
            summary["targets"] = {
                "calories": targets.target_calories,
                "protein_g": targets.target_protein_g,
            }

        if weekly_summary:
            summary["weekly_summary"] = {
                "trends": weekly_summary.trends,
                "ai_notes": weekly_summary.ai_notes,
            }

        return summary

    except Exception as e:
        return {"error": f"Failed to get progress: {str(e)}"}


@tool
def update_user_profile(field: str, value: str) -> Dict[str, Any]:
    """
    Update a specific field in the user's profile.

    Args:
        field: Profile field to update (e.g., 'primary_goal', 'activity_level', 'dietary_preferences', 'weight', 'foods_to_avoid')
        value: New value for the field

    Returns:
        Confirmation of the update
    """
    memory = _get_memory()
    if not memory:
        return {"error": "No user context available"}

    try:
        profile = memory.load_user_profile()
        if profile is None:
            return {"error": "No profile found to update"}

        profile_dict = profile.model_dump()

        # Handle list fields
        list_fields = {"dietary_preferences", "health_conditions", "favorite_cuisines"}
        if field in list_fields:
            # Parse comma-separated values
            profile_dict[field] = [v.strip() for v in value.split(",")]
        elif field in profile_dict:
            # Handle numeric fields
            if field in ("weight", "height"):
                profile_dict[field] = float(value)
            elif field == "age":
                profile_dict[field] = int(value)
            else:
                profile_dict[field] = value
        else:
            return {"error": f"Unknown profile field: {field}"}

        from shared.schemas import UserProfile
        updated_profile = UserProfile(**profile_dict)
        memory.save_user_profile(updated_profile)

        return {
            "updated": True,
            "field": field,
            "new_value": profile_dict[field],
            "message": f"Profile field '{field}' updated successfully",
        }

    except Exception as e:
        return {"error": f"Failed to update profile: {str(e)}"}


# List of all tools for the agent
ALL_TOOLS = [
    calculate_personalized_nutrition_targets,
    log_daily_intake,
    get_progress_summary,
    update_user_profile,
]
