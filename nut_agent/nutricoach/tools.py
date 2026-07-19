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
    MealPlan,
    WeeklySummary,
)
from nutricoach.food_vision.base import FoodAnalysisResult


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


def _recompute_daily_totals(log: DailyLog) -> None:
    """Recompute log totals from per-meal estimates (None if no meal has estimates)."""
    def _total(attr):
        vals = [getattr(m, attr) for m in log.meals if getattr(m, attr) is not None]
        return round(sum(vals), 1) if vals else None

    cal = _total("estimated_calories")
    log.total_calories = int(cal) if cal is not None else None
    log.total_protein_g = _total("estimated_protein_g")
    log.total_carbs_g = _total("estimated_carbs_g")
    log.total_fat_g = _total("estimated_fat_g")


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
    meal_type: Optional[str] = None,
    estimated_calories: Optional[int] = None,
    estimated_protein_g: Optional[float] = None,
    estimated_carbs_g: Optional[float] = None,
    estimated_fat_g: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Log the user's daily food intake, weight, and notes.

    Args:
        meals_description: Description of meals eaten (will be parsed into structured data)
        weight_kg: Optional current weight in kg
        energy_level: Optional energy level ('low', 'moderate', 'high')
        notes: Optional additional notes about the day
        meal_type: Optional meal slot ('breakfast', 'lunch', 'dinner', 'snack')
        estimated_calories: Your estimate of the calories in this meal (always provide it)
        estimated_protein_g: Your estimate of protein grams
        estimated_carbs_g: Your estimate of carb grams
        estimated_fat_g: Your estimate of fat grams

    Returns:
        Confirmation of logged data with updated daily totals
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
            meal_type=meal_type or "logged",
            description=meals_description[:500],
            estimated_calories=estimated_calories,
            estimated_protein_g=estimated_protein_g,
            estimated_carbs_g=estimated_carbs_g,
            estimated_fat_g=estimated_fat_g,
        ))
        _recompute_daily_totals(log)

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
            "total_calories_today": log.total_calories,
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


@tool
def analyze_food_image(
    image_path: str,
    method: str = "rag_vlm",
) -> Dict[str, Any]:
    """
    Analyze a food photo to identify ingredients, estimate portions, and compute calories/macros.

    Args:
        image_path: Path to the food image file
        method: Analysis method — 'vlm_claude' (pure LLM), 'rag_vlm' (RAG-enhanced, recommended),
                'clip_ensemble' (CLIP + LLM), 'rf_detr' (object detection)

    Returns:
        Dictionary with detected food items and nutritional breakdown
    """
    import os

    if not os.path.exists(image_path):
        return {"error": f"Image not found: {image_path}"}

    try:
        analyzer = None

        if method == "vlm_claude":
            from nutricoach.food_vision.vlm_analyzer import VLMAnalyzer
            analyzer = VLMAnalyzer()
        elif method == "rag_vlm":
            from nutricoach.food_vision.rag_vlm_analyzer import RAGVLMAnalyzer
            analyzer = RAGVLMAnalyzer()
        elif method == "clip_ensemble":
            from nutricoach.food_vision.clip_analyzer import CLIPFoodAnalyzer
            analyzer = CLIPFoodAnalyzer()
        elif method == "rf_detr":
            from nutricoach.food_vision.rf_detr_analyzer import RFDETRAnalyzer
            analyzer = RFDETRAnalyzer()
        else:
            return {"error": f"Unknown method: {method}. Use 'vlm_claude', 'rag_vlm', 'clip_ensemble', or 'rf_detr'"}

        result = analyzer.analyze(image_path)

        # Also log the meal if we have a user context
        memory = _get_memory()
        if memory and result.food_items and not result.error:
            today = date.today().isoformat()
            existing_log = memory.load_daily_log(today)
            log = existing_log if existing_log else DailyLog(date=today)

            description = ", ".join(
                f"{f.name} ({f.quantity_grams:.0f}g, {f.calories:.0f}kcal)"
                for f in result.food_items
            )
            log.meals.append(MealEntry(
                meal_type="photo_analysis",
                description=description[:500],
                estimated_calories=int(result.total_calories),
                estimated_protein_g=result.total_protein_g,
                estimated_carbs_g=result.total_carbs_g,
                estimated_fat_g=result.total_fat_g,
            ))
            _recompute_daily_totals(log)
            memory.save_daily_log(log)

        return result.to_dict()

    except ImportError as e:
        return {"error": f"Missing dependency for method '{method}': {str(e)}"}
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}


@tool
def log_water_intake(amount_ml: float) -> Dict[str, Any]:
    """
    Add water intake to today's log.

    Args:
        amount_ml: Amount of water just drunk, in milliliters (a glass is ~250ml)

    Returns:
        Updated water total for today and remaining amount vs target
    """
    memory = _get_memory()
    if not memory:
        return {"error": "No user context available"}

    try:
        today = date.today().isoformat()
        log = memory.load_daily_log(today) or DailyLog(date=today)
        log.water_intake_ml = (log.water_intake_ml or 0) + amount_ml
        memory.save_daily_log(log)

        result = {
            "logged": True,
            "water_today_ml": log.water_intake_ml,
        }
        targets = memory.load_nutrition_targets()
        if targets:
            result["target_water_ml"] = targets.target_water_ml
            result["remaining_ml"] = max(0, targets.target_water_ml - log.water_intake_ml)
        return result

    except Exception as e:
        return {"error": f"Failed to log water: {str(e)}"}


@tool
def lookup_food_nutrition(food_name: str, grams: float = 100.0) -> Dict[str, Any]:
    """
    Look up calories and macros for a food in the local nutrition database
    (USDA/CIQUAL per-100g values, no API cost). Use for questions like
    "how many calories in 150g of salmon?".

    Args:
        food_name: Food to look up (fuzzy matching supported)
        grams: Portion size in grams (default 100)

    Returns:
        Nutrition values for the requested portion, or closest matches if not found
    """
    from nutricoach.food_vision.nutrition_db import NutritionDB

    try:
        db = NutritionDB()
        matched_name, info = db.lookup_with_name(food_name)
        if info is None:
            return {
                "found": False,
                "food_name": food_name,
                "message": "Not in local database; estimate from your own knowledge.",
            }
        factor = grams / 100.0
        return {
            "found": True,
            "query": food_name,
            "matched_food": matched_name,
            "grams": grams,
            "calories": round(info.calories * factor, 1),
            "protein_g": round(info.protein_g * factor, 1),
            "carbs_g": round(info.carbs_g * factor, 1),
            "fat_g": round(info.fat_g * factor, 1),
        }
    except Exception as e:
        return {"error": f"Lookup failed: {str(e)}"}


@tool
def get_remaining_daily_budget() -> Dict[str, Any]:
    """
    Compute what the user can still eat and drink today: nutrition targets
    minus everything logged so far. Use for questions like "what's left for
    dinner?" or "can I have a snack?".

    Returns:
        Remaining calories, macros, and water for today
    """
    memory = _get_memory()
    if not memory:
        return {"error": "No user context available"}

    try:
        targets = memory.load_nutrition_targets()
        if targets is None:
            return {"error": "No nutrition targets set. Calculate targets first."}

        log = memory.load_todays_log()
        consumed = {
            "calories": (log.total_calories or 0) if log else 0,
            "protein_g": (log.total_protein_g or 0) if log else 0,
            "carbs_g": (log.total_carbs_g or 0) if log else 0,
            "fat_g": (log.total_fat_g or 0) if log else 0,
            "water_ml": (log.water_intake_ml or 0) if log else 0,
        }
        meals_logged = len(log.meals) if log else 0

        return {
            "date": date.today().isoformat(),
            "meals_logged": meals_logged,
            "consumed": consumed,
            "remaining": {
                "calories": round(targets.target_calories - consumed["calories"]),
                "protein_g": round(targets.target_protein_g - consumed["protein_g"], 1),
                "carbs_g": round(targets.target_carbs_g - consumed["carbs_g"], 1),
                "fat_g": round(targets.target_fat_g - consumed["fat_g"], 1),
                "water_ml": round(targets.target_water_ml - consumed["water_ml"]),
            },
            "note": "Remaining values can be negative if the user exceeded a target.",
        }

    except Exception as e:
        return {"error": f"Failed to compute budget: {str(e)}"}


@tool
def save_meal_plan(plan_text: str, notes: str = "") -> Dict[str, Any]:
    """
    Save the meal plan agreed with the user for the current week. Call this
    after presenting a meal plan the user accepts, passing the full plan text.
    The stored plan is used for daily compliance tracking and grocery lists.

    Args:
        plan_text: The complete meal plan (all days and meals)
        notes: Optional notes (constraints, substitutions)

    Returns:
        Confirmation with the week id
    """
    memory = _get_memory()
    if not memory:
        return {"error": "No user context available"}

    try:
        from datetime import datetime
        week_id = date.today().strftime("%G-W%V")
        plan = MealPlan(
            week_id=week_id,
            plan_text=plan_text,
            created_at=datetime.now().isoformat(),
            notes=notes,
        )
        memory.save_meal_plan(plan)
        return {"saved": True, "week_id": week_id, "message": f"Meal plan saved for week {week_id}"}
    except Exception as e:
        return {"error": f"Failed to save meal plan: {str(e)}"}


@tool
def get_meal_plan(week_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieve the stored meal plan for a week. Use it to answer "what's for
    dinner today?", check compliance, or build a grocery/shopping list from
    the plan's ingredients.

    Args:
        week_id: Week in YYYY-Wnn format (default: current week)

    Returns:
        The stored meal plan, or a message if none exists
    """
    memory = _get_memory()
    if not memory:
        return {"error": "No user context available"}

    try:
        plan = memory.load_meal_plan(week_id)
        if plan is None:
            return {
                "found": False,
                "message": "No meal plan stored for this week. Offer to create one.",
            }
        return {
            "found": True,
            "week_id": plan.week_id,
            "plan_text": plan.plan_text,
            "notes": plan.notes,
            "created_at": plan.created_at,
        }
    except Exception as e:
        return {"error": f"Failed to load meal plan: {str(e)}"}


@tool
def generate_weekly_summary() -> Dict[str, Any]:
    """
    Aggregate the last 7 daily logs into a weekly summary (average calories,
    compliance, weight change) and store it. Use when the user asks for a
    weekly review or at the end of the week.

    Returns:
        The computed weekly statistics
    """
    memory = _get_memory()
    if not memory:
        return {"error": "No user context available"}

    try:
        from datetime import datetime, timedelta

        logs = memory.load_recent_daily_logs(n=7)
        if not logs:
            return {"error": "No daily logs available for a weekly summary."}

        today = date.today()
        week_id = today.strftime("%G-W%V")
        start = today - timedelta(days=today.weekday())

        calories = [l.total_calories for l in logs if l.total_calories is not None]
        protein = [l.total_protein_g for l in logs if l.total_protein_g is not None]
        compliance = [l.compliance_score for l in logs if l.compliance_score is not None]
        weights = [(l.date, l.weight_kg) for l in logs if l.weight_kg is not None]
        weights.sort(key=lambda t: t[0])

        summary = WeeklySummary(
            week_id=week_id,
            start_date=start.isoformat(),
            end_date=(start + timedelta(days=6)).isoformat(),
            avg_daily_calories=round(sum(calories) / len(calories), 1) if calories else None,
            avg_daily_protein_g=round(sum(protein) / len(protein), 1) if protein else None,
            avg_compliance_score=round(sum(compliance) / len(compliance), 2) if compliance else None,
            weight_start=weights[0][1] if weights else None,
            weight_end=weights[-1][1] if weights else None,
            weight_change=round(weights[-1][1] - weights[0][1], 1) if len(weights) > 1 else None,
            days_logged=len(logs),
        )
        memory.save_weekly_summary(summary)

        return {
            "saved": True,
            "week_id": week_id,
            "days_logged": summary.days_logged,
            "avg_daily_calories": summary.avg_daily_calories,
            "avg_daily_protein_g": summary.avg_daily_protein_g,
            "avg_compliance_score": summary.avg_compliance_score,
            "weight_change_kg": summary.weight_change,
        }

    except Exception as e:
        return {"error": f"Failed to generate weekly summary: {str(e)}"}


# List of all tools for the agent
ALL_TOOLS = [
    calculate_personalized_nutrition_targets,
    log_daily_intake,
    get_progress_summary,
    update_user_profile,
    analyze_food_image,
    log_water_intake,
    lookup_food_nutrition,
    get_remaining_daily_budget,
    save_meal_plan,
    get_meal_plan,
    generate_weekly_summary,
]
