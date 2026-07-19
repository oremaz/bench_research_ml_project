"""
Pydantic models for structured data in NutriCoach and Recipe Lab.
"""

from datetime import date, datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class MealScheduleEntry(BaseModel):
    enabled: bool = True
    location: str = "home"
    cooking_time: str = "15"


class SnackSchedule(BaseModel):
    enabled: bool = True
    frequency: str = "2"
    type: str = "healthy"


class MealSchedule(BaseModel):
    breakfast: MealScheduleEntry = Field(default_factory=MealScheduleEntry)
    lunch: MealScheduleEntry = Field(default_factory=MealScheduleEntry)
    dinner: MealScheduleEntry = Field(default_factory=MealScheduleEntry)
    snacks: SnackSchedule = Field(default_factory=SnackSchedule)


class UserProfile(BaseModel):
    """Static user profile from registration."""
    weight: float
    height: float
    age: int
    gender: str
    activity_level: str = "moderate"
    primary_goal: str = "maintain_weight"
    dietary_preferences: List[str] = Field(default_factory=list)
    cooking_experience: str = "intermediate"
    budget_range: str = "$100-150"
    meal_schedule: MealSchedule = Field(default_factory=MealSchedule)
    health_conditions: List[str] = Field(default_factory=list)
    water_intake_goal: str = "8 glasses"
    favorite_cuisines: List[str] = Field(default_factory=list)
    foods_to_avoid: str = ""
    additional_notes: str = ""


class NutritionTargets(BaseModel):
    """Computed daily nutrition targets."""
    target_calories: int
    target_protein_g: float
    target_carbs_g: float
    target_fat_g: float
    target_water_ml: float
    bmr: float
    tdee: float
    bmi: float
    bmi_classification: str
    healthy_weight_range: Dict[str, float] = Field(default_factory=dict)


class MealEntry(BaseModel):
    """A single meal logged by the user."""
    meal_type: str  # breakfast, lunch, dinner, snack
    description: str = ""
    followed_plan: bool = True
    differences: str = ""
    estimated_calories: Optional[int] = None
    estimated_protein_g: Optional[float] = None
    estimated_carbs_g: Optional[float] = None
    estimated_fat_g: Optional[float] = None


class DailyLog(BaseModel):
    """Daily tracking log."""
    date: str  # YYYY-MM-DD format
    weight_kg: Optional[float] = None
    meals: List[MealEntry] = Field(default_factory=list)
    total_calories: Optional[int] = None
    total_protein_g: Optional[float] = None
    total_carbs_g: Optional[float] = None
    total_fat_g: Optional[float] = None
    water_intake_ml: Optional[float] = None
    energy_level: Optional[str] = None  # low, moderate, high
    notes: str = ""
    compliance_score: Optional[float] = None  # 0-1


class WeeklySummary(BaseModel):
    """AI-generated weekly trend analysis."""
    week_id: str  # YYYY-Wnn format
    start_date: str
    end_date: str
    avg_daily_calories: Optional[float] = None
    avg_daily_protein_g: Optional[float] = None
    avg_compliance_score: Optional[float] = None
    weight_start: Optional[float] = None
    weight_end: Optional[float] = None
    weight_change: Optional[float] = None
    days_logged: int = 0
    trends: str = ""
    ai_notes: str = ""


class MealPlan(BaseModel):
    """A weekly meal plan agreed with the agent."""
    week_id: str  # YYYY-Wnn format
    plan_text: str
    created_at: str = ""
    notes: str = ""


class ConversationEntry(BaseModel):
    """Index entry for significant conversation decisions."""
    timestamp: str
    summary: str
    key_decisions: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
