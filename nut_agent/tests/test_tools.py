"""Tests for nutricoach.tools — daily-routine tools added in v2.1
(water, food lookup, remaining budget, meal plans, weekly summary, totals).

All tools run against a temporary secrets directory; no network or GPU needed.
"""

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import nutricoach.tools as tools_mod
from nutricoach.tools import (
    calculate_personalized_nutrition_targets,
    log_daily_intake,
    log_water_intake,
    lookup_food_nutrition,
    get_remaining_daily_budget,
    save_meal_plan,
    get_meal_plan,
    generate_weekly_summary,
    ALL_TOOLS,
    set_current_user,
)
from shared.memory import MemoryManager
from shared.schemas import DailyLog, MealEntry


@pytest.fixture
def user_env(tmp_path, monkeypatch):
    monkeypatch.setattr(tools_mod, "SECRETS_DIR", tmp_path)
    set_current_user("tooluser")
    yield tmp_path
    set_current_user(None)


def _memory(tmp_path):
    return MemoryManager("tooluser", tmp_path)


class TestToolRegistry:
    def test_all_tools_registered(self):
        names = {t.name for t in ALL_TOOLS}
        expected = {
            "calculate_personalized_nutrition_targets", "log_daily_intake",
            "get_progress_summary", "update_user_profile", "analyze_food_image",
            "log_water_intake", "lookup_food_nutrition", "get_remaining_daily_budget",
            "save_meal_plan", "get_meal_plan", "generate_weekly_summary",
        }
        assert names == expected


class TestLogDailyIntakeTotals:
    def test_totals_recomputed_from_estimates(self, user_env):
        r1 = log_daily_intake.func(
            "oatmeal with berries", meal_type="breakfast",
            estimated_calories=350, estimated_protein_g=12.0,
            estimated_carbs_g=60.0, estimated_fat_g=7.0,
        )
        assert r1["logged"] and r1["total_calories_today"] == 350

        r2 = log_daily_intake.func(
            "chicken salad", meal_type="lunch",
            estimated_calories=450, estimated_protein_g=40.0,
            estimated_carbs_g=15.0, estimated_fat_g=25.0,
        )
        assert r2["total_calories_today"] == 800

        log = _memory(user_env).load_todays_log()
        assert log.total_calories == 800
        assert log.total_protein_g == 52.0
        assert len(log.meals) == 2

    def test_totals_none_without_estimates(self, user_env):
        log_daily_intake.func("some unspecified meal")
        log = _memory(user_env).load_todays_log()
        assert log.total_calories is None


class TestWaterIntake:
    def test_accumulates(self, user_env):
        r1 = log_water_intake.func(250)
        r2 = log_water_intake.func(500)
        assert r1["water_today_ml"] == 250
        assert r2["water_today_ml"] == 750

    def test_reports_remaining_when_targets_exist(self, user_env):
        calculate_personalized_nutrition_targets.func(
            weight_kg=70, height_cm=175, age=30, gender="male",
            activity_level="moderate", weight_goal="maintain",
        )
        r = log_water_intake.func(1000)
        assert r["target_water_ml"] == 70 * 35
        assert r["remaining_ml"] == 70 * 35 - 1000


class TestLookupFoodNutrition:
    def test_exact_food(self, user_env):
        r = lookup_food_nutrition.func("chicken breast", grams=200)
        assert r["found"] is True
        assert r["calories"] == pytest.approx(330, abs=1)
        assert r["protein_g"] == pytest.approx(62, abs=1)

    def test_fuzzy_match(self, user_env):
        r = lookup_food_nutrition.func("grilled chicken breasts")
        assert r["found"] is True

    def test_unknown_food(self, user_env):
        r = lookup_food_nutrition.func("xyzzy nonexistent dish 42")
        assert r["found"] is False


class TestRemainingBudget:
    def test_requires_targets(self, user_env):
        r = get_remaining_daily_budget.func()
        assert "error" in r

    def test_computes_remaining(self, user_env):
        calculate_personalized_nutrition_targets.func(
            weight_kg=80, height_cm=180, age=30, gender="male",
            activity_level="moderate", weight_goal="lose",
        )
        log_daily_intake.func(
            "big breakfast", estimated_calories=600,
            estimated_protein_g=30.0, estimated_carbs_g=70.0, estimated_fat_g=20.0,
        )
        log_water_intake.func(500)

        r = get_remaining_daily_budget.func()
        assert r["consumed"]["calories"] == 600
        assert r["consumed"]["water_ml"] == 500
        # targets: 2259 kcal (computed and validated earlier in the suite)
        assert r["remaining"]["calories"] == 2259 - 600
        assert r["meals_logged"] == 1


class TestMealPlan:
    def test_save_and_get_roundtrip(self, user_env):
        plan = "Monday: oats + chicken salad + salmon. Tuesday: eggs + soup + stir fry."
        r = save_meal_plan.func(plan, notes="no shellfish")
        assert r["saved"] is True
        week_id = r["week_id"]
        assert week_id == date.today().strftime("%G-W%V")

        g = get_meal_plan.func()
        assert g["found"] is True
        assert g["plan_text"] == plan
        assert g["notes"] == "no shellfish"

        g2 = get_meal_plan.func(week_id)
        assert g2["found"] is True

    def test_get_without_plan(self, user_env):
        g = get_meal_plan.func()
        assert g["found"] is False

    def test_plan_appears_in_context(self, user_env):
        save_meal_plan.func("Mon: pancakes")
        ctx = _memory(user_env).assemble_context()
        assert "MEAL PLAN" in ctx
        assert "pancakes" in ctx


class TestWeeklySummary:
    def test_requires_logs(self, user_env):
        r = generate_weekly_summary.func()
        assert "error" in r

    def test_aggregates_recent_logs(self, user_env):
        memory = _memory(user_env)
        from datetime import timedelta
        today = date.today()
        for i, (cal, weight) in enumerate([(2000, 80.0), (2200, 79.6), (1800, 79.2)]):
            d = (today - timedelta(days=i)).isoformat()
            log = DailyLog(date=d, weight_kg=weight, total_calories=cal,
                           compliance_score=0.8)
            memory.save_daily_log(log)

        r = generate_weekly_summary.func()
        assert r["saved"] is True
        assert r["days_logged"] == 3
        assert r["avg_daily_calories"] == 2000
        assert r["avg_compliance_score"] == 0.8
        # i=0 is today: weight went from 79.2 (2 days ago) to 80.0 (today)
        assert r["weight_change_kg"] == pytest.approx(0.8)

        stored = memory.load_current_week_summary()
        assert stored is not None
        assert stored.days_logged == 3
