"""Tests for shared.memory module."""

import sys
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.memory import MemoryManager
from shared.schemas import (
    UserProfile,
    NutritionTargets,
    DailyLog,
    MealEntry,
    WeeklySummary,
    ConversationEntry,
)


def _make_profile(**overrides):
    defaults = dict(weight=75.0, height=175.0, age=30, gender="male")
    defaults.update(overrides)
    return UserProfile(**defaults)


def _make_targets(**overrides):
    defaults = dict(
        target_calories=2100,
        target_protein_g=120.0,
        target_carbs_g=260.0,
        target_fat_g=58.0,
        target_water_ml=2625.0,
        bmr=1700.0,
        tdee=2635.0,
        bmi=24.5,
        bmi_classification="Normal weight",
    )
    defaults.update(overrides)
    return NutritionTargets(**defaults)


class TestUserProfile:
    def test_save_and_load(self, tmp_path):
        mm = MemoryManager("alice", tmp_path)
        profile = _make_profile(age=25)
        mm.save_user_profile(profile)

        loaded = mm.load_user_profile()
        assert loaded is not None
        assert loaded.age == 25
        assert loaded.weight == 75.0

    def test_load_nonexistent(self, tmp_path):
        mm = MemoryManager("bob", tmp_path)
        assert mm.load_user_profile() is None


class TestNutritionTargets:
    def test_save_and_load(self, tmp_path):
        mm = MemoryManager("alice", tmp_path)
        targets = _make_targets(target_calories=2200)
        mm.save_nutrition_targets(targets)

        loaded = mm.load_nutrition_targets()
        assert loaded is not None
        assert loaded.target_calories == 2200

    def test_load_nonexistent(self, tmp_path):
        mm = MemoryManager("bob", tmp_path)
        assert mm.load_nutrition_targets() is None


class TestDailyLogs:
    def test_save_and_load(self, tmp_path):
        mm = MemoryManager("alice", tmp_path)
        log = DailyLog(
            date="2026-03-22",
            weight_kg=74.5,
            meals=[MealEntry(meal_type="breakfast", description="Oatmeal with berries")],
            total_calories=450,
            compliance_score=0.9,
        )
        mm.save_daily_log(log)

        loaded = mm.load_daily_log("2026-03-22")
        assert loaded is not None
        assert loaded.weight_kg == 74.5
        assert len(loaded.meals) == 1
        assert loaded.meals[0].description == "Oatmeal with berries"

    def test_load_nonexistent_date(self, tmp_path):
        mm = MemoryManager("alice", tmp_path)
        assert mm.load_daily_log("2099-01-01") is None

    def test_load_recent_logs_ordering(self, tmp_path):
        mm = MemoryManager("alice", tmp_path)

        # Save logs for 5 days
        for day in range(20, 25):
            log = DailyLog(date=f"2026-03-{day}", total_calories=1800 + day * 10)
            mm.save_daily_log(log)

        recent = mm.load_recent_daily_logs(n=3)
        assert len(recent) == 3
        # Should be newest first (sorted by filename descending)
        assert recent[0].date == "2026-03-24"
        assert recent[1].date == "2026-03-23"
        assert recent[2].date == "2026-03-22"

    def test_load_recent_logs_empty(self, tmp_path):
        mm = MemoryManager("alice", tmp_path)
        assert mm.load_recent_daily_logs() == []


class TestWeeklySummary:
    def test_save_and_load(self, tmp_path):
        mm = MemoryManager("alice", tmp_path)
        summary = WeeklySummary(
            week_id="2026-W12",
            start_date="2026-03-16",
            end_date="2026-03-22",
            avg_daily_calories=2050.0,
            avg_compliance_score=0.85,
            weight_start=75.0,
            weight_end=74.5,
            weight_change=-0.5,
            days_logged=7,
            trends="Steady calorie intake, good compliance",
        )
        mm.save_weekly_summary(summary)

        loaded = mm.load_weekly_summary("2026-W12")
        assert loaded is not None
        assert loaded.avg_daily_calories == 2050.0
        assert loaded.weight_change == -0.5

    def test_load_nonexistent(self, tmp_path):
        mm = MemoryManager("alice", tmp_path)
        assert mm.load_weekly_summary("2099-W01") is None


class TestConversationIndex:
    def test_append_and_load(self, tmp_path):
        mm = MemoryManager("alice", tmp_path)
        entry = ConversationEntry(
            timestamp="2026-03-22T10:00:00",
            summary="Set initial nutrition targets",
            key_decisions=["Target 2100 cal/day"],
            tags=["targets", "setup"],
        )
        mm.append_conversation_entry(entry)

        entries = mm.load_conversation_index()
        assert len(entries) == 1
        assert entries[0].summary == "Set initial nutrition targets"

    def test_max_50_entries(self, tmp_path):
        mm = MemoryManager("alice", tmp_path)
        for i in range(60):
            entry = ConversationEntry(
                timestamp=f"2026-03-22T{i:02d}:00:00",
                summary=f"Entry {i}",
            )
            mm.append_conversation_entry(entry)

        entries = mm.load_conversation_index()
        assert len(entries) == 50
        # Should keep the most recent
        assert entries[-1].summary == "Entry 59"
        assert entries[0].summary == "Entry 10"


class TestAssembleContext:
    def test_empty_context(self, tmp_path):
        mm = MemoryManager("alice", tmp_path)
        assert mm.assemble_context() == ""

    def test_with_profile_only(self, tmp_path):
        mm = MemoryManager("alice", tmp_path)
        mm.save_user_profile(_make_profile(age=28, gender="female"))

        ctx = mm.assemble_context()
        assert "PROFILE:" in ctx
        assert "28yo female" in ctx

    def test_with_full_data(self, tmp_path):
        mm = MemoryManager("alice", tmp_path)
        mm.save_user_profile(_make_profile())
        mm.save_nutrition_targets(_make_targets())

        # Save a log for today
        today = date.today().isoformat()
        log = DailyLog(
            date=today,
            weight_kg=74.5,
            meals=[MealEntry(meal_type="breakfast", description="Eggs and toast")],
            total_calories=350,
        )
        mm.save_daily_log(log)

        ctx = mm.assemble_context()
        assert "USER CONTEXT:" in ctx
        assert "PROFILE:" in ctx
        assert "DAILY TARGETS:" in ctx
        assert "TODAY" in ctx
        assert "breakfast" in ctx

    def test_with_weekly_summary(self, tmp_path):
        mm = MemoryManager("alice", tmp_path)
        mm.save_user_profile(_make_profile())

        week_id = date.today().strftime("%G-W%V")
        summary = WeeklySummary(
            week_id=week_id,
            start_date="2026-03-16",
            end_date="2026-03-22",
            avg_daily_calories=2000.0,
            avg_compliance_score=0.9,
            weight_change=-0.3,
            trends="Good progress",
        )
        mm.save_weekly_summary(summary)

        ctx = mm.assemble_context()
        assert "WEEK" in ctx
        assert "Good progress" in ctx


class TestMigration:
    def test_migrate_from_legacy(self, tmp_path):
        mm = MemoryManager("alice", tmp_path)
        legacy = {
            "weight": 65.0,
            "height": 165.0,
            "age": 28,
            "gender": "female",
            "activity_level": "active",
            "primary_goal": "lose_weight",
            "dietary_preferences": ["vegetarian"],
            "favorite_cuisines": ["Italian", "Thai"],
            "foods_to_avoid": "mushrooms",
        }
        mm.migrate_from_legacy(legacy)

        profile = mm.load_user_profile()
        assert profile is not None
        assert profile.weight == 65.0
        assert profile.gender == "female"
        assert "vegetarian" in profile.dietary_preferences

    def test_migrate_skips_if_exists(self, tmp_path):
        mm = MemoryManager("alice", tmp_path)
        mm.save_user_profile(_make_profile(age=30))

        # Try to migrate with different data
        mm.migrate_from_legacy({"weight": 99, "height": 199, "age": 99, "gender": "male"})

        # Original should be preserved
        profile = mm.load_user_profile()
        assert profile.age == 30
