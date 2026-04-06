"""
Structured memory system for NutriCoach.
Replaces brute-force message replay with bounded, per-user file storage.
"""

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

from .schemas import (
    UserProfile,
    NutritionTargets,
    DailyLog,
    WeeklySummary,
    ConversationEntry,
)

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages per-user structured memory on disk."""

    def __init__(self, username: str, base_dir: Path):
        self.username = username
        self.user_dir = base_dir / username
        self.user_dir.mkdir(parents=True, exist_ok=True)
        (self.user_dir / "daily_logs").mkdir(exist_ok=True)
        (self.user_dir / "weekly_summaries").mkdir(exist_ok=True)

    # --- User Profile ---

    def load_user_profile(self) -> Optional[UserProfile]:
        path = self.user_dir / "user_profile.json"
        data = self._read_json(path)
        if data is None:
            return None
        return UserProfile(**data)

    def save_user_profile(self, profile: UserProfile) -> None:
        path = self.user_dir / "user_profile.json"
        self._write_json(path, profile.model_dump())

    # --- Nutrition Targets ---

    def load_nutrition_targets(self) -> Optional[NutritionTargets]:
        path = self.user_dir / "nutrition_targets.json"
        data = self._read_json(path)
        if data is None:
            return None
        return NutritionTargets(**data)

    def save_nutrition_targets(self, targets: NutritionTargets) -> None:
        path = self.user_dir / "nutrition_targets.json"
        self._write_json(path, targets.model_dump())

    # --- Daily Logs ---

    def load_daily_log(self, log_date: str) -> Optional[DailyLog]:
        """Load a daily log by date string (YYYY-MM-DD)."""
        path = self.user_dir / "daily_logs" / f"{log_date}.json"
        data = self._read_json(path)
        if data is None:
            return None
        return DailyLog(**data)

    def load_todays_log(self) -> Optional[DailyLog]:
        return self.load_daily_log(date.today().isoformat())

    def save_daily_log(self, log: DailyLog) -> None:
        path = self.user_dir / "daily_logs" / f"{log.date}.json"
        self._write_json(path, log.model_dump())

    def load_recent_daily_logs(self, n: int = 3) -> List[DailyLog]:
        """Load the most recent n daily logs, sorted newest first."""
        logs_dir = self.user_dir / "daily_logs"
        if not logs_dir.exists():
            return []

        log_files = sorted(logs_dir.glob("*.json"), reverse=True)
        logs = []
        for f in log_files[:n]:
            data = self._read_json(f)
            if data is not None:
                logs.append(DailyLog(**data))
        return logs

    # --- Weekly Summaries ---

    def load_weekly_summary(self, week_id: str) -> Optional[WeeklySummary]:
        path = self.user_dir / "weekly_summaries" / f"{week_id}.json"
        data = self._read_json(path)
        if data is None:
            return None
        return WeeklySummary(**data)

    def load_current_week_summary(self) -> Optional[WeeklySummary]:
        week_id = date.today().strftime("%G-W%V")
        return self.load_weekly_summary(week_id)

    def save_weekly_summary(self, summary: WeeklySummary) -> None:
        path = self.user_dir / "weekly_summaries" / f"{summary.week_id}.json"
        self._write_json(path, summary.model_dump())

    # --- Conversation Index ---

    def load_conversation_index(self) -> List[ConversationEntry]:
        path = self.user_dir / "conversation_index.json"
        data = self._read_json(path)
        if data is None:
            return []
        return [ConversationEntry(**entry) for entry in data]

    def append_conversation_entry(self, entry: ConversationEntry) -> None:
        entries = self.load_conversation_index()
        entries.append(entry)
        # Keep only last 50 entries
        entries = entries[-50:]
        path = self.user_dir / "conversation_index.json"
        self._write_json(path, [e.model_dump() for e in entries])

    # --- Context Assembly ---

    def assemble_context(self) -> str:
        """
        Build a bounded context string for LLM injection.
        Includes: user profile + targets + today's log + recent summaries + weekly trends.
        Designed to stay under ~1000 tokens.
        """
        sections = []

        # User profile
        profile = self.load_user_profile()
        if profile:
            sections.append(self._format_profile_context(profile))

        # Nutrition targets
        targets = self.load_nutrition_targets()
        if targets:
            sections.append(self._format_targets_context(targets))

        # Today's log
        todays_log = self.load_todays_log()
        if todays_log:
            sections.append(self._format_daily_log_context(todays_log))

        # Recent daily logs (last 3, excluding today)
        recent_logs = self.load_recent_daily_logs(n=4)
        recent_logs = [l for l in recent_logs if l.date != date.today().isoformat()][:3]
        if recent_logs:
            summaries = []
            for log in recent_logs:
                parts = [f"  {log.date}:"]
                if log.total_calories is not None:
                    parts.append(f"calories={log.total_calories}")
                if log.compliance_score is not None:
                    parts.append(f"compliance={log.compliance_score:.0%}")
                if log.weight_kg is not None:
                    parts.append(f"weight={log.weight_kg}kg")
                summaries.append(" ".join(parts))
            sections.append("RECENT DAYS:\n" + "\n".join(summaries))

        # Weekly summary
        weekly = self.load_current_week_summary()
        if weekly:
            sections.append(self._format_weekly_context(weekly))

        if not sections:
            return ""

        return "USER CONTEXT:\n" + "\n\n".join(sections)

    # --- Migration ---

    def migrate_from_legacy(self, legacy_user_info: Dict[str, Any]) -> None:
        """
        One-time migration from legacy users.json profile to structured memory.
        Only migrates if user_profile.json doesn't already exist.
        """
        if (self.user_dir / "user_profile.json").exists():
            return

        # Map legacy fields to UserProfile
        try:
            meal_schedule_data = legacy_user_info.get("meal_schedule", {})
            profile = UserProfile(
                weight=legacy_user_info.get("weight", 70),
                height=legacy_user_info.get("height", 170),
                age=legacy_user_info.get("age", 30),
                gender=legacy_user_info.get("gender", "male"),
                activity_level=legacy_user_info.get("activity_level", "moderate"),
                primary_goal=legacy_user_info.get("primary_goal", "maintain_weight"),
                dietary_preferences=legacy_user_info.get("dietary_preferences", []),
                cooking_experience=legacy_user_info.get("cooking_experience", "intermediate"),
                budget_range=legacy_user_info.get("budget_range", "$100-150"),
                health_conditions=legacy_user_info.get("health_conditions", []),
                water_intake_goal=legacy_user_info.get("water_intake_goal", "8 glasses"),
                favorite_cuisines=legacy_user_info.get("favorite_cuisines", []),
                foods_to_avoid=legacy_user_info.get("foods_to_avoid", ""),
                additional_notes=legacy_user_info.get("additional_notes", ""),
            )
            self.save_user_profile(profile)
            logger.info(f"Migrated legacy profile for user '{self.username}'")
        except Exception as e:
            logger.error(f"Failed to migrate profile for '{self.username}': {e}")

    # --- Private helpers ---

    def _format_profile_context(self, profile: UserProfile) -> str:
        lines = [
            "PROFILE:",
            f"  {profile.age}yo {profile.gender}, {profile.weight}kg, {profile.height}cm",
            f"  Activity: {profile.activity_level}, Goal: {profile.primary_goal}",
        ]
        if profile.dietary_preferences:
            lines.append(f"  Diet: {', '.join(profile.dietary_preferences)}")
        if profile.health_conditions:
            lines.append(f"  Health: {', '.join(profile.health_conditions)}")
        if profile.foods_to_avoid:
            lines.append(f"  Avoid: {profile.foods_to_avoid}")
        return "\n".join(lines)

    def _format_targets_context(self, targets: NutritionTargets) -> str:
        return (
            f"DAILY TARGETS: {targets.target_calories}cal, "
            f"P:{targets.target_protein_g:.0f}g, "
            f"C:{targets.target_carbs_g:.0f}g, "
            f"F:{targets.target_fat_g:.0f}g, "
            f"Water:{targets.target_water_ml:.0f}ml"
        )

    def _format_daily_log_context(self, log: DailyLog) -> str:
        lines = [f"TODAY ({log.date}):"]
        if log.meals:
            for meal in log.meals:
                status = "followed" if meal.followed_plan else "deviated"
                lines.append(f"  {meal.meal_type}: {meal.description[:60]} ({status})")
        if log.total_calories is not None:
            lines.append(f"  Total: {log.total_calories}cal")
        if log.weight_kg is not None:
            lines.append(f"  Weight: {log.weight_kg}kg")
        return "\n".join(lines)

    def _format_weekly_context(self, summary: WeeklySummary) -> str:
        lines = [f"WEEK ({summary.week_id}):"]
        if summary.avg_daily_calories is not None:
            lines.append(f"  Avg calories: {summary.avg_daily_calories:.0f}")
        if summary.avg_compliance_score is not None:
            lines.append(f"  Avg compliance: {summary.avg_compliance_score:.0%}")
        if summary.weight_change is not None:
            direction = "+" if summary.weight_change > 0 else ""
            lines.append(f"  Weight change: {direction}{summary.weight_change:.1f}kg")
        if summary.trends:
            lines.append(f"  Trends: {summary.trends[:100]}")
        return "\n".join(lines)

    def _read_json(self, path: Path) -> Optional[Any]:
        try:
            if path.exists() and path.stat().st_size > 0:
                with open(path, "r") as f:
                    return json.loads(f.read().strip())
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to read {path}: {e}")
        return None

    def _write_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
