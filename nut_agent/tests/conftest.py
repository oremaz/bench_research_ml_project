"""Shared test fixtures."""

import json
import pytest
from pathlib import Path


@pytest.fixture
def sample_user_profile():
    return {
        "weight": 75.0,
        "height": 175.0,
        "age": 30,
        "gender": "male",
        "activity_level": "moderate",
        "primary_goal": "maintain_weight",
        "dietary_preferences": ["vegetarian"],
        "cooking_experience": "intermediate",
        "budget_range": "$100-150",
        "meal_schedule": {
            "breakfast": {"enabled": True, "location": "home", "cooking_time": "15"},
            "lunch": {"enabled": True, "location": "work", "cooking_time": "10"},
            "dinner": {"enabled": True, "location": "home", "cooking_time": "30"},
            "snacks": {"enabled": True, "frequency": "2", "type": "healthy"},
        },
        "health_conditions": [],
        "water_intake_goal": "8 glasses",
        "favorite_cuisines": ["Italian", "Japanese"],
        "foods_to_avoid": "shellfish",
        "additional_notes": "",
    }


@pytest.fixture
def temp_secrets_dir(tmp_path):
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    return secrets


@pytest.fixture
def users_file(temp_secrets_dir):
    return temp_secrets_dir / "users.json"


@pytest.fixture
def sample_users_json(users_file, sample_user_profile):
    """Create a users.json with one test user (legacy SHA-256 hash)."""
    import hashlib

    profile = sample_user_profile.copy()
    profile["password_hash"] = hashlib.sha256("testpass123".encode()).hexdigest()
    profile["registration_date"] = "2026-03-01T10:00:00"

    data = {"testuser": profile}
    with open(users_file, "w") as f:
        json.dump(data, f)
    return users_file
