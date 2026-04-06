"""Tests for shared.utils module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.utils import calculate_bmi, validate_nutrition_targets


class TestCalculateBMI:
    def test_normal_weight(self):
        result = calculate_bmi(70, 175)
        assert result["classification"] == "Normal weight"
        assert 22.0 <= result["bmi"] <= 23.0

    def test_underweight(self):
        result = calculate_bmi(50, 180)
        assert result["classification"] == "Underweight"
        assert result["bmi"] < 18.5

    def test_overweight(self):
        result = calculate_bmi(85, 170)
        assert result["classification"] == "Overweight"
        assert 25.0 <= result["bmi"] < 30.0

    def test_obese(self):
        result = calculate_bmi(120, 170)
        assert result["classification"] == "Obese"
        assert result["bmi"] >= 30.0

    def test_healthy_weight_range(self):
        result = calculate_bmi(70, 175)
        hw = result["healthy_weight_range"]
        assert hw["min"] < hw["max"]
        assert hw["min"] > 0
        # 175cm -> 1.75m, healthy range ~56.7 to 76.3
        assert 55 < hw["min"] < 60
        assert 74 < hw["max"] < 78


class TestValidateNutritionTargets:
    def test_valid_targets(self):
        targets = {"target_calories": 2000, "target_protein_g": 120}
        result = validate_nutrition_targets(targets)
        assert result["is_valid"] is True
        assert len(result["warnings"]) == 0
        assert result["targets"]["target_calories"] == 2000

    def test_low_calories_warning(self):
        targets = {"target_calories": 800, "target_protein_g": 100}
        result = validate_nutrition_targets(targets)
        assert result["is_valid"] is False
        assert result["targets"]["target_calories"] == 1200  # clamped
        assert any("low" in w.lower() for w in result["warnings"])

    def test_high_calories_warning(self):
        targets = {"target_calories": 5000, "target_protein_g": 100}
        result = validate_nutrition_targets(targets)
        assert result["is_valid"] is False
        assert result["targets"]["target_calories"] == 4000  # clamped

    def test_low_protein_warning(self):
        targets = {"target_calories": 2000, "target_protein_g": 30}
        result = validate_nutrition_targets(targets)
        assert result["is_valid"] is False
        assert any("protein" in w.lower() for w in result["warnings"])

    def test_high_protein_warning(self):
        targets = {"target_calories": 2000, "target_protein_g": 250}
        result = validate_nutrition_targets(targets)
        assert result["is_valid"] is False
        assert any("protein" in w.lower() for w in result["warnings"])

    def test_missing_values_default_to_zero(self):
        result = validate_nutrition_targets({})
        assert result["targets"]["target_calories"] == 1200  # clamped from 0
