"""
Utility functions for the nutritionist agent.
"""

import json
from typing import Dict, Any, List
from datetime import datetime

def calculate_bmi(weight_kg: float, height_cm: float) -> Dict[str, Any]:
    """
    Calculate BMI and classification.
    
    Args:
        weight_kg: Weight in kilograms
        height_cm: Height in centimeters
        
    Returns:
        Dictionary with BMI and classification
    """
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    
    if bmi < 18.5:
        classification = "Underweight"
    elif bmi < 25:
        classification = "Normal weight"
    elif bmi < 30:
        classification = "Overweight"
    else:
        classification = "Obese"
    
    return {
        "bmi": round(bmi, 1),
        "classification": classification,
        "healthy_weight_range": {
            "min": round(18.5 * (height_m ** 2), 1),
            "max": round(24.9 * (height_m ** 2), 1)
        }
    }

def validate_nutrition_targets(targets: Dict[str, float]) -> Dict[str, Any]:
    """
    Validate and adjust nutrition targets if needed.
    
    Args:
        targets: Dictionary with nutrition targets
        
    Returns:
        Validated and adjusted targets with warnings if any
    """
    warnings = []
    adjusted_targets = targets.copy()
    
    # Check for reasonable calorie range
    calories = targets.get('target_calories', 0)
    if calories < 1200:
        warnings.append("Calorie target is very low. Consider consulting a healthcare provider.")
        adjusted_targets['target_calories'] = 1200
    elif calories > 4000:
        warnings.append("Calorie target is very high. Please verify your activity level.")
        adjusted_targets['target_calories'] = 4000
    
    # Check protein targets
    protein = targets.get('target_protein_g', 0)
    if protein < 50:
        warnings.append("Protein target may be too low for optimal health.")
    elif protein > 200:
        warnings.append("Very high protein target. Ensure adequate hydration.")
    
    return {
        'targets': adjusted_targets,
        'warnings': warnings,
        'is_valid': len(warnings) == 0
    }

def export_meal_plan_to_json(meal_plan: Dict[str, Any], filename: str = None) -> str:
    """
    Export meal plan to JSON format.
    
    Args:
        meal_plan: Meal plan dictionary
        filename: Optional filename
        
    Returns:
        JSON string of the meal plan
    """
    export_data = {
        'meal_plan': meal_plan,
        'export_date': datetime.now().isoformat(),
        'version': '1.0'
    }
    
    json_string = json.dumps(export_data, indent=2)
    
    if filename:
        with open(filename, 'w') as f:
            f.write(json_string)
    
    return json_string
