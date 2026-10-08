"""Validate photo corrections and recompute nutrients without another API call."""

import math

from .base import FoodAnalysisResult, FoodItem
from .nutrition_db import NutritionDB, NutrientInfo

NUTRIENT_FIELDS = ("calories", "protein_g", "carbs_g", "fat_g")


def photo_edit_rows(original: dict) -> list:
    rows = []
    for index, item in enumerate(original["food_items"]):
        grams = float(item["quantity_grams"])
        if not math.isfinite(grams) or grams <= 0:
            raise ValueError("The photo estimate contains an invalid quantity. Please analyze again.")
        density = {field: float(item[field]) * 100 / grams for field in NUTRIENT_FIELDS}
        NutrientInfo(**density).scaled(grams)
        rows.append({"original_index": index, "name": item["name"], "quantity_grams": grams,
                     "nutrition_source": "Photo estimate", **density})
    return rows


def apply_photo_edits(original: dict, rows: list) -> FoodAnalysisResult:
    if not rows:
        raise ValueError("Add at least one ingredient before saving the meal.")
    result = FoodAnalysisResult(method=original["method"])
    database = NutritionDB()
    for row in rows:
        name = row.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Every ingredient needs a name.")
        name = name.strip()
        source = row.get("nutrition_source")
        if source == "Photo estimate":
            index = row.get("original_index")
            if not isinstance(index, (int, float)) or not math.isfinite(index) or int(index) != index:
                raise ValueError("Choose a database food or Manual for new ingredients.")
            if not 0 <= index < len(original["food_items"]):
                raise ValueError("Invalid original ingredient.")
            item = original["food_items"][int(index)]
            if name != item["name"].strip():
                raise ValueError("For renamed ingredients, choose a matching database food or Manual nutrition.")
            density = NutrientInfo(**{field: float(item[field]) * 100 / float(item["quantity_grams"])
                                      for field in NUTRIENT_FIELDS})
        elif source == "Manual":
            density = NutrientInfo(**{field: row[field] for field in NUTRIENT_FIELDS})
        elif isinstance(source, str) and source.startswith("Database: ") and source[10:] in database.db:
            density = database.db[source[10:]]
        else:
            raise ValueError("Choose a nutrition source for every ingredient.")
        grams = float(row["quantity_grams"])
        result.food_items.append(FoodItem(name=name, quantity_grams=grams,
                                          portion_description=f"Reviewed by user; {source}",
                                          **density.scaled(grams)))
    result.compute_totals()
    return result
