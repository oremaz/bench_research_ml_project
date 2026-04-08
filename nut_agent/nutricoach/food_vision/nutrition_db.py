"""
Local nutrition database for calorie/macro lookup.

Contains common foods with per-100g nutritional values.
Can be extended with USDA FoodData Central or CIQUAL data.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import difflib


@dataclass
class NutrientInfo:
    """Nutritional values per 100g of food."""
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float


# Common foods database — values per 100g
# Sources: USDA FoodData Central, CIQUAL (French database)
FOOD_DB: Dict[str, NutrientInfo] = {
    # Proteins
    "chicken breast": NutrientInfo(165, 31.0, 0.0, 3.6),
    "chicken thigh": NutrientInfo(209, 26.0, 0.0, 10.9),
    "grilled chicken": NutrientInfo(165, 31.0, 0.0, 3.6),
    "roasted chicken": NutrientInfo(190, 28.0, 0.0, 7.4),
    "beef steak": NutrientInfo(271, 26.0, 0.0, 18.0),
    "ground beef": NutrientInfo(254, 17.2, 0.0, 20.0),
    "pork chop": NutrientInfo(231, 25.7, 0.0, 13.5),
    "salmon": NutrientInfo(208, 20.4, 0.0, 13.4),
    "tuna": NutrientInfo(130, 28.0, 0.0, 1.0),
    "shrimp": NutrientInfo(99, 24.0, 0.2, 0.3),
    "egg": NutrientInfo(155, 13.0, 1.1, 11.0),
    "boiled egg": NutrientInfo(155, 13.0, 1.1, 11.0),
    "fried egg": NutrientInfo(196, 13.6, 0.8, 15.3),
    "tofu": NutrientInfo(76, 8.0, 1.9, 4.8),
    "turkey breast": NutrientInfo(135, 30.0, 0.0, 1.0),
    # Carbs / Grains
    "white rice": NutrientInfo(130, 2.7, 28.2, 0.3),
    "brown rice": NutrientInfo(123, 2.6, 25.6, 1.0),
    "pasta": NutrientInfo(131, 5.0, 25.0, 1.1),
    "spaghetti": NutrientInfo(131, 5.0, 25.0, 1.1),
    "bread": NutrientInfo(265, 9.0, 49.0, 3.2),
    "whole wheat bread": NutrientInfo(247, 13.0, 41.0, 3.4),
    "baguette": NutrientInfo(274, 9.0, 56.0, 1.0),
    "croissant": NutrientInfo(406, 8.2, 45.5, 21.0),
    "oatmeal": NutrientInfo(68, 2.5, 12.0, 1.4),
    "quinoa": NutrientInfo(120, 4.4, 21.3, 1.9),
    "couscous": NutrientInfo(112, 3.8, 23.2, 0.2),
    "potato": NutrientInfo(77, 2.0, 17.5, 0.1),
    "mashed potato": NutrientInfo(83, 1.9, 14.0, 2.3),
    "french fries": NutrientInfo(312, 3.4, 41.4, 15.0),
    "sweet potato": NutrientInfo(86, 1.6, 20.1, 0.1),
    "noodles": NutrientInfo(138, 4.5, 25.6, 2.1),
    # Vegetables
    "broccoli": NutrientInfo(34, 2.8, 7.0, 0.4),
    "spinach": NutrientInfo(23, 2.9, 3.6, 0.4),
    "carrot": NutrientInfo(41, 0.9, 9.6, 0.2),
    "tomato": NutrientInfo(18, 0.9, 3.9, 0.2),
    "cucumber": NutrientInfo(15, 0.7, 3.6, 0.1),
    "bell pepper": NutrientInfo(31, 1.0, 6.0, 0.3),
    "onion": NutrientInfo(40, 1.1, 9.3, 0.1),
    "mushroom": NutrientInfo(22, 3.1, 3.3, 0.3),
    "zucchini": NutrientInfo(17, 1.2, 3.1, 0.3),
    "green beans": NutrientInfo(31, 1.8, 7.0, 0.1),
    "corn": NutrientInfo(96, 3.4, 21.0, 1.5),
    "peas": NutrientInfo(81, 5.4, 14.5, 0.4),
    "lettuce": NutrientInfo(15, 1.4, 2.9, 0.2),
    "cabbage": NutrientInfo(25, 1.3, 5.8, 0.1),
    "cauliflower": NutrientInfo(25, 1.9, 5.0, 0.3),
    "avocado": NutrientInfo(160, 2.0, 8.5, 14.7),
    "eggplant": NutrientInfo(25, 1.0, 6.0, 0.2),
    "mixed salad": NutrientInfo(20, 1.5, 3.5, 0.2),
    "green salad": NutrientInfo(15, 1.4, 2.9, 0.2),
    # Fruits
    "apple": NutrientInfo(52, 0.3, 13.8, 0.2),
    "banana": NutrientInfo(89, 1.1, 22.8, 0.3),
    "orange": NutrientInfo(47, 0.9, 11.8, 0.1),
    "strawberry": NutrientInfo(32, 0.7, 7.7, 0.3),
    "grape": NutrientInfo(69, 0.7, 18.1, 0.2),
    "watermelon": NutrientInfo(30, 0.6, 7.6, 0.2),
    "mango": NutrientInfo(60, 0.8, 15.0, 0.4),
    "pineapple": NutrientInfo(50, 0.5, 13.1, 0.1),
    "blueberry": NutrientInfo(57, 0.7, 14.5, 0.3),
    # Dairy
    "cheese": NutrientInfo(402, 25.0, 1.3, 33.1),
    "cheddar cheese": NutrientInfo(402, 25.0, 1.3, 33.1),
    "mozzarella": NutrientInfo(280, 28.0, 3.1, 17.1),
    "parmesan": NutrientInfo(431, 38.5, 4.1, 29.0),
    "cream cheese": NutrientInfo(342, 5.9, 4.1, 34.2),
    "yogurt": NutrientInfo(59, 3.5, 4.7, 3.3),
    "greek yogurt": NutrientInfo(97, 9.0, 3.6, 5.0),
    "milk": NutrientInfo(42, 3.4, 5.0, 1.0),
    "butter": NutrientInfo(717, 0.9, 0.1, 81.1),
    # Legumes
    "lentils": NutrientInfo(116, 9.0, 20.1, 0.4),
    "chickpeas": NutrientInfo(164, 8.9, 27.4, 2.6),
    "black beans": NutrientInfo(132, 8.9, 23.7, 0.5),
    "kidney beans": NutrientInfo(127, 8.7, 22.8, 0.5),
    "hummus": NutrientInfo(166, 7.9, 14.3, 9.6),
    # Common dishes
    "pizza": NutrientInfo(266, 11.0, 33.0, 10.4),
    "pizza slice": NutrientInfo(266, 11.0, 33.0, 10.4),
    "hamburger": NutrientInfo(295, 17.0, 24.0, 14.0),
    "cheeseburger": NutrientInfo(303, 17.0, 24.0, 15.0),
    "hot dog": NutrientInfo(290, 10.0, 24.0, 18.0),
    "sandwich": NutrientInfo(250, 12.0, 28.0, 10.0),
    "burrito": NutrientInfo(206, 9.0, 25.0, 8.0),
    "sushi roll": NutrientInfo(140, 5.0, 25.0, 2.0),
    "sushi": NutrientInfo(140, 5.0, 25.0, 2.0),
    "fried rice": NutrientInfo(163, 4.0, 22.0, 6.5),
    "stir fry": NutrientInfo(120, 10.0, 8.0, 5.0),
    "soup": NutrientInfo(45, 2.5, 6.0, 1.0),
    "chicken soup": NutrientInfo(55, 5.0, 5.0, 1.5),
    "vegetable soup": NutrientInfo(40, 1.5, 7.0, 0.5),
    "ramen": NutrientInfo(188, 5.0, 26.0, 7.0),
    "pad thai": NutrientInfo(155, 6.0, 22.0, 5.0),
    "curry": NutrientInfo(130, 8.0, 10.0, 7.0),
    "chicken curry": NutrientInfo(150, 12.0, 8.0, 8.0),
    "stew": NutrientInfo(100, 8.0, 8.0, 4.0),
    "lasagna": NutrientInfo(135, 8.0, 13.0, 5.5),
    "mac and cheese": NutrientInfo(164, 6.5, 18.0, 7.5),
    "tacos": NutrientInfo(226, 9.0, 20.0, 12.0),
    "omelette": NutrientInfo(154, 11.0, 1.6, 12.0),
    "pancake": NutrientInfo(227, 6.3, 28.0, 10.0),
    "waffle": NutrientInfo(291, 7.9, 33.0, 14.1),
    "crepe": NutrientInfo(160, 4.5, 20.0, 6.5),
    # Snacks / Sides
    "chips": NutrientInfo(536, 7.0, 53.0, 35.0),
    "popcorn": NutrientInfo(375, 11.0, 74.0, 4.5),
    "nuts": NutrientInfo(607, 20.0, 21.0, 54.0),
    "almonds": NutrientInfo(579, 21.2, 21.6, 49.9),
    "peanut butter": NutrientInfo(588, 25.1, 20.0, 50.4),
    "granola bar": NutrientInfo(471, 10.0, 64.0, 20.0),
    "chocolate": NutrientInfo(546, 5.0, 60.0, 31.0),
    "dark chocolate": NutrientInfo(546, 5.0, 60.0, 31.0),
    "ice cream": NutrientInfo(207, 3.5, 24.0, 11.0),
    "cake": NutrientInfo(347, 5.0, 52.0, 14.0),
    "cookie": NutrientInfo(488, 5.7, 64.0, 23.0),
    "brownie": NutrientInfo(405, 5.0, 50.0, 21.0),
    "donut": NutrientInfo(452, 5.0, 51.0, 25.0),
    # Sauces / Condiments
    "ketchup": NutrientInfo(112, 1.7, 26.0, 0.1),
    "mayonnaise": NutrientInfo(680, 1.0, 0.6, 75.0),
    "olive oil": NutrientInfo(884, 0.0, 0.0, 100.0),
    "soy sauce": NutrientInfo(53, 8.1, 4.9, 0.0),
    "vinaigrette": NutrientInfo(200, 0.3, 8.0, 18.0),
    # Drinks
    "orange juice": NutrientInfo(45, 0.7, 10.4, 0.2),
    "coffee": NutrientInfo(2, 0.3, 0.0, 0.0),
    "tea": NutrientInfo(1, 0.0, 0.3, 0.0),
    "smoothie": NutrientInfo(65, 1.5, 13.0, 0.5),
    "beer": NutrientInfo(43, 0.5, 3.6, 0.0),
    "wine": NutrientInfo(83, 0.1, 2.6, 0.0),
}


class NutritionDB:
    """Lookup nutritional info for food items with fuzzy matching."""

    def __init__(self, extra_foods: Optional[Dict[str, NutrientInfo]] = None):
        self.db = dict(FOOD_DB)
        if extra_foods:
            self.db.update(extra_foods)
        self._names = list(self.db.keys())

    def lookup(self, food_name: str) -> Optional[NutrientInfo]:
        """Exact or fuzzy match a food name and return per-100g nutrients."""
        key = food_name.lower().strip()
        if key in self.db:
            return self.db[key]
        # Fuzzy match
        matches = difflib.get_close_matches(key, self._names, n=1, cutoff=0.6)
        if matches:
            return self.db[matches[0]]
        return None

    def lookup_with_name(self, food_name: str) -> Tuple[Optional[str], Optional[NutrientInfo]]:
        """Return (matched_name, nutrients) or (None, None)."""
        key = food_name.lower().strip()
        if key in self.db:
            return key, self.db[key]
        matches = difflib.get_close_matches(key, self._names, n=1, cutoff=0.6)
        if matches:
            return matches[0], self.db[matches[0]]
        return None, None

    def enrich_food_item(self, name: str, quantity_grams: float) -> dict:
        """Return calorie/macro estimates for a given food and quantity."""
        info = self.lookup(name)
        if info is None:
            # Fallback: rough average food estimate
            info = NutrientInfo(150, 8.0, 15.0, 6.0)
        factor = quantity_grams / 100.0
        return {
            "calories": round(info.calories * factor, 1),
            "protein_g": round(info.protein_g * factor, 1),
            "carbs_g": round(info.carbs_g * factor, 1),
            "fat_g": round(info.fat_g * factor, 1),
        }
