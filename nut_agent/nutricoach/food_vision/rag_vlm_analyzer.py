"""
Method 4: RAG-Enhanced VLM (DietAI24-inspired).

Inspired by DietAI24 (Nature 2025), this method grounds VLM food analysis
in a structured nutrition database using retrieval-augmented generation.

Pipeline:
  1. VLM identifies food items from the image
  2. For each item, retrieve top nutrition DB matches (semantic search)
  3. VLM reasons over image + retrieved nutrition data to produce final estimates
  4. Cross-validate portions against known serving size standards

This approach reduces hallucination in calorie estimates by grounding
the LLM's reasoning in real nutrition data.

Requires:
  pip install openai sentence-transformers
  OPENROUTER_API_KEY env var
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

from .base import (
    FoodAnalyzer,
    FoodAnalysisResult,
    FoodItem,
    encode_image_to_base64,
    get_image_media_type,
)
from .nutrition_db import NutritionDB, FOOD_DB, NutrientInfo

logger = logging.getLogger(__name__)

# Standard serving sizes for cross-validation (grams)
STANDARD_SERVINGS: Dict[str, Tuple[float, float]] = {
    # (typical_min_g, typical_max_g) for one serving
    "chicken breast": (120, 220),
    "beef steak": (150, 250),
    "salmon": (120, 200),
    "rice": (130, 250),
    "pasta": (140, 250),
    "bread": (25, 50),
    "broccoli": (70, 150),
    "salad": (80, 200),
    "pizza": (100, 150),  # per slice
    "soup": (200, 350),
    "egg": (45, 60),
    "cheese": (20, 40),
    "potato": (100, 200),
    "fruit": (80, 180),
}


class RAGVLMAnalyzer(FoodAnalyzer):
    """
    RAG-enhanced VLM for food analysis, inspired by DietAI24.

    Grounds calorie estimates in a real nutrition database to reduce
    hallucination and improve accuracy.
    """

    method_name = "rag_vlm"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "anthropic/claude-opus-4-6",
        use_embeddings: bool = False,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model = model
        self.use_embeddings = use_embeddings
        self.nutrition_db = NutritionDB()
        self._client = None
        self._embedder = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
            )
        return self._client

    def _retrieve_nutrition_context(self, food_names: List[str]) -> str:
        """
        Retrieve relevant nutrition data for identified foods.
        Returns a formatted context string for the LLM.
        """
        context_parts = []

        for name in food_names:
            matched_name, info = self.nutrition_db.lookup_with_name(name)

            if info:
                context_parts.append(
                    f"DATABASE MATCH for '{name}' → '{matched_name}':\n"
                    f"  Per 100g: {info.calories} kcal, "
                    f"P:{info.protein_g}g, C:{info.carbs_g}g, F:{info.fat_g}g"
                )

                # Add standard serving info if available
                for key, (min_g, max_g) in STANDARD_SERVINGS.items():
                    if key in (matched_name or "").lower() or key in name.lower():
                        context_parts.append(
                            f"  Standard serving: {min_g}-{max_g}g"
                        )
                        break
            else:
                # Provide similar foods for context
                similar = []
                for db_name, db_info in list(FOOD_DB.items())[:5]:
                    if any(word in db_name for word in name.lower().split()):
                        similar.append(
                            f"    {db_name}: {db_info.calories} kcal/100g"
                        )
                if similar:
                    context_parts.append(
                        f"NO EXACT MATCH for '{name}'. Similar foods:\n"
                        + "\n".join(similar[:3])
                    )
                else:
                    context_parts.append(
                        f"NO MATCH for '{name}'. Use your best nutritional knowledge."
                    )

        return "\n\n".join(context_parts)

    def _cross_validate_portions(self, items: List[dict]) -> List[dict]:
        """
        Cross-validate estimated portions against standard serving sizes.
        Flags unrealistic estimates.
        """
        validated = []
        for item in items:
            name = item.get("name", "").lower()
            grams = item.get("quantity_grams", 150)

            # Check against standard servings
            warning = None
            for key, (min_g, max_g) in STANDARD_SERVINGS.items():
                if key in name:
                    if grams < min_g * 0.3:
                        warning = f"Very small portion ({grams}g < typical min {min_g}g)"
                        grams = min_g  # Correct to minimum
                    elif grams > max_g * 3:
                        warning = f"Very large portion ({grams}g > 3x typical max {max_g}g)"
                        grams = max_g * 2  # Cap at 2x max
                    break

            item["quantity_grams"] = grams
            if warning:
                item["portion_warning"] = warning
                logger.info("Portion validation: %s - %s", name, warning)

            validated.append(item)

        return validated

    def analyze(self, image_path: str) -> FoodAnalysisResult:
        """Run RAG-enhanced VLM analysis."""
        start = time.time()
        result = FoodAnalysisResult(method=self.method_name)
        raw_parts = []

        try:
            if not self.api_key:
                raise ValueError("OPENROUTER_API_KEY not set.")

            client = self._get_client()
            b64 = encode_image_to_base64(image_path)
            media_type = get_image_media_type(image_path)

            # Step 1: Identify food items
            identify_prompt = """Look at this meal photo and list ALL food items visible.
Be specific about preparation method (grilled, fried, steamed, etc.).
Return ONLY a JSON array of strings: ["item1", "item2", ...]"""

            response = client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": identify_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                    ],
                }],
                max_tokens=500,
                temperature=0.1,
            )

            raw1 = response.choices[0].message.content.strip()
            raw_parts.append(f"IDENTIFY:\n{raw1}")
            food_names = self._parse_json_array(raw1)

            if not food_names:
                result.error = "Could not identify foods"
                result.elapsed_seconds = time.time() - start
                return result

            # Step 2: Retrieve nutrition context from DB
            nutrition_context = self._retrieve_nutrition_context(food_names)
            raw_parts.append(f"RAG CONTEXT:\n{nutrition_context}")

            # Step 3: RAG-grounded estimation
            rag_prompt = f"""You are an expert nutritionist. Analyze this meal photo using the nutrition database data below.

NUTRITION DATABASE RESULTS:
{nutrition_context}

IDENTIFIED FOODS: {json.dumps(food_names)}

For each food item:
1. Estimate the portion size in grams (use the image for visual estimation and the database for standard serving sizes)
2. Calculate calories and macros using the per-100g values from the database
3. Account for cooking method (oils, sauces add calories)

IMPORTANT: Use the database values as your ground truth for per-100g nutrition.
Multiply by (estimated_grams / 100) for final values.

Return ONLY a JSON array:
[
  {{
    "name": "food item",
    "quantity_grams": 180,
    "portion_description": "1 medium serving",
    "confidence": 0.8,
    "calories": 297,
    "protein_g": 55.8,
    "carbs_g": 0.0,
    "fat_g": 6.5,
    "db_match": "chicken breast",
    "reasoning": "~180g breast, 165 kcal/100g × 1.8 = 297 kcal"
  }}
]"""

            response2 = client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": rag_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                    ],
                }],
                max_tokens=2000,
                temperature=0.1,
            )

            raw2 = response2.choices[0].message.content.strip()
            raw_parts.append(f"RAG ESTIMATION:\n{raw2}")
            items = self._parse_json_array(raw2)

            if not items:
                result.error = "Could not parse RAG estimation"
                result.elapsed_seconds = time.time() - start
                return result

            # Step 4: Cross-validate portions
            items = self._cross_validate_portions(items)

            # Build FoodItems
            for item in items:
                result.food_items.append(FoodItem(
                    name=item.get("name", "unknown"),
                    quantity_grams=item.get("quantity_grams", 100),
                    confidence=item.get("confidence", 0.5),
                    calories=item.get("calories", 0),
                    protein_g=item.get("protein_g", 0),
                    carbs_g=item.get("carbs_g", 0),
                    fat_g=item.get("fat_g", 0),
                    portion_description=item.get("portion_description", ""),
                ))

            result.compute_totals()
            result.raw_response = "\n---\n".join(raw_parts)

        except Exception as e:
            logger.error("RAG VLM analysis failed: %s", e)
            result.error = str(e)

        result.elapsed_seconds = time.time() - start
        return result

    def _parse_json_array(self, text: str):
        """Parse a JSON array from LLM response."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        start_idx = cleaned.find("[")
        end_idx = cleaned.rfind("]")
        if start_idx != -1 and end_idx > start_idx:
            try:
                return json.loads(cleaned[start_idx:end_idx + 1])
            except json.JSONDecodeError:
                pass

        return None
