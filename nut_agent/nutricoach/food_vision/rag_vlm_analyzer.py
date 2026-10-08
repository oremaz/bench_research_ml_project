"""
Method 4: VLM analysis with retrieved local nutrition references.

Pipeline:
  1. VLM identifies food items from the image
  2. Retrieve per-100g nutrition references using exact/fuzzy name matching
  3. VLM reasons over image + retrieved nutrition data to produce final estimates
  4. Apply heuristic limits based on hard-coded serving ranges

The VLM receives the image in both API calls and generates the final nutrients.
Retrieval does not use embeddings; use_embeddings currently has no effect.
Missing matches fall back to model knowledge. When the portion heuristic changes
grams, nutrients are scaled by the same ratio to preserve internal consistency.
Providing references does not establish improved nutrition or portion accuracy.

Requires:
  pip install openai
  OPENROUTER_API_KEY env var
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

from shared.config import OPENROUTER_MODEL_ID, OPENROUTER_REASONING_EFFORT, OPENROUTER_MAX_OUTPUT_TOKENS, openrouter_reasoning

from .base import (
    FoodAnalyzer,
    FoodAnalysisResult,
    FoodItem,
    encode_image_to_base64,
    get_image_media_type,
)
from .nutrition_db import NutritionDB, FOOD_DB, NutrientInfo, STANDARD_SERVINGS

logger = logging.getLogger(__name__)


class RAGVLMAnalyzer(FoodAnalyzer):
    """
    VLM food analysis with exact/fuzzy nutrition lookup in the prompt.
    """

    method_name = "rag_vlm"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = OPENROUTER_MODEL_ID,
        use_embeddings: bool = False,
        reasoning_effort: str = OPENROUTER_REASONING_EFFORT,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model = model or OPENROUTER_MODEL_ID
        self.reasoning_effort = reasoning_effort
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
        Adjust extreme gram estimates using hard-coded serving ranges.
        Scale nutrients proportionally when a weight is adjusted.
        """
        validated = []
        for item in items:
            name = item.get("name", "").lower()
            original_grams = float(item.get("quantity_grams", 150))
            if original_grams <= 0:
                raise ValueError("Estimated portions must be positive")
            grams = original_grams

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
                for field in ("calories", "protein_g", "carbs_g", "fat_g"):
                    item[field] = float(item.get(field, 0)) * grams / original_grams
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
                max_tokens=OPENROUTER_MAX_OUTPUT_TOKENS,
                extra_body=openrouter_reasoning(self.reasoning_effort),
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
                max_tokens=OPENROUTER_MAX_OUTPUT_TOKENS,
                extra_body=openrouter_reasoning(self.reasoning_effort),
                temperature=0.1,
            )

            raw2 = response2.choices[0].message.content.strip()
            raw_parts.append(f"RAG ESTIMATION:\n{raw2}")
            items = self._parse_json_array(raw2)

            if not items:
                result.error = "Could not parse RAG estimation"
                result.elapsed_seconds = time.time() - start
                return result

            # Step 4: Apply portion heuristics
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
