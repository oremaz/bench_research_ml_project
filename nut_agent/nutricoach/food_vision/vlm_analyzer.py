"""
Method 2: Pure vLLM approach via OpenRouter API.

Uses Claude Opus 4.6 with intelligently chained prompts:
  Step 1: Identify all food items visible on the plate
  Step 2: Estimate portion sizes for each item
  Step 3: Compute detailed calorie/macro breakdown

All reasoning is done by the LLM — no CV models needed.
This is the simplest approach and serves as a strong baseline.

Requires:
  pip install openai  (OpenRouter uses OpenAI-compatible API)
  OPENROUTER_API_KEY env var
"""

import json
import logging
import os
import time
from typing import Optional

from .base import (
    FoodAnalyzer,
    FoodAnalysisResult,
    FoodItem,
    encode_image_to_base64,
    get_image_media_type,
)

logger = logging.getLogger(__name__)

# --- Prompt Chain ---

STEP1_IDENTIFY = """You are an expert nutritionist analyzing a photo of a meal.

Look at this image carefully and list ALL food items you can identify.
For each item, be as specific as possible (e.g., "grilled chicken breast" not just "chicken").

Return ONLY a JSON array of strings. Example:
["grilled chicken breast", "steamed white rice", "sautéed broccoli", "olive oil dressing"]

Be thorough — include sauces, condiments, garnishes, and drinks if visible."""

STEP2_PORTIONS = """You are an expert nutritionist estimating portion sizes from a meal photo.

The following food items were identified in this image:
{food_items}

For each item, estimate:
1. The weight in grams (be realistic — a typical chicken breast is 150-200g, a cup of rice is ~180g cooked)
2. A human-readable portion description (e.g., "1 medium breast", "3/4 cup")
3. Your confidence (0.0 to 1.0) in this estimate

Return ONLY a JSON array of objects:
[
  {{"name": "grilled chicken breast", "quantity_grams": 180, "portion_description": "1 medium breast", "confidence": 0.8}},
  ...
]

Use visual cues: plate size (standard dinner plate ~26cm), utensils, and food proportions relative to each other."""

STEP3_NUTRITION = """You are an expert nutritionist computing the nutritional breakdown of a meal.

Here are the food items with estimated portions:
{portions}

For each item, provide calories, protein (g), carbs (g), and fat (g).
Use standard USDA/nutrition database values, adjusted for cooking method.

Return ONLY a JSON array:
[
  {{
    "name": "grilled chicken breast",
    "quantity_grams": 180,
    "portion_description": "1 medium breast",
    "confidence": 0.8,
    "calories": 297,
    "protein_g": 55.8,
    "carbs_g": 0.0,
    "fat_g": 6.5
  }},
  ...
]

Be precise. Account for cooking oils, sauces, and preparation methods."""


class VLMAnalyzer(FoodAnalyzer):
    """
    Pure vision-language model approach using Claude Opus via OpenRouter.

    Uses a 3-step prompt chain:
    1. Food identification (vision)
    2. Portion estimation (vision + reasoning)
    3. Nutrition computation (reasoning)
    """

    method_name = "vlm_claude"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "anthropic/claude-opus-4-6",
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model = model
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai is required. Install with: pip install openai")
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        return self._client

    def _call_vlm(self, messages: list, max_tokens: int = 2000) -> str:
        """Make a single API call to the VLM."""
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

    def _make_image_message(self, image_path: str, text: str) -> dict:
        """Create a message with image content."""
        b64 = encode_image_to_base64(image_path)
        media_type = get_image_media_type(image_path)
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{b64}",
                    },
                },
            ],
        }

    def analyze(self, image_path: str) -> FoodAnalysisResult:
        """Run the 3-step prompt chain on a food image."""
        start = time.time()
        result = FoodAnalysisResult(method=self.method_name)
        raw_parts = []

        try:
            if not self.api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY not set. "
                    "Set it as an environment variable or pass api_key to VLMAnalyzer."
                )

            # Step 1: Identify food items
            msg1 = self._make_image_message(image_path, STEP1_IDENTIFY)
            raw1 = self._call_vlm([msg1])
            raw_parts.append(f"STEP1:\n{raw1}")
            food_names = self._parse_json(raw1, list)

            if not food_names:
                result.error = "Could not identify any food items"
                result.elapsed_seconds = time.time() - start
                return result

            # Step 2: Estimate portions (with image context)
            step2_prompt = STEP2_PORTIONS.format(food_items=json.dumps(food_names))
            msg2 = self._make_image_message(image_path, step2_prompt)
            raw2 = self._call_vlm([msg2])
            raw_parts.append(f"STEP2:\n{raw2}")
            portions = self._parse_json(raw2, list)

            if not portions:
                # Fallback: use food names with default portions
                portions = [
                    {"name": n, "quantity_grams": 150, "portion_description": "estimated", "confidence": 0.3}
                    for n in food_names
                ]

            # Step 3: Compute nutrition
            step3_prompt = STEP3_NUTRITION.format(portions=json.dumps(portions, indent=2))
            raw3 = self._call_vlm([{"role": "user", "content": step3_prompt}])
            raw_parts.append(f"STEP3:\n{raw3}")
            nutrition = self._parse_json(raw3, list)

            if not nutrition:
                nutrition = portions  # Fallback to portions without nutrition

            # Build FoodItems
            for item in nutrition:
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
            logger.error("VLM analysis failed: %s", e)
            result.error = str(e)

        result.elapsed_seconds = time.time() - start
        return result

    def _parse_json(self, text: str, expected_type: type):
        """Extract and parse JSON from LLM response, handling markdown fences."""
        # Strip markdown code fences
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last fence lines
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, expected_type):
                return parsed
        except json.JSONDecodeError:
            pass

        # Try to find JSON array/object in the text
        for start_char, end_char in [("[", "]"), ("{", "}")]:
            start_idx = cleaned.find(start_char)
            end_idx = cleaned.rfind(end_char)
            if start_idx != -1 and end_idx > start_idx:
                try:
                    parsed = json.loads(cleaned[start_idx:end_idx + 1])
                    if isinstance(parsed, expected_type):
                        return parsed
                except json.JSONDecodeError:
                    continue

        logger.warning("Could not parse JSON from VLM response: %.100s...", text)
        return None


class VLMAnalyzerSingleShot(FoodAnalyzer):
    """
    Simpler single-prompt variant — sends one comprehensive prompt.
    Faster and cheaper but potentially less accurate than the chained approach.
    """

    method_name = "vlm_claude_single"

    SINGLE_PROMPT = """You are an expert nutritionist. Analyze this meal photo and provide a complete nutritional breakdown.

For each food item visible:
1. Identify the food (be specific about preparation method)
2. Estimate the portion size in grams
3. Calculate calories, protein (g), carbs (g), and fat (g)

Return ONLY a JSON object:
{{
  "items": [
    {{
      "name": "food name",
      "quantity_grams": 180,
      "portion_description": "1 medium serving",
      "confidence": 0.8,
      "calories": 250,
      "protein_g": 30.0,
      "carbs_g": 5.0,
      "fat_g": 12.0
    }}
  ]
}}

Be thorough — include sauces, condiments, garnishes, drinks. Use standard nutrition values."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "anthropic/claude-opus-4-6",
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model = model
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def analyze(self, image_path: str) -> FoodAnalysisResult:
        start = time.time()
        result = FoodAnalysisResult(method=self.method_name)

        try:
            if not self.api_key:
                raise ValueError("OPENROUTER_API_KEY not set.")

            b64 = encode_image_to_base64(image_path)
            media_type = get_image_media_type(image_path)

            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.SINGLE_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                    ],
                }],
                max_tokens=2000,
                temperature=0.1,
            )

            raw = response.choices[0].message.content.strip()
            result.raw_response = raw
            parsed = VLMAnalyzer._parse_json(None, raw, dict)

            if parsed and "items" in parsed:
                for item in parsed["items"]:
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
            else:
                result.error = "Could not parse response"

        except Exception as e:
            result.error = str(e)

        result.elapsed_seconds = time.time() - start
        return result
