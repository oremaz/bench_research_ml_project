"""
Method 2: Single-shot vision-language model baseline via OpenRouter.

One image prompt produces foods, estimated grams, and calorie/macro values.
No local nutrition database is queried. Requires an OpenRouter API key and
an image-capable model; this module does not use the vLLM library.
"""

import json
import logging
import os
import time
from typing import Optional

from shared.config import OPENROUTER_MODEL_ID, OPENROUTER_REASONING_EFFORT, OPENROUTER_MAX_OUTPUT_TOKENS, openrouter_reasoning

from .base import (
    FoodAnalyzer,
    FoodAnalysisResult,
    FoodItem,
    encode_image_to_base64,
    get_image_media_type,
)

logger = logging.getLogger(__name__)


class VLMAnalyzerSingleShot(FoodAnalyzer):
    """Identify foods and estimate portions and nutrients in one image API call."""

    method_name = "vlm_single"

    SINGLE_PROMPT = """You are an expert nutritionist. Analyze this meal photo and provide a complete nutritional breakdown.

For each food item visible:
1. Identify the food (be specific about preparation method)
2. Estimate the portion size in grams
3. Calculate calories, protein (g), carbs (g), and fat (g)

Return ONLY a JSON object:
{
  "items": [
    {
      "name": "food name",
      "quantity_grams": 180,
      "portion_description": "1 medium serving",
      "confidence": 0.8,
      "calories": 250,
      "protein_g": 30.0,
      "carbs_g": 5.0,
      "fat_g": 12.0
    }
  ]
}

Be thorough — include sauces, condiments, garnishes, drinks. Use standard nutrition values."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = OPENROUTER_MODEL_ID,
        base_url: str = "https://openrouter.ai/api/v1",
        reasoning_effort: str = OPENROUTER_REASONING_EFFORT,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model = model or OPENROUTER_MODEL_ID
        self.base_url = base_url
        self.reasoning_effort = reasoning_effort
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
                max_tokens=OPENROUTER_MAX_OUTPUT_TOKENS,
                extra_body=openrouter_reasoning(self.reasoning_effort),
                temperature=0.1,
            )

            raw = response.choices[0].message.content.strip()
            result.raw_response = raw
            parsed = self._parse_json(raw, dict)

            if parsed and isinstance(parsed.get("items"), list) and parsed["items"]:
                if not all(isinstance(item, dict) for item in parsed["items"]):
                    raise ValueError("Food items must be JSON objects")
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


VLMAnalyzer = VLMAnalyzerSingleShot
