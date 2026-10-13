"""
Single-shot vision-language baseline via OpenRouter, Groq or local Qwen.

One image prompt produces foods, estimated grams, and calorie/macro values.
No local nutrition database is queried. Hosted inference requires the selected
provider's API key; local inference uses cached Transformers weights.
"""

import json
import logging
import os
import time
from typing import Optional

from shared.config import OPENROUTER_REASONING_EFFORT, OPENROUTER_MAX_OUTPUT_TOKENS
from shared.llm_provider import create_client, default_model, request_settings, LOCAL_MODEL_REVISION

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
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        provider: str = "openrouter",
        device: Optional[str] = None,
        quantization: str = "none",
        revision: str = LOCAL_MODEL_REVISION,
        max_output_tokens: int = OPENROUTER_MAX_OUTPUT_TOKENS,
    ):
        self.provider = provider
        self.api_key = api_key or os.environ.get(f"{provider.upper()}_API_KEY", "")
        self.model = model or default_model(provider)
        self.reasoning_effort = reasoning_effort if reasoning_effort is not None else (
            "none" if provider == "local" else OPENROUTER_REASONING_EFFORT)
        self.device, self.quantization, self.revision = device, quantization, revision
        self.max_output_tokens = max_output_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = create_client(self.provider, self.api_key, model=self.model,
                                         device=self.device, quantization=self.quantization, revision=self.revision)
        return self._client

    def _request_options(self):
        token_field = "max_completion_tokens" if self.provider == "groq" else "max_tokens"
        return {token_field: self.max_output_tokens, "temperature": 0.1,
                **request_settings(self.provider, self.reasoning_effort)}

    def analyze(self, image_path: str) -> FoodAnalysisResult:
        start = time.time()
        result = FoodAnalysisResult(method=self.method_name)

        try:
            if self.provider != "local" and not self.api_key:
                raise ValueError(f"{self.provider.upper()}_API_KEY not set.")

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
                **self._request_options(),
            )

            raw = (response.choices[0].message.content or "").strip()
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
