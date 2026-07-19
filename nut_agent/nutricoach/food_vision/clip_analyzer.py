"""
Method 3: CLIP Zero-Shot + LLM Ensemble.

Pipeline:
  1. CLIP vision encoder extracts image embeddings
  2. Zero-shot classification against a food label bank (150+ food categories)
  3. Top-K food candidates are sent to an LLM for portion/nutrition reasoning
  4. Nutrition DB provides final calorie/macro values

This hybrid approach combines CLIP's visual understanding with LLM reasoning,
avoiding the cost of sending the full image to the LLM API.

Requires:
  pip install transformers torch Pillow openai
"""

import json
import logging
import os
import time
from typing import List, Optional, Tuple

import torch
from PIL import Image

from .base import FoodAnalyzer, FoodAnalysisResult, FoodItem
from .nutrition_db import NutritionDB, FOOD_DB
from shared.config import OPENROUTER_VISION_MODEL, OPENROUTER_VISION_FALLBACKS

logger = logging.getLogger(__name__)

# Extended food label bank for CLIP zero-shot classification
FOOD_LABELS = [
    # Proteins
    "grilled chicken breast", "roasted chicken", "fried chicken", "chicken thigh",
    "chicken wing", "chicken nuggets", "beef steak", "ground beef patty",
    "pork chop", "grilled salmon", "tuna steak", "fried fish", "shrimp",
    "boiled egg", "fried egg", "scrambled eggs", "omelette", "tofu",
    "turkey breast", "lamb chop", "sausage", "bacon",
    # Carbs / Grains
    "white rice", "brown rice", "fried rice", "pasta", "spaghetti with sauce",
    "noodles", "bread slice", "baguette", "croissant", "tortilla",
    "pancakes", "waffles", "oatmeal", "quinoa", "couscous",
    "mashed potatoes", "baked potato", "french fries", "sweet potato",
    # Vegetables
    "broccoli", "steamed broccoli", "spinach", "carrots", "tomato slices",
    "cucumber", "bell pepper", "onion rings", "mushrooms", "zucchini",
    "green beans", "corn on the cob", "peas", "mixed salad", "coleslaw",
    "roasted vegetables", "grilled vegetables", "cauliflower", "asparagus",
    "avocado", "eggplant",
    # Fruits
    "apple", "banana", "orange slices", "strawberries", "grapes",
    "watermelon", "mango", "pineapple", "blueberries", "mixed fruit",
    # Dairy
    "cheese slice", "mozzarella", "cream cheese", "yogurt", "butter pat",
    # Complete dishes
    "pizza slice", "hamburger", "cheeseburger", "hot dog", "sandwich",
    "burrito", "tacos", "sushi rolls", "ramen bowl", "pad thai",
    "chicken curry", "stir fry", "soup bowl", "lasagna", "mac and cheese",
    "caesar salad", "greek salad", "fried rice plate",
    # Snacks / Desserts
    "chips", "popcorn", "french fries", "onion rings",
    "cake slice", "brownie", "cookie", "donut", "ice cream",
    "chocolate", "granola bar", "trail mix",
    # Sauces / Condiments
    "ketchup", "mayonnaise", "mustard", "soy sauce", "salad dressing",
    "guacamole", "hummus", "salsa",
    # Drinks
    "glass of orange juice", "cup of coffee", "glass of milk",
    "smoothie", "glass of water", "glass of wine", "beer glass",
]

# Map CLIP labels back to nutrition DB keys
LABEL_TO_DB_KEY = {
    "grilled chicken breast": "chicken breast",
    "roasted chicken": "roasted chicken",
    "fried chicken": "chicken thigh",
    "chicken wing": "chicken thigh",
    "chicken nuggets": "chicken thigh",
    "beef steak": "beef steak",
    "ground beef patty": "ground beef",
    "grilled salmon": "salmon",
    "tuna steak": "tuna",
    "fried fish": "salmon",
    "boiled egg": "boiled egg",
    "fried egg": "fried egg",
    "scrambled eggs": "egg",
    "spaghetti with sauce": "pasta",
    "bread slice": "bread",
    "tomato slices": "tomato",
    "onion rings": "onion",
    "corn on the cob": "corn",
    "mixed salad": "mixed salad",
    "roasted vegetables": "stir fry",
    "grilled vegetables": "stir fry",
    "orange slices": "orange",
    "cheese slice": "cheese",
    "pizza slice": "pizza",
    "hamburger": "hamburger",
    "sushi rolls": "sushi",
    "ramen bowl": "ramen",
    "soup bowl": "soup",
    "caesar salad": "mixed salad",
    "greek salad": "mixed salad",
    "fried rice plate": "fried rice",
    "cake slice": "cake",
    "glass of orange juice": "orange juice",
    "cup of coffee": "coffee",
    "glass of milk": "milk",
    "glass of water": "tea",
    "glass of wine": "wine",
    "beer glass": "beer",
}


JINA_EMBED_MODEL = "jinaai/jina-embeddings-v5-omni-small"
# Cosine similarities from an embedding model need sharpening before softmax;
# CLIP bakes this in with its learned logit scale (~100).
JINA_LOGIT_SCALE = 30.0


class CLIPFoodAnalyzer(FoodAnalyzer):
    """
    Hybrid approach: zero-shot food classification + LLM portion reasoning.
    Vision backend is either CLIP ViT-B/32 or jina-embeddings-v5-omni-small.
    """

    method_name = "clip_ensemble"

    def __init__(
        self,
        clip_model: str = "openai/clip-vit-base-patch32",
        top_k: int = 5,
        confidence_threshold: float = 0.05,
        openrouter_api_key: Optional[str] = None,
        llm_model: str = None,
        device: Optional[str] = None,
        backend: str = "clip",
    ):
        if backend not in ("clip", "jina"):
            raise ValueError(f"backend must be 'clip' or 'jina', got {backend!r}")
        self.backend = backend
        self.clip_model_name = clip_model
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self.api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.llm_model = llm_model or OPENROUTER_VISION_MODEL
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.nutrition_db = NutritionDB()
        self._clip_model = None
        self._clip_processor = None
        self._jina_model = None
        self._label_emb = None

    def _load_clip(self):
        """Lazy-load CLIP model."""
        if self._clip_model is not None:
            return

        from transformers import CLIPProcessor, CLIPModel

        self._clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        self._clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device)
        self._clip_model.eval()
        logger.info("Loaded CLIP model: %s on %s", self.clip_model_name, self.device)

    def _load_jina(self):
        """Lazy-load the Jina omni embedding model and cache label embeddings."""
        if self._jina_model is not None:
            return

        from sentence_transformers import SentenceTransformer

        self._jina_model = SentenceTransformer(
            JINA_EMBED_MODEL,
            trust_remote_code=True,
            model_kwargs={"default_task": "retrieval"},
            device=self.device,
        )
        prompts = [f"a photo of {label}" for label in FOOD_LABELS]
        self._label_emb = torch.tensor(
            self._jina_model.encode_query(prompts, normalize_embeddings=True)
        )
        logger.info("Loaded Jina model: %s on %s", JINA_EMBED_MODEL, self.device)

    def _classify_food(self, image_path: str) -> List[Tuple[str, float]]:
        """Run zero-shot classification and return top-K (label, score) pairs."""
        image = Image.open(image_path).convert("RGB")

        if self.backend == "jina":
            self._load_jina()
            img_emb = torch.tensor(
                self._jina_model.encode_document([image], normalize_embeddings=True)
            )[0]
            logits = JINA_LOGIT_SCALE * (self._label_emb @ img_emb)
            probs = logits.softmax(dim=-1)
        else:
            self._load_clip()
            text_prompts = [f"a photo of {label}" for label in FOOD_LABELS]

            inputs = self._clip_processor(
                text=text_prompts,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._clip_model(**inputs)
                logits = outputs.logits_per_image[0]
                probs = logits.softmax(dim=-1).cpu()

        # Get top-K
        top_values, top_indices = probs.topk(self.top_k)
        results = []
        for score, idx in zip(top_values.tolist(), top_indices.tolist()):
            if score >= self.confidence_threshold:
                results.append((FOOD_LABELS[idx], score))

        return results

    def _llm_refine_portions(
        self, clip_results: List[Tuple[str, float]], image_path: str
    ) -> List[dict]:
        """Use LLM to refine CLIP detections and estimate portions."""
        if not self.api_key:
            # Fallback: use default portions without LLM
            return self._default_portions(clip_results)

        from .base import encode_image_to_base64, get_image_media_type

        items_text = "\n".join(
            f"- {name} (confidence: {score:.2f})" for name, score in clip_results
        )

        prompt = f"""A food image was analyzed by a vision model. These food items were detected:

{items_text}

Looking at the image, refine this list:
1. Remove false positives (items that aren't actually in the image)
2. Add any items the model missed
3. Estimate portion sizes in grams for each item
4. Provide a portion description

Return ONLY a JSON array:
[
  {{"name": "food item", "quantity_grams": 150, "portion_description": "1 serving", "confidence": 0.8}},
  ...
]"""

        try:
            from openai import OpenAI

            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
            )

            b64 = encode_image_to_base64(image_path)
            media_type = get_image_media_type(image_path)

            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                    ],
                }],
                max_tokens=1500,
                temperature=0.1,
                extra_body={"models": OPENROUTER_VISION_FALLBACKS},
            )

            raw = response.choices[0].message.content.strip()

            # Parse JSON
            cleaned = raw
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                cleaned = "\n".join(lines)

            start_idx = cleaned.find("[")
            end_idx = cleaned.rfind("]")
            if start_idx != -1 and end_idx > start_idx:
                return json.loads(cleaned[start_idx:end_idx + 1])

        except Exception as e:
            logger.warning("LLM refinement failed, using defaults: %s", e)

        return self._default_portions(clip_results)

    def _default_portions(self, clip_results: List[Tuple[str, float]]) -> List[dict]:
        """Fallback portion estimates without LLM."""
        return [
            {
                "name": name,
                "quantity_grams": 150,
                "portion_description": "1 estimated serving",
                "confidence": score,
            }
            for name, score in clip_results
        ]

    def analyze(self, image_path: str) -> FoodAnalysisResult:
        """Run CLIP classification + LLM refinement + nutrition lookup."""
        start = time.time()
        result = FoodAnalysisResult(method=self.method_name)

        try:
            # Step 1: CLIP zero-shot classification
            clip_results = self._classify_food(image_path)

            if not clip_results:
                result.error = "CLIP could not identify any food items"
                result.elapsed_seconds = time.time() - start
                return result

            result.raw_response = f"CLIP top-{self.top_k}: {clip_results}"

            # Step 2: LLM refinement of portions
            portions = self._llm_refine_portions(clip_results, image_path)

            # Step 3: Nutrition DB lookup
            for item in portions:
                name = item.get("name", "unknown")
                grams = item.get("quantity_grams", 150)

                # Map CLIP label to DB key
                db_key = LABEL_TO_DB_KEY.get(name, name)
                nutrients = self.nutrition_db.enrich_food_item(db_key, grams)

                result.food_items.append(FoodItem(
                    name=name,
                    quantity_grams=grams,
                    confidence=item.get("confidence", 0.5),
                    calories=nutrients["calories"],
                    protein_g=nutrients["protein_g"],
                    carbs_g=nutrients["carbs_g"],
                    fat_g=nutrients["fat_g"],
                    portion_description=item.get("portion_description", ""),
                ))

            result.compute_totals()

        except Exception as e:
            logger.error("CLIP ensemble analysis failed: %s", e)
            result.error = str(e)

        result.elapsed_seconds = time.time() - start
        return result
