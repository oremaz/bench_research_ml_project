"""RGB-only research baselines with locally trained or published weights."""

import math
import re
import time
from pathlib import Path

from .base import FoodAnalyzer, FoodAnalysisResult, FoodItem

TARGETS = ("calories", "mass_g", "fat_g", "carbs_g", "protein_g")
RGB_CHECKPOINT = Path(__file__).parent / "results/rgb_regression/checkpoint.pt"
FOOD_R1_MODEL = "zy12123/Food-R1"
FOOD_R1_REVISION = "c70e0d6585b1e81923432df46014d6ce32855e3f"


def validate_regression_split(provenance, official_train, test_ids):
    train = set(provenance["train_ids"])
    validation = set(provenance["validation_ids"])
    fitted = train | validation
    if not train or not validation or train & validation or fitted & set(test_ids) or not fitted <= set(official_train):
        raise ValueError("RGB checkpoint train/validation IDs violate official test separation")


def total_result(method, values, started, raw_response=""):
    if set(values) != set(TARGETS) or any(not math.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError("Invalid dish nutrition prediction")
    result = FoodAnalysisResult(method=method, elapsed_seconds=time.perf_counter() - started,
                                raw_response=raw_response, food_items=[FoodItem(
                                    name="Whole dish", quantity_grams=values["mass_g"],
                                    calories=values["calories"], fat_g=values["fat_g"],
                                    carbs_g=values["carbs_g"], protein_g=values["protein_g"])])
    result.compute_totals()
    return result


class RGBRegressionAnalyzer(FoodAnalyzer):
    method_name = "rgb_regression"

    def __init__(self, checkpoint=RGB_CHECKPOINT, device=None):
        self.checkpoint = Path(checkpoint)
        self.device = device
        self.encoder = None

    def _load_model(self):
        if self.encoder is not None:
            return
        import torch
        from timm.data import create_transform
        from ml_pipeline.pipelines_torch.vision_models import get_model

        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        if state["targets"] != list(TARGETS) or state["format_version"] != 1:
            raise ValueError("Incompatible RGB regression checkpoint")
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = get_model(state["backbone"], num_classes=0, pretrained=False,
                                 img_size=state["image_size"])
        self.encoder.load_state_dict(state["encoder_state"])
        self.encoder.to(self.device).eval()
        self.transform = create_transform(**state["preprocessing"], is_training=False)
        self.state = state

    def analyze(self, image_path):
        import torch
        from PIL import Image

        started = time.perf_counter()
        self._load_model()
        with Image.open(image_path) as image:
            pixels = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            features = self.encoder(pixels).cpu().double()
        state = self.state
        features = (features - state["feature_mean"]) / state["feature_scale"]
        values = (features @ state["coefficient"].T + state["intercept"])[0]
        values = (values * state["target_scale"] + state["target_mean"]).clamp_min(0)
        return total_result(self.method_name, dict(zip(TARGETS, values.tolist())), started)


def parse_food_r1(text):
    answer = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if len(answer) != 1:
        raise ValueError("Expected one complete Food-R1 answer block")
    number = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    patterns = {
        "mass_g": rf"weighs\s+{number}\s*(?:g|grams)\b",
        "calories": rf"(?:provides|contains|has)\s+(?:about\s+)?{number}\s*kcal\b",
        "fat_g": rf"{number}\s*(?:g|grams)\s+of\s+fat\b",
        "carbs_g": rf"{number}\s*(?:g|grams)\s+of\s+carbohydrates?\b",
        "protein_g": rf"{number}\s*(?:g|grams)\s+of\s+protein\b",
    }
    values = {}
    for target, pattern in patterns.items():
        matches = re.findall(pattern, answer[0], flags=re.IGNORECASE)
        if len(matches) != 1:
            raise ValueError(f"Missing or ambiguous Food-R1 field: {target}")
        values[target] = float(matches[0])
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("Invalid Food-R1 nutrition values")
    return values


class FoodR1Analyzer(FoodAnalyzer):
    method_name = "food_r1"
    PROMPT = """Estimate total nutrition for the entire pictured meal using only its RGB image.
Return one <answer> block with this numerical template, replacing every placeholder:
<answer>The dish weighs MASS g in total and provides about KCAL kcal, including FAT g
of fat, CARB g of carbohydrate, and PROTEIN g of protein overall.</answer>
Report food mass in grams, energy in kilocalories, and all macronutrients in grams.
Do not estimate a named ingredient separately."""

    def __init__(self, device=None, max_new_tokens=512):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model = None

    def _load_model(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(FOOD_R1_MODEL, revision=FOOD_R1_REVISION)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            FOOD_R1_MODEL, revision=FOOD_R1_REVISION,
            torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
            attn_implementation="sdpa").to(self.device).eval()

    def analyze(self, image_path):
        import torch
        from PIL import Image

        started = time.perf_counter()
        self._load_model()
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            image.thumbnail((1024, 1024))
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": self.PROMPT}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt").to(self.device)
        inputs.pop("token_type_ids", None)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens)
        text = self.processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:],
                                           skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return total_result(self.method_name, parse_food_r1(text), started, text)
