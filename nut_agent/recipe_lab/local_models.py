"""Cached local embeddings and LightGBM models for Recipe Lab."""

import ast
import hashlib
import os
import re
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from shared.config import SECRETS_DIR


EMBEDDING_MODEL_ID = "LiquidAI/LFM2.5-Embedding-350M"
EMBEDDING_MODEL_REVISION = "f35ae2c91d687658dbf1f2b449382f0b019b9808"
CACHE_VERSION = "3"
DATA_PATH = Path(__file__).resolve().parents[2] / "ml_pipeline" / "recipes_df.csv"
CACHE_DIR = SECRETS_DIR / "recipe_lab_cache"
MEAL_TYPES = {"Breakfast recipes": "breakfast", "Lunch recipes": "lunch/dinner", "Dinner recipes": "lunch/dinner"}
TIME_LABELS = ("<15 min", "15-30 min", "30-60 min", ">60 min")
NUTRIENT_TARGETS = ("kcal", "fat", "saturates", "carbs", "sugars", "fibre", "protein", "salt")


def _cache_key(data_path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(EMBEDDING_MODEL_ID.encode())
    digest.update(EMBEDDING_MODEL_REVISION.encode())
    digest.update(CACHE_VERSION.encode())
    with data_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:20]


def _save_atomic(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        writer(tmp_path)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _parse_minutes(value) -> float:
    text = str(value).lower()
    hours = re.search(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)", text)
    minutes = re.search(r"(\d+(?:\.\d+)?)\s*(?:minutes?|mins?)", text)
    if hours or minutes:
        return (float(hours.group(1)) * 60 if hours else 0) + (float(minutes.group(1)) if minutes else 0)
    return 0.0


def _total_minutes(value) -> float:
    try:
        parts = ast.literal_eval(value) if isinstance(value, str) else value
        return sum(_parse_minutes(part) for part in parts.values()) if isinstance(parts, dict) else 0.0
    except (ValueError, SyntaxError, TypeError):
        return 0.0


def load_or_train_models(data_path: Path = DATA_PATH, cache_dir: Path = CACHE_DIR):
    """Embed training recipes once, then fit and cache compatible LightGBM models."""
    if not data_path.exists():
        raise FileNotFoundError(f"Training recipes not found: {data_path}")
    key = _cache_key(data_path)
    embedding_path = cache_dir / f"embeddings_{key}.npz"
    model_path = cache_dir / f"lightgbm_{key}.joblib"
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(
        EMBEDDING_MODEL_ID,
        revision=EMBEDDING_MODEL_REVISION,
        trust_remote_code=True,
    )

    if model_path.exists():
        return encoder, joblib.load(model_path)

    data = pd.read_csv(data_path, usecols=["recipe_text", "difficult", "subcategory", "times", "nutrients"])
    if embedding_path.exists():
        with np.load(embedding_path) as cached:
            embeddings = cached["embeddings"]
    else:
        texts = data["recipe_text"].fillna("").astype(str).tolist()
        embeddings = np.asarray(encoder.encode(
            texts, batch_size=16, prompt_name="document", normalize_embeddings=True,
            show_progress_bar=False,
        ), dtype=np.float32)
        _save_atomic(embedding_path, lambda path: _save_embeddings(path, embeddings))
    if len(embeddings) != len(data):
        raise ValueError("Cached embeddings do not match the training recipes")

    models = {}
    difficulty = data["difficult"].isin(["Easy", "More effort", "A challenge"]).to_numpy()
    difficulty_labels = data.loc[difficulty, "difficult"].replace({"A challenge": "More effort"})
    models["difficulty"] = _fit(embeddings[difficulty], difficulty_labels)
    meal = data["subcategory"].isin(MEAL_TYPES).to_numpy()
    models["meal_type"] = _fit(embeddings[meal], data.loc[meal, "subcategory"].map(MEAL_TYPES))
    minutes = data["times"].map(_total_minutes).to_numpy()
    valid_time = minutes > 0
    time_bins = np.searchsorted([15, 30, 60], minutes[valid_time], side="right")
    models["time_class"] = _fit(embeddings[valid_time], np.asarray(TIME_LABELS)[time_bins])
    nutrient_values = data["nutrients"].map(_parse_nutrients)
    valid_nutrients = nutrient_values.notna().to_numpy()
    if valid_nutrients.any():
        from lightgbm import LGBMRegressor
        from sklearn.multioutput import MultiOutputRegressor

        targets = np.asarray(nutrient_values[valid_nutrients].tolist(), dtype=np.float32)
        model = MultiOutputRegressor(LGBMRegressor(
            n_estimators=120, learning_rate=0.05, num_leaves=15,
            random_state=42, verbosity=-1, n_jobs=4,
        ))
        model.fit(embeddings[valid_nutrients], targets)
        models["nutrients"] = model
    _save_atomic(model_path, lambda path: joblib.dump(models, path))
    return encoder, models


def _save_embeddings(path: Path, embeddings: np.ndarray) -> None:
    with path.open("wb") as output:
        np.savez_compressed(output, embeddings=embeddings)


def _fit(embeddings: np.ndarray, labels):
    from lightgbm import LGBMClassifier

    if len(set(labels)) < 2:
        raise ValueError("At least two labeled classes are required to train LightGBM")
    model = LGBMClassifier(n_estimators=120, learning_rate=0.05, num_leaves=15,
                           class_weight="balanced", random_state=42, verbosity=-1, n_jobs=4)
    model.fit(embeddings, labels)
    return model


def _parse_nutrients(value):
    try:
        data = ast.literal_eval(value) if isinstance(value, str) else value
    except (ValueError, SyntaxError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    values = []
    for target in NUTRIENT_TARGETS:
        match = re.search(r"(-?\d+\.?\d*)", str(data.get(target, "")))
        if not match:
            return None
        values.append(float(match.group(1)))
    return values
