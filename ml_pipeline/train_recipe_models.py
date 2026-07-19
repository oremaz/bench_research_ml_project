"""
Train the recipe models consumed by nut_agent/recipe_lab (difficulty, binary
meal type, total time class, per-serving nutrients) on locally computed
embeddings.

The embedding backend defaults to the winner of bench_recipe_methods.py and
can be overridden with RECIPE_EMBEDDING_MODEL. Existing task checkpoint dirs
are cleared on each run because checkpoints are only valid for the embeddings
they were trained on. recipe_models_meta.json records the embedding backend,
label order, nutrient targets, and test metrics for the predictor.

Run:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run python ml_pipeline/train_recipe_models.py
"""

import ast
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

ML_PIPELINE_DIR = Path(__file__).parent
os.chdir(ML_PIPELINE_DIR)
sys.path.insert(0, str(ML_PIPELINE_DIR))
sys.path.insert(0, str(ML_PIPELINE_DIR.parent / "nut_agent"))

import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score

from utils.data import load_csv, LabelEncoderHelper, filter_meal_types
from utils.metrics import METRIC_REGISTRY
from utils.utils import load_model_by_name
from pipelines_torch.models import MODEL_REGISTRY
from pipelines_torch.benchmark import BenchmarkRunner

from recipe_lab.predictor import LocalEmbedder, LOCAL_EMBEDDING_MODEL

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBEDDING_MODEL = os.environ.get("RECIPE_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL)

# Per-task model families served by the predictor; set from the
# bench_recipe_methods.py winners among deployable registry families.
# Nutrients additionally get a fine-tuned bge-base regression head (the only
# method with meaningful test signal); the registry family is the fallback.
DEPLOYED_MODELS = {
    "difficulty": "stacking",
    "meal_type": "lightgbm",
    "time_class": "lightgbm",
    "nutrients": "catboost",
}
NUTRIENTS_FINETUNE_DIR = "nutrients_bge_regressor"

NUTRIENT_TARGETS = ["kcal", "fat", "saturates", "carbs", "sugars", "fibre", "protein", "salt"]

TASKS = {
    "difficulty": {"path_start": "difficulty_train", "labels": ["Easy", "More effort"]},
    "meal_type": {"path_start": "meal_train", "labels": ["Breakfast", "Lunch/Dinner"]},
    "time_class": {"path_start": "total_time_train", "labels": ["<15 min", "15-30 min", "30-60 min", ">60 min"]},
    "nutrients": {"path_start": "nutrients_train", "targets": NUTRIENT_TARGETS},
}

metrics_cls = [METRIC_REGISTRY["f1"], METRIC_REGISTRY["recall"],
               METRIC_REGISTRY["precision"], METRIC_REGISTRY["accuracy"]]
metrics_reg = [METRIC_REGISTRY["mae"], METRIC_REGISTRY["mse"]]


def _parse_minutes(s):
    if not s:
        return 0.0
    s = str(s).strip()
    if "No Time" in s or s == "":
        return 0.0
    hrs = re.search(r"(\d+\.?\d*)\s*hr", s)
    mins = re.search(r"(\d+\.?\d*)\s*min", s)
    total = 0.0
    if hrs:
        total += float(hrs.group(1)) * 60
    if mins:
        total += float(mins.group(1))
    if total == 0.0:
        num = re.search(r"(\d+\.?\d*)", s)
        if num:
            total = float(num.group(1))
    return total


def _parse_total_time(x):
    try:
        d = ast.literal_eval(x) if isinstance(x, str) else x
        if not isinstance(d, dict):
            return 0.0
        return sum(_parse_minutes(v) for v in d.values())
    except Exception:
        return 0.0


def time_bin(t):
    if t < 15:
        return 0
    elif t < 30:
        return 1
    elif t < 60:
        return 2
    return 3


def parse_nutrients(x):
    """Return {target: float} for a nutrients cell, or None if unusable."""
    try:
        d = ast.literal_eval(x) if isinstance(x, str) else x
    except Exception:
        return None
    if not isinstance(d, dict) or not d:
        return None
    out = {}
    for key in NUTRIENT_TARGETS:
        v = d.get(key)
        if v is None:
            return None
        m = re.search(r"(-?\d+\.?\d*)", str(v))
        if not m:
            return None
        out[key] = float(m.group(1))
    return out


def embed_texts(embedder, df, cache_name):
    """Embed recipe_text with an on-disk cache to make reruns cheap."""
    backend_slug = "jina" if embedder.is_jina else "bge"
    cache = ML_PIPELINE_DIR / "results" / f"_emb_cache_{backend_slug}_{cache_name}.npy"
    if backend_slug == "bge":
        legacy = ML_PIPELINE_DIR / "results" / f"_emb_cache_{cache_name}.npy"
        if legacy.exists() and not cache.exists():
            cache = legacy
    texts = df["recipe_text"].fillna("").tolist()
    if cache.exists():
        emb = np.load(cache)
        if emb.shape[0] == len(texts):
            return emb
    emb = embedder.embed(texts, batch_size=32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, emb)
    return emb


CLS_MODEL_NAMES = ("lightgbm", "xgboost", "catboost", "stacking")


def train_nutrients_finetune(train_texts, y_train, out_dir,
                             epochs=5, lr=2e-5, batch_size=16, max_len=512):
    """Fine-tune bge-base with a multi-target regression head on standardized
    targets and save model + tokenizer + target scaler for the predictor."""
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(LOCAL_EMBEDDING_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        LOCAL_EMBEDDING_MODEL, num_labels=y_train.shape[1], problem_type="regression",
    ).to(DEVICE)

    mu = y_train.mean(axis=0)
    sd = y_train.std(axis=0) + 1e-8
    y_std = (y_train - mu) / sd

    enc = tok(list(train_texts), truncation=True, max_length=max_len,
              padding=True, return_tensors="pt")
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"],
                       torch.tensor(y_std, dtype=torch.float32))
    gen = torch.Generator().manual_seed(SEED)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen)

    loss_fn = torch.nn.SmoothL1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    for _ in range(epochs):
        for ids, mask, yb in loader:
            opt.zero_grad()
            logits = model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE)).logits
            loss = loss_fn(logits, yb.to(DEVICE))
            loss.backward()
            opt.step()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    with open(out_dir / "regressor_meta.json", "w") as f:
        json.dump({
            "base_model": LOCAL_EMBEDDING_MODEL,
            "targets": NUTRIENT_TARGETS,
            "mu": mu.tolist(),
            "sd": sd.tolist(),
            "max_len": max_len,
        }, f, indent=2)
    return model, tok, mu, sd


def _hf_regressor_predict(model, tok, mu, sd, texts, max_len=512, batch_size=64):
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(list(texts[i:i + batch_size]), truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to(model.device)
            preds.append(model(**enc).logits.cpu().numpy())
    return np.concatenate(preds) * sd + mu


def model_configs(input_dim, num_classes):
    return [
        {"name": "lightgbm", "class": MODEL_REGISTRY["lightgbm_classifier"], "params": {}},
        {"name": "xgboost", "class": MODEL_REGISTRY["xgboost_classifier"], "params": {}},
        {"name": "catboost", "class": MODEL_REGISTRY["catboost_classifier"], "params": {}},
        {"name": "stacking", "class": MODEL_REGISTRY["stacking_classifier"], "params": {}},
        {"name": "mlp", "class": MODEL_REGISTRY["mlp_classifier"],
         "params": {"input_dim": input_dim, "num_classes": num_classes}},
    ]


def evaluate_checkpoint(path_start, model_name, X_test, y_test):
    model = load_model_by_name(
        MODEL_REGISTRY[f"{model_name}_classifier"],
        model_name, {}, path_start=path_start, task_type="classification",
    )
    preds = model.predict(X_test)
    preds = np.asarray(preds)
    # CatBoost returns (n, 1) label columns; only argmax true probability grids
    if preds.ndim > 1 and preds.shape[1] == 1:
        preds = preds.reshape(-1)
    elif preds.ndim > 1:
        preds = preds.argmax(axis=1)
    return {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "f1_macro": round(float(f1_score(y_test, preds, average="macro")), 4),
    }


def main():
    print(f"Device: {DEVICE} | embedder: {EMBEDDING_MODEL}")
    embedder = LocalEmbedder(model_name=EMBEDDING_MODEL, device=DEVICE)

    # Checkpoints are only valid for the embeddings they were trained on
    for task in TASKS.values():
        stale = ML_PIPELINE_DIR / "results" / task["path_start"]
        if stale.exists():
            shutil.rmtree(stale)

    train_df = load_csv("recipes_df.csv")
    test_df = load_csv("recipes_df_test_bis.csv")
    meal_test_df = load_csv("recipes_df_test.csv")

    X_train_all = embed_texts(embedder, train_df, "train")
    X_test_all = embed_texts(embedder, test_df, "test_bis")
    X_meal_test_all = embed_texts(embedder, meal_test_df, "test")
    print("Embeddings:", X_train_all.shape, X_test_all.shape, X_meal_test_all.shape)

    test_metrics = {}

    # --- Task 1: difficulty (Easy vs More effort, 'A challenge' merged) ---
    diff_mask = train_df["difficult"].notna().values
    y_diff_raw = train_df.loc[diff_mask, "difficult"].replace({"A challenge": "More effort"}).values
    le_diff = LabelEncoderHelper()
    le_diff.fit(y_diff_raw)
    assert le_diff.classes() == TASKS["difficulty"]["labels"], le_diff.classes()
    y_diff = le_diff.transform(y_diff_raw)
    X_diff = X_train_all[diff_mask]
    print(f"\n[difficulty] n={len(y_diff)} classes={le_diff.classes()}")
    BenchmarkRunner(
        model_configs=model_configs(X_diff.shape[1], 2), augmentations=[None],
        metrics=metrics_cls, task_type="classification", device=DEVICE,
        epochs=150, batch_size=32, use_kfold=False, use_class_weights=True,
        learning_rate=3e-4, path_start=TASKS["difficulty"]["path_start"], random_state=SEED,
    ).run(X_diff, y_diff)

    t_mask = test_df["difficult"].notna().values
    yt = le_diff.transform(test_df.loc[t_mask, "difficult"].replace({"A challenge": "More effort"}).values)
    test_metrics["difficulty"] = {
        m: evaluate_checkpoint(TASKS["difficulty"]["path_start"], m, X_test_all[t_mask], yt)
        for m in CLS_MODEL_NAMES
    }
    print("[difficulty] test:", test_metrics["difficulty"])

    # --- Task 2: meal type, binary (Breakfast vs Lunch/Dinner) ---
    to_binary = {"Lunch": "Lunch/Dinner", "Dinner": "Lunch/Dinner"}
    meal_df = filter_meal_types(train_df.reset_index())
    y_meal_raw = meal_df["meal_type"].replace(to_binary).values
    le_meal = LabelEncoderHelper()
    le_meal.fit(y_meal_raw)
    assert le_meal.classes() == TASKS["meal_type"]["labels"], le_meal.classes()
    y_meal = le_meal.transform(y_meal_raw)
    X_meal = X_train_all[meal_df["index"].values]
    print(f"\n[meal_type] n={len(y_meal)} classes={le_meal.classes()}")
    BenchmarkRunner(
        model_configs=model_configs(X_meal.shape[1], 2), augmentations=[None],
        metrics=metrics_cls, task_type="classification", device=DEVICE,
        epochs=600, batch_size=32, early_stopping=40, use_kfold=False,
        use_class_weights=True, learning_rate=2e-4, weight_decay=1e-4,
        path_start=TASKS["meal_type"]["path_start"], random_state=SEED,
    ).run(X_meal, y_meal)

    meal_test = meal_test_df.reset_index()
    meal_test = meal_test[meal_test["subcategory"].isin(["Breakfast", "Dinner", "Lunch"])]
    yt = le_meal.transform(meal_test["subcategory"].replace(to_binary).values)
    Xt = X_meal_test_all[meal_test["index"].values]
    test_metrics["meal_type"] = {
        m: evaluate_checkpoint(TASKS["meal_type"]["path_start"], m, Xt, yt)
        for m in CLS_MODEL_NAMES
    }
    print("[meal_type] test:", test_metrics["meal_type"])

    # --- Task 3: total time class (4 bins) ---
    tt = train_df["times"].apply(_parse_total_time)
    time_mask = (tt > 0).values
    y_time = tt[time_mask].apply(time_bin).values.astype(int)
    X_time = X_train_all[time_mask]
    print(f"\n[time_class] n={len(y_time)} bins={dict(zip(*np.unique(y_time, return_counts=True)))}")
    BenchmarkRunner(
        model_configs=model_configs(X_time.shape[1], 4), augmentations=[None],
        metrics=metrics_cls, task_type="classification", device=DEVICE,
        epochs=150, batch_size=32, use_kfold=False, use_class_weights=True,
        learning_rate=1e-4, weight_decay=5e-4,
        path_start=TASKS["time_class"]["path_start"], random_state=SEED,
    ).run(X_time, y_time)

    tt_test = test_df["times"].apply(_parse_total_time)
    tmask = (tt_test > 0).values
    yt = tt_test[tmask].apply(time_bin).values.astype(int)
    test_metrics["time_class"] = {
        m: evaluate_checkpoint(TASKS["time_class"]["path_start"], m, X_test_all[tmask], yt)
        for m in CLS_MODEL_NAMES
    }
    print("[time_class] test:", test_metrics["time_class"])

    # --- Task 4: per-serving nutrients (multi-target regression) ---
    nut_tr = train_df["nutrients"].apply(parse_nutrients)
    nut_te = test_df["nutrients"].apply(parse_nutrients)
    tr_mask = nut_tr.notna().values
    te_mask = nut_te.notna().values
    y_nut = np.array([[n[t] for t in NUTRIENT_TARGETS] for n in nut_tr[tr_mask]])
    X_nut = X_train_all[tr_mask]
    print(f"\n[nutrients] n={len(y_nut)} targets={NUTRIENT_TARGETS}")
    BenchmarkRunner(
        model_configs=[
            {"name": "lightgbm", "class": MODEL_REGISTRY["lightgbm_regressor"], "params": {}},
            {"name": "catboost", "class": MODEL_REGISTRY["catboost_regressor"], "params": {}},
            {"name": "stacking", "class": MODEL_REGISTRY["stacking_regressor"], "params": {}},
        ],
        augmentations=[None], metrics=metrics_reg, task_type="regression",
        device=DEVICE, epochs=150, batch_size=32, use_kfold=False,
        use_class_weights=False, path_start=TASKS["nutrients"]["path_start"],
        random_state=SEED,
    ).run(X_nut, y_nut)

    y_nut_test = np.array([[n[t] for t in NUTRIENT_TARGETS] for n in nut_te[te_mask]])
    test_metrics["nutrients"] = {}
    for m in ("lightgbm", "catboost", "stacking"):
        nut_model = load_model_by_name(
            MODEL_REGISTRY[f"{m}_regressor"], m, {},
            path_start=TASKS["nutrients"]["path_start"], task_type="regression",
        )
        preds = np.asarray(nut_model.predict(X_test_all[te_mask]))
        test_metrics["nutrients"][m] = {
            t: {
                "mae": round(float(mean_absolute_error(y_nut_test[:, j], preds[:, j])), 3),
                "r2": round(float(r2_score(y_nut_test[:, j], preds[:, j])), 3),
            }
            for j, t in enumerate(NUTRIENT_TARGETS)
        }

    # Primary nutrients model: fine-tuned bge-base regression head on raw text
    ft_dir = ML_PIPELINE_DIR / "results" / NUTRIENTS_FINETUNE_DIR
    tr_texts = train_df["recipe_text"].fillna("").values[tr_mask]
    te_texts = test_df["recipe_text"].fillna("").values[te_mask]
    print(f"\n[nutrients] fine-tuning {LOCAL_EMBEDDING_MODEL} regression head...")
    ft_model, ft_tok, mu, sd = train_nutrients_finetune(tr_texts, y_nut, ft_dir)
    preds = _hf_regressor_predict(ft_model, ft_tok, mu, sd, te_texts)
    test_metrics["nutrients"]["bge_finetune"] = {
        t: {
            "mae": round(float(mean_absolute_error(y_nut_test[:, j], preds[:, j])), 3),
            "r2": round(float(r2_score(y_nut_test[:, j], preds[:, j])), 3),
        }
        for j, t in enumerate(NUTRIENT_TARGETS)
    }
    del ft_model
    torch.cuda.empty_cache()
    print("[nutrients] test:", test_metrics["nutrients"])

    meta = {
        "embedding": {
            "backend": "sentence-transformers",
            "model": EMBEDDING_MODEL,
            "dim": int(X_train_all.shape[1]),
            "normalize": True,
        },
        "model_name": "lightgbm",
        "nutrients_model_name": DEPLOYED_MODELS["nutrients"],
        "tasks": {
            task: {
                **cfg,
                "model_name": DEPLOYED_MODELS[task],
                **({"model_dir": NUTRIENTS_FINETUNE_DIR} if task == "nutrients" else {}),
            }
            for task, cfg in TASKS.items()
        },
        "seed": SEED,
        "test_metrics": test_metrics,
    }
    meta_path = ML_PIPELINE_DIR / "results" / "recipe_models_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nWrote {meta_path}")
    print("Training complete.")


if __name__ == "__main__":
    main()
