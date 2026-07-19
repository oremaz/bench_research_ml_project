"""
Benchmark embedding backends and prediction methods for the Recipe Lab tasks.

Compares BAAI/bge-base-en-v1.5 against jinaai/jina-embeddings-v5-omni-small
across classifiers (logistic probe, kNN, LightGBM, XGBoost, MLP, fine-tuned
bge-base) on difficulty, meal type (binary and 3-class) and total-time class,
plus multi-target nutrient regression (per-serving kcal/fat/carbs/... from the
recipes' ground-truth nutrients column).

Writes results to ml_pipeline/results/bench_recipe_methods.json and prints
markdown tables. Deployment training stays in train_recipe_models.py.

Run:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run python ml_pipeline/bench_recipe_methods.py
"""

import ast
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ML_PIPELINE_DIR = Path(__file__).parent
os.chdir(ML_PIPELINE_DIR)
sys.path.insert(0, str(ML_PIPELINE_DIR))
sys.path.insert(0, str(ML_PIPELINE_DIR.parent / "nut_agent"))

import torch
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier

from utils.data import load_csv, filter_meal_types

from recipe_lab.predictor import LocalEmbedder

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BACKENDS = {
    "bge": "BAAI/bge-base-en-v1.5",
    "jina": "jinaai/jina-embeddings-v5-omni-small",
}

NUTRIENT_TARGETS = ["kcal", "fat", "saturates", "carbs", "sugars", "fibre", "protein", "salt"]

RESULTS_PATH = ML_PIPELINE_DIR / "results" / "bench_recipe_methods.json"


def save_results(results):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    tmp.replace(RESULTS_PATH)


# --- data preparation (mirrors train_recipe_models.py) ---

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


def build_datasets():
    """Return dict of task -> (train_idx, y_train, test_split, test_idx, y_test, labels)."""
    train_df = load_csv("recipes_df.csv")
    test_df = load_csv("recipes_df_test_bis.csv")
    meal_test_df = load_csv("recipes_df_test.csv")

    tasks = {}

    # difficulty: Easy vs More effort ('A challenge' merged), test on test_bis
    tr_mask = train_df["difficult"].notna().values
    y_tr = (train_df.loc[tr_mask, "difficult"] != "Easy").astype(int).values
    te_mask = test_df["difficult"].notna().values
    y_te = (test_df.loc[te_mask, "difficult"] != "Easy").astype(int).values
    tasks["difficulty"] = {
        "train_idx": np.where(tr_mask)[0], "y_train": y_tr,
        "test_split": "test_bis", "test_idx": np.where(te_mask)[0], "y_test": y_te,
        "labels": ["Easy", "More effort"],
    }

    # meal type from subcategory, OOD test on recipes_df_test
    meal_df = filter_meal_types(train_df.reset_index())
    meal_te = meal_test_df.reset_index()
    meal_te = meal_te[meal_te["subcategory"].isin(["Breakfast", "Dinner", "Lunch"])]

    order3 = ["Breakfast", "Dinner", "Lunch"]
    tasks["meal_3class"] = {
        "train_idx": meal_df["index"].values,
        "y_train": meal_df["meal_type"].map(order3.index).values,
        "test_split": "test", "test_idx": meal_te["index"].values,
        "y_test": meal_te["subcategory"].map(order3.index).values,
        "labels": order3,
    }
    tasks["meal_binary"] = {
        "train_idx": meal_df["index"].values,
        "y_train": (meal_df["meal_type"] != "Breakfast").astype(int).values,
        "test_split": "test", "test_idx": meal_te["index"].values,
        "y_test": (meal_te["subcategory"] != "Breakfast").astype(int).values,
        "labels": ["Breakfast", "Lunch/Dinner"],
    }

    # total time class, test on test_bis
    tt = train_df["times"].apply(_parse_total_time)
    tr_mask = (tt > 0).values
    tt_test = test_df["times"].apply(_parse_total_time)
    te_mask = (tt_test > 0).values
    tasks["time_class"] = {
        "train_idx": np.where(tr_mask)[0],
        "y_train": tt[tr_mask].apply(time_bin).values.astype(int),
        "test_split": "test_bis", "test_idx": np.where(te_mask)[0],
        "y_test": tt_test[te_mask].apply(time_bin).values.astype(int),
        "labels": ["<15 min", "15-30 min", "30-60 min", ">60 min"],
    }

    # nutrients regression (per serving), test on test_bis
    nut_tr = train_df["nutrients"].apply(parse_nutrients)
    nut_te = test_df["nutrients"].apply(parse_nutrients)
    tr_mask = nut_tr.notna().values
    te_mask = nut_te.notna().values
    tasks["nutrients"] = {
        "train_idx": np.where(tr_mask)[0],
        "y_train": np.array([[n[t] for t in NUTRIENT_TARGETS] for n in nut_tr[tr_mask]]),
        "test_split": "test_bis", "test_idx": np.where(te_mask)[0],
        "y_test": np.array([[n[t] for t in NUTRIENT_TARGETS] for n in nut_te[te_mask]]),
        "labels": NUTRIENT_TARGETS,
    }

    dfs = {"train": train_df, "test_bis": test_df, "test": meal_test_df}
    return tasks, dfs


def embed_split(backend, model_name, df, split):
    cache = ML_PIPELINE_DIR / "results" / f"_emb_cache_{backend}_{split}.npy"
    # bge caches from train_recipe_models.py keep their original names
    legacy = {"train": "train", "test_bis": "test_bis", "test": "test"}
    if backend == "bge":
        old = ML_PIPELINE_DIR / "results" / f"_emb_cache_{legacy[split]}.npy"
        if old.exists() and not cache.exists():
            cache = old
    texts = df["recipe_text"].fillna("").tolist()
    if cache.exists():
        emb = np.load(cache)
        if emb.shape[0] == len(texts):
            return emb
    embedder = LocalEmbedder(model_name=model_name, device=DEVICE)
    emb = embedder.embed(texts, batch_size=32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, emb)
    return emb


# --- classifiers on frozen embeddings ---

def make_classifiers(seed=SEED):
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier

    from pipelines_torch.models import StackingEnsembleClassifierWrapper

    return {
        "logreg": GridSearchCV(
            LogisticRegression(max_iter=3000, class_weight="balanced"),
            {"C": [0.25, 1.0, 4.0, 16.0]},
            cv=3, scoring="f1_macro", n_jobs=-1,
        ),
        "knn": KNeighborsClassifier(n_neighbors=15, weights="distance", metric="cosine"),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=300, class_weight="balanced", random_state=seed, verbosity=-1
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=300, eval_metric="mlogloss", random_state=seed, verbosity=0
        ),
        "catboost": CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6, auto_class_weights="Balanced",
            random_seed=seed, verbose=False, allow_writing_files=False,
        ),
        "stacking": StackingEnsembleClassifierWrapper(random_state=seed).model,
        "mlp": MLPClassifier(
            hidden_layer_sizes=(256,), max_iter=500, early_stopping=True, random_state=seed
        ),
    }


def eval_classifier(name, clf, X_tr, y_tr, X_te, y_te, do_cv=True):
    t0 = time.time()
    clf.fit(X_tr, y_tr)
    fit_s = time.time() - t0
    preds = clf.predict(X_te)
    cv_acc = None
    if do_cv:
        cv_estimator = clf.best_estimator_ if hasattr(clf, "best_estimator_") else clf
        cv = cross_val_score(
            cv_estimator, X_tr, y_tr, scoring="accuracy",
            cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1,
        )
        cv_acc = round(float(cv.mean()), 4)
    return {
        "test_accuracy": round(float(accuracy_score(y_te, preds)), 4),
        "test_f1_macro": round(float(f1_score(y_te, preds, average="macro")), 4),
        "cv_accuracy": cv_acc,
        "fit_seconds": round(fit_s, 1),
    }, preds


# --- fine-tuned bge-base classifier (text -> logits, no frozen embeddings) ---

def finetune_bge(train_texts, y_train, test_texts, y_test, num_labels,
                 epochs=3, lr=2e-5, batch_size=16, max_len=512):
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
    model = AutoModelForSequenceClassification.from_pretrained(
        "BAAI/bge-base-en-v1.5", num_labels=num_labels
    ).to(DEVICE)

    enc = tok(list(train_texts), truncation=True, max_length=max_len,
              padding=True, return_tensors="pt")
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(y_train))
    gen = torch.Generator().manual_seed(SEED)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen)

    counts = np.bincount(y_train, minlength=num_labels).astype(np.float64)
    weights = torch.tensor(counts.sum() / (num_labels * counts), dtype=torch.float32).to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    t0 = time.time()
    model.train()
    for _ in range(epochs):
        for ids, mask, yb in loader:
            opt.zero_grad()
            logits = model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE)).logits
            loss = loss_fn(logits, yb.to(DEVICE))
            loss.backward()
            opt.step()
    fit_s = time.time() - t0

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(test_texts), 64):
            enc = tok(list(test_texts[i:i + 64]), truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to(DEVICE)
            preds.append(model(**enc).logits.argmax(-1).cpu().numpy())
    preds = np.concatenate(preds)

    del model
    torch.cuda.empty_cache()
    return {
        "test_accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "test_f1_macro": round(float(f1_score(y_test, preds, average="macro")), 4),
        "cv_accuracy": None,
        "fit_seconds": round(fit_s, 1),
    }, preds


def finetune_bge_regressor(train_texts, y_train, test_texts, y_test,
                           epochs=5, lr=2e-5, batch_size=16, max_len=512):
    """Fine-tune bge-base with a multi-target regression head on standardized targets."""
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
    model = AutoModelForSequenceClassification.from_pretrained(
        "BAAI/bge-base-en-v1.5", num_labels=y_train.shape[1],
        problem_type="regression",
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

    t0 = time.time()
    model.train()
    for _ in range(epochs):
        for ids, mask, yb in loader:
            opt.zero_grad()
            logits = model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE)).logits
            loss = loss_fn(logits, yb.to(DEVICE))
            loss.backward()
            opt.step()
    fit_s = time.time() - t0

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(test_texts), 64):
            enc = tok(list(test_texts[i:i + 64]), truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to(DEVICE)
            preds.append(model(**enc).logits.cpu().numpy())
    preds = np.concatenate(preds) * sd + mu

    del model
    torch.cuda.empty_cache()

    out = {"fit_seconds": round(fit_s, 1), "per_target": {}}
    for j, t in enumerate(NUTRIENT_TARGETS):
        out["per_target"][t] = {
            "mae": round(float(mean_absolute_error(y_test[:, j], preds[:, j])), 3),
            "r2": round(float(r2_score(y_test[:, j], preds[:, j])), 3),
        }
    out["mean_r2"] = round(float(np.mean([v["r2"] for v in out["per_target"].values()])), 3)
    out["kcal_mae"] = out["per_target"]["kcal"]["mae"]
    return out


# --- nutrient regression ---

def make_regressors(seed=SEED):
    import lightgbm as lgb
    from catboost import CatBoostRegressor
    from sklearn.multioutput import MultiOutputRegressor

    from pipelines_torch.models import StackingEnsembleRegressorWrapper

    return {
        "ridge": RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]),
        "knn": KNeighborsRegressor(n_neighbors=10, weights="distance", metric="cosine"),
        "lightgbm": MultiOutputRegressor(
            lgb.LGBMRegressor(n_estimators=400, random_state=seed, verbosity=-1)
        ),
        "catboost": MultiOutputRegressor(CatBoostRegressor(
            iterations=500, learning_rate=0.05, depth=6,
            random_seed=seed, verbose=False, allow_writing_files=False,
        )),
        "stacking": StackingEnsembleRegressorWrapper(random_state=seed).model,
    }


def eval_regressor(reg, X_tr, y_tr, X_te, y_te):
    t0 = time.time()
    reg.fit(X_tr, y_tr)
    fit_s = time.time() - t0
    preds = np.asarray(reg.predict(X_te))
    out = {"fit_seconds": round(fit_s, 1), "per_target": {}}
    for j, t in enumerate(NUTRIENT_TARGETS):
        out["per_target"][t] = {
            "mae": round(float(mean_absolute_error(y_te[:, j], preds[:, j])), 3),
            "r2": round(float(r2_score(y_te[:, j], preds[:, j])), 3),
        }
    out["mean_r2"] = round(float(np.mean([v["r2"] for v in out["per_target"].values()])), 3)
    out["kcal_mae"] = out["per_target"]["kcal"]["mae"]
    return out


def main():
    print(f"Device: {DEVICE}")
    tasks, dfs = build_datasets()
    for name, t in tasks.items():
        print(f"[{name}] train n={len(t['y_train'])} test n={len(t['y_test'])} ({t['test_split']})")

    embeddings = {}
    for backend, model_name in BACKENDS.items():
        for split, df in dfs.items():
            print(f"Embedding {backend}/{split}...")
            embeddings[(backend, split)] = embed_split(backend, model_name, df, split)
        torch.cuda.empty_cache()

    # Merge with an existing results file so reruns only compute missing combos
    results = {"classification": {}, "nutrients": {}, "finetune": {}}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            prev = json.load(f)
        for key in results:
            results[key].update(prev.get(key, {}))

    cls_tasks = ["difficulty", "meal_binary", "meal_3class", "time_class"]
    for task in cls_tasks:
        t = tasks[task]
        results["classification"].setdefault(task, {})
        for backend in BACKENDS:
            X_tr = embeddings[(backend, "train")][t["train_idx"]]
            X_te = embeddings[(backend, t["test_split"])][t["test_idx"]]
            done = results["classification"][task].setdefault(backend, {})
            for mname, clf in make_classifiers().items():
                if mname in done:
                    continue
                # stacking already cross-validates internally; skip the outer CV
                try:
                    metrics, preds = eval_classifier(
                        mname, clf, X_tr, t["y_train"], X_te, t["y_test"],
                        do_cv=(mname != "stacking"))
                except Exception as e:
                    print(f"[{task}][{backend}][{mname}] FAILED: {e}")
                    continue
                # 3-class predictions mapped to binary for a fair binary-vs-3class read
                if task == "meal_3class":
                    bin_true = (tasks["meal_binary"]["y_test"] > 0).astype(int)
                    bin_pred = (preds > 0).astype(int)
                    metrics["implied_binary_accuracy"] = round(
                        float(accuracy_score(bin_true, bin_pred)), 4)
                results["classification"][task][backend][mname] = metrics
                save_results(results)
                print(f"[{task}][{backend}][{mname}] {metrics}")

    # fine-tuned bge-base on raw text (independent of frozen embeddings)
    for task in ["difficulty", "meal_binary", "time_class"]:
        if task in results["finetune"]:
            continue
        t = tasks[task]
        tr_texts = dfs["train"]["recipe_text"].fillna("").values[t["train_idx"]]
        te_texts = dfs[t["test_split"]]["recipe_text"].fillna("").values[t["test_idx"]]
        metrics, _ = finetune_bge(tr_texts, t["y_train"], te_texts, t["y_test"], len(t["labels"]))
        results["finetune"][task] = metrics
        save_results(results)
        print(f"[{task}][bge_finetune] {metrics}")

    if "nutrients" not in results["finetune"]:
        t = tasks["nutrients"]
        tr_texts = dfs["train"]["recipe_text"].fillna("").values[t["train_idx"]]
        te_texts = dfs[t["test_split"]]["recipe_text"].fillna("").values[t["test_idx"]]
        metrics = finetune_bge_regressor(tr_texts, t["y_train"], te_texts, t["y_test"])
        results["finetune"]["nutrients"] = metrics
        save_results(results)
        print(f"[nutrients][bge_finetune] kcal_mae={metrics['kcal_mae']} mean_r2={metrics['mean_r2']}")

    t = tasks["nutrients"]
    y_tr, y_te = t["y_train"], t["y_test"]
    mean_pred = np.tile(y_tr.mean(axis=0), (len(y_te), 1))
    baseline = {"per_target": {}}
    for j, tgt in enumerate(NUTRIENT_TARGETS):
        baseline["per_target"][tgt] = {
            "mae": round(float(mean_absolute_error(y_te[:, j], mean_pred[:, j])), 3),
            "r2": round(float(r2_score(y_te[:, j], mean_pred[:, j])), 3),
        }
    baseline["kcal_mae"] = baseline["per_target"]["kcal"]["mae"]
    results["nutrients"]["baseline_mean"] = baseline
    for backend in BACKENDS:
        X_tr = embeddings[(backend, "train")][t["train_idx"]]
        X_te = embeddings[(backend, t["test_split"])][t["test_idx"]]
        done = results["nutrients"].setdefault(backend, {})
        for mname, reg in make_regressors().items():
            if mname in done:
                continue
            try:
                metrics = eval_regressor(reg, X_tr, y_tr, X_te, y_te)
            except Exception as e:
                print(f"[nutrients][{backend}][{mname}] FAILED: {e}")
                continue
            results["nutrients"][backend][mname] = metrics
            save_results(results)
            print(f"[nutrients][{backend}][{mname}] kcal_mae={metrics['kcal_mae']} "
                  f"mean_r2={metrics['mean_r2']}")

    results["meta"] = {
        "seed": SEED,
        "backends": BACKENDS,
        "nutrient_targets": NUTRIENT_TARGETS,
        "train_sizes": {k: int(len(v["y_train"])) for k, v in tasks.items()},
        "test_sizes": {k: int(len(v["y_test"])) for k, v in tasks.items()},
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
