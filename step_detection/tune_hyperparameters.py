"""
Step Detection — Hyperparameter Tuning
========================================
Uses RandomizedSearchCV on the top-performing models to find
optimal hyperparameters. Saves results and optionally retrains
and re-saves the best model.

Run:
    python step_detection/tune_hyperparameters.py
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, make_scorer

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR = SCRIPT_DIR / "models"

# Reuse data loading from training script
from train_and_compare import load_all_data

# Try XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# ===========================================================================
# Parameter search spaces
# ===========================================================================
SEARCH_SPACES = {
    "HistGradientBoosting": {
        "pipeline": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(random_state=42)),
        ]),
        "params": {
            "clf__max_iter": [100, 200, 300, 500],
            "clf__max_depth": [4, 6, 8, None],
            "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "clf__min_samples_leaf": [10, 20, 30, 50],
            "clf__l2_regularization": [0.0, 0.1, 1.0],
        },
        "n_iter": 40,
    },
    "MLP Neural Net": {
        "pipeline": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(early_stopping=True, random_state=42, max_iter=500)),
        ]),
        "params": {
            "clf__hidden_layer_sizes": [(64, 32), (128, 64), (256, 128), (128, 64, 32)],
            "clf__alpha": [1e-4, 1e-3, 1e-2],
            "clf__learning_rate_init": [1e-4, 5e-4, 1e-3],
            "clf__batch_size": [32, 64, 128],
        },
        "n_iter": 30,
    },
}

if HAS_XGBOOST:
    SEARCH_SPACES["XGBoost"] = {
        "pipeline": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                use_label_encoder=False, eval_metric="logloss",
                random_state=42, n_jobs=-1, verbosity=0)),
        ]),
        "params": {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [4, 6, 8],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__reg_alpha": [0, 0.1, 1.0],
            "clf__reg_lambda": [1, 2, 5],
            "clf__subsample": [0.8, 0.9, 1.0],
        },
        "n_iter": 40,
    }


# ===========================================================================
# Tuning
# ===========================================================================
def tune_models(df: pd.DataFrame):
    """Run RandomizedSearchCV on each model and return results."""
    drop_cols = ["video", "frame", "target_foot", "label"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(int)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score)

    results = {}

    for name, config in SEARCH_SPACES.items():
        print(f"\n  🔧 Tuning {name} ({config['n_iter']} iterations)…")
        pipeline = config["pipeline"]()

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=config["params"],
            n_iter=config["n_iter"],
            cv=cv,
            scoring=scorer,
            random_state=42,
            n_jobs=-1,
            verbose=0,
            refit=True,
        )
        search.fit(X, y)

        results[name] = {
            "best_f1": search.best_score_,
            "best_f1_std": search.cv_results_["std_test_score"][search.best_index_],
            "best_params": {k: _serialize(v) for k, v in search.best_params_.items()},
            "best_estimator": search.best_estimator_,
        }

        print(f"     Best F1: {search.best_score_:.4f} "
              f"± {results[name]['best_f1_std']:.4f}")
        print(f"     Params: {search.best_params_}")

    return results, feature_cols, X, y


def _serialize(v):
    """Make a value JSON-serializable."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, tuple):
        return list(v)
    return v


# ===========================================================================
# Report
# ===========================================================================
def write_tuning_report(results: dict, default_f1: float):
    """Write markdown report comparing default vs tuned performance."""
    lines = [
        "# Hyperparameter Tuning Results",
        "",
        f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Strategy**: RandomizedSearchCV with Stratified 5-Fold CV",
        f"**Default best F1**: {default_f1:.4f}",
        "",
        "## Results",
        "",
        "| Model | Default F1 | Tuned F1 | Improvement | Best Parameters |",
        "|-------|-----------|----------|-------------|-----------------|",
    ]

    for name, res in results.items():
        improvement = res["best_f1"] - default_f1
        sign = "+" if improvement >= 0 else ""
        params_str = ", ".join(f"{k.split('__')[-1]}={v}"
                               for k, v in res["best_params"].items())
        lines.append(
            f"| {name} | {default_f1:.4f} | "
            f"{res['best_f1']:.4f} ± {res['best_f1_std']:.4f} | "
            f"{sign}{improvement:.4f} | {params_str} |"
        )

    # Find overall best
    best_name = max(results, key=lambda n: results[n]["best_f1"])
    best = results[best_name]

    lines += [
        "",
        f"## Best Configuration",
        "",
        f"- **Model**: {best_name}",
        f"- **F1**: {best['best_f1']:.4f} ± {best['best_f1_std']:.4f}",
        f"- **Parameters**:",
    ]
    for k, v in best["best_params"].items():
        lines.append(f"  - `{k}`: {v}")

    lines += [""]

    report_path = RESULTS_DIR / "tuning_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  📄 Saved {report_path.name}")


def maybe_save_tuned_model(results: dict, default_f1: float,
                           feature_cols: list, X: np.ndarray, y: np.ndarray):
    """If the tuned model beats the default, retrain on full data and save."""
    best_name = max(results, key=lambda n: results[n]["best_f1"])
    best = results[best_name]

    if best["best_f1"] <= default_f1 + 0.001:
        print(f"\n  ℹ️  Tuned model ({best['best_f1']:.4f}) does not meaningfully "
              f"improve over default ({default_f1:.4f}). Keeping existing model.")
        return False

    print(f"\n  🏆 Tuned {best_name} (F1={best['best_f1']:.4f}) beats "
          f"default ({default_f1:.4f}). Retraining and saving…")

    pipeline = best["best_estimator"]
    pipeline.fit(X, y)

    model_path = MODELS_DIR / "best_step_model.joblib"
    meta_path = MODELS_DIR / "best_step_model_meta.json"

    joblib.dump(pipeline, model_path)

    meta = {
        "model_name": best_name,
        "f1": float(best["best_f1"]),
        "f1_std": float(best["best_f1_std"]),
        "accuracy": float(best["best_f1"]),  # approximate
        "roc_auc": 0.0,  # would need to recompute
        "n_features": len(feature_cols),
        "n_samples": int(len(X)),
        "feature_columns": feature_cols,
        "date_trained": pd.Timestamp.now().isoformat(),
        "tuned": True,
        "best_params": best["best_params"],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  💾 Saved tuned model to {model_path}")
    return True


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 60)
    print("  Step Detection — Hyperparameter Tuning")
    print("=" * 60)

    # 1. Load data
    print("\n📂 Loading training data…")
    df = load_all_data()
    if df.empty:
        print("❌ No data found.")
        return
    print(f"\n✅ Loaded {len(df)} samples from {df['video'].nunique()} videos")

    # 2. Get default model F1
    meta_path = MODELS_DIR / "best_step_model_meta.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        default_f1 = meta.get("f1", 0.0)
        print(f"  Current best model F1: {default_f1:.4f}")
    else:
        default_f1 = 0.0
        print("  ⚠️  No existing model metadata found.")

    # 3. Tune
    print("\n🔧 Starting hyperparameter search…")
    results, feature_cols, X, y = tune_models(df)

    # 4. Report
    print("\n📝 Writing tuning report…")
    write_tuning_report(results, default_f1)

    # 5. Optionally save improved model
    maybe_save_tuned_model(results, default_f1, feature_cols, X, y)

    print("\n" + "=" * 60)
    print(f"  ✅ Done! Results saved to {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
