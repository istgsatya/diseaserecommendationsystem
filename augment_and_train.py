"""Augment single-sample-per-class symptom matrix and train a classifier.

The original dataset has exactly one row per disease (class) and ~1300 binary symptom columns.
This script generates synthetic patient presentations by randomly subsampling
true symptoms (simulating partial presentation) and optionally injecting a few
false-positive symptoms (noise). It then trains and evaluates a RandomForest
and performs lightweight hyperparameter tuning.

DISCLAIMER: Synthetic augmentation cannot replace real clinical data. Metrics
obtained here reflect performance on artificial samples and do NOT indicate
clinical diagnostic validity.
"""

from __future__ import annotations

import os
import json
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from chardet import detect
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DATASET_PRIMARY = "dataset/trainings.csv"
DATASET_FALLBACK = "data/dataset/trainings.csv"
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


@dataclass
class AugmentConfig:
    n_per_class: int = 20            # synthetic samples per original disease
    min_keep_frac: float = 0.6        # lower bound for symptom retention probability
    max_keep_frac: float = 0.85       # upper bound for symptom retention probability
    fp_symptom_rate: float = 0.01     # probability each absent symptom becomes a false positive
    max_fp_symptoms: int = 5          # hard cap on false positives per synthetic sample
    ensure_min_pos: int = 1           # guarantee at least this many true symptoms retained


def load_original_matrix() -> pd.DataFrame:
    path = DATASET_PRIMARY if os.path.exists(DATASET_PRIMARY) else DATASET_FALLBACK
    raw = open(path, 'rb').read()
    enc = detect(raw)['encoding'] or 'utf-8'
    df = pd.read_csv(path, encoding=enc)
    return df


def generate_synthetic_cases(df: pd.DataFrame, cfg: AugmentConfig) -> pd.DataFrame:
    """Create synthetic patient rows.

    Strategy per disease row:
      1. Identify positive symptom indices.
      2. Sample a keep probability from [min_keep_frac, max_keep_frac].
      3. Retain each positive symptom with that probability.
      4. If retained set empty, force random positive symptoms until ensure_min_pos.
      5. Inject limited false positives among negatives with fp_symptom_rate.
    """
    prognosis_col = df.columns[-1]
    features = df.columns[:-1]
    rows: List[dict] = []
    for _, row in df.iterrows():
        label = row[prognosis_col]
        pos_symptoms = [f for f in features if row[f] == 1]
        neg_symptoms = [f for f in features if row[f] == 0]
        for _ in range(cfg.n_per_class):
            keep_p = random.uniform(cfg.min_keep_frac, cfg.max_keep_frac)
            retained = [s for s in pos_symptoms if random.random() < keep_p]
            if len(retained) < cfg.ensure_min_pos:
                # force add random positives
                needed = cfg.ensure_min_pos - len(retained)
                retained.extend(random.sample(pos_symptoms, min(needed, len(pos_symptoms))))
            # false positives
            fp_candidates = [s for s in neg_symptoms if random.random() < cfg.fp_symptom_rate]
            if len(fp_candidates) > cfg.max_fp_symptoms:
                fp_candidates = random.sample(fp_candidates, cfg.max_fp_symptoms)
            symptom_set = set(retained) | set(fp_candidates)
            sample = {f: (1 if f in symptom_set else 0) for f in features}
            sample[prognosis_col] = label
            rows.append(sample)
    synth_df = pd.DataFrame(rows, columns=list(features) + [prognosis_col])
    return synth_df


def train_and_evaluate(df: pd.DataFrame) -> Tuple[RandomForestClassifier, dict]:
    prognosis_col = df.columns[-1]
    X = df.drop(columns=[prognosis_col]).values
    y_raw = df[prognosis_col].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    base_metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro"),
    }

    # Lightweight hyperparameter search
    param_dist = {
        "n_estimators": [300, 500, 800],
        "max_depth": [None, 20, 40, 60],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=12,
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED),
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=0,
    )
    search.fit(X_train, y_train)
    best_rf = search.best_estimator_
    tuned_preds = best_rf.predict(X_test)
    tuned_metrics = {
        "accuracy": accuracy_score(y_test, tuned_preds),
        "macro_f1": f1_score(y_test, tuned_preds, average="macro"),
    }

    # Persist artifacts
    import joblib
    joblib.dump(best_rf, os.path.join(ARTIFACT_DIR, "rf_tuned_augmented.pkl"))
    joblib.dump(le, os.path.join(ARTIFACT_DIR, "label_encoder_augmented.pkl"))
    # Save feature columns for later inference
    feature_cols_path = os.path.join(ARTIFACT_DIR, "feature_cols_augmented.json")
    with open(feature_cols_path, "w") as f:
        json.dump(list(df.columns[:-1]), f, indent=2)

    report = classification_report(y_test, tuned_preds, target_names=le.inverse_transform(sorted(set(y_test))), zero_division=0)
    metrics = {
        "base": base_metrics,
        "tuned": tuned_metrics,
        "best_params": search.best_params_,
        "classes": len(le.classes_),
    }
    return best_rf, {"metrics": metrics, "report": report}


def main():
    original_df = load_original_matrix()
    cfg = AugmentConfig()
    synth_df = generate_synthetic_cases(original_df, cfg)
    print(f"Original rows: {original_df.shape[0]}, Synthetic rows: {synth_df.shape[0]}")
    model, info = train_and_evaluate(synth_df)
    print("Metrics (base vs tuned):", info["metrics"])
    print("Classification report (tuned):")
    print(info["report"][:1200])  # truncate huge output for console
    # Save augmented dataset for transparency
    synth_path = os.path.join(ARTIFACT_DIR, "augmented_dataset.csv")
    synth_df.to_csv(synth_path, index=False)
    # Note limitations
    with open(os.path.join(ARTIFACT_DIR, "augmentation_manifest.json"), "w") as f:
        json.dump({
            "warning": "Synthetic data generated; not clinically validated.",
            "config": cfg.__dict__,
            "original_rows": int(original_df.shape[0]),
            "synthetic_rows": int(synth_df.shape[0]),
        }, f, indent=2)


if __name__ == "__main__":
    main()
