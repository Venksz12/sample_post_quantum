from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .config import (
    DATASET_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    RISK_BY_ATTACK,
    SEVERITY_BY_ATTACK,
    LOW_MAX,
    MEDIUM_MAX,
)


@dataclass
class ModelBundle:
    model: object
    metrics: dict
    feature_columns: list[str]
    classes: list[str]


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Put the supplied CSV in data/."
        )
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    clean = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    for col in FEATURE_COLUMNS:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean[TARGET_COLUMN] = clean[TARGET_COLUMN].astype(str).str.strip()
    clean = clean.dropna()
    clean = clean[clean[TARGET_COLUMN].isin(RISK_BY_ATTACK)]
    return clean


def train_model(path: Path = DATASET_PATH, random_state: int = 42) -> ModelBundle:
    df = load_dataset(path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Two-stage holdout: final test set is untouched by model fitting.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.20, stratify=y_trainval,
        random_state=random_state
    )

    extra = ExtraTreesClassifier(
        n_estimators=450,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    forest = RandomForestClassifier(
        n_estimators=350,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state + 1,
        n_jobs=-1,
    )

    # Soft voting gives a robust ensemble while retaining probability estimates.
    model = VotingClassifier(
        estimators=[("extra_trees", extra), ("random_forest", forest)],
        voting="soft",
        weights=[2, 1],
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    classes = list(model.classes_)
    test_report = classification_report(
        y_test, test_pred, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, test_pred, labels=classes)

    metrics = {
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "validation_accuracy": float(accuracy_score(y_val, val_pred)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_macro_f1": float(f1_score(y_test, test_pred, average="macro")),
        "test_weighted_f1": float(f1_score(y_test, test_pred, average="weighted")),
        "classification_report": test_report,
        "confusion_matrix": cm.tolist(),
        "labels": classes,
    }
    return ModelBundle(model=model, metrics=metrics,
                       feature_columns=FEATURE_COLUMNS, classes=classes)


def _probability_risk(classes, probabilities) -> float:
    weighted = 0.0
    total = 0.0
    for cls, prob in zip(classes, probabilities):
        severity = SEVERITY_BY_ATTACK.get(str(cls), 0.70)
        weighted += float(prob) * severity
        total += float(prob)
    return float(np.clip(weighted / total if total else 1.0, 0.0, 1.0))


def score_request(bundle: ModelBundle, features: dict,
                  signature_valid: bool) -> dict:
    row = pd.DataFrame([{c: float(features[c]) for c in bundle.feature_columns}])
    predicted = str(bundle.model.predict(row)[0])
    probabilities = bundle.model.predict_proba(row)[0]

    probability_risk = _probability_risk(bundle.classes, probabilities)

    # Cryptographic verification is a hard gate, never merely an ML feature.
    if not signature_valid:
        return {
            "attack_type": "Cryptographic Tampering",
            "risk_score": 1.0,
            "risk_level": "High",
            "decision": "BLOCK",
            "probability_risk": probability_risk,
            "class_probabilities": {
                str(c): float(p) for c, p in zip(bundle.classes, probabilities)
            },
        }

    # Blend model confidence with a deterministic severity prior. This avoids
    # pretending the model's class probability itself is a calibrated risk score.
    class_prior = RISK_BY_ATTACK.get(predicted, 0.75)
    risk = float(np.clip(0.65 * probability_risk + 0.35 * class_prior, 0, 1))

    # Additional deterministic controls are deliberately transparent.
    if features["certificate_valid"] < 0.5:
        risk = max(risk, 0.72)
    if features["device_change"] >= 1 and features["failed_attempts"] >= 2:
        risk = max(risk, 0.78)
    if features["request_frequency"] >= 60:
        risk = max(risk, 0.82)

    if risk <= LOW_MAX:
        level, decision = "Low", "ALLOW"
    elif risk <= MEDIUM_MAX:
        level, decision = "Medium", "FLAG"
    else:
        level, decision = "High", "BLOCK"

    return {
        "attack_type": predicted,
        "risk_score": risk,
        "risk_level": level,
        "decision": decision,
        "probability_risk": probability_risk,
        "class_probabilities": {
            str(c): float(p) for c, p in zip(bundle.classes, probabilities)
        },
    }
