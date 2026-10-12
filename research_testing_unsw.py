"""PQ-Auth Sentinel research runner for the existing synthetic CSV and UNSW-NB15.

Expected files in project_root/data/:
  pq_ai_authentication_dataset_20000.csv
  UNSW_NB15_training-set.csv
  UNSW_NB15_testing-set.csv

Examples (run from project root):
  python research_testing_adapted.py --dataset both
  python research_testing_adapted.py --dataset unsw
  python research_testing_adapted.py --dataset synthetic

UNSW-NB15 is a network-intrusion dataset, not an authentication telemetry dataset.
Its results are reported as a separate benchmark and do not replace the app's
seven-feature authentication model. For UNSW, the primary task is multiclass
classification using attack_cat; id and binary label are excluded from features.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
SYNTHETIC_FEATURES = [
    "request_frequency", "inter_request_time_sec", "failed_attempts",
    "session_duration_min", "device_change", "certificate_valid",
    "request_pattern_score",
]
SYNTHETIC_TARGET = "attack_type"
SYNTHETIC_FILE = "pq_ai_authentication_dataset_20000.csv"
UNSW_TRAIN_FILE = "UNSW_NB15_training-set.csv"
UNSW_TEST_FILE = "UNSW_NB15_testing-set.csv"


def project_root() -> Path:
    return Path(__file__).resolve().parent


def find_file(root: Path, filename: str) -> Path:
    path = root / "data" / filename
    if not path.exists():
        raise FileNotFoundError(f"Required dataset not found: {path}\nPut the CSV in the project's data folder.")
    return path


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # Normalize whitespace-only categorical values while preserving actual missing values.
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
    return df


def read_csv(path: Path) -> pd.DataFrame:
    # low_memory=False avoids mixed-type inference warnings on network datasets.
    return clean_frame(pd.read_csv(path, low_memory=False))


def make_preprocessor(X: pd.DataFrame, scale_numeric: bool = False) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    num_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(num_steps)
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # scikit-learn older than 1.2
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", encoder),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ], remainder="drop")


def make_models(preprocessor: ColumnTransformer) -> dict[str, Any]:
    # Moderate tree counts keep this usable on a normal student laptop.
    return {
        "Logistic Regression": Pipeline([
            ("preprocess", clone(preprocessor)),
            ("model", LogisticRegression(max_iter=1200, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("preprocess", clone(preprocessor)),
            ("model", DecisionTreeClassifier(max_depth=24, min_samples_leaf=2,
                                               class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", clone(preprocessor)),
            ("model", RandomForestClassifier(n_estimators=100, max_features="sqrt",
                                               min_samples_leaf=2, class_weight="balanced",
                                               n_jobs=-1, random_state=RANDOM_STATE)),
        ]),
        "Extra Trees": Pipeline([
            ("preprocess", clone(preprocessor)),
            ("model", ExtraTreesClassifier(n_estimators=100, max_features="sqrt",
                                             min_samples_leaf=2, class_weight="balanced",
                                             n_jobs=-1, random_state=RANDOM_STATE)),
        ]),
    }


def metric_row(name: str, y_true: pd.Series, pred: np.ndarray, split: str) -> dict[str, Any]:
    return {
        "Algorithm": name,
        f"{split} Accuracy": float(accuracy_score(y_true, pred)),
        f"{split} Macro F1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        f"{split} Weighted F1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        f"{split} Precision Macro": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        f"{split} Recall Macro": float(recall_score(y_true, pred, average="macro", zero_division=0)),
    }


def evaluate(train_X: pd.DataFrame, train_y: pd.Series,
             val_X: pd.DataFrame, val_y: pd.Series,
             test_X: pd.DataFrame, test_y: pd.Series,
             output_dir: Path, dataset_label: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor = make_preprocessor(train_X, scale_numeric=True)
    models = make_models(preprocessor)
    rows: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    failures: list[str] = []

    for name, model in models.items():
        print(f"[{dataset_label}] Training {name}...", flush=True)
        try:
            model.fit(train_X, train_y)
            val_pred = model.predict(val_X)
            test_pred = model.predict(test_X)
            fitted[name] = model
            row = {"Algorithm": name, "Status": "OK"}
            row.update(metric_row(name, val_y, val_pred, "Validation"))
            row.update(metric_row(name, test_y, test_pred, "Test"))
            rows.append(row)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            rows.append({"Algorithm": name, "Status": f"ERROR: {type(exc).__name__}: {exc}"})
            print(f"[{dataset_label}] {name} failed: {exc}", file=sys.stderr, flush=True)

    # Optional voting model. Use the two tree pipelines; preprocessing is fitted within each pipeline.
    if "Random Forest" in fitted and "Extra Trees" in fitted:
        try:
            voting = VotingClassifier(
                estimators=[("rf", clone(models["Random Forest"])), ("et", clone(models["Extra Trees"]))],
                voting="soft", weights=[1, 1], n_jobs=1,
            )
            print(f"[{dataset_label}] Training Soft Voting...", flush=True)
            voting.fit(train_X, train_y)
            vp = voting.predict(val_X)
            tp = voting.predict(test_X)
            fitted["Soft Voting (RF + Extra Trees)"] = voting
            row = {"Algorithm": "Soft Voting (RF + Extra Trees)", "Status": "OK"}
            row.update(metric_row(row["Algorithm"], val_y, vp, "Validation"))
            row.update(metric_row(row["Algorithm"], test_y, tp, "Test"))
            rows.append(row)
        except Exception as exc:
            failures.append(f"Soft Voting: {type(exc).__name__}: {exc}")
            rows.append({"Algorithm": "Soft Voting (RF + Extra Trees)", "Status": f"ERROR: {type(exc).__name__}: {exc}"})

    scores = pd.DataFrame(rows)
    scores.to_csv(output_dir / "algorithm_comparison.csv", index=False)
    successful = scores[scores["Status"] == "OK"].copy()
    if successful.empty:
        raise RuntimeError("Every classifier failed. " + " | ".join(failures))
    selected_name = successful.sort_values(
        ["Validation Macro F1", "Validation Accuracy"], ascending=False
    ).iloc[0]["Algorithm"]
    selected = fitted[selected_name]
    pred = selected.predict(test_X)
    labels = sorted(pd.Series(test_y).astype(str).unique().tolist())
    y_test_strings = pd.Series(test_y).astype(str)
    pred_strings = pd.Series(pred).astype(str)
    cm = confusion_matrix(y_test_strings, pred_strings, labels=labels)
    report = classification_report(y_test_strings, pred_strings, labels=labels,
                                   output_dict=True, zero_division=0)
    metrics = {
        "dataset": dataset_label,
        "target": "attack_type" if dataset_label == "synthetic" else "attack_cat",
        "selected_algorithm": selected_name,
        "test_accuracy": float(accuracy_score(test_y, pred)),
        "test_macro_f1": float(f1_score(test_y, pred, average="macro", zero_division=0)),
        "test_weighted_f1": float(f1_score(test_y, pred, average="weighted", zero_division=0)),
        "test_precision_macro": float(precision_score(test_y, pred, average="macro", zero_division=0)),
        "test_recall_macro": float(recall_score(test_y, pred, average="macro", zero_division=0)),
        "train_rows": int(len(train_y)), "validation_rows": int(len(val_y)),
        "test_rows": int(len(test_y)), "feature_count": int(train_X.shape[1]),
        "features": list(train_X.columns), "labels": labels,
        "confusion_matrix": cm.tolist(), "classification_report": report,
        "failures": failures,
    }
    (output_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(report).T.to_csv(output_dir / "selected_model_classification_report.csv")
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(output_dir / "selected_model_confusion_matrix.csv")
    pd.DataFrame({"Actual": y_test_strings, "Predicted": pred_strings}).to_csv(
        output_dir / "selected_model_test_predictions.csv", index=False)
    # Class counts are useful for documenting imbalance.
    pd.DataFrame({"class": pd.Series(train_y).value_counts().index,
                  "training_count": pd.Series(train_y).value_counts().values}).to_csv(
        output_dir / "training_class_distribution.csv", index=False)
    print(f"\n[{dataset_label}] Selected by validation macro-F1: {selected_name}")
    print(f"[{dataset_label}] Official/held-out test accuracy: {metrics['test_accuracy']:.4f}")
    print(f"[{dataset_label}] Test macro-F1: {metrics['test_macro_f1']:.4f}")
    print(scores.to_string(index=False))
    create_excel_report(output_dir, scores, metrics)
    create_pdf_report(output_dir, scores, metrics)
    return metrics


def create_excel_report(output_dir: Path, scores: pd.DataFrame, metrics: dict[str, Any]) -> None:
    """Write a compact Excel report with model comparison and confusion matrix."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.chart import BarChart, Reference
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    dash = wb.active
    dash.title = "Summary"
    dash.append(["PQ-Auth Sentinel Research Report", ""])
    dash.append(["Dataset", metrics["dataset"]])
    dash.append(["Target", metrics["target"]])
    dash.append(["Selected model", metrics["selected_algorithm"]])
    dash.append(["Test accuracy", metrics["test_accuracy"]])
    dash.append(["Test macro-F1", metrics["test_macro_f1"]])
    dash.append(["Test weighted F1", metrics["test_weighted_f1"]])
    dash.append(["Training rows", metrics["train_rows"]])
    dash.append(["Validation rows", metrics["validation_rows"]])
    dash.append(["Test rows", metrics["test_rows"]])
    dash.append(["Feature count", metrics["feature_count"]])
    comp = wb.create_sheet("Algorithm Comparison")
    comp.append(list(scores.columns))
    for row in scores.itertuples(index=False, name=None):
        comp.append([None if pd.isna(v) else v for v in row])
    cm = wb.create_sheet("Confusion Matrix")
    cm.append(["Actual / Predicted"] + metrics["labels"])
    for label, values in zip(metrics["labels"], metrics["confusion_matrix"]):
        cm.append([label] + values)
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="294766")
            cell.alignment = Alignment(wrap_text=True)
        for col in range(1, ws.max_column + 1):
            width = max(12, min(40, max(len(str(ws.cell(r, col).value or "")) for r in range(1, ws.max_row + 1)) + 2))
            ws.column_dimensions[get_column_letter(col)].width = width
    dash["B5"].number_format = "0.00%"
    dash["B6"].number_format = "0.000"
    dash["B7"].number_format = "0.000"
    if "Test Accuracy" in scores.columns:
        chart = BarChart()
        chart.type = "bar"
        chart.title = "Held-out test accuracy by model"
        chart.x_axis.title = "Accuracy"
        chart.y_axis.title = "Algorithm"
        col = list(scores.columns).index("Test Accuracy") + 1
        chart.add_data(Reference(comp, min_col=col, min_row=1, max_row=len(scores) + 1), titles_from_data=True)
        chart.set_categories(Reference(comp, min_col=1, min_row=2, max_row=len(scores) + 1))
        dash.add_chart(chart, "D2")
    wb.save(output_dir / "PQ_Auth_Sentinel_Research_Dashboard.xlsx")


def create_pdf_report(output_dir: Path, scores: pd.DataFrame, metrics: dict[str, Any]) -> None:
    """Write a compact PDF summary; full tables remain available in CSV/XLSX."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.units import mm

    path = output_dir / "PQ_Auth_Sentinel_Research_Report.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=landscape(A4), leftMargin=10*mm, rightMargin=10*mm, topMargin=10*mm, bottomMargin=10*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ResearchTitle", parent=styles["Title"], alignment=TA_CENTER))
    story = [Paragraph("PQ-Auth Sentinel — Research Testing Report", styles["ResearchTitle"]),
             Paragraph(f"Dataset: {metrics['dataset']} | Target: {metrics['target']}", styles["Normal"]),
             Spacer(1, 4*mm), Paragraph("Selected model and held-out test results", styles["Heading2"])]
    summary = [["Measure", "Result"], ["Selected model", str(metrics["selected_algorithm"])],
               ["Test accuracy", f"{metrics['test_accuracy']:.2%}"],
               ["Test macro-F1", f"{metrics['test_macro_f1']:.3f}"],
               ["Test weighted F1", f"{metrics['test_weighted_f1']:.3f}"],
               ["Train / validation / test rows", f"{metrics['train_rows']} / {metrics['validation_rows']} / {metrics['test_rows']}"],
               ["Input feature count", str(metrics["feature_count"])]]
    table = Table(summary, colWidths=[65*mm, 105*mm])
    table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#294766")),
                               ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                               ("GRID", (0,0), (-1,-1), .4, colors.grey),
                               ("PADDING", (0,0), (-1,-1), 5)]))
    story += [table, Spacer(1, 4*mm), Paragraph("Algorithm comparison", styles["Heading2"])]
    wanted = [c for c in ["Algorithm", "Validation Accuracy", "Validation Macro F1", "Test Accuracy", "Test Macro F1", "Status"] if c in scores.columns]
    rows = [wanted]
    for _, row in scores.iterrows():
        vals = []
        for col in wanted:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(f"{val:.3f}")
            else:
                vals.append(str(val))
        rows.append(vals)
    comp = Table(rows, repeatRows=1)
    comp.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#294766")),
                              ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                              ("FONTSIZE", (0,0), (-1,-1), 7),
                              ("GRID", (0,0), (-1,-1), .35, colors.grey),
                              ("VALIGN", (0,0), (-1,-1), "TOP")]))
    story += [comp, Spacer(1, 4*mm), Paragraph("Research limitation", styles["Heading2"]),
              Paragraph("UNSW-NB15 is a network intrusion detection benchmark. Its metrics do not directly validate ML-DSA signatures or the application's seven authentication telemetry features. Report results on each dataset separately.", styles["BodyText"])]
    doc.build(story)


def load_synthetic(path: Path):
    df = read_csv(path)
    required = SYNTHETIC_FEATURES + [SYNTHETIC_TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Synthetic CSV missing columns {missing}. Available: {list(df.columns)}")
    X = df[SYNTHETIC_FEATURES].copy()
    for col in SYNTHETIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    y = df[SYNTHETIC_TARGET].astype("string").str.strip()
    valid = y.notna() & (y != "") & (y.str.lower() != "nan")
    return X.loc[valid].reset_index(drop=True), y.loc[valid].astype(str).reset_index(drop=True)


def run_synthetic(root: Path, out_root: Path) -> dict[str, Any]:
    path = find_file(root, SYNTHETIC_FILE)
    print(f"Synthetic dataset: {path}")
    X, y = load_synthetic(path)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=RANDOM_STATE, stratify=y_trainval)
    return evaluate(X_train, y_train, X_val, y_val, X_test, y_test,
                    out_root / "synthetic", "synthetic")


def run_unsw(root: Path, out_root: Path) -> dict[str, Any]:
    train_path = find_file(root, UNSW_TRAIN_FILE)
    test_path = find_file(root, UNSW_TEST_FILE)
    print(f"UNSW-NB15 training CSV: {train_path}")
    print(f"UNSW-NB15 testing CSV:  {test_path}")
    train_df = read_csv(train_path)
    test_df = read_csv(test_path)
    target = "attack_cat"
    for name, df in (("training", train_df), ("testing", test_df)):
        if target not in df.columns:
            raise ValueError(f"{name} CSV does not contain '{target}'. Columns: {list(df.columns)}")
    # Do not use id, binary label, or attack_cat as input when attack_cat is the target.
    excluded = {"id", "label", "attack_cat"}
    feature_cols = [c for c in train_df.columns if c not in excluded and c in test_df.columns]
    if not feature_cols:
        raise ValueError("No shared UNSW feature columns found between training and testing CSVs.")
    train_y = train_df[target].astype("string").str.strip().fillna("Unknown").astype(str)
    test_y = test_df[target].astype("string").str.strip().fillna("Unknown").astype(str)
    # Drop rows with missing targets; preserve official split, do not mix train and test.
    train_mask = train_y.str.lower().ne("nan") & train_y.ne("")
    test_mask = test_y.str.lower().ne("nan") & test_y.ne("")
    train_df, train_y = train_df.loc[train_mask].reset_index(drop=True), train_y.loc[train_mask].reset_index(drop=True)
    test_df, test_y = test_df.loc[test_mask].reset_index(drop=True), test_y.loc[test_mask].reset_index(drop=True)
    train_X = train_df[feature_cols].copy()
    test_X = test_df[feature_cols].copy()
    # Ensure columns with numeric-looking values are treated consistently across official splits.
    for col in feature_cols:
        train_numeric = pd.to_numeric(train_X[col], errors="coerce")
        test_numeric = pd.to_numeric(test_X[col], errors="coerce")
        # Convert to numeric only if both sets are at least mostly numeric.
        if train_numeric.notna().mean() > 0.95 and test_numeric.notna().mean() > 0.95:
            train_X[col] = train_numeric
            test_X[col] = test_numeric
        else:
            train_X[col] = train_X[col].astype("string").fillna("__MISSING__").astype(str)
            test_X[col] = test_X[col].astype("string").fillna("__MISSING__").astype(str)
    # Create validation from the official training split only; official test remains untouched.
    train_X, val_X, train_y, val_y = train_test_split(
        train_X, train_y, test_size=0.20, random_state=RANDOM_STATE, stratify=train_y)
    # A class can appear in official test but not train; the classifier will simply not predict it.
    print(f"UNSW rows: train={len(train_X)}, validation={len(val_X)}, official_test={len(test_X)}")
    print("UNSW target class counts (training split):")
    print(train_y.value_counts().to_string())
    return evaluate(train_X, train_y, val_X, val_y, test_X, test_y,
                    out_root / "unsw_nb15", "unsw_nb15")


def main() -> int:
    parser = argparse.ArgumentParser(description="PQ-Auth Sentinel research testing for synthetic and UNSW-NB15 CSVs")
    parser.add_argument("--dataset", choices=["synthetic", "unsw", "both"], default="both",
                        help="Which dataset to evaluate (default: both separately)")
    parser.add_argument("--data-dir", default="data", help="Dataset folder relative to project root, or absolute path")
    parser.add_argument("--output", default="research_outputs", help="Output folder relative to project root, or absolute path")
    args = parser.parse_args()
    root = project_root()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    # Dataset discovery is anchored at project root/data by default. Allow an explicit alternate data folder.
    root_for_data = root
    if data_dir != root / "data":
        root_for_data = root
        # find_file() is deliberately not used for custom data-dir paths.
    out_root = Path(args.output)
    if not out_root.is_absolute():
        out_root = root / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    def required_file(name: str) -> Path:
        candidate = data_dir / name
        if not candidate.exists():
            raise FileNotFoundError(f"Required dataset not found: {candidate}\nPlace the CSV in the same data folder or pass --data-dir.")
        return candidate

    try:
        results = {}
        if args.dataset in ("synthetic", "both"):
            synthetic_path = required_file(SYNTHETIC_FILE)
            X, y = load_synthetic(synthetic_path)
            X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
            X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=RANDOM_STATE, stratify=y_trainval)
            results["synthetic"] = evaluate(X_train, y_train, X_val, y_val, X_test, y_test, out_root / "synthetic", "synthetic")
        if args.dataset in ("unsw", "both"):
            train_path, test_path = required_file(UNSW_TRAIN_FILE), required_file(UNSW_TEST_FILE)
            # run_unsw expects standard project/data paths; load and evaluate directly for custom --data-dir too.
            tr, te = read_csv(train_path), read_csv(test_path)
            if "attack_cat" not in tr.columns or "attack_cat" not in te.columns:
                raise ValueError("Both UNSW CSVs must contain the 'attack_cat' column.")
            excluded = {"id", "label", "attack_cat"}
            features = [c for c in tr.columns if c not in excluded and c in te.columns]
            ty = tr["attack_cat"].astype("string").str.strip()
            ey = te["attack_cat"].astype("string").str.strip()
            tm, em = ty.notna() & ty.ne("") & ty.str.lower().ne("nan"), ey.notna() & ey.ne("") & ey.str.lower().ne("nan")
            tr, ty = tr.loc[tm].reset_index(drop=True), ty.loc[tm].astype(str).reset_index(drop=True)
            te, ey = te.loc[em].reset_index(drop=True), ey.loc[em].astype(str).reset_index(drop=True)
            tx, ex = tr[features].copy(), te[features].copy()
            for col in features:
                tn, en = pd.to_numeric(tx[col], errors="coerce"), pd.to_numeric(ex[col], errors="coerce")
                if tn.notna().mean() > 0.95 and en.notna().mean() > 0.95:
                    tx[col], ex[col] = tn, en
                else:
                    tx[col] = tx[col].astype("string").fillna("__MISSING__").astype(str)
                    ex[col] = ex[col].astype("string").fillna("__MISSING__").astype(str)
            tx, vx, ty, vy = train_test_split(tx, ty, test_size=0.20, random_state=RANDOM_STATE, stratify=ty)
            results["unsw_nb15"] = evaluate(tx, ty, vx, vy, ex, ey, out_root / "unsw_nb15", "unsw_nb15")
        (out_root / "run_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nReports and CSV results saved under: {out_root}")
        print("Note: UNSW-NB15 evaluates network intrusion categories; it is not a direct validation of ML-DSA authentication or the app's seven authentication features.")
        return 0
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        if os.environ.get("PQ_AUTH_DEBUG") == "1":
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
