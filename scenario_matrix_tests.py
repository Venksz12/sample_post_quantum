"""
PQ-Auth Sentinel — 3x3 / 4x3 Scenario Matrix Test Runner
========================================================

Tests the combination of:
  Application choice × Authentication type

The screenshots show FOUR application choices and THREE authentication
types, which makes 12 combinations. To run exactly 9 cases, set
RUN_FIRST_THREE_APPS_ONLY = True below (the fourth app is then excluded).

Outputs are saved in ./test_case_results/:
  - test_case_results.csv
  - test_case_results.xlsx (with charts)
  - test_case_results_dashboard.png
  - test_case_results_report.pdf (if reportlab is installed)

Run from the project root:
    python scenario_matrix_tests.py

This is a controlled integration-test harness, not a live attack simulator.
It can optionally load the project's ML model and optionally run ML-DSA
smoke tests if the app modules and liboqs are available.
"""

from __future__ import annotations

import itertools
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------- CONFIGURATION -----------------------

RUN_FIRST_THREE_APPS_ONLY = True  # True = 9 cases; False = all 12 screenshot combinations

APPLICATION_OPTIONS = [
    "PQ Research Portal",
    "Student Authentication Portal",
    "Secure API Gateway",
    "Research Data Portal",
]

AUTHENTICATION_TYPES = [
    "Secure Login",
    "Session Authentication",
    "API Authentication",
]

FEATURE_COLUMNS = [
    "request_frequency",
    "inter_request_time_sec",
    "failed_attempts",
    "session_duration_min",
    "device_change",
    "certificate_valid",
    "request_pattern_score",
]

# Feature profiles make the test cases reproducible. These are controlled
# synthetic inputs, not measurements from a live browser or network.
TEST_PROFILES = [
    {
        "case": "Baseline / normal",
        "telemetry": {
            "request_frequency": 3.0,
            "inter_request_time_sec": 20.0,
            "failed_attempts": 0,
            "session_duration_min": 12.0,
            "device_change": 0,
            "certificate_valid": 1,
            "request_pattern_score": 0.08,
        },
    },
    {
        "case": "Elevated / suspicious",
        "telemetry": {
            "request_frequency": 9.0,
            "inter_request_time_sec": 2.5,
            "failed_attempts": 3,
            "session_duration_min": 3.0,
            "device_change": 1,
            "certificate_valid": 1,
            "request_pattern_score": 0.62,
        },
    },
    {
        "case": "High-rate / untrusted context",
        "telemetry": {
            "request_frequency": 18.0,
            "inter_request_time_sec": 0.6,
            "failed_attempts": 5,
            "session_duration_min": 1.5,
            "device_change": 1,
            "certificate_valid": 0,
            "request_pattern_score": 0.92,
        },
    },
]

RANDOM_STATE = 42


def find_dataset(project_root: Path) -> Path | None:
    candidates = [
        project_root / "data" / "pq_ai_authentication_dataset_20000.csv",
        project_root / "data" / "pq_ai_authentication_dataset_20000 (1).csv",
        project_root / "pq_ai_authentication_dataset_20000.csv",
        project_root / "pq_ai_authentication_dataset_20000 (1).csv",
    ]
    return next((p for p in candidates if p.exists()), None)


def load_optional_model(project_root: Path):
    """
    Prefer the project's trained model, so scenario predictions are aligned
    with the app. If unavailable, train a small comparison model from the
    project's CSV. Returns (model, status string).
    """
    # First try the app's own training function.
    try:
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from app.ml_engine import train_model  # type: ignore
        bundle = train_model()
        if hasattr(bundle, "model"):
            return bundle.model, "Loaded model from app.ml_engine.train_model()"
    except Exception as exc:
        app_error = str(exc)
    else:
        app_error = "app.ml_engine did not return a bundle with .model"

    dataset = find_dataset(project_root)
    if dataset is None:
        return None, f"Project model unavailable ({app_error}); dataset CSV not found."

    try:
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline

        df = pd.read_csv(dataset)
        df.columns = [str(c).strip() for c in df.columns]
        missing = [c for c in FEATURE_COLUMNS + ["attack_type"] if c not in df.columns]
        if missing:
            return None, f"Dataset missing columns {missing}; model not loaded."

        X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        y = df["attack_type"].astype(str).str.strip()
        valid = y.notna() & (y != "") & (y.str.lower() != "nan")
        X, y = X.loc[valid], y.loc[valid]
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", VotingClassifier(
                estimators=[
                    ("extra", ExtraTreesClassifier(
                        n_estimators=180, max_features="sqrt", min_samples_leaf=2,
                        class_weight="balanced", random_state=42, n_jobs=-1
                    )),
                    ("rf", RandomForestClassifier(
                        n_estimators=160, max_features="sqrt", min_samples_leaf=2,
                        class_weight="balanced", random_state=43, n_jobs=-1
                    )),
                ],
                voting="soft",
                weights=[2, 1],
                n_jobs=-1,
            )),
        ])
        model.fit(X, y)
        return model, f"Trained fallback soft-voting model on full CSV ({dataset.name}); no held-out metrics calculated in this harness."
    except Exception as exc:
        return None, f"Model unavailable: {exc}"


def optional_crypto_test() -> tuple[str, str]:
    """Return availability and a real sign/verify smoke test when OQS exists."""
    try:
        import oqs  # type: ignore
        if "ML-DSA-65" not in list(oqs.get_enabled_sig_mechanisms()):
            return "NOT AVAILABLE", "ML-DSA-65 is not enabled in this liboqs build."
        with oqs.Signature("ML-DSA-65") as signer:
            public_key = signer.generate_keypair()
            secret_key = signer.export_secret_key()
        message = b"PQ-Auth Sentinel 3x3 scenario matrix test"
        with oqs.Signature("ML-DSA-65", secret_key=secret_key) as signer:
            signature = signer.sign(message)
        with oqs.Signature("ML-DSA-65") as verifier:
            valid = bool(verifier.verify(message, signature, public_key))
            tampered = bool(verifier.verify(message + b"-changed", signature, public_key))
        passed = valid and not tampered
        return ("PASS" if passed else "FAIL",
                f"valid_signature={valid}; tampered_message_rejected={not tampered}")
    except Exception as exc:
        return "NOT RUN", f"liboqs/oqs unavailable or failed: {exc}"


def classify_risk(model, telemetry: dict) -> tuple[str, float | None, float | None, str]:
    if model is None:
        # Deterministic transparent baseline policy used only when model is unavailable.
        score = (
            0.30 * min(float(telemetry["request_frequency"]) / 20.0, 1.0)
            + 0.25 * min(2.0 / max(float(telemetry["inter_request_time_sec"]), 0.1), 1.0)
            + 0.20 * min(float(telemetry["failed_attempts"]) / 5.0, 1.0)
            + 0.15 * float(telemetry["device_change"])
            + 0.10 * (1.0 - float(telemetry["certificate_valid"]))
        )
        score = float(np.clip(score, 0, 1))
        decision = "BLOCK" if score >= 0.64 else "FLAG" if score >= 0.34 else "ALLOW"
        return "Rules-only fallback", None, score, decision

    row = pd.DataFrame([{c: telemetry[c] for c in FEATURE_COLUMNS}], columns=FEATURE_COLUMNS)
    try:
        prediction = str(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0] if hasattr(model, "predict_proba") else None
        classes = list(model.classes_) if hasattr(model, "classes_") else []
        if probabilities is not None:
            pmap = {str(k): float(v) for k, v in zip(classes, probabilities)}
            # A non-Normal probability is a risk proxy, not a calibrated security probability.
            score = float(np.clip(1.0 - pmap.get("Normal", 0.0), 0, 1))
        else:
            score = 0.0 if prediction.lower() == "normal" else 0.7
        decision = "BLOCK" if score >= 0.64 else "FLAG" if score >= 0.34 else "ALLOW"
        return prediction, float(np.max(probabilities)) if probabilities is not None else None, score, decision
    except Exception as exc:
        return f"MODEL ERROR: {exc}", None, None, "NOT EVALUATED"


def build_test_cases(apps: list[str]) -> pd.DataFrame:
    rows = []
    for app_name, auth_type, profile in itertools.product(apps, AUTHENTICATION_TYPES, TEST_PROFILES):
        rows.append({
            "Application": app_name,
            "Authentication Type": auth_type,
            "Profile": profile["case"],
            "Telemetry": profile["telemetry"],
        })
    return pd.DataFrame(rows)


def execute_matrix(project_root: Path) -> tuple[pd.DataFrame, str, tuple[str, str]]:
    model, model_status = load_optional_model(project_root)
    crypto_result = optional_crypto_test()
    apps = APPLICATION_OPTIONS[:3] if RUN_FIRST_THREE_APPS_ONLY else APPLICATION_OPTIONS
    test_inputs = build_test_cases(apps)
    results = []

    for number, row in enumerate(test_inputs.to_dict(orient="records"), start=1):
        telemetry = row["Telemetry"]
        prediction, confidence, score, decision = classify_risk(model, telemetry)

        # A test is considered executed if a model or transparent fallback produced
        # a decision. This is execution status, not a claim that the security control
        # detected a real attack.
        executed = decision in {"ALLOW", "FLAG", "BLOCK"}
        results.append({
            "Test Case ID": f"TC-{number:02d}",
            "Application": row["Application"],
            "Authentication Type": row["Authentication Type"],
            "Input Profile": row["Profile"],
            "Model Prediction": prediction,
            "Model Confidence": confidence,
            "Risk Proxy": round(score, 4) if score is not None else None,
            "Policy Suggestion": decision,
            "Execution Status": "EXECUTED" if executed else "ERROR",
            "ML-DSA Smoke Test": crypto_result[0],
            "Notes": "Controlled telemetry profile; no real browser/network attack generated.",
        })

    return pd.DataFrame(results), model_status, crypto_result


def save_csv(results: pd.DataFrame, out_dir: Path) -> Path:
    path = out_dir / "test_case_results.csv"
    df = results.copy()
    df["Model Confidence"] = pd.to_numeric(df["Model Confidence"], errors="coerce")
    df.to_csv(path, index=False)
    return path


def save_excel(results: pd.DataFrame, out_dir: Path, model_status: str, crypto_result: tuple[str, str]) -> Path:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.chart import BarChart, Reference
    from openpyxl.utils import get_column_letter

    path = out_dir / "test_case_results.xlsx"
    wb = Workbook()
    dash = wb.active
    dash.title = "Dashboard"
    details = wb.create_sheet("Test Case Results")

    dash["A1"] = "PQ-Auth Sentinel — Scenario Matrix Dashboard"
    dash.merge_cells("A1:H1")
    dash["A1"].font = Font(size=16, bold=True, color="FFFFFF")
    dash["A1"].fill = PatternFill("solid", fgColor="14233B")
    dash["A2"] = f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    dash.merge_cells("A2:H2")
    dash["A4"] = "Metric"
    dash["B4"] = "Value"
    metrics = [
        ("Test cases", len(results)),
        ("Executed", int((results["Execution Status"] == "EXECUTED").sum())),
        ("Errors", int((results["Execution Status"] == "ERROR").sum())),
        ("Applications", results["Application"].nunique()),
        ("Authentication types", results["Authentication Type"].nunique()),
        ("Model status", model_status),
        ("ML-DSA smoke test", crypto_result[0]),
        ("ML-DSA detail", crypto_result[1]),
    ]
    for item in metrics:
        dash.append(list(item))

    dash["D4"] = "Policy Suggestion"
    dash["E4"] = "Count"
    counts = results["Policy Suggestion"].value_counts()
    for r, decision in enumerate(["ALLOW", "FLAG", "BLOCK", "NOT EVALUATED"], start=5):
        dash.cell(r, 4, decision)
        dash.cell(r, 5, int(counts.get(decision, 0)))

    chart = BarChart()
    chart.type = "col"
    chart.title = "Decision suggestion distribution"
    chart.y_axis.title = "Test cases"
    chart.x_axis.title = "Policy suggestion"
    chart.add_data(Reference(dash, min_col=5, min_row=4, max_row=8), titles_from_data=True)
    chart.set_categories(Reference(dash, min_col=4, min_row=5, max_row=8))
    chart.height, chart.width = 7, 13
    dash.add_chart(chart, "G4")

    # Average risk by profile
    dash["D12"] = "Profile"
    dash["E12"] = "Average Risk Proxy"
    profile_means = results.groupby("Input Profile", dropna=False)["Risk Proxy"].mean()
    for r, (profile, mean) in enumerate(profile_means.items(), start=13):
        dash.cell(r, 4, str(profile))
        dash.cell(r, 5, float(mean) if pd.notna(mean) else 0.0)
    chart2 = BarChart()
    chart2.type = "bar"
    chart2.title = "Average risk proxy by profile"
    chart2.x_axis.title = "Risk proxy"
    chart2.y_axis.title = "Input profile"
    chart2.add_data(Reference(dash, min_col=5, min_row=12, max_row=12 + len(profile_means)), titles_from_data=True)
    chart2.set_categories(Reference(dash, min_col=4, min_row=13, max_row=12 + len(profile_means)))
    chart2.height, chart2.width = 7, 13
    dash.add_chart(chart2, "G20")

    # Full test case sheet
    export = results.copy()
    export["Model Confidence"] = pd.to_numeric(export["Model Confidence"], errors="coerce")
    details.append(list(export.columns))
    for row in export.itertuples(index=False, name=None):
        details.append(list(row))

    for ws in [dash, details]:
        ws.freeze_panes = "A2"
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="294766")
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        for col in range(1, ws.max_column + 1):
            max_len = max(len(str(ws.cell(r, col).value or "")) for r in range(1, ws.max_row + 1))
            ws.column_dimensions[get_column_letter(col)].width = min(max(max_len + 2, 12), 44)
        for row in ws.iter_rows():
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
    wb.save(path)
    return path


def save_chart_png(results: pd.DataFrame, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = out_dir / "test_case_results_dashboard.png"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    decision_counts = results["Policy Suggestion"].value_counts()
    axes[0].bar(decision_counts.index.astype(str), decision_counts.values)
    axes[0].set_title("Policy suggestion distribution")
    axes[0].set_ylabel("Number of test cases")
    axes[0].tick_params(axis="x", rotation=20)

    risk = results.groupby("Input Profile")["Risk Proxy"].mean()
    axes[1].bar(risk.index.astype(str), risk.values)
    axes[1].set_title("Average risk proxy by input profile")
    axes[1].set_ylabel("Risk proxy")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle("PQ-Auth Sentinel — Scenario Matrix Test Results")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def save_pdf(results: pd.DataFrame, out_dir: Path, model_status: str, crypto_result: tuple[str, str]) -> Path | None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.units import mm
    except ImportError:
        return None

    path = out_dir / "test_case_results_report.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=landscape(A4), leftMargin=8*mm, rightMargin=8*mm, topMargin=8*mm, bottomMargin=8*mm)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("PQ-Auth Sentinel — Scenario Matrix Test Report", styles["Title"]),
        Paragraph(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", styles["Normal"]),
        Spacer(1, 4*mm),
        Paragraph(f"Model status: {model_status}", styles["BodyText"]),
        Paragraph(f"ML-DSA smoke test: {crypto_result[0]} — {crypto_result[1]}", styles["BodyText"]),
        Spacer(1, 3*mm),
    ]
    table_data = [["ID", "Application", "Authentication", "Profile", "Prediction", "Risk", "Decision", "Execution"]]
    for _, row in results.iterrows():
        table_data.append([
            str(row["Test Case ID"]), str(row["Application"]), str(row["Authentication Type"]),
            str(row["Input Profile"]), str(row["Model Prediction"]),
            "" if pd.isna(row["Risk Proxy"]) else f'{float(row["Risk Proxy"]):.3f}',
            str(row["Policy Suggestion"]), str(row["Execution Status"]),
        ])
    table = Table(table_data, repeatRows=1, colWidths=[14*mm, 34*mm, 34*mm, 34*mm, 32*mm, 15*mm, 25*mm, 23*mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#294766")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTSIZE", (0,0), (-1,-1), 7),
        ("GRID", (0,0), (-1,-1), .35, colors.grey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F2F5F9")]),
    ]))
    story.append(table)
    story += [
        Spacer(1, 4*mm),
        Paragraph(
            "Interpretation: these are controlled input combinations and model/rule suggestions, not observed "
            "production attacks. The ML risk proxy is 1 − P(Normal) where model probabilities are available; "
            "it is not a calibrated probability of compromise. Passing this suite is not security certification.",
            styles["BodyText"],
        ),
    ]
    doc.build(story)
    return path


def main() -> int:
    project_root = Path(__file__).resolve().parent
    out_dir = project_root / "test_case_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    apps = APPLICATION_OPTIONS[:3] if RUN_FIRST_THREE_APPS_ONLY else APPLICATION_OPTIONS
    expected_count = len(apps) * len(AUTHENTICATION_TYPES) * len(TEST_PROFILES)
    print("=" * 80)
    print("PQ-AUTH SENTINEL — SCENARIO MATRIX TEST RUNNER")
    print("=" * 80)
    print(f"Applications: {len(apps)} | Authentication types: {len(AUTHENTICATION_TYPES)} | Profiles: {len(TEST_PROFILES)}")
    print(f"Expected test cases: {expected_count}")
    print("Loading project model / CSV fallback...")
    results, model_status, crypto_result = execute_matrix(project_root)

    # Persist evidence.
    csv_path = save_csv(results, out_dir)
    excel_path = save_excel(results, out_dir, model_status, crypto_result)
    chart_path = save_chart_png(results, out_dir)
    pdf_path = save_pdf(results, out_dir, model_status, crypto_result)

    print("\nMODEL STATUS")
    print(model_status)
    print("\nTEST CASE RESULTS")
    display_cols = [
        "Test Case ID", "Application", "Authentication Type", "Input Profile",
        "Model Prediction", "Risk Proxy", "Policy Suggestion", "Execution Status",
    ]
    print(results[display_cols].to_string(index=False))
    print("\nFILES CREATED")
    print(f"CSV:   {csv_path}")
    print(f"Excel: {excel_path}")
    print(f"Chart: {chart_path}")
    print(f"PDF:   {pdf_path if pdf_path else 'Not created (install reportlab)'}")
    print("\nNote: controlled research test inputs only; not production security certification.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nFATAL ERROR: {exc}", file=sys.stderr)
        if "--debug" in sys.argv:
            traceback.print_exc()
        raise SystemExit(1)
