from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DATASET_PATH = DATA_DIR / "pq_ai_authentication_dataset_20000.csv"

APP_TITLE = "PQ-Auth Sentinel"
APP_SUBTITLE = "Post-Quantum Authentication + Behavioural Threat Intelligence"

MLDSA_ALGORITHM = "ML-DSA-65"

FEATURE_COLUMNS = [
    "request_frequency",
    "inter_request_time_sec",
    "failed_attempts",
    "session_duration_min",
    "device_change",
    "certificate_valid",
    "request_pattern_score",
]

TARGET_COLUMN = "attack_type"

RISK_BY_ATTACK = {
    "Normal": 0.08,
    "Replay": 0.78,
    "Flooding": 0.88,
    "Spoofing": 0.82,
}

SEVERITY_BY_ATTACK = {
    "Normal": 0.05,
    "Replay": 0.82,
    "Flooding": 0.94,
    "Spoofing": 0.88,
}

LOW_MAX = 0.34
MEDIUM_MAX = 0.64

MAX_SESSION_TRANSACTIONS = 500
