import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import FEATURE_COLUMNS
from app.ml_engine import load_dataset


def test_dataset_feature_contract():
    df = load_dataset()
    assert len(df) == 20000
    assert list(df[FEATURE_COLUMNS].columns) == FEATURE_COLUMNS
    assert set(df["attack_type"].unique()) == {"Normal", "Replay", "Flooding", "Spoofing"}


def test_no_target_leakage():
    forbidden = {
        "expected_risk_score",
        "expected_risk_level",
        "expected_decision",
        "attack_type",
        "signature_valid",
    }
    assert not (set(FEATURE_COLUMNS) & forbidden)
