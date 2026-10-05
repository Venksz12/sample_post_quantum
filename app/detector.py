from __future__ import annotations


def classify_security_condition(
    *,
    signature_valid: bool,
    replay_detected: bool,
    telemetry: dict,
    ml_result: dict,
) -> dict:
    """
    Combine deterministic security controls and ML prediction.

    Cryptographic failures always take priority over ML.
    Replay detection always takes priority over ML.
    """

    # ---------------------------------------------------------
    # HARD SECURITY GATES
    # ---------------------------------------------------------

    if not signature_valid:
        return {
            "condition": "TAMPERED / INVALID SIGNATURE",
            "decision": "BLOCK",
            "reason": "ML-DSA signature verification failed.",
            "risk_score": 1.0,
            "risk_level": "High",
        }

    if replay_detected:
        return {
            "condition": "REPLAY ATTACK",
            "decision": "BLOCK",
            "reason": "Authentication nonce has already been used.",
            "risk_score": 1.0,
            "risk_level": "High",
        }

    # ---------------------------------------------------------
    # ML LAYER
    # ---------------------------------------------------------

    attack_type = ml_result.get(
        "attack_type",
        "Normal",
    )

    risk_score = float(
        ml_result.get("risk_score", 0.0)
    )

    if risk_score >= 0.64:
        decision = "BLOCK"
        risk_level = "High"

    elif risk_score >= 0.34:
        decision = "FLAG"
        risk_level = "Medium"

    else:
        decision = "ALLOW"
        risk_level = "Low"

    if attack_type == "Normal" and decision == "ALLOW":
        condition = "NORMAL AUTHENTICATION"

    elif attack_type == "Flooding":
        condition = "FLOODING / HIGH-RATE ACTIVITY"

    elif attack_type == "Spoofing":
        condition = "SPOOFING / ANOMALOUS BEHAVIOUR"

    elif attack_type == "Replay":
        condition = "REPLAY-LIKE BEHAVIOUR"

    else:
        condition = "SUSPICIOUS BEHAVIOUR"

    return {
        "condition": condition,
        "decision": decision,
        "reason": (
            f"Behavioural model classified the request as "
            f"{attack_type}."
        ),
        "risk_score": risk_score,
        "risk_level": risk_level,
    }