from __future__ import annotations

from .telemetry import (

    generate_normal_telemetry,

    generate_suspicious_telemetry,

    generate_flooding_telemetry,

)

from .telemetry import (

    record_auth_event,

    get_live_metrics,

)

from .detector import classify_security_condition



import json

import math

import secrets

from datetime import datetime, timezone

import pandas as pd

import streamlit as st



from .config import APP_SUBTITLE, MLDSA_ALGORITHM, FEATURE_COLUMNS

from .crypto import (

    KeyMaterial, PQCryptoUnavailable, canonical_message, generate_keypair,

    sign_message, verify_message, short_digest, available_algorithms

)

from .ml_engine import ModelBundle, score_request

from .session import (

    add_transaction, new_request_id, new_signature_id, new_session_id,

    now_utc, transaction_frame

)

from .ui import hero, info_card, status_pill





def render_overview(bundle: ModelBundle) -> None:

    hero(

        "Authentication, rebuilt around two independent proofs.",

        "A database-free research portal that verifies the cryptographic request first, then evaluates behavioural risk. The private signing key never becomes part of a transaction.",

        "POST-QUANTUM SECURITY LAB"

    )



    tx = transaction_frame()

    total = len(tx)

    valid = int((tx["signature_valid"] == True).sum()) if total else 0

    blocked = int((tx["decision"] == "BLOCK").sum()) if total else 0

    flagged = int((tx["decision"] == "FLAG").sum()) if total else 0



    c1, c2, c3, c4 = st.columns(4)

    with c1: info_card("Session requests", str(total), "In-memory demo ledger")

    with c2: info_card("Cryptographically valid", str(valid), "ML-DSA verification")

    with c3: info_card("Flagged", str(flagged), "Requires policy review")

    with c4: info_card("Blocked", str(blocked), "Hard security gate")



    st.markdown("### Transaction path")

    st.markdown("""

    <div class="flow">

      <div class="node">Request portal</div><div class="arrow">→</div>

      <div class="node">Canonical message</div><div class="arrow">→</div>

      <div class="node">ML-DSA-65</div><div class="arrow">→</div>

      <div class="node">Verifier</div><div class="arrow">→</div>

      <div class="node">Behaviour model</div><div class="arrow">→</div>

      <div class="node">Risk policy</div><div class="arrow">→</div>

      <div class="node">Decision</div>

    </div>

    """, unsafe_allow_html=True)



    st.markdown("### Model assurance")

    m = bundle.metrics

    a, b, c = st.columns(3)

    a.metric("Test accuracy", f"{m['test_accuracy']:.2%}")

    b.metric("Macro F1", f"{m['test_macro_f1']:.3f}")

    c.metric("Test samples", f"{m['test_rows']:,}")



    st.info(

        "Academic note: these metrics are measured on the supplied synthetic dataset. "

        "They demonstrate prototype discrimination, not production detection performance."

    )



    st.markdown("### Security boundaries")

    left, right = st.columns(2)

    with left:

        st.markdown("""

        <div class="card">

          <b>Cryptographic gate</b>

          <p class="section-note">

          If the signed message changes, verification fails and the request is blocked.

          ML is never allowed to override a failed signature.

          </p>

        </div>

        """, unsafe_allow_html=True)

    with right:

        st.markdown("""

        <div class="card">

          <b>Behavioural gate</b>

          <p class="section-note">

          A valid signature proves possession of the signing key; it does not prove

          that the behaviour is benign. The model therefore operates after verification.

          </p>

        </div>

        """, unsafe_allow_html=True)





def _ensure_identity() -> KeyMaterial | None:

    """

    Return the active ML-DSA identity.



    No warning/error styling is used here because the absence of an

    identity is an expected first-run state, not a system failure.

    """

    key = st.session_state.get("key_material")



    if key is None:

        st.info(

            "No authentication identity is active yet. "

            "Open Identity & Keys and generate an ML-DSA-65 keypair "

            "before creating a signed authentication transaction.",

            icon="ℹ️",

        )

        return None



    return key





def render_identity() -> None:

    hero(

        "Identity & key ceremony",

        "Create a local demonstration identity and an ML-DSA-65 keypair. Only the public key belongs in the verification record.",

        "01 · CRYPTOGRAPHIC IDENTITY"

    )



    key = st.session_state.get("key_material")

    if key is not None:

        st.success(f"Identity active · key fingerprint {key.key_fingerprint}")

        a, b, c = st.columns(3)

        a.metric("Algorithm", key.algorithm)

        b.metric("Public key bytes", len(key.public_key))

        c.metric("Secret key bytes", len(key.secret_key))



        st.markdown("### Public verification material")

        st.code(json.dumps(key.public_record(), indent=2), language="json")



        st.warning(

            "The secret key is intentionally not displayed or written to the transaction. "

            "Use “Reset identity” only when you want a fresh demo keypair."

        )

        if st.button("Reset identity", type="secondary"):

            st.session_state.key_material = None

            st.session_state.identity = None

            st.rerun()

        return



    st.markdown("""

    <div class="card">

      <b>What happens here</b>

      <p class="section-note">

      1. liboqs generates an ML-DSA-65 keypair.<br>

      2. The public key is exposed to the verifier.<br>

      3. The secret key remains in the current application session.<br>

      4. Each authentication request is signed independently.

      </p>

    </div>

    """, unsafe_allow_html=True)



    st.write("")

    user_id = st.text_input("User ID", value="USER-PORTAL-001")

    device_id = st.text_input("Device ID", value="DEV-PORTAL-001")



    if st.button("Generate ML-DSA identity", type="primary", use_container_width=True):

        try:

            key = generate_keypair()

            st.session_state.key_material = key

            st.session_state.identity = {"user_id": user_id.strip(), "device_id": device_id.strip()}

            st.rerun()

        except PQCryptoUnavailable as exc:

            st.error(str(exc))

            st.code(

                'python -c "import oqs; print([x for x in oqs.get_enabled_sig_mechanisms() if \'ML-DSA\' in x])"',

                language="bash"

            )





def _generate_application_telemetry() -> dict:

    """

    Generate behavioural telemetry internally.



    Normal users never manually enter the ML feature values.

    """



    return generate_normal_telemetry()





def render_request(bundle: ModelBundle) -> None:

    hero(

        "Create authentication request",

        "Authenticate a user through an automatic post-quantum security pipeline. "

        "Security metadata and behavioural telemetry are generated by the application.",

        "02 · AUTHENTICATION REQUEST",

    )



    key = _ensure_identity()



    if key is None:

        return



    identity = st.session_state.get("identity", {})



    st.markdown("### Authentication details")



    c1, c2 = st.columns(2)



    with c1:

        user_id = st.text_input(

            "User ID",

            value=identity.get("user_id", "USER-PORTAL-001"),

            help="The identity requesting authentication.",

        )



    with c2:

        application = st.selectbox(

            "Application",

            [

                "PQ Research Portal",

                "Student Authentication Portal",

                "Secure API Gateway",

                "Research Data Portal",

            ],

        )



    authentication_type = st.selectbox(

        "Authentication type",

        [

            "Secure Login",

            "Session Authentication",

            "API Authentication",

        ],

    )



    st.markdown("### Security pipeline")



    st.info(

        "The application automatically generates the request ID, session ID, "

        "nonce, timestamp and behavioural telemetry. These values are not "

        "manually entered by the user.",

        icon="🔐",

    )



    st.markdown(

        """

        <div class="flow">

            <div class="node">Request</div>

            <div class="arrow">→</div>

            <div class="node">ML-DSA-65</div>

            <div class="arrow">→</div>

            <div class="node">Verify</div>

            <div class="arrow">→</div>

            <div class="node">Replay</div>

            <div class="arrow">→</div>

            <div class="node">Behaviour ML</div>

            <div class="arrow">→</div>

            <div class="node">Risk</div>

            <div class="arrow">→</div>

            <div class="node">Decision</div>

        </div>

        """,

        unsafe_allow_html=True,

    )



    st.write("")



    if st.button(

        "AUTHENTICATE REQUEST",

        type="primary",

        use_container_width=True,

    ):

        if not user_id.strip():

            st.error("Please enter a User ID.")

            return



        # ---------------------------------------------------------

        # 1. Generate security identifiers automatically

        # ---------------------------------------------------------



        request_id = new_request_id()

        signature_id = new_signature_id()

        session_id = new_session_id()



        import secrets

        nonce = secrets.token_hex(32)



        timestamp = now_utc()



        # ---------------------------------------------------------

        # 2. Generate behavioural telemetry automatically

        # ---------------------------------------------------------



        telemetry = _generate_application_telemetry()



        # ---------------------------------------------------------

        # 3. Build authentication payload

        # ---------------------------------------------------------



        payload = {

            "request_id": request_id,

            "signature_id": signature_id,

            "user_id": user_id.strip(),

            "application": application,

            "authentication_type": authentication_type,

            "session_id": session_id,

            "timestamp": timestamp,

            "nonce": nonce,

            "telemetry": telemetry,

        }



        # ---------------------------------------------------------

        # 4. Canonicalize the payload

        # ---------------------------------------------------------



        message = canonical_message(payload)



        # ---------------------------------------------------------

        # 5. ML-DSA-65 signing

        # ---------------------------------------------------------



        try:

            signature = sign_message(message, key)

        except PQCryptoUnavailable as exc:

            st.error(str(exc))

            return

        except Exception as exc:

            st.error(f"Signing failed: {exc}")

            return



        # ---------------------------------------------------------

        # 6. Independent signature verification

        # ---------------------------------------------------------



        signature_valid = verify_message(

            message,

            signature,

            key.public_key_b64,

            key.algorithm,

        )



        # ---------------------------------------------------------

        # 7. Replay protection

        # ---------------------------------------------------------



        if "used_nonces" not in st.session_state:

            st.session_state.used_nonces = set()



        replay_detected = nonce in st.session_state.used_nonces



        if not replay_detected:

            st.session_state.used_nonces.add(nonce)



        # ---------------------------------------------------------

        # 8. Behavioural ML

        # ---------------------------------------------------------



        if signature_valid and not replay_detected:

            result = score_request(

                bundle,

                telemetry,

                True,

            )

        else:

            result = {

                "attack_type": "Cryptographic Failure",

                "risk_score": 1.0,

                "risk_level": "High",

                "decision": "BLOCK",

                "probability_risk": 1.0,

                "class_probabilities": {},

            }



        # ---------------------------------------------------------

        # 9. Enforce security decision hierarchy

        # ---------------------------------------------------------



        if not signature_valid:

            decision = "BLOCK"

            risk_level = "High"

            risk_score = 1.0



        elif replay_detected:

            decision = "BLOCK"

            risk_level = "High"

            risk_score = 1.0



        else:

            risk_score = float(result["risk_score"])



            if risk_score >= 0.64:

                decision = "BLOCK"

            elif risk_score >= 0.34:

                decision = "FLAG"

            else:

                decision = "ALLOW"



            if risk_score >= 0.64:

                risk_level = "High"

            elif risk_score >= 0.34:

                risk_level = "Medium"

            else:

                risk_level = "Low"



        # ---------------------------------------------------------

        # 10. Store transaction

        # ---------------------------------------------------------



        tx = {

            **payload,

            "algorithm": key.algorithm,

            "key_fingerprint": key.key_fingerprint,

            "message": message,

            "signature": signature,

            "public_key": key.public_key_b64,

            "signature_valid": bool(signature_valid),

            "signature_status": (

                "VALID" if signature_valid else "INVALID"

            ),

            "replay_detected": bool(replay_detected),

            "attack_type": result["attack_type"],

            "risk_score": risk_score,

            "risk_level": risk_level,

            "decision": decision,

            "probability_risk": result.get(

                "probability_risk",

                risk_score,

            ),

            "class_probabilities": result.get(

                "class_probabilities",

                {},

            ),

            "tampered": False,

        }



        add_transaction(tx)



        # Keep the transaction available for the verifier page.

        st.session_state.last_transaction = tx



        # ---------------------------------------------------------

        # 11. Show result immediately

        # ---------------------------------------------------------



        st.success("Authentication pipeline completed.")



        st.markdown("## Authentication Result")



        if decision == "ALLOW":

            st.success("✓ ACCESS ALLOWED")

        elif decision == "FLAG":

            st.warning("⚠ AUTHENTICATION FLAGGED")

        else:

            st.error("✕ ACCESS BLOCKED")



        a, b, c = st.columns(3)



        with a:

            st.metric(

                "Risk Score",

                f"{risk_score:.3f}",

            )



        with b:

            st.metric(

                "Risk Level",

                risk_level.upper(),

            )



        with c:

            st.metric(

                "Decision",

                decision,

            )



        st.markdown("### Security Verification")



        x, y, z = st.columns(3)



        with x:

            st.metric(

                "ML-DSA-65",

                "VALID" if signature_valid else "INVALID",

            )



        with y:

            st.metric(

                "Replay Check",

                "PASSED" if not replay_detected else "DETECTED",

            )



        with z:

            st.metric(

                "Behaviour",

                result["attack_type"],

            )



        st.markdown("### Transaction identifiers")



        a, b, c = st.columns(3)



        with a:

            st.caption("REQUEST ID")

            st.code(request_id)



        with b:

            st.caption("SIGNATURE ID")

            st.code(signature_id)



        with c:

            st.caption("SESSION ID")

            st.code(session_id)



        with st.expander("View technical details"):

            st.json(

                {

                    "algorithm": key.algorithm,

                    "key_fingerprint": key.key_fingerprint,

                    "signature_valid": signature_valid,

                    "replay_detected": replay_detected,

                    "attack_type": result["attack_type"],

                    "risk_score": risk_score,

                    "risk_level": risk_level,

                    "decision": decision,

                    "telemetry": telemetry,

                }

            )





def render_verifier(bundle: ModelBundle) -> None:

    hero(

        "Verification console",

        "The verifier receives only public verification material plus the signed message. It performs the cryptographic decision first, then the behavioural decision.",

        "03 · INDEPENDENT VERIFICATION"

    )



    tx = st.session_state.get("last_transaction")

    if not tx:

        st.info("Create an authentication request first.")

        return



    st.markdown(f"### {tx['signature_id']}")

    a, b, c, d = st.columns(4)

    a.metric("Request", tx["request_id"])

    b.metric("Signature", tx["signature_id"])

    c.metric("Algorithm", tx["algorithm"])

    d.metric("Crypto status", tx["signature_status"])



    if tx["signature_valid"]:

        st.success("ML-DSA signature verification succeeded.")

    else:

        st.error("ML-DSA signature verification failed. Policy blocks the request.")



    if tx["signature_valid"]:

        st.markdown("### Behavioural analysis")

        x, y, z = st.columns(3)

        x.metric("Prediction", tx["attack_type"])

        y.metric("Risk score", f"{tx['risk_score']:.3f}")

        z.metric("Decision", tx["decision"])



        st.markdown(

            f"Risk level: {status_pill(tx['risk_level'])}",

            unsafe_allow_html=True

        )



        probs = pd.DataFrame(

            [{"Attack class": k, "Model probability": v}

             for k, v in tx["class_probabilities"].items()]

        ).sort_values("Model probability", ascending=False)

        st.bar_chart(probs.set_index("Attack class"))

    else:

        st.markdown(

            f"Risk level: {status_pill('High')} &nbsp; Decision: {status_pill('BLOCK')}",

            unsafe_allow_html=True

        )



    with st.expander("Signed envelope"):

        st.code(json.dumps({

            "request_id": tx["request_id"],

            "signature_id": tx["signature_id"],

            "algorithm": tx["algorithm"],

            "key_fingerprint": tx["key_fingerprint"],

            "message_sha256_prefix": short_digest(tx["message"]),

            "signature_base64": tx["signature"],

            "public_key_base64": tx["public_key"],

        }, indent=2), language="json")





def render_demo_lab(bundle: ModelBundle) -> None:

    hero(

        "Security demonstration lab",

        "Three controlled scenarios make the project's research contribution visible: genuine authentication, valid-key malicious behaviour, and message tampering.",

        "04 · PRESENTATION MODE"

    )



    st.markdown("### Scenario A · Genuine request")

    if st.button("Generate genuine scenario", use_container_width=True):

        st.session_state["demo_request"] = "normal"

        st.session_state["demo_values"] = {

            "request_frequency": 4.5, "inter_request_time_sec": 18.0,

            "failed_attempts": 0, "session_duration_min": 24.0,

            "device_change": 0, "certificate_valid": 1, "request_pattern_score": 0.12

        }



    st.markdown("### Scenario B · Valid signature + hostile behaviour")

    if st.button("Generate replay-style scenario", use_container_width=True):

        st.session_state["demo_request"] = "replay"

        st.session_state["demo_values"] = {

            "request_frequency": 8.0, "inter_request_time_sec": 1.0,

            "failed_attempts": 5, "session_duration_min": 2.0,

            "device_change": 1, "certificate_valid": 1, "request_pattern_score": 0.91

        }



    st.markdown("### Scenario C · Tampered message")

    if st.button("Generate tamper scenario", use_container_width=True):

        st.session_state["demo_request"] = "tamper"

        st.session_state["demo_values"] = {

            "request_frequency": 5.0, "inter_request_time_sec": 15.0,

            "failed_attempts": 0, "session_duration_min": 20.0,

            "device_change": 0, "certificate_valid": 1, "request_pattern_score": 0.15

        }



    scenario = st.session_state.get("demo_request")

    values = st.session_state.get("demo_values")

    if scenario and values:

        st.divider()

        st.write(f"Selected scenario: **{scenario.upper()}**")

        st.json(values)

        if st.button("Run selected scenario", type="primary", use_container_width=True):

            key = _ensure_identity()

            if key is None:

                return



            request_id = new_request_id()

            signature_id = new_signature_id()

            payload = {

                "request_id": request_id,

                "signature_id": signature_id,

                "user_id": st.session_state.identity["user_id"],

                "device_id": st.session_state.identity["device_id"],

                "session_id": new_session_id(),

                "timestamp": now_utc(),

                "telemetry": values,

            }

            message = canonical_message(payload)

            signature = sign_message(message, key)

            verify_payload = message

            if scenario == "tamper":

                verify_payload = message + "tampered"

            valid = verify_message(verify_payload, signature, key.public_key_b64, key.algorithm)

            result = score_request(bundle, values, valid)

            add_transaction({

                **payload,

                "algorithm": key.algorithm,

                "key_fingerprint": key.key_fingerprint,

                "message": verify_payload,

                "signature": signature,

                "public_key": key.public_key_b64,

                "signature_valid": bool(valid),

                "signature_status": "VALID" if valid else "INVALID",

                **result,

                "tampered": scenario == "tamper",

            })

            st.rerun()





def render_live_monitor(bundle: ModelBundle) -> None:
    hero(
        "Live security monitor",
        "Observe authentication activity captured by this Streamlit session. "
        "The monitor derives operational telemetry from actual authentication events "
        "instead of asking the user to enter ML feature values.",
        "05 · LIVE SECURITY MONITOR"
    )

    metrics = get_live_metrics(st)
    tx = transaction_frame()

    st.markdown("### Current authentication activity")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Requests / 60 sec",
            metrics["request_frequency"],
            help="Authentication events observed by this application during the last 60 seconds.",
        )

    with c2:
        st.metric(
            "Avg. inter-request time",
            f"{metrics['inter_request_time_sec']:.2f} s",
            help="Average time between observed authentication events in this session.",
        )

    with c3:
        st.metric(
            "Failed attempts",
            metrics["failed_attempts"],
            help="Authentication events recorded as unsuccessful in this session.",
        )

    with c4:
        st.metric(
            "Session duration",
            f"{metrics['session_duration_min']:.2f} min",
        )

    st.markdown("### Current security state")

    if tx.empty:
        st.info(
            "No authentication activity has been observed yet. "
            "Create an authentication request to populate the monitor."
        )
        return

    latest = tx.iloc[-1].to_dict()

    x, y, z, w = st.columns(4)

    with x:
        st.metric("Latest prediction", str(latest.get("attack_type", "Unknown")))

    with y:
        st.metric(
            "Risk score",
            f"{float(latest.get('risk_score', 0.0)):.3f}",
        )

    with z:
        st.metric("Risk level", str(latest.get("risk_level", "Unknown")).upper())

    with w:
        st.metric("Decision", str(latest.get("decision", "Unknown")))

    decision = str(latest.get("decision", "UNKNOWN")).upper()

    if decision == "ALLOW":
        st.success("✓ Current state: authentication allowed.")
    elif decision == "FLAG":
        st.warning("⚠ Current state: authentication requires review.")
    else:
        st.error("✕ Current state: authentication blocked.")

    st.markdown("### Security controls")

    a, b, c = st.columns(3)

    with a:
        crypto_state = "VALID" if bool(latest.get("signature_valid", False)) else "INVALID"
        st.metric("ML-DSA-65", crypto_state)

    with b:
        replay_state = (
            "DETECTED"
            if bool(latest.get("replay_detected", False))
            else "PASSED"
        )
        st.metric("Replay check", replay_state)

    with c:
        st.metric(
            "Observed events",
            len(tx),
            help="Total authentication transactions stored in the current Streamlit session.",
        )

    st.markdown("### Recent authentication activity")

    display_cols = [
        "timestamp",
        "request_id",
        "user_id",
        "signature_status",
        "attack_type",
        "risk_score",
        "risk_level",
        "decision",
    ]

    available = [column for column in display_cols if column in tx.columns]

    if available:
        recent = tx[available].tail(10).copy()

        if "risk_score" in recent.columns:
            recent["risk_score"] = recent["risk_score"].astype(float).round(3)

        st.dataframe(
            recent,
            use_container_width=True,
            hide_index=True,
            column_config={
                "risk_score": st.column_config.NumberColumn(format="%.3f"),
            },
        )

    st.caption(
        "Prototype boundary: this monitor observes authentication events generated "
        "inside the application session. It is not a network-wide intrusion sensor."
    )

    if st.button(
        "Refresh security monitor",
        type="secondary",
        use_container_width=True,
    ):
        st.rerun()


def render_transactions() -> None:

    hero(

        "Transaction explorer",

        "A transparent session ledger for the prototype. It is intentionally not a database: restart the Streamlit process and the demo ledger resets.",

        "06 · TRANSACTION TRACE"

    )



    df = transaction_frame()

    if df.empty:

        st.info("No authentication transactions have been generated in this session.")

        return



    display_cols = [

        "timestamp", "request_id", "signature_id", "user_id",

        "signature_status", "attack_type", "risk_score", "risk_level", "decision"

    ]

    shown = df[[c for c in display_cols if c in df.columns]].copy()

    st.dataframe(

        shown,

        use_container_width=True,

        hide_index=True,

        column_config={

            "risk_score": st.column_config.NumberColumn(format="%.3f"),

        },

    )



    ids = df["signature_id"].tolist()

    selected = st.selectbox("Inspect signature transaction", ids)

    row = df[df["signature_id"] == selected].iloc[0].to_dict()

    st.code(json.dumps(row, indent=2, default=str), language="json")





def render_model(bundle: ModelBundle) -> None:

    hero(

        "Model laboratory",

        "The classifier uses behavioural telemetry only. Attack labels, expected risk, expected decision, and cryptographic outcomes are kept out of the feature matrix.",

        "07 · ML ASSURANCE"

    )

    m = bundle.metrics

    a, b, c, d = st.columns(4)

    a.metric("Rows", f"{m['dataset_rows']:,}")

    b.metric("Validation accuracy", f"{m['validation_accuracy']:.2%}")

    c.metric("Test accuracy", f"{m['test_accuracy']:.2%}")

    d.metric("Macro F1", f"{m['test_macro_f1']:.3f}")



    st.markdown("### Feature contract")

    st.dataframe(

        pd.DataFrame({

            "Feature": bundle.feature_columns,

            "Role": [

                "Rate signal", "Timing signal", "Failure signal",

                "Session signal", "Context signal", "Trust signal",

                "Pattern signal"

            ],

        }),

        use_container_width=True, hide_index=True

    )



    st.markdown("### Test-set confusion matrix")

    cm = pd.DataFrame(

        m["confusion_matrix"],

        index=m["labels"],

        columns=m["labels"]

    )

    st.dataframe(cm, use_container_width=True)



    report = pd.DataFrame(m["classification_report"]).T.reset_index(names="class")

    st.dataframe(report, use_container_width=True)





def render_about() -> None:

    hero(

        "Engineering notes",

        "The implementation is intentionally conservative: cryptographic correctness is a hard prerequisite, while ML is a second-layer behavioural control.",

        "08 · DESIGN NOTES"

    )

    st.markdown("""

    <div class="card">

      <b>Why ML-DSA-65?</b>

      <p class="section-note">

      ML-DSA is the NIST-standardized lattice-based signature family. This prototype uses

      the 65 parameter set through the Open Quantum Safe Python binding and explicitly

      checks that the mechanism is enabled rather than silently falling back.

      </p>

      <b>Why an ensemble?</b>

      <p class="section-note">

      Extra Trees and Random Forest are strong, interpretable baselines for structured

      tabular telemetry. Soft voting combines their probability estimates while keeping

      the model easy to explain in a college viva.

      </p>

      <b>Why no database yet?</b>

      <p class="section-note">

      The current version isolates the portal workflow and cryptographic protocol.

      Transactions live only in Streamlit session state. A later persistence layer can

      replace this ledger without changing the signing/verifying contract.

      </p>

      <b>What this is not</b>

      <p class="section-note">

      It is a research/college prototype, not a production identity provider. A production

      deployment would add hardware-backed keys, authenticated transport, durable audit

      storage, replay state, authorization, key rotation, rate limiting, and operational

      monitoring.

      </p>

    </div>

    """, unsafe_allow_html=True)
