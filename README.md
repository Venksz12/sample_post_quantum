# PQ-Auth Sentinel

A database-free Streamlit research prototype for post-quantum authentication with ML-DSA-65 and behavioural attack classification.

## Core workflow

Request Portal
→ canonical transaction
→ ML-DSA-65 signature
→ independent verification
→ behavioural feature extraction
→ Extra Trees + Random Forest soft-voting classifier
→ probability-weighted risk
→ deterministic policy
→ session transaction explorer

## Security design

- ML-DSA-65 is a hard cryptographic gate.
- The private key stays in Streamlit session memory.
- Transaction records contain the public key and signature, never the private key.
- The ML model does not use attack labels, risk score, risk level, decision, or signature validity as behavioural features.
- Tampering with a signed message causes verification failure.
- A valid signature does not automatically mean the request is benign; behavioural analysis is deliberately separate.

## Data

`data/pq_ai_authentication_dataset_20000.csv` is the supplied 20,000-row dataset.

The model feature contract is:

- request_frequency
- inter_request_time_sec
- failed_attempts
- session_duration_min
- device_change
- certificate_valid
- request_pattern_score

Target:

- attack_type

## Important prototype limitation

This version deliberately has no database. Streamlit session state acts as the temporary transaction ledger. Restarting the application clears the ledger and the in-memory private key.

For a production system, add durable audit storage, authenticated transport, hardware-backed or OS-protected private keys, replay protection, authorization, rate limiting, key rotation, and operational monitoring.

## ML-DSA dependency

The application intentionally fails closed if ML-DSA-65 is not enabled. It does not substitute RSA, ECDSA, Ed25519, or a homemade signature.

The `liboqs-python` package may build/install a matching liboqs release automatically, but source builds require Git, CMake and a C/C++ compiler. See the installation guide included with this project.

## Run

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/main.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/main.py
```

If `liboqs-python` cannot build automatically, install Git, CMake and a C/C++ compiler, then rerun:

```bash
pip install liboqs-python
```

Verify ML-DSA-65:

```bash
python -c "import oqs; print([x for x in oqs.get_enabled_sig_mechanisms() if 'ML-DSA' in x])"
```

You should see `ML-DSA-65`.

Then launch:

```bash
streamlit run app/main.py
```

## Presentation flow

1. Open **Identity** and generate an ML-DSA-65 identity.
2. Open **Request Portal** and create a genuine request.
3. Open **Verifier** and show `VALID → behavioural analysis → ALLOW`.
4. Use **Demo Lab → replay-style scenario** to show a valid signature with hostile behaviour.
5. Use **Demo Lab → tamper scenario** to show `INVALID → BLOCK`.
6. Open **Transactions** to inspect the signed envelope.
7. Open **Model Lab** to show the held-out test metrics and confusion matrix.

## Files

- `app/main.py` — Streamlit entry point
- `app/crypto.py` — ML-DSA key generation, signing and verification
- `app/ml_engine.py` — dataset loading, training, scoring and evaluation
- `app/session.py` — database-free session transaction ledger
- `app/views.py` — application screens
- `app/ui.py` — visual system
- `app/config.py` — configuration
- `data/pq_ai_authentication_dataset_20000.csv` — supplied training dataset
- `requirements.txt` — dependencies
- `run.py` — convenience launcher
