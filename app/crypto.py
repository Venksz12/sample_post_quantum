from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import Optional

try:
    import oqs
except Exception as exc:  # pragma: no cover - environment dependent
    oqs = None
    OQS_IMPORT_ERROR = exc
else:
    OQS_IMPORT_ERROR = None

from .config import MLDSA_ALGORITHM


class PQCryptoUnavailable(RuntimeError):
    pass


def _require_oqs() -> None:
    if oqs is None:
        raise PQCryptoUnavailable(
            "liboqs-python is not available. Install dependencies and make sure "
            "ML-DSA-65 is enabled in liboqs."
        )


def available_algorithms() -> list[str]:
    if oqs is None:
        return []
    try:
        return list(oqs.get_enabled_sig_mechanisms())
    except Exception:
        return []


def ensure_mldsa65() -> None:
    _require_oqs()
    algorithms = available_algorithms()
    if MLDSA_ALGORITHM not in algorithms:
        raise PQCryptoUnavailable(
            f"{MLDSA_ALGORITHM} is not enabled by the installed liboqs build. "
            f"Enabled signature mechanisms include: {', '.join(algorithms[:12])}"
        )


@dataclass
class KeyMaterial:
    algorithm: str
    public_key_b64: str
    secret_key_b64: str

    @property
    def public_key(self) -> bytes:
        return base64.b64decode(self.public_key_b64)

    @property
    def secret_key(self) -> bytes:
        return base64.b64decode(self.secret_key_b64)

    @property
    def key_fingerprint(self) -> str:
        return hashlib.sha256(self.public_key).hexdigest()[:16].upper()

    def public_record(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "public_key": self.public_key_b64,
            "key_fingerprint": self.key_fingerprint,
        }


def generate_keypair() -> KeyMaterial:
    ensure_mldsa65()
    with oqs.Signature(MLDSA_ALGORITHM) as signer:
        public_key = signer.generate_keypair()
        secret_key = signer.export_secret_key()
    return KeyMaterial(
        algorithm=MLDSA_ALGORITHM,
        public_key_b64=base64.b64encode(public_key).decode("ascii"),
        secret_key_b64=base64.b64encode(secret_key).decode("ascii"),
    )


def sign_message(message: str, key: KeyMaterial) -> str:
    ensure_mldsa65()
    payload = message.encode("utf-8")
    with oqs.Signature(key.algorithm, secret_key=key.secret_key) as signer:
        signature = signer.sign(payload)
    return base64.b64encode(signature).decode("ascii")


def verify_message(message: str, signature_b64: str, public_key_b64: str,
                   algorithm: str = MLDSA_ALGORITHM) -> bool:
    ensure_mldsa65()
    try:
        signature = base64.b64decode(signature_b64, validate=True)
        public_key = base64.b64decode(public_key_b64, validate=True)
        with oqs.Signature(algorithm) as verifier:
            return bool(verifier.verify(
                message.encode("utf-8"), signature, public_key
            ))
    except Exception:
        return False


def canonical_message(payload: dict) -> str:
    """Stable JSON representation: every verifier signs/verifies the same bytes."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def short_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20].upper()

def build_signed_package(
    *,
    request_id: str,
    signature_id: str,
    session_id: str,
    timestamp: str,
    nonce: str,
    message: dict,
    signature: bytes,
    public_key: bytes,
    key_fingerprint: str,
    algorithm: str = "ML-DSA-65",
) -> dict:
    """
    Build the portable signed authentication envelope.

    The signature covers the canonical message. The private key
    is never included in this package.
    """

    return {
        "version": "1.0",
        "algorithm": algorithm,
        "request_id": request_id,
        "signature_id": signature_id,
        "session_id": session_id,
        "timestamp": timestamp,
        "nonce": nonce,
        "message": message,
        "signature": base64.b64encode(signature).decode("ascii"),
        "public_key": base64.b64encode(public_key).decode("ascii"),
        "key_fingerprint": key_fingerprint,
    }

def decode_signed_package(package: dict):
    """
    Decode the base64-encoded signature and public key
    from an uploaded authentication package.
    """

    if not isinstance(package, dict):
        raise ValueError("Authentication package must be a JSON object.")

    required = [
        "version",
        "algorithm",
        "request_id",
        "signature_id",
        "session_id",
        "timestamp",
        "nonce",
        "message",
        "signature",
        "public_key",
        "key_fingerprint",
    ]

    missing = [field for field in required if field not in package]

    if missing:
        raise ValueError(
            f"Authentication package is missing: {', '.join(missing)}"
        )

    if package["algorithm"] != "ML-DSA-65":
        raise ValueError(
            f"Unsupported signature algorithm: {package['algorithm']}"
        )

    try:
        signature = base64.b64decode(package["signature"])
        public_key = base64.b64decode(package["public_key"])
    except Exception as exc:
        raise ValueError("Invalid base64 signature or public key.") from exc

    return {
        "message": package["message"],
        "signature": signature,
        "public_key": public_key,
    }

def package_to_json(package: dict) -> bytes:
    """
    Convert a signed authentication package to deterministic,
    human-readable JSON bytes.
    """

    return json.dumps(
        package,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")

def package_from_json(data: bytes) -> dict:
    """
    Parse an uploaded authentication package.
    """

    try:
        package = json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise ValueError("Uploaded file is not valid UTF-8 JSON.") from exc

    if not isinstance(package, dict):
        raise ValueError("Uploaded package must contain a JSON object.")

    return package