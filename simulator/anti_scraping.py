"""Anti-scraping helpers for encrypting API payloads.

Encryption scheme
-----------------
- Algorithm: AES-256-GCM (authenticated encryption)
- Key: Base64-encoded 32 byte secret provided via ``API_ENCRYPTION_KEY``
- Nonce: Random 96-bit value generated per response
- Associated data (AAD): Static string ``b\"tariff-simulate:v1\"`` for replay protection
- Output envelope:
    {
        "version": 1,
        "algorithm": "AES-256-GCM",
        "ciphertext": "<base64-url token>",
        "nonce": "<base64-url nonce>",
        "issued_at": "<ISO8601 UTC timestamp>",
        "key_id": "<optional identifier>"
    }

Client-side example (JavaScript)
--------------------------------
```js
async function decryptEnvelope(envelope, base64Key) {
  const decoder = (b64) => Uint8Array.from(atob(b64.replace(/-/g, '+').replace(/_/g, '/')), c => c.charCodeAt(0));
  const keyBytes = decoder(base64Key);
  const cryptoKey = await crypto.subtle.importKey(
    'raw',
    keyBytes,
    { name: 'AES-GCM', length: 256 },
    false,
    ['decrypt']
  );
  const nonce = decoder(envelope.nonce);
  const ciphertext = decoder(envelope.ciphertext);
  const aad = new TextEncoder().encode('tariff-simulate:v1');
  const plaintextBuffer = await crypto.subtle.decrypt(
    { name: 'AES-GCM', iv: nonce, additionalData: aad },
    cryptoKey,
    ciphertext
  );
  const plaintext = new TextDecoder().decode(plaintextBuffer);
  return JSON.parse(plaintext);
}
```

The ``base64Key`` must match the server-side ``API_ENCRYPTION_KEY`` value.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, Mapping, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

AAD = b"tariff-simulate:v1"
_KEY_BITS: Optional[int] = None


class EncryptionConfigError(RuntimeError):
    """Raised when encryption settings are invalid or missing."""


class EncryptionExecutionError(RuntimeError):
    """Raised when encryption or serialization fails."""


@dataclass
class EncryptionEnvelope:
    algorithm: str
    ciphertext: str
    nonce: str
    issued_at: datetime
    version: int = 1
    key_id: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        # Convert datetime to ISO8601 string for JSON transport
        payload["issued_at"] = self.issued_at.isoformat()
        return payload


def _decode_base64_key(value: str) -> bytes:
    try:
        key_bytes = base64.urlsafe_b64decode(value)
    except (ValueError, base64.binascii.Error) as exc:  # pragma: no cover - defensive
        raise EncryptionConfigError("API_ENCRYPTION_KEY must be URL-safe base64 encoded.") from exc
    if len(key_bytes) not in (16, 24, 32):  # 128, 192, 256 bit AES keys
        raise EncryptionConfigError("API_ENCRYPTION_KEY must decode to 16, 24, or 32 bytes.")
    return key_bytes


@lru_cache(maxsize=1)
def _get_cipher() -> AESGCM:
    key_b64 = os.getenv("API_ENCRYPTION_KEY")
    if not key_b64:
        raise EncryptionConfigError("API_ENCRYPTION_KEY environment variable is required for encryption.")
    key = _decode_base64_key(key_b64)
    global _KEY_BITS
    _KEY_BITS = len(key) * 8
    return AESGCM(key)


def _get_key_bits() -> int:
    if _KEY_BITS is None:
        _get_cipher()
    assert _KEY_BITS is not None  # for mypy/static
    return _KEY_BITS


def encrypt_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Encrypt the provided payload and return an envelope dictionary."""

    try:
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EncryptionExecutionError("Failed to serialize payload for encryption.") from exc

    cipher = _get_cipher()
    nonce = os.urandom(12)  # 96-bit nonce recommended for GCM
    ciphertext = cipher.encrypt(nonce, serialized, AAD)

    envelope = EncryptionEnvelope(
        algorithm=f"AES-{_get_key_bits()}-GCM",
        ciphertext=base64.urlsafe_b64encode(ciphertext).decode("ascii"),
        nonce=base64.urlsafe_b64encode(nonce).decode("ascii"),
        issued_at=datetime.now(timezone.utc),
        key_id=os.getenv("API_ENCRYPTION_KEY_ID"),
    )
    return envelope.as_dict()


def decrypt_payload(envelope: Mapping[str, Any]) -> Dict[str, Any]:
    """Decrypt an envelope back to a dictionary (primarily for diagnostics/tests)."""

    cipher = _get_cipher()
    try:
        nonce_b64 = envelope["nonce"]
        ciphertext_b64 = envelope["ciphertext"]
    except KeyError as exc:  # pragma: no cover - defensive
        raise EncryptionExecutionError(f"Envelope missing field: {exc.args[0]}") from exc

    nonce = base64.urlsafe_b64decode(nonce_b64)
    ciphertext = base64.urlsafe_b64decode(ciphertext_b64)

    plaintext = cipher.decrypt(nonce, ciphertext, AAD)
    try:
        return json.loads(plaintext.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:  # pragma: no cover - defensive
        raise EncryptionExecutionError("Failed to deserialize decrypted payload.") from exc
