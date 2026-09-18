"""Protected secret storage for source connections (API keys / tokens).

Secrets are encrypted at rest with Fernet. The encryption key comes from
SECRET_ENCRYPTION_KEY; if unset, a stable dev key is derived from local
settings so secrets survive restarts without manual setup. The derived
fallback is documented as a dev-profile convenience — set a real key in
production.
"""

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken

from app.config import get_settings


def _fernet() -> Fernet:
    settings = get_settings()
    explicit = getattr(settings, "secret_encryption_key", None)
    if explicit:
        key_material = explicit.get_secret_value() if hasattr(explicit, "get_secret_value") else explicit
        key = key_material.encode()
        if len(key) != 44 or not key.endswith(b"="):
            # Accept any passphrase by deriving a proper Fernet key from it.
            key = base64.urlsafe_b64encode(hashlib.sha256(key).digest())
    else:
        seed = f"{settings.database_url}|{settings.app_name}|automl-secrets".encode()
        key = base64.urlsafe_b64encode(hashlib.sha256(seed).digest())
    return Fernet(key)


def encrypt_secret(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt_secret(token: str) -> str:
    try:
        return _fernet().decrypt(token.encode()).decode()
    except InvalidToken as exc:
        raise ValueError("Stored secret cannot be decrypted (encryption key changed?).") from exc


def mask_secret(plaintext: str) -> str:
    if len(plaintext) <= 8:
        return "••••••••"
    return f"{plaintext[:4]}…{plaintext[-4:]}"
