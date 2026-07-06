"""
AgriDoctor AI - Security utilities: password hashing (PBKDF2), JWT, auth deps.
"""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from .config import get_settings
from .db import get_db

HASH_ALGORITHM = "sha256"
HASH_ITERATIONS = 100_000

# auto_error=False so we can support optional (anonymous) auth on the analyze route.
_bearer = HTTPBearer(auto_error=False)


# --------------------------------------------------------------------------- #
# Passwords
# --------------------------------------------------------------------------- #
def get_password_hash(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        HASH_ALGORITHM, password.encode("utf-8"), salt, HASH_ITERATIONS
    )
    return f"{salt.hex()}:{digest.hex()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        salt_hex, stored_hash = hashed_password.split(":")
        computed = hashlib.pbkdf2_hmac(
            HASH_ALGORITHM,
            plain_password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            HASH_ITERATIONS,
        ).hex()
        return secrets.compare_digest(stored_hash, computed)
    except (ValueError, AttributeError):
        return False


# --------------------------------------------------------------------------- #
# JWT
# --------------------------------------------------------------------------- #
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    settings = get_settings()
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    to_encode["exp"] = expire
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)


def _decode_user_id(token: str) -> Optional[str]:
    settings = get_settings()
    try:
        payload = jwt.decode(
            token, settings.secret_key, algorithms=[settings.algorithm]
        )
        return payload.get("sub")
    except JWTError:
        return None


def _load_user(user_id: str) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, email, full_name, role FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return dict(row) if row else None


# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #
async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict:
    """Require a valid token; raise 401 otherwise."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user_id = _decode_user_id(credentials.credentials)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = _load_user(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[dict]:
    """Return the user if a valid token is present, else None (no error)."""
    if credentials is None:
        return None
    user_id = _decode_user_id(credentials.credentials)
    if not user_id:
        return None
    return _load_user(user_id)
