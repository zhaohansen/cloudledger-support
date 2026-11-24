# app/security.py
# 密码哈希与 JWT 签发/校验。
# 设计要点：JWT 只放 user_id（sub）和过期时间，角色、租户每次请求从数据库读，
# 这样管理员改角色或禁用账号后立即生效，避免"旧 Token 带着旧权限"。

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

# 兼容性补丁：passlib 1.7.4 会读取 bcrypt.__about__.__version__，bcrypt>=4 已移除该属性
import bcrypt as _bcrypt_mod
import types as _types

if not hasattr(_bcrypt_mod, "__about__"):  # pragma: no cover - 环境兼容
    _bcrypt_mod.__about__ = _types.SimpleNamespace(
        __version__=getattr(_bcrypt_mod, "__version__", "4.x")
    )

from app.config import get_settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class TokenError(Exception):
    """Token 无效或过期。"""


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return pwd_context.verify(password, password_hash)
    except Exception:
        return False


def create_access_token(user_id: str, expires_minutes: int | None = None) -> str:
    settings = get_settings()
    minutes = expires_minutes if expires_minutes is not None else settings.jwt_expire_minutes
    payload: dict[str, Any] = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=minutes),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.resolve_jwt_secret(), algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    settings = get_settings()
    try:
        return jwt.decode(
            token,
            settings.resolve_jwt_secret(),
            algorithms=[settings.jwt_algorithm],
        )
    except JWTError as exc:
        raise TokenError(str(exc)) from exc
