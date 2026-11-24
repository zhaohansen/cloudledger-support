# tests/helpers.py
# 测试辅助：按用户名取用户/租户上下文与 Token，避免每个用例重复登录。

from __future__ import annotations

from sqlalchemy import select

from app.db import AsyncSessionLocal
from app.deps import UserContext
from app.models import Tenant, User
from app.security import create_access_token

_USER_IDS: dict[str, str] = {}


async def get_user(username: str) -> User:
    async with AsyncSessionLocal() as db:
        user = await db.scalar(select(User).where(User.username == username))
        if user is None:
            raise AssertionError(f"测试用户不存在：{username}")
        _USER_IDS[username] = user.id
        return user


async def ctx_for(username: str) -> UserContext:
    async with AsyncSessionLocal() as db:
        user = await db.scalar(select(User).where(User.username == username))
        if user is None:
            raise AssertionError(f"测试用户不存在：{username}")
        tenant = await db.get(Tenant, user.tenant_id)
    return UserContext(
        user_id=user.id,
        username=user.username,
        display_name=user.display_name,
        role=user.role,
        tenant_id=user.tenant_id,
        tenant_name=tenant.name if tenant else "",
    )


async def token_for(username: str) -> str:
    if username not in _USER_IDS:
        user = await get_user(username)
        _USER_IDS[username] = user.id
    return create_access_token(_USER_IDS[username])


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def login(client, username: str, password: str):
    return await client.post("/api/auth/login", json={"username": username, "password": password})
