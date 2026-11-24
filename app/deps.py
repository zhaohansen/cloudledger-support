# app/deps.py
# FastAPI 依赖：当前用户、角色校验、数据库会话。
# 关键点：身份只来自 Authorization Token；请求体/查询参数里的 user_id、tenant_id 一律不信任。

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models import Tenant, User
from app.security import TokenError, decode_access_token

bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class UserContext:
    """一次请求内的可信身份快照。"""

    user_id: str
    username: str
    display_name: str
    role: str
    tenant_id: str
    tenant_name: str = ""

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def is_agent(self) -> bool:
        return self.role == "agent"

    @property
    def is_customer(self) -> bool:
        return self.role == "customer"


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> UserContext:
    """校验 Token 并加载数据库中的最新用户状态（包含禁用检查）。"""
    invalid = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效或已过期的认证凭证",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None or not credentials.credentials:
        raise invalid

    try:
        payload = decode_access_token(credentials.credentials)
    except TokenError:
        raise invalid

    user_id = payload.get("sub")
    if not user_id:
        raise invalid

    result = await db.execute(
        select(User, Tenant.name)
        .outerjoin(Tenant, Tenant.id == User.tenant_id)
        .where(User.id == user_id)
    )
    row = result.one_or_none()
    if row is None:
        raise invalid
    user, tenant_name = row
    # 禁用账号即使持有未过期 Token 也必须拒绝
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="账号已被禁用")

    return UserContext(
        user_id=user.id,
        username=user.username,
        display_name=user.display_name,
        role=user.role,
        tenant_id=user.tenant_id,
        tenant_name=tenant_name or "",
    )


def require_roles(*roles: str):
    """角色校验依赖工厂。管理员是否额外放行由调用方决定，避免隐式越权。"""

    async def _checker(user: UserContext = Depends(get_current_user)) -> UserContext:
        if user.role not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权执行该操作")
        return user

    return _checker
