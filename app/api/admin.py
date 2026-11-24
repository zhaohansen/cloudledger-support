# app/api/admin.py
# 管理员接口：租户、用户、客服授权范围。
# 所有路由都通过 require_roles("admin") 做后端校验。

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.db import get_db
from app.deps import UserContext, require_roles
from app.models import AgentTenantGrant, Tenant, User
from app.security import hash_password

router = APIRouter(prefix="/api/admin", tags=["admin"])

VALID_ROLES = {"customer", "agent", "admin"}


class UserCreateRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=64)
    password: str = Field(..., min_length=6, max_length=128)
    display_name: str = Field("", max_length=64)
    email: str = Field("", max_length=128)
    role: str = Field(..., max_length=16)
    tenant_id: str = Field(..., max_length=32)


class UserUpdateRequest(BaseModel):
    display_name: str | None = Field(None, max_length=64)
    email: str | None = Field(None, max_length=128)
    role: str | None = Field(None, max_length=16)
    is_active: bool | None = None
    tenant_id: str | None = Field(None, max_length=32)
    password: str | None = Field(None, min_length=6, max_length=128)


class GrantCreateRequest(BaseModel):
    agent_user_id: str = Field(..., max_length=32)
    tenant_id: str = Field(..., max_length=32)


def _serialize_user(user: User, tenant_name: str = "") -> dict:
    return {
        "id": user.id,
        "username": user.username,
        "display_name": user.display_name,
        "email": user.email,
        "role": user.role,
        "tenant_id": user.tenant_id,
        "tenant_name": tenant_name,
        "is_active": user.is_active,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


@router.get("/tenants")
async def list_tenants(
    user: UserContext = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(select(Tenant).order_by(Tenant.created_at.asc()))).scalars().all()
    return {
        "items": [
            {"id": t.id, "name": t.name, "code": t.code, "is_platform": t.is_platform} for t in rows
        ]
    }


@router.get("/users")
async def list_users(
    user: UserContext = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    rows = (
        await db.execute(select(User, Tenant.name).join(Tenant, User.tenant_id == Tenant.id).order_by(User.created_at))
    ).all()
    return {"items": [_serialize_user(u, name) for u, name in rows]}


@router.post("/users")
async def create_user(
    req: UserCreateRequest,
    user: UserContext = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    if req.role not in VALID_ROLES:
        raise HTTPException(status_code=422, detail="角色不合法")
    if await db.get(Tenant, req.tenant_id) is None:
        raise HTTPException(status_code=422, detail="租户不存在")
    if await db.scalar(select(User).where(User.username == req.username)):
        raise HTTPException(status_code=409, detail="用户名已存在")

    password_hash = await run_in_threadpool(hash_password, req.password)
    new_user = User(
        username=req.username.strip(),
        display_name=req.display_name,
        email=req.email,
        role=req.role,
        tenant_id=req.tenant_id,
        password_hash=password_hash,
    )
    db.add(new_user)
    await db.flush()
    tenant = await db.get(Tenant, req.tenant_id)
    return _serialize_user(new_user, tenant.name if tenant else "")


@router.patch("/users/{user_id}")
async def update_user(
    user_id: str,
    req: UserUpdateRequest,
    current: UserContext = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    target = await db.get(User, user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="用户不存在")

    if req.role is not None:
        if req.role not in VALID_ROLES:
            raise HTTPException(status_code=422, detail="角色不合法")
        target.role = req.role
    if req.tenant_id is not None:
        if await db.get(Tenant, req.tenant_id) is None:
            raise HTTPException(status_code=422, detail="租户不存在")
        target.tenant_id = req.tenant_id
    if req.is_active is not None:
        if target.id == current.user_id and not req.is_active:
            raise HTTPException(status_code=400, detail="不能禁用当前登录的管理员账号")
        target.is_active = req.is_active
    if req.display_name is not None:
        target.display_name = req.display_name
    if req.email is not None:
        target.email = req.email
    if req.password:
        target.password_hash = await run_in_threadpool(hash_password, req.password)

    await db.flush()
    tenant = await db.get(Tenant, target.tenant_id)
    return _serialize_user(target, tenant.name if tenant else "")


@router.get("/agent-grants")
async def list_grants(
    user: UserContext = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    rows = (
        await db.execute(
            select(AgentTenantGrant, User.username, Tenant.name)
            .join(User, AgentTenantGrant.agent_user_id == User.id)
            .join(Tenant, AgentTenantGrant.tenant_id == Tenant.id)
            .order_by(AgentTenantGrant.created_at)
        )
    ).all()
    return {
        "items": [
            {
                "id": grant.id,
                "agent_user_id": grant.agent_user_id,
                "agent_username": username,
                "tenant_id": grant.tenant_id,
                "tenant_name": tenant_name,
            }
            for grant, username, tenant_name in rows
        ]
    }


@router.post("/agent-grants")
async def create_grant(
    req: GrantCreateRequest,
    user: UserContext = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    agent = await db.get(User, req.agent_user_id)
    if agent is None or agent.role != "agent":
        raise HTTPException(status_code=422, detail="只能给客服角色授权")
    if await db.get(Tenant, req.tenant_id) is None:
        raise HTTPException(status_code=422, detail="租户不存在")

    existing = await db.scalar(
        select(AgentTenantGrant).where(
            AgentTenantGrant.agent_user_id == req.agent_user_id,
            AgentTenantGrant.tenant_id == req.tenant_id,
        )
    )
    if existing:
        raise HTTPException(status_code=409, detail="该授权已存在")

    grant = AgentTenantGrant(agent_user_id=req.agent_user_id, tenant_id=req.tenant_id)
    db.add(grant)
    await db.flush()
    return {"id": grant.id, "agent_user_id": grant.agent_user_id, "tenant_id": grant.tenant_id}


@router.delete("/agent-grants/{grant_id}")
async def delete_grant(
    grant_id: str,
    user: UserContext = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    grant = await db.get(AgentTenantGrant, grant_id)
    if grant is None:
        raise HTTPException(status_code=404, detail="授权不存在")
    await db.delete(grant)
    return {"ok": True}
