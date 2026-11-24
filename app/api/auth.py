# app/api/auth.py
# 登录与当前用户信息。

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.deps import UserContext, get_current_user
from app.db import get_db
from app.logging_config import get_logger
from app.models import User
from app.security import create_access_token, verify_password

router = APIRouter()
logger = get_logger(__name__)


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str
    username: str
    display_name: str
    role: str
    tenant_id: str
    tenant_name: str


@router.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    user = await db.scalar(select(User).where(User.username == req.username.strip()))
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")

    # bcrypt 是 CPU 密集型，放到线程池，避免阻塞事件循环
    ok = await run_in_threadpool(verify_password, req.password, user.password_hash)
    if not ok:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="账号已被禁用，请联系管理员")

    from app.config import get_settings
    from app.models import Tenant

    settings = get_settings()
    tenant = await db.get(Tenant, user.tenant_id)
    token = create_access_token(user.id)
    logger.info("auth.login_success", user_id=user.id, role=user.role)
    return LoginResponse(
        access_token=token,
        expires_in=settings.jwt_expire_minutes * 60,
        user_id=user.id,
        username=user.username,
        display_name=user.display_name,
        role=user.role,
        tenant_id=user.tenant_id,
        tenant_name=tenant.name if tenant else "",
    )


@router.get("/me")
async def me(user: UserContext = Depends(get_current_user)):
    return {
        "user_id": user.user_id,
        "username": user.username,
        "display_name": user.display_name,
        "role": user.role,
        "tenant_id": user.tenant_id,
        "tenant_name": user.tenant_name,
    }
