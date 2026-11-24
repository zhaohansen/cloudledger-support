# app/services/permissions.py
# 权限判定的唯一入口。API 路由、工具函数、检索都调用这里，避免规则分散。
# 核心原则：tenant_id / user_id / role 一律来自认证上下文，不接受请求参数指定。

from __future__ import annotations

from fastapi import HTTPException, status
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import UserContext
from app.models import AgentTenantGrant, Tenant, Ticket


async def agent_tenant_ids(db: AsyncSession, agent_user_id: str) -> list[str]:
    tenant_ids = await db.scalars(
        select(AgentTenantGrant.tenant_id).where(AgentTenantGrant.agent_user_id == agent_user_id)
    )
    return list(tenant_ids)


async def all_tenant_ids(db: AsyncSession) -> list[str]:
    return list(await db.scalars(select(Tenant.id)))


async def allowed_kb_tenant_ids(db: AsyncSession, user: UserContext) -> list[str]:
    """知识库中该用户可访问的「私有资料」租户范围。

    - 管理员：所有租户
    - 客服：被明确授权的租户
    - 客户：自己所属租户
    公共资料不在此列表中，由检索条件单独放行。
    """
    if user.is_admin:
        return await all_tenant_ids(db)
    if user.is_agent:
        return await agent_tenant_ids(db, user.user_id)
    return [user.tenant_id]


async def can_manage_knowledge(user: UserContext) -> bool:
    """知识文档的增删改仅管理员可操作；客服/客户只能通过问答使用。"""
    return user.is_admin


def ensure_can_manage_knowledge(user: UserContext) -> None:
    if not user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="仅管理员可管理知识文档")


async def ticket_visibility_clause(db: AsyncSession, user: UserContext):
    """返回工单列表查询的权限过滤条件。

    客户：只看自己提交的
    客服：只看被授权租户的
    管理员：全部
    """
    if user.is_admin:
        return True
    if user.is_agent:
        tenant_ids = await agent_tenant_ids(db, user.user_id)
        return Ticket.tenant_id.in_(tenant_ids)
    return Ticket.created_by == user.user_id


async def can_view_ticket(db: AsyncSession, user: UserContext, ticket: Ticket) -> bool:
    if user.is_admin:
        return True
    if user.is_agent:
        return ticket.tenant_id in await agent_tenant_ids(db, user.user_id)
    return ticket.created_by == user.user_id


async def ensure_can_view_ticket(db: AsyncSession, user: UserContext, ticket: Ticket) -> None:
    """无权限时客户统一返回 404，避免泄露"这张工单存在"。

    客服越权访问返回 403，便于内部排查授权配置。
    """
    if await can_view_ticket(db, user, ticket):
        return
    if user.is_customer:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="工单不存在")
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权访问该租户的工单")


async def can_access_tenant(db: AsyncSession, user: UserContext, tenant_id: str) -> bool:
    if user.is_admin:
        return True
    if user.is_agent:
        return tenant_id in await agent_tenant_ids(db, user.user_id)
    return tenant_id == user.tenant_id


async def ensure_tenant_access(db: AsyncSession, user: UserContext, tenant_id: str) -> None:
    if not await can_access_tenant(db, user, tenant_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权访问该租户")
