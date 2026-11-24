# app/services/tools.py
# Agent 可调用的工具。每个工具都强制接收 UserContext 并在服务端校验权限，
# 不依赖提示词约束；返回结构化结果，失败时 ok=False，Agent 不能据此宣称成功。

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import UserContext
from app.logging_config import get_logger
from app.services import tickets as ticket_service

logger = get_logger(__name__)


@dataclass
class ToolResult:
    ok: bool
    tool: str
    data: dict = field(default_factory=dict)
    error: str = ""
    code: str = ""


def _fail(tool: str, message: str, code: str = "tool_error") -> ToolResult:
    return ToolResult(ok=False, tool=tool, error=message, code=code)


async def tool_create_ticket(
    db: AsyncSession,
    user: UserContext,
    draft_id: str,
    request_id: str = "-",
    expected_version: int | None = None,
) -> ToolResult:
    """创建工单（写操作，幂等）。expected_version 用于拒绝旧版本确认。"""
    try:
        ticket, already = await ticket_service.create_ticket_from_draft(
            db,
            user,
            draft_id,
            expected_version,
            request_id=request_id,
        )
        return ToolResult(
            ok=True,
            tool="create_ticket",
            data={"ticket": ticket_service.serialize_ticket(ticket), "already_created": already},
        )
    except ticket_service.ToolError as exc:
        logger.warning("tool.create_ticket_failed", code=exc.code, detail=str(exc))
        return _fail("create_ticket", str(exc), exc.code)
    except HTTPException as exc:
        return _fail("create_ticket", str(exc.detail), "http_error")
    except Exception as exc:  # 数据库异常等不可预期错误：明确失败
        logger.error("tool.create_ticket_error", error=type(exc).__name__, detail=str(exc)[:200])
        return _fail("create_ticket", "系统暂时无法创建工单，请稍后重试", "internal_error")


async def tool_list_tickets(
    db: AsyncSession,
    user: UserContext,
    *,
    status_filter: str | None = None,
    limit: int = 10,
) -> ToolResult:
    """查询当前用户有权查看的工单列表。"""
    try:
        rows = await ticket_service.list_tickets(db, user, status_filter=status_filter, limit=min(limit, 50))
        return ToolResult(
            ok=True,
            tool="list_tickets",
            data={"items": [ticket_service.serialize_ticket(t) for t in rows], "count": len(rows)},
        )
    except Exception as exc:
        logger.error("tool.list_tickets_error", error=type(exc).__name__)
        return _fail("list_tickets", "查询工单失败，请稍后重试", "internal_error")


async def tool_get_ticket(db: AsyncSession, user: UserContext, ticket_no: str) -> ToolResult:
    """查询工单详情（含回复与状态记录），需通过权限校验。"""
    try:
        ticket = await ticket_service.get_ticket_by_no(db, user, ticket_no)
        detail = await ticket_service.get_ticket_detail(db, user, ticket)
        return ToolResult(ok=True, tool="get_ticket", data=detail)
    except HTTPException as exc:
        return _fail("get_ticket", str(exc.detail), "not_found_or_forbidden")
    except Exception as exc:
        logger.error("tool.get_ticket_error", error=type(exc).__name__)
        return _fail("get_ticket", "查询工单详情失败", "internal_error")


async def tool_supplement_ticket(
    db: AsyncSession,
    user: UserContext,
    ticket_no: str,
    content: str,
) -> ToolResult:
    """补充工单信息（写操作）。"""
    try:
        ticket = await ticket_service.get_ticket_by_no(db, user, ticket_no)
        # Agent 重跑同一条补充指令时可能重复调用工具：
        # 用“用户 + 工单 + 内容摘要”作为幂等键，避免重复追加同一条补充信息。
        digest = hashlib.sha1(content.strip().encode("utf-8")).hexdigest()[:16]
        request_key = f"agent-supplement:{user.user_id}:{ticket_no}:{digest}"
        await ticket_service.supplement_ticket(
            db, user, ticket, description=content, request_key=request_key
        )
        detail = await ticket_service.get_ticket_detail(db, user, ticket)
        return ToolResult(ok=True, tool="supplement_ticket", data=detail)
    except HTTPException as exc:
        return _fail("supplement_ticket", str(exc.detail), "http_error")
    except Exception as exc:
        logger.error("tool.supplement_ticket_error", error=type(exc).__name__)
        return _fail("supplement_ticket", "补充工单信息失败", "internal_error")
