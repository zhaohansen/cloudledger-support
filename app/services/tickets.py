# app/services/tickets.py
# 工单服务：草稿转工单、查询、状态流转、补充信息、附件。
# 所有函数都接收 UserContext，并在服务端做权限校验；路由和 Agent 工具复用同一套规则。

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import UserContext
from app.logging_config import get_logger
from app.models import (
    Ticket,
    TicketAttachment,
    TicketDraft,
    TicketEvent,
    TicketMessage,
    TicketSequence,
)
from app.services.permissions import (
    can_view_ticket,
    ensure_can_view_ticket,
    ticket_visibility_clause,
)

logger = get_logger(__name__)

TICKET_STATUSES = ("pending", "processing", "waiting_customer", "closed")

# 状态机：值 = 允许流转到的状态
STATUS_TRANSITIONS = {
    "pending": {"processing", "closed"},
    "processing": {"waiting_customer", "pending"},
    "waiting_customer": {"closed", "processing"},
    "closed": {"processing"},
}

# 角色允许的流转。客服只能推进被授权租户的工单；客户只能在"待确认"时确认或重新打开。
ROLE_TRANSITIONS = {
    "agent": {
        "pending": {"processing"},
        "processing": {"waiting_customer"},
        "waiting_customer": {"closed", "processing"},
        "closed": set(),
    },
    "admin": {
        "pending": {"processing", "closed"},
        "processing": {"waiting_customer", "pending"},
        "waiting_customer": {"closed", "processing"},
        "closed": {"processing"},
    },
    "customer": {
        "waiting_customer": {"closed", "processing"},
    },
}


class ToolError(Exception):
    """工具层可预期的业务错误（权限、状态不允许等）。"""

    def __init__(self, message: str, code: str = "tool_error"):
        super().__init__(message)
        self.code = code


def serialize_draft(draft: TicketDraft) -> dict:
    return {
        "id": draft.id,
        "title": draft.title,
        "description": draft.description,
        "product_version": draft.product_version,
        "error_message": draft.error_message,
        "tried_steps": draft.tried_steps,
        "category": draft.category,
        "priority": draft.priority,
        "status": draft.status,
        "version": draft.version,
        "tenant_id": draft.tenant_id,
        "created_at": draft.created_at.isoformat() if draft.created_at else None,
    }


def serialize_ticket(ticket: Ticket) -> dict:
    return {
        "id": ticket.id,
        "ticket_no": ticket.ticket_no,
        "title": ticket.title,
        "description": ticket.description,
        "product_version": ticket.product_version,
        "error_message": ticket.error_message,
        "tried_steps": ticket.tried_steps,
        "category": ticket.category,
        "priority": ticket.priority,
        "status": ticket.status,
        "tenant_id": ticket.tenant_id,
        "created_by": ticket.created_by,
        "assigned_to": ticket.assigned_to,
        "created_at": ticket.created_at.isoformat() if ticket.created_at else None,
        "updated_at": ticket.updated_at.isoformat() if ticket.updated_at else None,
        "closed_at": ticket.closed_at.isoformat() if ticket.closed_at else None,
    }


def serialize_message(message: TicketMessage) -> dict:
    return {
        "id": message.id,
        "author_id": message.author_id,
        "author_role": message.author_role,
        "content": message.content,
        "message_type": message.message_type,
        "created_at": message.created_at.isoformat() if message.created_at else None,
    }


async def _next_ticket_no(db: AsyncSession) -> str:
    """用数据库原子自增生成工单号。

    之前是“查当天最大号 + 1”，两个并发请求会拿到同一个号，只能靠唯一约束冲突后重试。
    现在用 ticket_sequences 的 UPSERT ... RETURNING，取号本身是原子的。
    """
    prefix = f"YZ{datetime.now(timezone.utc).strftime('%Y%m%d')}"
    dialect = db.get_bind().dialect.name
    if dialect == "sqlite":
        stmt = (
            sqlite_insert(TicketSequence)
            .values(key=prefix, value=1)
            .on_conflict_do_update(
                index_elements=[TicketSequence.key],
                set_={"value": TicketSequence.value + 1},
            )
            .returning(TicketSequence.value)
        )
    elif dialect == "postgresql":
        stmt = (
            pg_insert(TicketSequence)
            .values(key=prefix, value=1)
            .on_conflict_do_update(
                index_elements=[TicketSequence.key],
                set_={"value": TicketSequence.value + 1},
            )
            .returning(TicketSequence.value)
        )
    else:  # 兜底：其他数据库退回“查最大号”，并依赖唯一约束重试
        last = await db.scalar(
            select(Ticket.ticket_no)
            .where(Ticket.ticket_no.like(f"{prefix}%"))
            .order_by(Ticket.ticket_no.desc())
            .limit(1)
        )
        seq = int(last[-4:]) + 1 if last else 1
        return f"{prefix}{seq:04d}"

    value = await db.scalar(stmt)
    return f"{prefix}{int(value):04d}"


def build_idempotency_key(user: UserContext, session_id: str | None, description: str) -> str:
    """幂等键关联用户、租户、会话和问题内容摘要，重复确认不会重复建单。"""
    digest = hashlib.sha1((description or "").strip().encode("utf-8")).hexdigest()[:16]
    return f"draft:{user.user_id}:{user.tenant_id}:{session_id or '-'}:{digest}"


async def get_or_create_draft(
    db: AsyncSession,
    user: UserContext,
    *,
    session_id: str | None,
    title: str,
    description: str,
    product_version: str,
    error_message: str,
    tried_steps: str,
    category: str,
    priority: str,
) -> TicketDraft:
    """同一会话、同一问题内容只保留一份“进行中”的草稿。

    幂等键仍然关联用户、租户、会话和内容摘要；但取消/确认后再次发起时，
    不能因为旧 key 已存在而报唯一键冲突，因此新草稿会追加 :rN 后缀。
    """
    base_key = build_idempotency_key(user, session_id, description)
    existing_drafts = list(
        (
            await db.scalars(
                select(TicketDraft)
                .where(TicketDraft.idempotency_key.like(f"{base_key}%"))
                .order_by(TicketDraft.created_at.desc())
            )
        ).all()
    )
    for item in existing_drafts:
        if item.status == "draft":
            return item

    key = base_key if not existing_drafts else f"{base_key}:r{len(existing_drafts) + 1}"
    draft = TicketDraft(
        user_id=user.user_id,
        tenant_id=user.tenant_id,
        session_id=session_id,
        idempotency_key=key,
        title=title[:200],
        description=description,
        product_version=product_version,
        error_message=error_message,
        tried_steps=tried_steps,
        category=category,
        priority=priority,
    )
    try:
        async with db.begin_nested():
            db.add(draft)
            await db.flush()
    except IntegrityError:
        # 并发下另一个请求可能刚创建了同一份草稿，返回它
        existing = await db.scalar(
            select(TicketDraft).where(
                TicketDraft.idempotency_key == key,
                TicketDraft.status == "draft",
            )
        )
        if existing:
            return existing
        raise
    logger.info("ticket.draft_created", draft_id=draft.id, user_id=user.user_id)
    return draft


EDITABLE_DRAFT_FIELDS = {
    "title",
    "description",
    "product_version",
    "error_message",
    "tried_steps",
    "category",
    "priority",
}


def _require_expected_version(expected_version: int | None) -> int:
    """确认必须绑定用户实际看到的草稿版本。

    缺失版本时直接拒绝，不能退化成“读最新版本替用户确认”——那等于取消了
    “用户确认的是他看到的版本”这层保护。
    """
    if expected_version is None:
        raise ToolError(
            "确认工单必须提供用户看到的草稿版本 expected_version",
            "missing_expected_version",
        )
    if isinstance(expected_version, bool):
        raise ToolError("草稿版本号无效", "invalid_expected_version")
    try:
        value = int(expected_version)
    except (TypeError, ValueError):
        raise ToolError("草稿版本号无效", "invalid_expected_version") from None
    if value < 1:
        raise ToolError("草稿版本号无效", "invalid_expected_version")
    return value


async def update_draft(
    db: AsyncSession,
    user: UserContext,
    draft_id: str,
    *,
    expected_version: int | None = None,
    **fields,
) -> TicketDraft:
    """修改草稿：用条件 UPDATE 保证同一版本只有一个请求能改成功。

    先读后写的“Python 判断”在并发下会让两个请求都通过版本检查（丢失更新）；
    这里把版本条件放进 UPDATE 的 WHERE，数据库负责最终裁决。
    """
    draft = await db.get(TicketDraft, draft_id)
    if draft is None or draft.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="工单草稿不存在")
    if draft.status != "draft":
        raise HTTPException(status_code=409, detail="该草稿已确认或已取消，不能再修改")

    current_version = int(draft.version or 1)
    if expected_version is not None and current_version != int(expected_version):
        raise HTTPException(
            status_code=409,
            detail="草稿已被修改，请刷新后再编辑，避免覆盖别人的改动",
        )

    values = {
        key: str(value)[:2000]
        for key, value in fields.items()
        if key in EDITABLE_DRAFT_FIELDS and value is not None
    }
    if not values:
        return draft

    result = await db.execute(
        update(TicketDraft)
        .where(
            TicketDraft.id == draft.id,
            TicketDraft.user_id == user.user_id,
            TicketDraft.status == "draft",
            TicketDraft.version == current_version,
        )
        .values(**values, version=TicketDraft.version + 1)
        .execution_options(synchronize_session=False)
    )
    if result.rowcount != 1:
        await db.refresh(draft)
        if draft.status != "draft":
            raise HTTPException(status_code=409, detail="该草稿已确认或已取消，不能再修改")
        raise HTTPException(
            status_code=409,
            detail="草稿已被修改，请刷新后再编辑，避免覆盖别人的改动",
        )
    # 版本 +1 后刷新，返回值与数据库一致；旧版本上的确认会因此失效
    await db.refresh(draft)
    return draft


async def cancel_draft(
    db: AsyncSession,
    user: UserContext,
    draft_id: str,
    *,
    expected_version: int | None = None,
) -> TicketDraft:
    """取消草稿：与确认互斥，只有一个操作能改变草稿状态。"""
    draft = await db.get(TicketDraft, draft_id)
    if draft is None or draft.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="工单草稿不存在")
    if draft.status == "confirmed":
        raise HTTPException(status_code=409, detail="该草稿已创建工单，不能取消")
    if draft.status == "cancelled":
        return draft

    current_version = int(draft.version or 1)
    if expected_version is not None and current_version != int(expected_version):
        raise HTTPException(
            status_code=409,
            detail="草稿已被修改，请查看最新内容后再决定是否取消",
        )

    result = await db.execute(
        update(TicketDraft)
        .where(
            TicketDraft.id == draft.id,
            TicketDraft.user_id == user.user_id,
            TicketDraft.status == "draft",
            TicketDraft.version == current_version,
        )
        .values(status="cancelled")
        .execution_options(synchronize_session=False)
    )
    if result.rowcount != 1:
        await db.refresh(draft)
        if draft.status == "confirmed":
            raise HTTPException(status_code=409, detail="该草稿已创建工单，不能取消")
        if draft.status == "cancelled":
            return draft
        raise HTTPException(
            status_code=409,
            detail="草稿已被修改，请查看最新内容后再决定是否取消",
        )
    await db.refresh(draft)
    return draft


async def create_ticket_from_draft(
    db: AsyncSession,
    user: UserContext,
    draft_id: str,
    expected_version: int | None = None,
    *,
    request_id: str = "-",
) -> tuple[Ticket, bool]:
    """把草稿转成正式工单。返回 (ticket, already_created)。

    expected_version 是用户实际看到的草稿版本，必须提供；服务层再次校验，
    不能只依赖请求模型（Agent 工具、脚本都可能直接调用服务层）。

    并发正确性：
    1. 用条件 UPDATE（status='draft' AND version=expected）原子认领草稿，
       并发确认/编辑/取消只有一个能成功；不依赖进程内锁。
    2. tickets.draft_id 唯一约束是最终防线；
    3. ticket_sequences 原子取号，避免并发请求拿到同一个工单号。

    幂等：同一草稿、同一版本重复确认返回已有工单，already_created=True；
    若版本已变化（用户看到的不是最终内容），即使草稿已确认也返回冲突。
    """
    expected = _require_expected_version(expected_version)

    # 第一条语句就是条件写：只有“仍是 draft 且版本正是用户看到的版本”才认领成功。
    # 并发确认/编辑/取消由数据库裁决，不存在“先读、Python 判断、再写”的竞争窗口；
    # SQLite 下也尽量缩短读锁升级写锁的窗口。
    claim = await db.execute(
        update(TicketDraft)
        .where(
            TicketDraft.id == draft_id,
            TicketDraft.user_id == user.user_id,
            TicketDraft.tenant_id == user.tenant_id,
            TicketDraft.status == "draft",
            TicketDraft.version == expected,
        )
        .values(status="confirmed")
        .execution_options(synchronize_session=False)
    )
    claimed = claim.rowcount == 1

    draft = await db.get(TicketDraft, draft_id)
    if draft is None:
        raise ToolError("工单草稿不存在", "draft_not_found")
    if draft.user_id != user.user_id or draft.tenant_id != user.tenant_id:
        raise ToolError("无权确认该草稿", "forbidden")

    if not claimed:
        # 认领失败：按最新状态说明原因，绝不改成“读最新版本替用户确认”
        if draft.status == "cancelled":
            raise ToolError("该草稿已取消，请重新生成", "draft_cancelled")
        if int(draft.version or 1) != expected:
            raise ToolError(
                "草稿内容已更新，旧确认不能提交新内容，请重新核对后确认",
                "draft_version_conflict",
            )
        if draft.status == "confirmed":
            existing = await db.scalar(select(Ticket).where(Ticket.draft_id == draft.id))
            if existing:
                return existing, True
            raise ToolError("草稿状态异常：已确认但找不到对应工单", "inconsistent_state")
        raise ToolError("草稿状态已变化，请刷新后重试", "draft_state_conflict")

    # 认领成功后，本事务内读到的是与 expected 对应的内容；后续插入失败要撤销认领，
    # 否则调用方（Agent 节点）可能把“已确认但无工单”的草稿提交进数据库。
    try:
        if not draft.title.strip() or not draft.description.strip():
            raise ToolError("工单标题和问题描述不能为空", "invalid_params")
        for _ in range(5):
            # 取号放在 savepoint 外：插入失败回滚 savepoint 时序列不回退，下一次换新号
            ticket_no = await _next_ticket_no(db)
            ticket = Ticket(
                ticket_no=ticket_no,
                draft_id=draft.id,
                tenant_id=draft.tenant_id,
                created_by=user.user_id,
                title=draft.title,
                description=draft.description,
                product_version=draft.product_version or "未提供",
                error_message=draft.error_message,
                tried_steps=draft.tried_steps,
                category=draft.category,
                priority=draft.priority,
                status="pending",
            )
            try:
                # 先创建 savepoint 再 add/flush，避免 begin_nested 触发外层 pending 对象的预 flush
                async with db.begin_nested():
                    db.add(ticket)
                    await db.flush()
            except IntegrityError:
                # savepoint 回滚后 session 可继续使用；先看是否已被其他请求创建
                existing = await db.scalar(select(Ticket).where(Ticket.draft_id == draft.id))
                if existing:
                    claimed = False  # 已有工单，保留 confirmed 状态
                    return existing, True
                # 否则是工单号冲突，换号重试
                continue

            db.add(
                TicketEvent(
                    ticket_id=ticket.id,
                    actor_id=user.user_id,
                    action="create",
                    from_status="",
                    to_status="pending",
                    note=f"由会话草稿创建，draft_version={expected}，request_id={request_id}",
                )
            )
            logger.info(
                "ticket.created",
                ticket_no=ticket.ticket_no,
                tenant_id=ticket.tenant_id,
                user_id=user.user_id,
                draft_version=expected,
                request_id=request_id,
            )
            claimed = False  # 成功返回，不需要撤销
            return ticket, False

        raise ToolError("工单号生成冲突，创建失败，请稍后重试", "ticket_no_conflict")
    finally:
        if claimed:
            # 只撤销本事务自己写下的认领，避免把并发请求的结果改回去
            try:
                await db.execute(
                    update(TicketDraft)
                    .where(
                        TicketDraft.id == draft.id,
                        TicketDraft.status == "confirmed",
                        TicketDraft.version == expected,
                    )
                    .values(status="draft")
                    .execution_options(synchronize_session=False)
                )
                await db.flush()
            except Exception as exc:  # 撤销失败只记录，原始异常继续抛出
                logger.error("ticket.claim_revert_failed", error=type(exc).__name__)


async def list_tickets(
    db: AsyncSession,
    user: UserContext,
    *,
    status_filter: str | None = None,
    category: str | None = None,
    limit: int = 50,
) -> list[Ticket]:
    stmt = select(Ticket).order_by(Ticket.created_at.desc()).limit(limit)
    stmt = stmt.where(await ticket_visibility_clause(db, user))
    if status_filter:
        stmt = stmt.where(Ticket.status == status_filter)
    if category:
        stmt = stmt.where(Ticket.category == category)
    return list((await db.execute(stmt)).scalars().all())


async def get_ticket(db: AsyncSession, user: UserContext, ticket_id: str) -> Ticket:
    ticket = await db.get(Ticket, ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="工单不存在")
    await ensure_can_view_ticket(db, user, ticket)
    return ticket


async def get_ticket_by_no(db: AsyncSession, user: UserContext, ticket_no: str) -> Ticket:
    ticket = await db.scalar(select(Ticket).where(Ticket.ticket_no == ticket_no.strip().upper()))
    if ticket is None:
        raise HTTPException(status_code=404, detail="工单不存在")
    if not await can_view_ticket(db, user, ticket):
        raise HTTPException(status_code=404, detail="工单不存在")
    return ticket


async def get_ticket_detail(db: AsyncSession, user: UserContext, ticket: Ticket) -> dict:
    messages = (
        await db.execute(
            select(TicketMessage)
            .where(TicketMessage.ticket_id == ticket.id)
            .order_by(TicketMessage.created_at.asc())
        )
    ).scalars().all()
    events = (
        await db.execute(
            select(TicketEvent).where(TicketEvent.ticket_id == ticket.id).order_by(TicketEvent.created_at.asc())
        )
    ).scalars().all()
    attachments = (
        await db.execute(
            select(TicketAttachment)
            .where(TicketAttachment.ticket_id == ticket.id)
            .order_by(TicketAttachment.created_at.asc())
        )
    ).scalars().all()

    data = serialize_ticket(ticket)
    data["messages"] = [serialize_message(m) for m in messages]
    data["events"] = [
        {
            "id": e.id,
            "actor_id": e.actor_id,
            "action": e.action,
            "from_status": e.from_status,
            "to_status": e.to_status,
            "note": e.note,
            "created_at": e.created_at.isoformat() if e.created_at else None,
        }
        for e in events
    ]
    data["attachments"] = [
        {
            "id": a.id,
            "filename": a.filename,
            "size": a.size,
            "content_type": a.content_type,
            "created_at": a.created_at.isoformat() if a.created_at else None,
        }
        for a in attachments
    ]
    return data


async def _find_saved_message(
    db: AsyncSession, ticket_id: str, request_key: str | None
) -> TicketMessage | None:
    """重试时查询同一工单下已经保存的消息。"""
    if request_key is None:
        return None
    message = await db.scalar(
        select(TicketMessage).where(TicketMessage.request_key == request_key)
    )
    if message is not None and message.ticket_id != ticket_id:
        raise HTTPException(status_code=409, detail="幂等键已被其他工单使用")
    return message


async def add_message(
    db: AsyncSession,
    user: UserContext,
    ticket: Ticket,
    content: str,
    *,
    message_type: str = "reply",
    request_key: str | None = None,
) -> TicketMessage:
    if not content.strip():
        raise HTTPException(status_code=422, detail="回复内容不能为空")
    if ticket.status == "closed" and user.role != "admin":
        raise HTTPException(status_code=409, detail="工单已关闭，如需继续处理请先由管理员重新打开")

    ticket_id = ticket.id
    # 幂等键：客户端重试同一次回复时返回已写入的消息，不产生重复副作用
    existing = await _find_saved_message(db, ticket_id, request_key)
    if existing is not None:
        return existing

    try:
        async with db.begin_nested():
            message = TicketMessage(
                ticket_id=ticket.id,
                author_id=user.user_id,
                author_role=user.role,
                content=content.strip()[:5000],
                message_type=message_type,
                request_key=request_key,
            )
            db.add(message)
            # 客户补充信息后，工单回到处理中，提醒客服重新查看
            if user.is_customer and ticket.status == "waiting_customer":
                ticket.status = "processing"
                db.add(
                    TicketEvent(
                        ticket_id=ticket.id,
                        actor_id=user.user_id,
                        action="customer_reply",
                        from_status="waiting_customer",
                        to_status="processing",
                        note="客户补充信息",
                    )
                )
            await db.flush()
    except IntegrityError:
        # 并发重试撞唯一键：回滚 savepoint 后返回已存在的那条消息
        existing = await _find_saved_message(db, ticket_id, request_key)
        if existing is not None:
            await db.refresh(ticket)
            return existing
        raise
    return message


async def update_status(
    db: AsyncSession,
    user: UserContext,
    ticket: Ticket,
    new_status: str,
    note: str = "",
) -> Ticket:
    if new_status not in TICKET_STATUSES:
        raise HTTPException(status_code=422, detail=f"非法状态：{new_status}")
    if new_status == ticket.status:
        return ticket

    role_rules = ROLE_TRANSITIONS.get(user.role, {})
    allowed = role_rules.get(ticket.status, set())
    if new_status not in allowed or new_status not in STATUS_TRANSITIONS.get(ticket.status, set()):
        raise HTTPException(
            status_code=403 if user.is_customer else 409,
            detail=f"当前角色不允许把工单从 {ticket.status} 变更为 {new_status}",
        )

    old_status = ticket.status
    changes = {"status": new_status}
    if new_status == "closed":
        changes["closed_at"] = datetime.now(timezone.utc)
    if new_status == "processing":
        changes["closed_at"] = None
        if user.is_agent and not ticket.assigned_to:
            changes["assigned_to"] = user.user_id

    changed = await db.scalar(
        update(Ticket)
        .where(
            Ticket.id == ticket.id,
            Ticket.status == old_status,
            Ticket.updated_at == ticket.updated_at,
        )
        .values(**changes)
        .returning(Ticket.id)
        .execution_options(synchronize_session=False)
    )
    if changed is None:
        raise HTTPException(status_code=409, detail="工单状态已更新，请刷新后重试")
    await db.refresh(ticket)
    db.add(
        TicketEvent(
            ticket_id=ticket.id,
            actor_id=user.user_id,
            action="status_change",
            from_status=old_status,
            to_status=new_status,
            note=note[:500],
        )
    )
    await db.flush()
    logger.info(
        "ticket.status_changed",
        ticket_no=ticket.ticket_no,
        from_status=old_status,
        to_status=new_status,
        actor_role=user.role,
    )
    return ticket


async def supplement_ticket(
    db: AsyncSession,
    user: UserContext,
    ticket: Ticket,
    *,
    description: str | None = None,
    error_message: str | None = None,
    tried_steps: str | None = None,
    request_key: str | None = None,
) -> Ticket:
    """客户补充工单信息（工具与 API 复用）。

    request_key 用于防止客户端重试或 Agent 重跑导致重复追加；
    重复请求直接返回原工单，不重复改写描述/报错/步骤。
    """
    if ticket.status == "closed":
        raise HTTPException(status_code=409, detail="工单已关闭，不能补充信息")

    ticket_id = ticket.id
    if await _find_saved_message(db, ticket_id, request_key):
        return ticket

    try:
        async with db.begin_nested():
            if description:
                ticket.description = (ticket.description + "\n\n【补充】" + description.strip())[:8000]
            if error_message:
                ticket.error_message = (ticket.error_message + "\n" + error_message.strip())[:4000]
            if tried_steps:
                ticket.tried_steps = (ticket.tried_steps + "\n" + tried_steps.strip())[:4000]

            detail_text = description or error_message or tried_steps or "补充信息"
            db.add(
                TicketMessage(
                    ticket_id=ticket.id,
                    author_id=user.user_id,
                    author_role=user.role,
                    content=f"补充信息：{detail_text}"[:5000],
                    message_type="supplement",
                    request_key=request_key,
                )
            )
            db.add(
                TicketEvent(
                    ticket_id=ticket.id,
                    actor_id=user.user_id,
                    action="supplement",
                    from_status=ticket.status,
                    to_status=ticket.status,
                    note="客户补充工单信息" if user.is_customer else "客服补充工单信息",
                )
            )
            await db.flush()
    except IntegrityError:
        existing = await _find_saved_message(db, ticket_id, request_key)
        if existing is not None:
            await db.refresh(ticket)
            return ticket
        raise
    return ticket


async def add_attachment(
    db: AsyncSession,
    user: UserContext,
    ticket: Ticket,
    *,
    filename: str,
    stored_path: str,
    content_type: str,
    size: int,
) -> TicketAttachment:
    attachment = TicketAttachment(
        ticket_id=ticket.id,
        filename=filename,
        stored_path=stored_path,
        content_type=content_type,
        size=size,
        uploaded_by=user.user_id,
    )
    db.add(attachment)
    db.add(
        TicketEvent(
            ticket_id=ticket.id,
            actor_id=user.user_id,
            action="attachment_added",
            from_status=ticket.status,
            to_status=ticket.status,
            note=filename,
        )
    )
    await db.flush()
    return attachment


async def ticket_counts(db: AsyncSession, user: UserContext) -> dict:
    """工作台统计（仅统计当前用户可见范围）。"""
    visibility = await ticket_visibility_clause(db, user)
    rows = (
        await db.execute(select(Ticket.status, func.count(Ticket.id)).where(visibility).group_by(Ticket.status))
    ).all()
    counts = {status: 0 for status in TICKET_STATUSES}
    for status_value, count in rows:
        counts[status_value] = count
    counts["total"] = sum(counts.values())
    return counts
