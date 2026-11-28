# app/api/tickets.py
# 工单 API：列表、详情、草稿编辑/确认、状态流转、回复、补充、附件。
# 每条路由都从认证上下文取身份，并在服务层校验工单可见性。

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.uploads import read_upload
from app.config import get_settings
from app.db import get_db
from app.deps import UserContext, get_current_user, require_roles
from app.models import TicketAttachment, new_id
from app.services import qa, tickets as ticket_service
from app.services.tools import tool_create_ticket

router = APIRouter(prefix="/api/tickets", tags=["tickets"])
settings = get_settings()

ATTACHMENT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".txt", ".log", ".csv"}


class DraftUpdateRequest(BaseModel):
    title: str | None = Field(None, max_length=200)
    description: str | None = Field(None, max_length=2000)
    product_version: str | None = Field(None, max_length=32)
    error_message: str | None = Field(None, max_length=2000)
    tried_steps: str | None = Field(None, max_length=2000)
    category: str | None = Field(None, max_length=32)
    priority: str | None = Field(None, max_length=16)
    # 前端保存时带上当前版本；版本不一致说明别人改过，拒绝覆盖
    expected_version: int | None = Field(None, ge=1, strict=True)


class ConfirmRequest(BaseModel):
    draft_id: str = Field(..., min_length=1, max_length=64)
    # 必填：用户实际看到的草稿版本。缺失/非法直接 422，不能绕过后端版本校验。
    # 前端、Agent 确认节点和工具层都必须传用户看到的版本，服务端不会读最新版本代填。
    expected_version: int = Field(..., ge=1, strict=True)


class StatusRequest(BaseModel):
    status: str = Field(..., max_length=24)
    note: str = Field("", max_length=500)


class MessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=5000)


class SupplementRequest(BaseModel):
    description: str | None = Field(None, max_length=3000)
    error_message: str | None = Field(None, max_length=2000)
    tried_steps: str | None = Field(None, max_length=2000)


@router.get("")
async def list_tickets(
    status: str | None = None,
    category: str | None = None,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    rows = await ticket_service.list_tickets(db, user, status_filter=status, category=category)
    return {"items": [ticket_service.serialize_ticket(t) for t in rows]}


@router.get("/counts")
async def counts(
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await ticket_service.ticket_counts(db, user)


# ── 草稿相关路由必须放在 /{ticket_id} 之前，避免 "drafts" 被当成工单 ID ──
@router.get("/drafts/{draft_id}")
async def get_draft(
    draft_id: str,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from app.models import TicketDraft

    draft = await db.get(TicketDraft, draft_id)
    if draft is None or draft.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="工单草稿不存在")
    return ticket_service.serialize_draft(draft)


@router.patch("/drafts/{draft_id}")
async def update_draft(
    draft_id: str,
    req: DraftUpdateRequest,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    draft = await ticket_service.update_draft(
        db,
        user,
        draft_id,
        expected_version=req.expected_version,
        **req.model_dump(exclude_none=True, exclude={"expected_version"}),
    )
    return ticket_service.serialize_draft(draft)


@router.post("/confirm")
async def confirm_draft(
    req: ConfirmRequest,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """用户明确确认后创建工单。重复调用返回同一张工单（幂等）。"""
    result = await tool_create_ticket(
        db, user, req.draft_id, expected_version=req.expected_version
    )
    if not result.ok:
        raise HTTPException(status_code=409, detail=result.error)
    # 在返回成功响应前提交；提交失败必须按失败返回，不能“嘴上成功”
    try:
        await db.commit()
    except Exception:
        await db.rollback()
        raise HTTPException(status_code=500, detail="工单创建提交失败，本次没有创建成功")
    return result.data


@router.get("/{ticket_id}")
async def ticket_detail(
    ticket_id: str,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    ticket = await ticket_service.get_ticket(db, user, ticket_id)
    return await ticket_service.get_ticket_detail(db, user, ticket)


@router.patch("/{ticket_id}/status")
async def update_status(
    ticket_id: str,
    req: StatusRequest,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    ticket = await ticket_service.get_ticket(db, user, ticket_id)
    updated = await ticket_service.update_status(db, user, ticket, req.status, req.note)
    return ticket_service.serialize_ticket(updated)


@router.post("/{ticket_id}/reply-draft")
async def draft_reply(
    ticket_id: str,
    user: UserContext = Depends(require_roles("agent", "admin")),
    db: AsyncSession = Depends(get_db),
):
    """用工单标题和版本生成供顾问审阅的回复，不写入对话或流转记录。"""
    ticket = await ticket_service.get_ticket(db, user, ticket_id)
    if ticket.status == "closed":
        raise HTTPException(status_code=409, detail="工单已关闭，重新打开后可生成草稿")
    result = await qa.answer_question(
        db, user, ticket.title, product_version=ticket.product_version or None,
        tenant_scope=ticket.tenant_id,
    )
    # 资料不足时把说明给顾问，保留手写回复入口。
    return {
        "draft": result.answer if result.citations else "",
        "message": result.answer if not result.citations else "",
        "citations": result.citations,
        "answer_mode": result.answer_mode,
    }


@router.post("/{ticket_id}/messages")
async def add_message(
    ticket_id: str,
    req: MessageRequest,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key", max_length=128),
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    ticket = await ticket_service.get_ticket(db, user, ticket_id)
    message = await ticket_service.add_message(
        db, user, ticket, req.content, request_key=idempotency_key
    )
    return ticket_service.serialize_message(message)


@router.post("/{ticket_id}/supplement")
async def supplement(
    ticket_id: str,
    req: SupplementRequest,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key", max_length=128),
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not any([req.description, req.error_message, req.tried_steps]):
        raise HTTPException(status_code=422, detail="至少提供一项补充内容")
    ticket = await ticket_service.get_ticket(db, user, ticket_id)
    await ticket_service.supplement_ticket(
        db,
        user,
        ticket,
        description=req.description,
        error_message=req.error_message,
        tried_steps=req.tried_steps,
        request_key=idempotency_key,
    )
    return await ticket_service.get_ticket_detail(db, user, ticket)


@router.post("/{ticket_id}/attachments")
async def upload_attachment(
    ticket_id: str,
    file: UploadFile = File(...),
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    ticket = await ticket_service.get_ticket(db, user, ticket_id)
    filename = Path(file.filename or "attachment").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ATTACHMENT_EXTENSIONS:
        raise HTTPException(status_code=422, detail=f"不支持的附件类型 {suffix}")

    content = await read_upload(file, label="附件")

    attachment_id = new_id()
    folder = settings.data_path / "attachments" / ticket.id
    folder.mkdir(parents=True, exist_ok=True)
    stored_path = folder / f"{attachment_id}{suffix}"
    stored_path.write_bytes(content)

    attachment = await ticket_service.add_attachment(
        db,
        user,
        ticket,
        filename=filename,
        stored_path=str(stored_path),
        content_type=file.content_type or "application/octet-stream",
        size=len(content),
    )
    return {
        "id": attachment.id,
        "filename": attachment.filename,
        "size": attachment.size,
        "content_type": attachment.content_type,
    }


@router.get("/{ticket_id}/attachments/{attachment_id}")
async def download_attachment(
    ticket_id: str,
    attachment_id: str,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    ticket = await ticket_service.get_ticket(db, user, ticket_id)
    attachment = await db.get(TicketAttachment, attachment_id)
    if attachment is None or attachment.ticket_id != ticket.id:
        raise HTTPException(status_code=404, detail="附件不存在")

    path = Path(attachment.stored_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="附件文件已丢失")
    # 强制下载，避免浏览器把用户上传的 HTML/SVG 当页面执行
    return FileResponse(
        path,
        filename=attachment.filename,
        media_type="application/octet-stream",
        content_disposition_type="attachment",
    )
