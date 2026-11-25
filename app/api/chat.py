# app/api/chat.py
# 会话、消息与 SSE 流式输出。
# 纯问答会消费模型流并逐块转发；有草稿/工单操作时仍先执行受控 Agent，
# 事件标记 buffered=True，避免把一次性结果伪装成逐 token 流式。

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.deps import UserContext, get_current_user
from app.logging_config import get_logger
from app.services import chat as chat_service

router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = get_logger(__name__)


class SessionCreateRequest(BaseModel):
    title: str = Field("新会话", max_length=120)


class MessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=2000)
    product_version: str | None = Field(None, max_length=32)
    use_web_search: bool = False


@router.post("/sessions")
async def create_session(
    req: SessionCreateRequest,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    session = await chat_service.create_session(db, user, req.title)
    return chat_service.serialize_session(session)


@router.get("/sessions")
async def list_sessions(
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    sessions = await chat_service.list_sessions(db, user)
    return {"items": [chat_service.serialize_session(s) for s in sessions]}


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await chat_service.delete_session(db, user, session_id)
    return {"deleted": True, "session_id": session_id}


@router.get("/sessions/{session_id}/messages")
async def list_messages(
    session_id: str,
    limit: int | None = Query(None, ge=1, le=100),
    before: str | None = Query(None, max_length=32),
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # 旧客户端不传分页参数时保留原有响应。
    if limit is None and before is None:
        return {"items": await chat_service.get_messages(db, user, session_id)}
    return await chat_service.get_message_page(db, user, session_id, limit or 30, before)


@router.post("/sessions/{session_id}/messages")
async def send_message(
    session_id: str,
    req: MessageRequest,
    request: Request,
    idempotency_key: str | None = Header(None, min_length=1, max_length=128),
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    chat_request = await chat_service.claim_chat_request(
        db, user, session_id, idempotency_key, req.content, req.product_version, "message"
    )
    if chat_request is not None and chat_request.result is not None:
        return chat_request.result
    request_id = getattr(request.state, "request_id", None) or request.headers.get("X-Request-ID", "-")
    return await chat_service.handle_message(
        db,
        user,
        session_id,
        req.content,
        product_version=req.product_version,
        request_id=request_id,
        chat_request=chat_request,
    )


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/sessions/{session_id}/stream")
async def stream_message(
    session_id: str,
    req: MessageRequest,
    request: Request,
    idempotency_key: str | None = Header(None, min_length=1, max_length=128),
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # 在响应开始前完成会话归属校验，这样 404 还能以正常 HTTP 状态返回
    await chat_service.get_session(db, user, session_id)
    request_id = getattr(request.state, "request_id", None) or request.headers.get("X-Request-ID", "-")
    chat_request = await chat_service.claim_chat_request(
        db, user, session_id, idempotency_key, req.content, req.product_version, "stream"
    )

    replay_result = chat_request.result if chat_request is not None else None

    async def event_generator():
        if replay_result is not None:
            yield _sse(replay_result["final"])
            yield _sse(replay_result["saved"])
            yield "data: [DONE]\n\n"
            return
        try:
            # 流式响应期间使用数据库会话，结束时释放。
            async with db:
                async for event in chat_service.stream_message(
                    db,
                    user,
                    session_id,
                    req.content,
                    product_version=req.product_version,
                    use_web_search=req.use_web_search,
                    request_id=request_id,
                    is_disconnected=request.is_disconnected,
                    idempotency_key=idempotency_key,
                ):
                    yield _sse(event)
        except HTTPException as exc:
            code, message = f"http_{exc.status_code}", str(exc.detail)
        except Exception as exc:
            logger.error("chat.stream_api_failed", error=type(exc).__name__)
            code, message = "internal_error", "服务异常，已停止生成；请稍后重试。"
        else:
            yield "data: [DONE]\n\n"
            return

        yield _sse({"type": "error", "code": code, "message": message, "partial": False})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
