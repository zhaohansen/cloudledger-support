# app/services/chat.py
# 会话与 Agent 调用：保存消息、恢复会话事实、执行 LangGraph、回写上下文。
# 会话归属在每次读写时校验，防止通过 session_id 读取别人的聊天记录。
#
# 流式规则：
# - 只有“纯问答”会直接走 qa.stream_answer，模型 token 真正逐块转发；
# - 有草稿、工单操作、转人工等受控流程仍然经过 LangGraph，事件会标记 buffered=True，
#   不把一次性结果伪装成逐 token 流式；
# - 客户端断开或超时时保存已生成的前缀，meta.stream_status=partial，且不保存引用列表。

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

from fastapi import HTTPException
from sqlalchemy import and_, delete, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.graph import build_agent_graph
from app.agent.intent import classify_intent
from app.agent.nodes import AgentRuntime, merge_facts
from app.config import get_settings
from app.deps import UserContext
from app.logging_config import get_logger
from app.models import ChatMessage, ChatRequest, ChatSession, TicketDraft
from app.services import qa as qa_service
from app.services.tickets import serialize_draft

settings = get_settings()
logger = get_logger(__name__)

DisconnectCheck = Callable[[], Awaitable[bool]] | None


@dataclass
class PreparedTurn:
    session: ChatSession
    context: dict
    facts: dict
    pending_draft: dict | None
    pending_draft_stale: bool
    history: list[dict]


async def create_session(db: AsyncSession, user: UserContext, title: str = "新会话") -> ChatSession:
    session = ChatSession(user_id=user.user_id, tenant_id=user.tenant_id, title=title[:120])
    db.add(session)
    await db.flush()
    return session


async def list_sessions(db: AsyncSession, user: UserContext, limit: int = 30) -> list[ChatSession]:
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == user.user_id, ChatSession.tenant_id == user.tenant_id)
        .order_by(ChatSession.updated_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def get_session(db: AsyncSession, user: UserContext, session_id: str) -> ChatSession:
    session = await db.get(ChatSession, session_id)
    # 会话不存在与不属于当前用户统一返回 404，避免枚举他人会话
    if session is None or session.user_id != user.user_id or session.tenant_id != user.tenant_id:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session


async def delete_session(db: AsyncSession, user: UserContext, session_id: str) -> None:
    """删除当前用户自己的会话及其消息与幂等记录（不影响已创建的正式工单）。"""
    session = await get_session(db, user, session_id)
    await db.execute(delete(ChatMessage).where(ChatMessage.session_id == session.id))
    await db.execute(delete(ChatRequest).where(ChatRequest.session_id == session.id))
    await db.delete(session)
    await db.commit()


async def get_messages(db: AsyncSession, user: UserContext, session_id: str) -> list[dict]:
    await get_session(db, user, session_id)
    rows = (
        await db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
        )
    ).scalars().all()
    return [serialize_chat_message(m) for m in rows]


async def get_message_page(
    db: AsyncSession,
    user: UserContext,
    session_id: str,
    limit: int,
    before: str | None,
) -> dict:
    await get_session(db, user, session_id)
    query = select(ChatMessage).where(ChatMessage.session_id == session_id)
    if before is not None:
        cursor = await db.get(ChatMessage, before)
        if cursor is None or cursor.session_id != session_id:
            raise HTTPException(status_code=422, detail="消息分页游标无效")
        query = query.where(or_(
            ChatMessage.created_at < cursor.created_at,
            and_(ChatMessage.created_at == cursor.created_at, ChatMessage.id < cursor.id),
        ))
    query = query.order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc()).limit(limit + 1)
    rows = list((await db.scalars(query)).all())
    page = rows[:limit]
    return {
        "items": [serialize_chat_message(message) for message in reversed(page)],
        "next_before": page[-1].id if len(rows) > limit else None,
    }


async def claim_chat_request(
    db: AsyncSession,
    user: UserContext,
    session_id: str,
    key: str | None,
    content: str,
    product_version: str | None,
    mode: str,
) -> ChatRequest | None:
    if key is None:
        return None
    await get_session(db, user, session_id)
    if not content.strip():
        raise HTTPException(status_code=422, detail="消息内容不能为空")
    payload = json.dumps([mode, content.strip(), product_version], ensure_ascii=False)
    fingerprint = hashlib.sha256(payload.encode()).hexdigest()
    request = ChatRequest(session_id=session_id, request_key=key, fingerprint=fingerprint)
    try:
        db.add(request)
        await db.commit()
    except IntegrityError:
        await db.rollback()
        existing = await db.get(ChatRequest, (session_id, key))
        if existing is None:
            raise
        if existing.fingerprint != fingerprint:
            raise HTTPException(status_code=409, detail="请求标识已用于其他消息")
        if existing.result is None:
            raise HTTPException(status_code=409, detail="相同请求正在处理或未完成，请先查看聊天历史")
        return existing
    return request


def serialize_chat_message(message: ChatMessage) -> dict:
    return {
        "id": message.id,
        "role": message.role,
        "content": message.content,
        "citations": message.citations_json or [],
        "meta": message.meta_json or {},
        "created_at": message.created_at.isoformat() if message.created_at else None,
    }


def serialize_session(session: ChatSession) -> dict:
    return {
        "id": session.id,
        "title": session.title,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "updated_at": session.updated_at.isoformat() if session.updated_at else None,
    }


async def _load_history(db: AsyncSession, session_id: str) -> list[dict]:
    """读取最近若干条历史消息（不含本次刚写入的消息），用于理解指代。"""
    rows = (
        await db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
            .limit(settings.chat_history_limit)
        )
    ).scalars().all()
    return [
        {"role": row.role, "content": row.content}
        for row in reversed(rows)
        if row.role in ("user", "assistant")
    ]


async def _prepare_turn(
    db: AsyncSession,
    user: UserContext,
    session_id: str,
    content: str,
    product_version: str | None,
) -> PreparedTurn:
    if not content:
        raise HTTPException(status_code=422, detail="消息内容不能为空")
    if len(content) > 2000:
        raise HTTPException(status_code=422, detail="单条消息不能超过 2000 字")

    session = await get_session(db, user, session_id)
    # 先读历史，再写入本次用户消息，避免模型把当前问题重复当成历史
    history = await _load_history(db, session.id)
    db.add(ChatMessage(session_id=session.id, role="user", content=content))

    try:
        context = json.loads(session.context_json or "{}")
    except json.JSONDecodeError:
        context = {}
    facts = dict(context.get("facts") or {})
    if product_version:
        facts["product_version"] = product_version.strip()

    pending_draft = None
    pending_draft_stale = False
    pending_id = context.get("pending_draft_id")
    seen_version = context.get("pending_draft_version")
    if pending_id:
        draft = await db.get(TicketDraft, pending_id)
        if draft and draft.status == "draft" and draft.user_id == user.user_id:
            pending_draft = serialize_draft(draft)
            # 记录用户上次实际看到的版本。历史会话缺失该字段时为 None，
            # 后续确认必须重新核对，不能用数据库最新版本代填。
            pending_draft["seen_version"] = (
                int(seen_version) if seen_version is not None else None
            )
            # 用户在页面上看到的版本与数据库不一致：说明草稿被改过，
            # 这次说“确认”也不能直接提交，必须重新核对。
            if seen_version is not None and int(seen_version) != int(draft.version or 1):
                pending_draft_stale = True

    return PreparedTurn(
        session=session,
        context=context,
        facts=facts,
        pending_draft=pending_draft,
        pending_draft_stale=pending_draft_stale,
        history=history,
    )


def _write_context(
    session: ChatSession,
    context: dict,
    facts: dict,
    *,
    pending_out: dict | None = None,
    keep_pending: bool = False,
    last_ticket_no: str | None = None,
) -> None:
    if keep_pending:
        pending_id = context.get("pending_draft_id")
        pending_version = context.get("pending_draft_version")
    else:
        pending_id = (pending_out or {}).get("id")
        pending_version = (pending_out or {}).get("version")
    session.context_json = json.dumps(
        {
            "facts": facts,
            "pending_draft_id": pending_id,
            "pending_draft_version": pending_version,
            "last_ticket_no": last_ticket_no,
        },
        ensure_ascii=False,
    )


async def _persist_assistant(
    db: AsyncSession,
    session: ChatSession,
    *,
    content: str,
    citations: list[dict],
    meta: dict,
) -> ChatMessage:
    assistant = ChatMessage(
        session_id=session.id,
        role="assistant",
        content=content,
        citations_json=citations or [],
        meta_json=meta or {},
    )
    db.add(assistant)
    await db.flush()
    return assistant


def _answer_meta_from_result(result_event: dict) -> dict:
    return {
        "answer_mode": result_event.get("answer_mode"),
        "search_reason": result_event.get("search_reason"),
        "version_conflict": result_event.get("version_conflict"),
        "matched_versions": result_event.get("matched_versions"),
        "retrieval_ms": result_event.get("retrieval_ms"),
        "embedding_backend": result_event.get("embedding_backend"),
        "retrieval_backend": result_event.get("retrieval_backend"),
        "rerank_status": result_event.get("rerank_status"),
        "search_detail": result_event.get("search_detail"),
        "used_citation_indices": result_event.get("used_citation_indices"),
        "citation_warnings": result_event.get("citation_warnings"),
        "partial": result_event.get("partial", False),
    }


def _partial_reply(parts: list[str]) -> str:
    notice = "（回答未完成：连接中断或超时，请重新提问。）"
    prefix = "".join(parts).strip()
    return f"{prefix}\n\n{notice}" if prefix else notice


async def _persist_partial_stream(
    db: AsyncSession,
    session: ChatSession,
    prepared: PreparedTurn,
    facts: dict,
    intent: str,
    parts: list[str],
    error_message: str,
    chat_request: ChatRequest | None,
) -> None:
    """硬取消/断开时保存已生成前缀：不保存引用，避免把半截回答包装成完整结论。"""
    reply = _partial_reply(parts)
    meta = {
        "intent": intent,
        "stream_status": "partial",
        "answer_mode": "llm_degraded",
        "partial": True,
        "stream_error": error_message,
    }
    try:
        assistant = await _persist_assistant(db, session, content=reply, citations=[], meta=meta)
        if chat_request is not None:
            chat_request.result = {
                "final": {"type": "result", "answer": reply, "citations": [], **meta},
                "saved": {"type": "saved", "message_id": assistant.id, "stream_status": "partial"},
            }
        _write_context(
            session,
            prepared.context,
            facts,
            keep_pending=True,
            last_ticket_no=prepared.context.get("last_ticket_no"),
        )
        await db.commit()
    except Exception as exc:  # 连接已断开，保存失败只记录，不掩盖原始取消
        logger.error("chat.persist_partial_failed", error=type(exc).__name__)


async def _run_agent_turn(
    db: AsyncSession,
    user: UserContext,
    prepared: PreparedTurn,
    content: str,
    request_id: str,
    stream_status: str,
) -> tuple[ChatMessage, dict]:
    """执行 LangGraph，更新会话并保存回复。"""
    session = prepared.session
    runtime = AgentRuntime(db=db, user=user, session_id=session.id, request_id=request_id)
    graph = build_agent_graph(runtime)
    initial_state = {
        "user_message": content,
        "facts": prepared.facts,
        "history": prepared.history,
        "pending_draft": prepared.pending_draft,
        "pending_draft_stale": prepared.pending_draft_stale,
        "last_ticket_no": prepared.context.get("last_ticket_no"),
    }
    final_state = await graph.ainvoke(initial_state)

    pending_out = final_state.get("pending_draft") or {}
    _write_context(
        session,
        prepared.context,
        final_state.get("facts") or prepared.facts,
        pending_out=pending_out,
        last_ticket_no=final_state.get("last_ticket_no"),
    )
    if session.title == "新会话":
        session.title = content[:30]

    reply = final_state.get("reply") or "抱歉，我没有理解你的问题，请换一种说法。"
    meta = {
        "intent": final_state.get("intent"),
        "intent_source": final_state.get("intent_source"),
        "answer_mode": final_state.get("answer_mode"),
        "draft": final_state.get("draft"),
        "ticket": final_state.get("ticket"),
        "tool_ok": final_state.get("tool_ok"),
        "error": final_state.get("error"),
        "stream_status": stream_status,
        **(final_state.get("answer_meta") or {}),
    }
    if stream_status == "buffered":
        meta["buffered"] = True
    assistant = await _persist_assistant(
        db,
        session,
        content=reply,
        citations=final_state.get("citations") or [],
        meta=meta,
    )
    return assistant, final_state


async def handle_message(
    db: AsyncSession,
    user: UserContext,
    session_id: str,
    content: str,
    *,
    product_version: str | None = None,
    request_id: str = "-",
    chat_request: ChatRequest | None = None,
) -> dict:
    content = (content or "").strip()
    prepared = await _prepare_turn(db, user, session_id, content, product_version)
    session = prepared.session

    assistant, final_state = await _run_agent_turn(
        db, user, prepared, content, request_id, stream_status="complete"
    )

    logger.info(
        "chat.handled",
        session_id=session.id,
        user_id=user.user_id,
        intent=final_state.get("intent"),
        answer_mode=final_state.get("answer_mode"),
        request_id=request_id,
    )

    response = {
        "message_id": assistant.id,
        "reply": assistant.content,
        "citations": assistant.citations_json,
        "draft": final_state.get("draft"),
        "ticket": final_state.get("ticket"),
        "intent": final_state.get("intent"),
        "intent_source": final_state.get("intent_source"),
        "answer_mode": final_state.get("answer_mode"),
        "tool_ok": final_state.get("tool_ok"),
        "error": final_state.get("error"),
        # 评测与排查需要看到实际检索后端、耗时和降级原因
        "answer_meta": final_state.get("answer_meta") or {},
        "session": serialize_session(session),
    }
    if chat_request is not None:
        chat_request.result = response
        await db.commit()
    return response


async def stream_message(
    db: AsyncSession,
    user: UserContext,
    session_id: str,
    content: str,
    *,
    product_version: str | None = None,
    use_web_search: bool = False,
    request_id: str = "-",
    is_disconnected: DisconnectCheck = None,
    idempotency_key: str | None = None,
) -> AsyncIterator[dict]:
    """SSE 事件生成器。

    事件类型：start / intent / stage / retrieval / citations / token / warning /
    error / result / saved。
    """
    content = (content or "").strip()
    started = time.perf_counter()
    # 生成器开始执行时重新加载请求记录，之后才能保存重试所需的结果。
    chat_request = None
    if idempotency_key is not None:
        chat_request = await db.get(ChatRequest, (session_id, idempotency_key))
    prepared = await _prepare_turn(db, user, session_id, content, product_version)
    session = prepared.session
    intent = classify_intent(
        content,
        has_pending_draft=bool(prepared.pending_draft),
        last_ticket_no=prepared.context.get("last_ticket_no"),
    )
    facts = merge_facts(prepared.facts, content, intent)

    # 有草稿或受控操作时一律走 LangGraph，避免绕过确认直接写库
    direct_qa = intent in ("question", "issue") and not prepared.pending_draft
    if intent == "issue" and direct_qa:
        # 故障排查需要版本；缺版本时交给 Agent 追问，不直接检索
        if not facts.get("product_version") and not facts.get("version_unknown"):
            direct_qa = False

    yield {
        "type": "start",
        "session_id": session.id,
        "intent": intent,
        "streaming": direct_qa,
    }
    # 兼容现有前端：meta 事件用于更新会话标题
    yield {"type": "meta", "session": serialize_session(session), "intent": intent}

    if direct_qa:
        version = None if facts.get("version_unknown") else facts.get("product_version")
        parts: list[str] = []
        final_event: dict | None = None
        partial = False
        error_message = ""

        answer_stream = qa_service.stream_answer(
            db,
            user,
            facts.get("description") or content,
            product_version=version,
            history=prepared.history,
            use_web_search=use_web_search,
        )
        try:
            async for event in answer_stream:
                if event.get("type") == "token":
                    parts.append(event.get("content") or "")
                if event.get("type") == "result":
                    final_event = event
                # 模型超时/异常时 qa 层会正常返回 result(partial=True) 而不是抛异常；
                # 这里必须同步标记，否则会被当成完整回答保存成 complete。
                if event.get("type") == "error" or (
                    event.get("type") == "result" and event.get("partial")
                ):
                    partial = True
                    error_message = event.get("message") or error_message or "回答未完整生成"
                yield event
                if is_disconnected is not None and await is_disconnected():
                    partial = True
                    error_message = "客户端连接已断开，停止继续生成"
                    break
        except asyncio.CancelledError:
            # 硬取消时也要尽量保存前缀，避免用户看到的内容与历史记录不一致
            partial = True
            error_message = "请求被取消，停止继续生成"
            await _persist_partial_stream(
                db, session, prepared, facts, intent, parts, error_message, chat_request
            )
            raise
        except Exception as exc:  # 流式阶段异常：保存已生成前缀，不假装完整
            partial = True
            error_message = f"{type(exc).__name__}: {str(exc)[:200]}"
            logger.error("chat.stream_exception", error=type(exc).__name__)
            yield {
                "type": "error",
                "code": "stream_exception",
                "message": "回答生成中断，已保存已生成的部分内容。",
                "partial": True,
            }
        finally:
            try:
                await answer_stream.aclose()
            except Exception:
                pass

        if final_event is not None and not partial and not error_message:
            reply = final_event.get("answer") or "".join(parts)
            citations = final_event.get("citations") or []
            meta = {
                "intent": intent,
                "stream_status": "complete",
                "draft": None,
                "ticket": None,
                "tool_ok": True,
                **_answer_meta_from_result(final_event),
            }
        else:
            reply = _partial_reply(parts)
            citations = []
            meta = {
                "intent": intent,
                "stream_status": "partial",
                "answer_mode": "llm_degraded",
                "draft": None,
                "ticket": None,
                "tool_ok": True,
                "partial": True,
                "stream_error": error_message,
            }

        assistant = await _persist_assistant(
            db, session, content=reply, citations=citations, meta=meta
        )
        _write_context(
            session,
            prepared.context,
            facts,
            keep_pending=True,
            last_ticket_no=prepared.context.get("last_ticket_no"),
        )
        if session.title == "新会话":
            session.title = content[:30]
        if chat_request is not None:
            result = {"type": "result", "answer": reply, "citations": citations, **meta}
            chat_request.result = {
                "final": result,
                "saved": {"type": "saved", "message_id": assistant.id, "stream_status": meta["stream_status"]},
            }
        await db.commit()

        logger.info(
            "chat.streamed",
            session_id=session.id,
            user_id=user.user_id,
            intent=intent,
            status=meta["stream_status"],
            request_id=request_id,
            total_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        yield {
            "type": "saved",
            "message_id": assistant.id,
            "stream_status": meta["stream_status"],
        }
        return

    # 工单等操作由 LangGraph 执行，完成后发送结果。
    yield {"type": "stage", "stage": "agent"}
    assistant, final_state = await _run_agent_turn(
        db, user, prepared, content, request_id, stream_status="buffered"
    )
    result = {
        "type": "result",
        "answer": assistant.content,
        "citations": assistant.citations_json,
        "answer_mode": final_state.get("answer_mode"),
        "intent": final_state.get("intent"),
        "intent_source": final_state.get("intent_source"),
        "draft": final_state.get("draft"),
        "ticket": final_state.get("ticket"),
        "tool_ok": final_state.get("tool_ok"),
        "error": final_state.get("error"),
        "partial": False,
        "streaming": False,
        "buffered": True,
    }
    saved = {"type": "saved", "message_id": assistant.id, "stream_status": "buffered"}
    if chat_request is not None:
        chat_request.result = {"final": result, "saved": saved}
    await db.commit()
    yield result
    yield saved
