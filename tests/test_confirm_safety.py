# tests/test_confirm_safety.py
# 覆盖确认误判与工单确认可靠性：否定、取消、疑问、修改、旧版本、重复确认、并发确认、他人草稿。

from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import func, select

from app.agent.intent import classify_intent, is_explicit_confirm
from app.db import AsyncSessionLocal
from app.models import Ticket, TicketDraft
from app.services.tickets import get_or_create_draft
from tests.helpers import auth, ctx_for, token_for


def test_confirm_intent_allowlist():
    """只有严格白名单里的完整指令才算确认。"""
    for text in ["确认创建", "确认提交", "确认无误", "就按这个创建", "提交吧", "好的，确认创建", "没问题，提交吧"]:
        assert is_explicit_confirm(text) is True, text
        assert classify_intent(text, has_pending_draft=True, last_ticket_no=None) == "confirm", text

    for text in ["不要提交", "先不创建吧", "取消", "算了，不建了", "别创建"]:
        assert is_explicit_confirm(text) is False, text
        assert classify_intent(text, has_pending_draft=True, last_ticket_no=None) == "cancel", text


def test_confirm_intent_rejects_question_and_edit():
    cases = {
        "这个能提交吗": "clarify",
        "确认一下怎么操作": "clarify",
        "确认之前先改一下描述": "edit_draft",
        "把优先级改成高": "edit_draft",
        "确认": "confirm",
        "可以": "clarify",
        "好": "clarify",
    }
    for text, expected in cases.items():
        assert classify_intent(text, has_pending_draft=True, last_ticket_no=None) == expected, text


async def _make_draft(username: str, description: str) -> dict:
    ctx = await ctx_for(username)
    async with AsyncSessionLocal() as db:
        draft = await get_or_create_draft(
            db,
            ctx,
            session_id=None,
            title="确认安全测试",
            description=description,
            product_version="v2.3",
            error_message="模拟报错",
            tried_steps="模拟步骤",
            category="other",
            priority="medium",
        )
        await db.commit()
        return {"id": draft.id, "version": draft.version}


async def test_old_draft_version_cannot_confirm_new_content(client):
    """草稿内容变了以后，旧版本的确认必须被拒绝。"""
    token = await token_for("customer_a2")
    draft = await _make_draft("customer_a2", "旧版本确认测试 " + uuid.uuid4().hex[:8])

    saved = await client.patch(
        f"/api/tickets/drafts/{draft['id']}",
        headers=auth(token),
        json={"description": "修改后的内容", "expected_version": draft["version"]},
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["version"] == draft["version"] + 1

    stale = await client.post(
        "/api/tickets/confirm",
        headers=auth(token),
        json={"draft_id": draft["id"], "expected_version": draft["version"]},
    )
    assert stale.status_code == 409
    assert "版本" in stale.json()["detail"] or "更新" in stale.json()["detail"]

    fresh = await client.post(
        "/api/tickets/confirm",
        headers=auth(token),
        json={"draft_id": draft["id"], "expected_version": draft["version"] + 1},
    )
    assert fresh.status_code == 200, fresh.text


async def test_cancelled_draft_cannot_confirm(client):
    token = await token_for("customer_a2")
    draft = await _make_draft("customer_a2", "取消后确认测试 " + uuid.uuid4().hex[:8])

    cancelled = await client.patch(
        f"/api/tickets/drafts/{draft['id']}",
        headers=auth(token),
        json={"description": "取消前修改", "expected_version": draft["version"]},
    )
    assert cancelled.status_code == 200
    # 通过取消接口或 Agent 取消都由服务层写 cancelled；这里直接调用服务层模拟
    from app.deps import UserContext  # noqa: F401
    from app.models import TicketDraft as DraftModel

    async with AsyncSessionLocal() as db:
        row = await db.get(DraftModel, draft["id"])
        row.status = "cancelled"
        await db.commit()

    response = await client.post(
        "/api/tickets/confirm",
        headers=auth(token),
        json={"draft_id": draft["id"], "expected_version": draft["version"] + 1},
    )
    assert response.status_code == 409
    assert "取消" in response.json()["detail"]


async def test_other_users_draft_cannot_be_confirmed(client):
    draft = await _make_draft("customer_a2", "他人草稿测试 " + uuid.uuid4().hex[:8])
    other_token = await token_for("customer_b")
    response = await client.post(
        "/api/tickets/confirm",
        headers=auth(other_token),
        json={"draft_id": draft["id"], "expected_version": draft["version"]},
    )
    assert response.status_code == 409
    assert "无权" in response.json()["detail"]


async def test_repeated_and_concurrent_confirm_creates_one_ticket(client):
    token = await token_for("customer_a2")
    draft = await _make_draft("customer_a2", "并发确认测试 " + uuid.uuid4().hex[:8])

    first = await client.post(
        "/api/tickets/confirm",
        headers=auth(token),
        json={"draft_id": draft["id"], "expected_version": draft["version"]},
    )
    second = await client.post(
        "/api/tickets/confirm",
        headers=auth(token),
        json={"draft_id": draft["id"], "expected_version": draft["version"]},
    )
    assert first.status_code == 200 and second.status_code == 200
    assert first.json()["ticket"]["ticket_no"] == second.json()["ticket"]["ticket_no"]
    assert second.json()["already_created"] is True

    # 新草稿：两个并发请求同时确认，也只能产生一张工单
    draft2 = await _make_draft("customer_a2", "并发确认测试二 " + uuid.uuid4().hex[:8])
    payload = {"draft_id": draft2["id"], "expected_version": draft2["version"]}
    responses = await asyncio.gather(
        client.post("/api/tickets/confirm", headers=auth(token), json=payload),
        client.post("/api/tickets/confirm", headers=auth(token), json=payload),
        return_exceptions=True,
    )
    ok = [r for r in responses if not isinstance(r, Exception) and r.status_code == 200]
    assert ok, responses
    no = {r.json()["ticket"]["ticket_no"] for r in ok}
    assert len(no) == 1, responses

    async with AsyncSessionLocal() as db:
        count = await db.scalar(
            select(func.count()).select_from(Ticket).where(Ticket.draft_id == draft2["id"])
        )
    assert count == 1


async def test_concurrent_different_drafts_get_distinct_numbers(client):
    token = await token_for("customer_a2")
    draft_a = await _make_draft("customer_a2", "不同草稿并发 A " + uuid.uuid4().hex[:8])
    draft_b = await _make_draft("customer_a2", "不同草稿并发 B " + uuid.uuid4().hex[:8])

    responses = await asyncio.gather(
        client.post(
            "/api/tickets/confirm",
            headers=auth(token),
            json={"draft_id": draft_a["id"], "expected_version": draft_a["version"]},
        ),
        client.post(
            "/api/tickets/confirm",
            headers=auth(token),
            json={"draft_id": draft_b["id"], "expected_version": draft_b["version"]},
        ),
        return_exceptions=True,
    )
    ok = [r for r in responses if not isinstance(r, Exception) and r.status_code == 200]
    assert len(ok) == 2, responses
    numbers = {r.json()["ticket"]["ticket_no"] for r in ok}
    assert len(numbers) == 2


async def test_agent_edit_and_cancel_never_create_ticket(client):
    """自然语言“先改一下”“不要提交”只改草稿/取消，不能创建工单。"""
    token = await token_for("customer_a2")
    session = await client.post("/api/chat/sessions", headers=auth(token), json={"title": "确认安全"})
    session_id = session.json()["id"]

    first = await client.post(
        f"/api/chat/sessions/{session_id}/messages",
        headers=auth(token),
        json={
            "content": "转人工。v2.3 新建商品保存提示编码重复，已尝试去掉空格，仍然报错。",
            "product_version": "v2.3",
        },
    )
    assert first.status_code == 200, first.text
    draft = first.json().get("draft")
    assert draft and draft["status"] == "draft", first.text
    original_version = draft["version"]

    edit = await client.post(
        f"/api/chat/sessions/{session_id}/messages",
        headers=auth(token),
        json={"content": "确认之前先改一下描述：补充一下，报错发生在保存按钮"},
    )
    assert edit.status_code == 200, edit.text
    assert edit.json().get("ticket") is None
    edited = edit.json().get("draft")
    assert edited and edited["version"] > original_version
    assert "补充一下" in edited["description"]

    cancel = await client.post(
        f"/api/chat/sessions/{session_id}/messages",
        headers=auth(token),
        json={"content": "先不创建吧"},
    )
    assert cancel.status_code == 200
    assert cancel.json().get("ticket") is None

    confirm_after_cancel = await client.post(
        f"/api/chat/sessions/{session_id}/messages",
        headers=auth(token),
        json={"content": "确认创建"},
    )
    assert confirm_after_cancel.status_code == 200
    assert confirm_after_cancel.json().get("ticket") is None

    async with AsyncSessionLocal() as db:
        count = await db.scalar(
            select(func.count()).select_from(Ticket).where(Ticket.draft_id == draft["id"])
        )
    assert count == 0


async def test_cancelled_then_new_draft_same_issue_is_allowed(client):
    """同一问题取消后重新发起，不应被旧幂等键卡住。"""
    description = "取消后重新发起 " + uuid.uuid4().hex[:8]
    draft1 = await _make_draft("customer_a2", description)
    async with AsyncSessionLocal() as db:
        row = await db.get(TicketDraft, draft1["id"])
        row.status = "cancelled"
        await db.commit()

    draft2 = await _make_draft("customer_a2", description)
    assert draft2["id"] != draft1["id"]
    assert draft2["version"] == 1


async def test_service_level_parallel_confirm_with_separate_sessions():
    """两个独立数据库会话同时确认：取号与唯一约束都要能正确恢复。"""
    from app.services.tickets import create_ticket_from_draft

    draft = await _make_draft("customer_a2", "服务层并发确认 " + uuid.uuid4().hex[:8])
    ctx = await ctx_for("customer_a2")
    barrier = asyncio.Barrier(2)
    results: list[dict] = []

    async def worker():
        async with AsyncSessionLocal() as db:
            await barrier.wait()
            try:
                ticket, already = await create_ticket_from_draft(
                    db, ctx, draft["id"], expected_version=draft["version"]
                )
                await db.commit()
                results.append({"ticket_no": ticket.ticket_no, "already": already})
            except Exception as exc:  # 让测试给出具体异常
                await db.rollback()
                results.append({"error": type(exc).__name__ + ": " + str(exc)[:120]})

    await asyncio.gather(worker(), worker())

    assert all("error" not in item for item in results), results
    assert len({item["ticket_no"] for item in results}) == 1, results

    async with AsyncSessionLocal() as db:
        count = await db.scalar(
            select(func.count()).select_from(Ticket).where(Ticket.draft_id == draft["id"])
        )
    assert count == 1
