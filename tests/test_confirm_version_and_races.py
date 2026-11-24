# tests/test_confirm_version_and_races.py
# 本轮修复的回归测试：
# 1. 确认必须提供用户看到的草稿版本（缺失、null、0、负数、字符串、小数都拒绝）；
# 2. 编辑 vs 确认、取消 vs 确认、两次编辑同一版本的竞争结果；
# 3. 工单内容必须对应被确认的草稿版本，工单/草稿/事件保持一致。

from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import func, select

from app.db import AsyncSessionLocal
from app.models import Ticket, TicketDraft, TicketEvent
from app.services.tickets import (
    ToolError,
    create_ticket_from_draft,
    get_or_create_draft,
)
from tests.helpers import auth, ctx_for, token_for


async def _make_draft(username: str, description: str) -> dict:
    ctx = await ctx_for(username)
    async with AsyncSessionLocal() as db:
        draft = await get_or_create_draft(
            db,
            ctx,
            session_id=None,
            title="版本与竞争测试",
            description=description,
            product_version="v2.3",
            error_message="模拟报错",
            tried_steps="模拟步骤",
            category="other",
            priority="medium",
        )
        await db.commit()
        return {"id": draft.id, "version": draft.version, "description": draft.description}


async def _draft_row(draft_id: str) -> TicketDraft | None:
    async with AsyncSessionLocal() as db:
        return await db.get(TicketDraft, draft_id)


async def _ticket_count(draft_id: str) -> int:
    async with AsyncSessionLocal() as db:
        return int(
            await db.scalar(
                select(func.count()).select_from(Ticket).where(Ticket.draft_id == draft_id)
            )
            or 0
        )


@pytest.mark.parametrize(
    "payload",
    [
        {},  # 缺失
        {"expected_version": None},  # null
        {"expected_version": 0},
        {"expected_version": -1},
        {"expected_version": "1"},  # 字符串即使是数字也拒绝
        {"expected_version": 1.5},
        {"expected_version": True},
    ],
)
async def test_confirm_requires_valid_expected_version(client, payload):
    """请求模型层就必须拒绝，不能进入服务层后再“猜”版本。"""
    token = await token_for("customer_a2")
    draft = await _make_draft("customer_a2", "缺版本测试 " + uuid.uuid4().hex[:8])

    response = await client.post(
        "/api/tickets/confirm",
        headers=auth(token),
        json={"draft_id": draft["id"], **payload},
    )
    assert response.status_code == 422, response.text
    assert await _ticket_count(draft["id"]) == 0
    row = await _draft_row(draft["id"])
    assert row is not None and row.status == "draft"


@pytest.mark.parametrize("bad_version", [None, 0, -3, "abc", True])
async def test_service_layer_rejects_missing_or_invalid_version(bad_version):
    """服务层同样拒绝：Agent 工具和脚本不能绕过请求模型直接建单。"""
    ctx = await ctx_for("customer_a2")
    draft = await _make_draft("customer_a2", "服务层版本校验 " + uuid.uuid4().hex[:8])
    async with AsyncSessionLocal() as db:
        with pytest.raises(ToolError) as excinfo:
            await create_ticket_from_draft(db, ctx, draft["id"], bad_version)
        await db.rollback()
    assert excinfo.value.code in {"missing_expected_version", "invalid_expected_version"}
    assert await _ticket_count(draft["id"]) == 0


async def test_correct_version_creates_ticket_matching_that_version(client):
    """确认 v2 后，工单内容必须是 v2 的内容，而不是更早或更新的版本。"""
    token = await token_for("customer_a2")
    draft = await _make_draft("customer_a2", "原始描述 " + uuid.uuid4().hex[:8])

    saved = await client.patch(
        f"/api/tickets/drafts/{draft['id']}",
        headers=auth(token),
        json={"description": "确认后应写入的 v2 描述", "expected_version": draft["version"]},
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["version"] == 2

    confirmed = await client.post(
        "/api/tickets/confirm",
        headers=auth(token),
        json={"draft_id": draft["id"], "expected_version": 2},
    )
    assert confirmed.status_code == 200, confirmed.text
    body = confirmed.json()
    assert body["already_created"] is False
    assert body["ticket"]["description"] == "确认后应写入的 v2 描述"

    row = await _draft_row(draft["id"])
    assert row is not None and row.status == "confirmed" and row.version == 2

    async with AsyncSessionLocal() as db:
        # 创建工单与写状态事件在同一事务中提交
        event_count = await db.scalar(
            select(func.count())
            .select_from(TicketEvent)
            .where(TicketEvent.ticket_id == body["ticket"]["id"])
        )
    assert event_count == 1


async def test_stale_version_after_edit_cannot_confirm_new_content(client):
    token = await token_for("customer_a2")
    draft = await _make_draft("customer_a2", "旧版本确认 " + uuid.uuid4().hex[:8])

    saved = await client.patch(
        f"/api/tickets/drafts/{draft['id']}",
        headers=auth(token),
        json={"description": "修改后的新内容", "expected_version": draft["version"]},
    )
    assert saved.status_code == 200

    stale = await client.post(
        "/api/tickets/confirm",
        headers=auth(token),
        json={"draft_id": draft["id"], "expected_version": draft["version"]},
    )
    assert stale.status_code == 409, stale.text
    assert await _ticket_count(draft["id"]) == 0


async def test_other_users_draft_cannot_be_confirmed_even_with_version(client):
    draft = await _make_draft("customer_a2", "他人草稿 " + uuid.uuid4().hex[:8])
    other_token = await token_for("customer_b")
    response = await client.post(
        "/api/tickets/confirm",
        headers=auth(other_token),
        json={"draft_id": draft["id"], "expected_version": draft["version"]},
    )
    assert response.status_code == 409
    assert "无权" in response.json()["detail"]
    assert await _ticket_count(draft["id"]) == 0


async def test_repeat_confirm_same_version_is_idempotent_and_flagged(client):
    """同一次操作重复提交：第二次明确标记 already_created=True。"""
    token = await token_for("customer_a2")
    draft = await _make_draft("customer_a2", "重复确认标记 " + uuid.uuid4().hex[:8])
    payload = {"draft_id": draft["id"], "expected_version": draft["version"]}

    first = await client.post("/api/tickets/confirm", headers=auth(token), json=payload)
    second = await client.post("/api/tickets/confirm", headers=auth(token), json=payload)
    assert first.status_code == 200 and second.status_code == 200
    assert first.json()["already_created"] is False
    assert second.json()["already_created"] is True
    assert first.json()["ticket"]["id"] == second.json()["ticket"]["id"]


async def test_edit_and_confirm_race_keeps_content_consistent(client):
    """编辑和确认同时发生：只能一方成功，工单内容必须对应被确认的版本。"""
    token = await token_for("customer_a2")
    draft = await _make_draft("customer_a2", "编辑确认竞争 " + uuid.uuid4().hex[:8])
    original = draft["description"]

    edit_response, confirm_response = await asyncio.gather(
        client.patch(
            f"/api/tickets/drafts/{draft['id']}",
            headers=auth(token),
            json={"description": "竞争中的编辑内容", "expected_version": draft["version"]},
        ),
        client.post(
            "/api/tickets/confirm",
            headers=auth(token),
            json={"draft_id": draft["id"], "expected_version": draft["version"]},
        ),
    )

    edit_ok = edit_response.status_code == 200
    confirm_ok = confirm_response.status_code == 200
    assert edit_ok + confirm_ok == 1, (edit_response.text, confirm_response.text)

    row = await _draft_row(draft["id"])
    assert row is not None
    tickets = await _ticket_count(draft["id"])

    if confirm_ok:
        assert edit_response.status_code == 409
        assert row.status == "confirmed" and row.version == draft["version"]
        assert tickets == 1
        async with AsyncSessionLocal() as db:
            ticket = await db.scalar(select(Ticket).where(Ticket.draft_id == draft["id"]))
        assert ticket is not None and ticket.description == original
    else:
        assert confirm_response.status_code == 409
        assert row.status == "draft" and row.version == draft["version"] + 1
        assert row.description == "竞争中的编辑内容"
        assert tickets == 0


async def _cancel_worker(draft_id: str, version: int) -> str:
    """独立会话取消草稿，用于和 API 确认请求并发。"""
    from app.services.tickets import cancel_draft

    ctx = await ctx_for("customer_a2")
    async with AsyncSessionLocal() as db:
        try:
            await cancel_draft(db, ctx, draft_id, expected_version=version)
            await db.commit()
            return "cancelled"
        except Exception:
            await db.rollback()
            return "blocked"


async def test_cancel_and_confirm_race_has_single_winner(client):
    """取消和确认同时发生：要么取消成功且无工单，要么确认成功且草稿已确认。"""
    token = await token_for("customer_a2")
    draft = await _make_draft("customer_a2", "取消确认竞争 " + uuid.uuid4().hex[:8])

    confirm_response, cancel_outcome = await asyncio.gather(
        client.post(
            "/api/tickets/confirm",
            headers=auth(token),
            json={"draft_id": draft["id"], "expected_version": draft["version"]},
        ),
        _cancel_worker(draft["id"], draft["version"]),
    )

    row = await _draft_row(draft["id"])
    tickets = await _ticket_count(draft["id"])
    assert row is not None
    if cancel_outcome == "cancelled":
        # 取消先赢：确认必须失败，数据库里不能有工单
        assert confirm_response.status_code == 409, confirm_response.text
        assert row.status == "cancelled" and tickets == 0
    else:
        # 确认先赢：取消被拒绝（或数据库忙），草稿与工单保持一致
        assert confirm_response.status_code == 200, confirm_response.text
        assert row.status == "confirmed" and tickets == 1


async def test_two_edits_same_version_only_one_wins(client):
    """两个请求基于同一版本修改：只有一个成功，另一个收到 409。"""
    token = await token_for("customer_a2")
    draft = await _make_draft("customer_a2", "双编辑竞争 " + uuid.uuid4().hex[:8])

    first, second = await asyncio.gather(
        client.patch(
            f"/api/tickets/drafts/{draft['id']}",
            headers=auth(token),
            json={"description": "编辑 A", "expected_version": draft["version"]},
        ),
        client.patch(
            f"/api/tickets/drafts/{draft['id']}",
            headers=auth(token),
            json={"description": "编辑 B", "expected_version": draft["version"]},
        ),
    )
    statuses = sorted([first.status_code, second.status_code])
    assert statuses == [200, 409], (first.text, second.text)

    row = await _draft_row(draft["id"])
    assert row is not None and row.version == draft["version"] + 1
    assert row.description in {"编辑 A", "编辑 B"}


async def test_chat_without_seen_version_cannot_confirm_latest(client):
    """历史会话没有记录“用户看到的版本”时，不能用数据库最新版本代填确认。"""
    import json

    from app.models import ChatSession

    token = await token_for("customer_a2")
    session = await client.post(
        "/api/chat/sessions", headers=auth(token), json={"title": "缺版本会话"}
    )
    session_id = session.json()["id"]
    created = await client.post(
        f"/api/chat/sessions/{session_id}/messages",
        headers=auth(token),
        json={
            "content": "转人工。v2.3 新建商品保存提示编码重复，已尝试去掉空格，仍然报错。",
            "product_version": "v2.3",
        },
    )
    assert created.status_code == 200, created.text
    draft = created.json().get("draft")
    assert draft and draft["status"] == "draft", created.text

    # 模拟旧版本会话：有草稿 ID，但没有 pending_draft_version
    async with AsyncSessionLocal() as db:
        row = await db.get(ChatSession, session_id)
        context = json.loads(row.context_json or "{}")
        context.pop("pending_draft_version", None)
        row.context_json = json.dumps(context, ensure_ascii=False)
        await db.commit()

    blocked = await client.post(
        f"/api/chat/sessions/{session_id}/messages",
        headers=auth(token),
        json={"content": "确认创建"},
    )
    assert blocked.status_code == 200, blocked.text
    body = blocked.json()
    assert body.get("ticket") is None
    # 必须把最新草稿重新展示给用户，而不是静默用最新版本创建
    assert body.get("draft") and body["draft"]["id"] == draft["id"]
    assert await _ticket_count(draft["id"]) == 0

    # 草稿已展示，用户再次明确确认后才创建
    confirmed = await client.post(
        f"/api/chat/sessions/{session_id}/messages",
        headers=auth(token),
        json={"content": "确认创建"},
    )
    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json().get("ticket") is not None


async def test_uncommitted_ticket_is_invisible_and_rollback_leaves_draft(client):
    """工单、事件与草稿状态同属一个事务：回滚后不留下任何“已创建”痕迹。"""
    token = await token_for("customer_a2")
    draft = await _make_draft("customer_a2", "事务回滚 " + uuid.uuid4().hex[:8])
    ctx = await ctx_for("customer_a2")

    async with AsyncSessionLocal() as db:
        ticket, already = await create_ticket_from_draft(
            db, ctx, draft["id"], draft["version"]
        )
        assert already is False
        assert ticket.ticket_no
        # 尚未提交：其他会话看不到这张工单
        assert await _ticket_count(draft["id"]) == 0
        await db.rollback()

    assert await _ticket_count(draft["id"]) == 0
    row = await _draft_row(draft["id"])
    assert row is not None and row.status == "draft"
