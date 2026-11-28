# tests/test_tickets.py
# 覆盖：草稿未确认不建单、重复确认幂等、工具失败不报成功、状态流转与客户查看。

from __future__ import annotations

import uuid

from sqlalchemy import func, select

from app.db import AsyncSessionLocal
from app.models import Ticket, TicketDraft
from app.services.tickets import get_or_create_draft
from tests.helpers import auth, ctx_for, token_for


async def _make_draft(username: str, description: str) -> dict:
    """直接通过服务层创建草稿，绕开自然语言意图识别，让测试聚焦幂等与权限。"""
    ctx = await ctx_for(username)
    async with AsyncSessionLocal() as db:
        draft = await get_or_create_draft(
            db,
            ctx,
            session_id=None,
            title="测试工单草稿 " + description[:20],
            description=description,
            product_version="v2.3",
            error_message="模拟报错",
            tried_steps="模拟已尝试步骤",
            category="other",
            priority="medium",
        )
        await db.commit()
        return {"id": draft.id, "title": draft.title, "version": draft.version}


async def test_draft_does_not_create_ticket_before_confirm(client):
    """9. 只生成草稿时，工单数量不变；明确确认后才出现工单。"""
    token = await token_for("customer_a2")
    before = (await client.get("/api/tickets", headers=auth(token))).json()["items"]
    draft = await _make_draft("customer_a2", "未确认测试 " + uuid.uuid4().hex[:8])

    after_draft = (await client.get("/api/tickets", headers=auth(token))).json()["items"]
    assert len(after_draft) == len(before)

    confirmed = await client.post(
        "/api/tickets/confirm",
        headers=auth(token),
        json={"draft_id": draft["id"], "expected_version": draft["version"]},
    )
    assert confirmed.status_code == 200, confirmed.text
    ticket = confirmed.json()["ticket"]
    assert ticket["ticket_no"].startswith("YZ")
    assert ticket["status"] == "pending"


async def test_repeated_confirm_creates_only_one_ticket(client):
    """10. 重复确认同一草稿只创建一张工单，返回同一单号。"""
    token = await token_for("customer_a")
    description = "重复确认测试 " + uuid.uuid4().hex[:8]
    draft = await _make_draft("customer_a", description)

    payload = {"draft_id": draft["id"], "expected_version": draft["version"]}
    first = await client.post("/api/tickets/confirm", headers=auth(token), json=payload)
    second = await client.post("/api/tickets/confirm", headers=auth(token), json=payload)
    assert first.status_code == 200
    assert second.status_code == 200
    first_ticket = first.json()["ticket"]
    second_ticket = second.json()["ticket"]
    assert first_ticket["ticket_no"] == second_ticket["ticket_no"]
    assert second.json()["already_created"] is True

    async with AsyncSessionLocal() as db:
        count = await db.scalar(
            select(func.count()).select_from(Ticket).where(Ticket.draft_id == draft["id"])
        )
    assert count == 1


async def test_tool_failure_returns_error_not_success(client):
    """11. 草稿不存在时确认接口返回 409，不返回工单号。"""
    token = await token_for("customer_a")
    response = await client.post(
        "/api/tickets/confirm",
        headers=auth(token),
        json={"draft_id": "not-exist-draft", "expected_version": 1},
    )
    assert response.status_code == 409
    body = response.json()
    assert "detail" in body and body["detail"]
    assert "ticket" not in body


async def test_status_flow_and_customer_view(client):
    """12. 客服推进状态、客户查看回复并确认解决；非法流转被拒绝。"""
    async with AsyncSessionLocal() as db:
        ticket = await db.scalar(
            select(Ticket).where(Ticket.title == "Excel 导入费用科目提示编码重复")
        )
    assert ticket is not None

    agent2 = await token_for("agent2")
    customer_b = await token_for("customer_b")

    started = await client.patch(
        f"/api/tickets/{ticket.id}/status",
        headers=auth(agent2),
        json={"status": "processing", "note": "客服已接手"},
    )
    assert started.status_code == 200, started.text

    reply = await client.post(
        f"/api/tickets/{ticket.id}/messages",
        headers=auth(agent2),
        json={"content": "请提供报错截图和导入模板的前两行，我们核对编码规则。"},
    )
    assert reply.status_code == 200

    waiting = await client.patch(
        f"/api/tickets/{ticket.id}/status",
        headers=auth(agent2),
        json={"status": "waiting_customer", "note": "等待客户确认真实编码"},
    )
    assert waiting.status_code == 200

    detail = await client.get(f"/api/tickets/{ticket.id}", headers=auth(customer_b))
    assert detail.status_code == 200
    detail_body = detail.json()
    assert detail_body["status"] == "waiting_customer"
    assert any("报错截图" in message["content"] for message in detail_body["messages"])
    assert any(event["to_status"] == "waiting_customer" for event in detail_body["events"])

    closed = await client.patch(
        f"/api/tickets/{ticket.id}/status",
        headers=auth(customer_b),
        json={"status": "closed", "note": "客户确认已解决"},
    )
    assert closed.status_code == 200
    assert closed.json()["status"] == "closed"

    # 客服角色不允许重新打开已关闭工单
    reopen = await client.patch(
        f"/api/tickets/{ticket.id}/status",
        headers=auth(agent2),
        json={"status": "processing", "note": "尝试重新打开"},
    )
    assert reopen.status_code == 409


async def test_customer_cannot_skip_to_closed_from_processing(client):
    """13. 客户只能从「待客户确认」关闭工单，不能直接关闭处理中工单。"""
    async with AsyncSessionLocal() as db:
        ticket = await db.scalar(select(Ticket).where(Ticket.title == "v2.2 报销单保存草稿后部门可用预算被冻结"))
    assert ticket is not None

    customer_a = await token_for("customer_a")
    # 把演示工单从待客户确认改为处理中（模拟客户反馈未解决）
    reopen = await client.patch(
        f"/api/tickets/{ticket.id}/status",
        headers=auth(customer_a),
        json={"status": "processing", "note": "问题仍未解决"},
    )
    assert reopen.status_code == 200

    illegal_close = await client.patch(
        f"/api/tickets/{ticket.id}/status",
        headers=auth(customer_a),
        json={"status": "closed", "note": "直接关闭"},
    )
    assert illegal_close.status_code == 403

    async with AsyncSessionLocal() as db:
        drafts = (await db.execute(select(TicketDraft))).scalars().all()
    assert drafts  # 演示数据至少有一份草稿记录


async def test_reply_draft_is_advisor_only_and_does_not_send_message(client, monkeypatch):
    from app.services import qa

    customer = auth(await token_for("customer_b"))
    wrong_agent = auth(await token_for("agent1"))
    advisor = auth(await token_for("agent2"))
    draft = await _make_draft("customer_b", "库存编码导入需复核" + uuid.uuid4().hex[:8])
    confirmed = await client.post(
        "/api/tickets/confirm", headers=customer,
        json={"draft_id": draft["id"], "expected_version": draft["version"]},
    )
    assert confirmed.status_code == 200
    ticket = confirmed.json()["ticket"]
    path = f"/api/tickets/{ticket['id']}"
    before = (await client.get(path, headers=advisor)).json()
    seen = []

    async def fake_answer(db, user, question, **scope):
        seen.append((question, scope))
        return qa.AnswerResult(
            answer="请核对编码格式 [1]", citations=[{"index": 1, "title": "导入指南"}],
            answer_mode="extractive",
        )

    monkeypatch.setattr(qa, "answer_question", fake_answer)
    assert (await client.post(path + "/reply-draft", headers=customer)).status_code == 403
    assert (await client.post(path + "/reply-draft", headers=wrong_agent)).status_code == 403
    response = await client.post(path + "/reply-draft", headers=advisor)
    assert response.status_code == 200
    assert response.json()["draft"] == "请核对编码格式 [1]"
    assert seen == [(ticket["title"], {"product_version": "v2.3", "tenant_scope": ticket["tenant_id"]})]
    after = (await client.get(path, headers=advisor)).json()
    assert after["messages"] == before["messages"]
    assert after["status"] == before["status"]

    async def no_evidence(*args, **kwargs):
        return qa.AnswerResult(answer="暂无可用资料", citations=[], answer_mode="no_evidence")

    monkeypatch.setattr(qa, "answer_question", no_evidence)
    response = await client.post(path + "/reply-draft", headers=advisor)
    assert response.json()["draft"] == ""
    assert response.json()["message"] == "暂无可用资料"
