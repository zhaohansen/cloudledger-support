# tests/test_chat.py
# 覆盖：无证据不编造、引用来自实际检索、会话不跨用户读取。

from __future__ import annotations

from tests.helpers import auth, token_for


async def _new_session(client, token: str) -> str:
    response = await client.post("/api/chat/sessions", headers=auth(token), json={"title": "测试会话"})
    assert response.status_code == 200
    return response.json()["id"]


async def test_no_evidence_does_not_fabricate(client):
    """14. 资料中没有依据时明确说明，不返回编造引用。"""
    token = await token_for("customer_a")
    session_id = await _new_session(client, token)

    response = await client.post(
        f"/api/chat/sessions/{session_id}/messages",
        headers=auth(token),
        json={"content": "量子纠缠通信设备的校准步骤是什么？"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer_mode"] == "no_evidence"
    assert body["citations"] == []
    assert "没有找到" in body["reply"]


async def test_citations_come_from_retrieved_documents(client):
    """15. 回答引用来自本次检索结果，编号连续且能对应资料与章节。"""
    token = await token_for("customer_a")
    session_id = await _new_session(client, token)

    response = await client.post(
        f"/api/chat/sessions/{session_id}/messages",
        headers=auth(token),
        json={"content": "v2.2 报销单保存草稿后为什么部门可用预算被冻结？", "product_version": "v2.2"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    citations = body["citations"]
    assert citations, "应命中 v2.2 预算控制相关资料"
    assert body["answer_mode"] in ("extractive", "llm")

    for index, citation in enumerate(citations, start=1):
        assert citation["index"] == index
        assert citation["document_id"]
        assert citation["title"]
        assert citation["snippet"]
        # 指定 v2.2 时只应引用 v2.2 或全版本资料
        assert citation["product_version"] in ("v2.2", "all")

    titles = " ".join(citation["title"] for citation in citations)
    assert "预算" in titles or "v2.2" in titles


async def test_chat_sessions_are_isolated_between_users(client):
    """16. 会话按用户隔离，其他用户读取返回 404。"""
    token_a = await token_for("customer_a")
    token_b = await token_for("customer_b")
    session_id = await _new_session(client, token_a)

    own = await client.get(f"/api/chat/sessions/{session_id}/messages", headers=auth(token_a))
    assert own.status_code == 200

    other = await client.get(f"/api/chat/sessions/{session_id}/messages", headers=auth(token_b))
    assert other.status_code == 404


async def test_old_tenant_session_hidden_after_tenant_change(client):
    from dataclasses import replace

    from app.db import AsyncSessionLocal
    from app.services.chat import get_session, list_sessions
    from fastapi import HTTPException
    import pytest
    from tests.helpers import ctx_for

    user = await ctx_for("customer_a")
    session_id = await _new_session(client, await token_for(user.username))
    other_tenant = await ctx_for("customer_b")
    moved_user = replace(user, tenant_id=other_tenant.tenant_id)
    assert moved_user.tenant_id != user.tenant_id
    async with AsyncSessionLocal() as db:
        sessions = await list_sessions(db, moved_user)
        assert session_id not in [session.id for session in sessions]
        with pytest.raises(HTTPException) as exc:
            await get_session(db, moved_user, session_id)
        assert exc.value.status_code == 404


async def test_chat_pagination_and_idempotent_message(client):
    token = await token_for("customer_a")
    session_id = await _new_session(client, token)
    url = f"/api/chat/sessions/{session_id}/messages"
    headers = {**auth(token), "Idempotency-Key": "pagination-turn"}
    first = await client.post(url, headers=headers, json={"content": "你好"})
    replay = await client.post(url, headers=headers, json={"content": "你好"})
    conflict = await client.post(url, headers=headers, json={"content": "谢谢"})
    assert first.status_code == replay.status_code == 200, (first.text, replay.text)
    assert first.json() == replay.json()
    assert conflict.status_code == 409

    first_page = (await client.get(url, headers=auth(token), params={"limit": 1})).json()
    assert [message["role"] for message in first_page["items"]] == ["assistant"]
    older_page = (await client.get(
        url, headers=auth(token), params={"limit": 1, "before": first_page["next_before"]}
    )).json()
    assert [message["role"] for message in older_page["items"]] == ["user"]
    assert older_page["next_before"] is None


async def test_streamed_turn_can_be_replayed_without_duplicate_messages(client):
    token = await token_for("customer_a")
    session_id = await _new_session(client, token)
    base = f"/api/chat/sessions/{session_id}"
    headers = {**auth(token), "Idempotency-Key": "stream-test"}
    payload = {"content": "你好"}
    first = await client.post(f"{base}/stream", headers=headers, json=payload)
    replay = await client.post(f"{base}/stream", headers=headers, json=payload)
    assert first.status_code == replay.status_code == 200, (first.text, replay.text)
    assert '"type": "saved"' in first.text and '"type": "saved"' in replay.text
    history = await client.get(f"{base}/messages", headers=auth(token))
    assert [message["role"] for message in history.json()["items"]] == ["user", "assistant"]


async def test_delete_session_owner_only(client):
    """用户可删除自己的会话，他人尝试删除返回 404。"""
    token_a = await token_for("customer_a")
    token_b = await token_for("customer_b")
    session_id = await _new_session(client, token_a)

    other = await client.delete(f"/api/chat/sessions/{session_id}", headers=auth(token_b))
    assert other.status_code == 404

    ok = await client.delete(f"/api/chat/sessions/{session_id}", headers=auth(token_a))
    assert ok.status_code == 200
    assert ok.json()["deleted"] is True

    after = await client.get(f"/api/chat/sessions/{session_id}/messages", headers=auth(token_a))
    assert after.status_code == 404
