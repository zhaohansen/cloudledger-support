# tests/test_qa_stream.py
# 覆盖：模型回答的引用校验与降级、真实 token 流式转发、写操作幂等。
# 模型客户端全部是测试替身，不发真实网络请求、不产生费用。

from __future__ import annotations

import json
import uuid

from app.db import AsyncSessionLocal
from app.services import qa as qa_service
from app.services.llm import LLMServiceError, LLMTimeout
from tests.helpers import auth, ctx_for, token_for

QUESTION = "v2.2 报销单保存草稿后为什么部门可用预算被冻结？"


class FakeLLM:
    configured = True

    def __init__(self, replies: list[str] | None = None, stream_parts: list[str] | None = None):
        self.replies = list(replies or [])
        self.stream_parts = list(stream_parts or [])
        self.complete_calls = 0
        self.stream_calls = 0

    async def complete(self, messages):
        self.complete_calls += 1
        if not self.replies:
            raise LLMTimeout("测试替身：模拟模型超时")
        return self.replies.pop(0)

    async def stream(self, messages):
        self.stream_calls += 1
        for part in self.stream_parts:
            yield part


def _enable_fake_llm(monkeypatch, fake: FakeLLM) -> None:
    """把 qa 模块切到 openai 模式并注入测试替身（不读取任何真实密钥）。"""
    monkeypatch.setattr(qa_service.settings, "llm_mode", "openai")
    monkeypatch.setattr(qa_service.settings, "llm_api_key", "test-key-not-real")
    monkeypatch.setattr(qa_service, "get_llm_client", lambda: fake)


async def _answer_with(monkeypatch, fake: FakeLLM, question: str = QUESTION):
    _enable_fake_llm(monkeypatch, fake)
    async with AsyncSessionLocal() as db:
        ctx = await ctx_for("customer_a")
        return await qa_service.answer_question(db, ctx, question, product_version="v2.2")


# ── 回答评测：引用必须真实，缺失时不得伪装 ──────────────────────────────


async def test_llm_answer_with_valid_citation_is_kept(monkeypatch):
    fake = FakeLLM(replies=["第一步：删除或撤回报销草稿释放预算 [1]。"])
    result = await _answer_with(monkeypatch, fake)

    assert result.answer_mode == "llm"
    assert result.used_citation_indices == [1]
    assert "[1]" in result.answer
    assert result.citations and result.citations[0]["index"] == 1
    assert fake.complete_calls == 1


async def test_llm_invented_citation_is_removed_and_falls_back(monkeypatch):
    """模型编造 [99] 且重试后仍无有效引用：删除非法编号并退回资料摘录。"""
    fake = FakeLLM(replies=["先删除数据库再重装 [99]。", "还是照上面做 [99]。"])
    result = await _answer_with(monkeypatch, fake)

    assert result.answer_mode == "llm_uncited"
    assert "[99]" not in result.answer
    assert "先删除数据库" not in result.answer
    assert "资料摘录" in result.answer
    assert result.citations, "摘录降级仍应保留真实检索引用"
    assert fake.complete_calls == 2


async def test_llm_timeout_degrades_with_explicit_note(monkeypatch):
    fake = FakeLLM(replies=[])
    result = await _answer_with(monkeypatch, fake)

    assert result.answer_mode == "llm_degraded"
    assert "降级" in result.answer
    assert result.citations


# ── SSE：事件顺序与真实流式转发 ────────────────────────────────────────


async def _new_session(client, token: str) -> str:
    response = await client.post(
        "/api/chat/sessions", headers=auth(token), json={"title": "流式测试会话"}
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


async def _collect_events(client, token: str, session_id: str, content: str) -> list[dict]:
    events: list[dict] = []
    async with client.stream(
        "POST",
        f"/api/chat/sessions/{session_id}/stream",
        headers=auth(token),
        json={"content": content, "product_version": "v2.2"},
    ) as response:
        assert response.status_code == 200, await response.aread()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                events.append({"type": "done"})
                continue
            events.append(json.loads(payload))
    return events


async def test_sse_extractive_events_and_done(client):
    token = await token_for("customer_a")
    session_id = await _new_session(client, token)
    events = await _collect_events(client, token, session_id, QUESTION)
    types = [event["type"] for event in events]

    for expected in ("start", "meta", "retrieval", "citations", "token", "result", "saved", "done"):
        assert expected in types, f"缺少 SSE 事件 {expected}：{types}"
    assert types[-1] == "done"

    result = next(event for event in events if event["type"] == "result")
    assert result["streaming"] is False, "摘录模式不是逐 token 流式，应明确标注"
    assert result["citations"]

    saved = next(event for event in events if event["type"] == "saved")
    assert saved["stream_status"] == "complete"


async def test_sse_forwards_real_model_tokens(monkeypatch, client):
    fake = FakeLLM(stream_parts=["第一步：", "打开预算管理 → 部门预算。", " [1]"])
    _enable_fake_llm(monkeypatch, fake)
    token = await token_for("customer_a")
    session_id = await _new_session(client, token)
    events = await _collect_events(client, token, session_id, QUESTION)

    tokens = [event for event in events if event["type"] == "token"]
    assert len(tokens) >= 3, f"应逐块转发模型输出，实际：{tokens}"
    assert "".join(event["content"] for event in tokens).strip() == "第一步：打开预算管理 → 部门预算。 [1]"

    result = next(event for event in events if event["type"] == "result")
    assert result["streaming"] is True
    assert result["answer_mode"] == "llm"
    assert result["timings"]["first_token_ms"] is not None
    assert result["timings"]["first_token_ms"] <= result["timings"]["total_ms"]
    assert fake.stream_calls == 1

    # 完整回答按完整状态保存，引用对应真实片段
    messages = await client.get(
        f"/api/chat/sessions/{session_id}/messages", headers=auth(token)
    )
    assistant = [m for m in messages.json()["items"] if m["role"] == "assistant"][-1]
    assert assistant["meta"]["stream_status"] == "complete"
    assert "[1]" in assistant["content"]
    assert assistant["citations"]


async def test_sse_uncited_answer_is_flagged_not_presented_as_success(monkeypatch, client):
    """模型返回越界引用：正文已流出，但必须告警、去掉假引用并标记未验证。"""
    fake = FakeLLM(stream_parts=["先删除数据库再重装 [99]。"])
    _enable_fake_llm(monkeypatch, fake)
    token = await token_for("customer_a")
    session_id = await _new_session(client, token)
    events = await _collect_events(client, token, session_id, QUESTION)

    warning = next(event for event in events if event["type"] == "warning")
    assert warning["code"] == "citation_missing"
    result = next(event for event in events if event["type"] == "result")
    assert result["answer_mode"] == "llm_uncited"
    assert result["citations"] == []
    assert "[99]" not in result["answer"]

    messages = await client.get(
        f"/api/chat/sessions/{session_id}/messages", headers=auth(token)
    )
    assistant = [m for m in messages.json()["items"] if m["role"] == "assistant"][-1]
    assert assistant["meta"]["answer_mode"] == "llm_uncited"
    assert assistant["citations"] == []


async def test_sse_partial_invalid_citation_emits_visible_warning(monkeypatch, client):
    """部分引用合法、部分越界：保留合法引用，同时明确提示哪些编号被剔除。"""
    fake = FakeLLM(stream_parts=["第一步 [1]，第二步参考 [99]。"])
    _enable_fake_llm(monkeypatch, fake)
    token = await token_for("customer_a")
    session_id = await _new_session(client, token)
    events = await _collect_events(client, token, session_id, QUESTION)

    warning = next(event for event in events if event["type"] == "warning")
    assert warning["code"] == "citation_invalid"
    assert "99" in warning["message"]
    result = next(event for event in events if event["type"] == "result")
    assert result["answer_mode"] == "llm"
    assert result["citation_warnings"] == [99]
    assert "[99]" not in result["answer"]


class ExplodingStreamLLM(FakeLLM):
    async def stream(self, messages):
        self.stream_calls += 1
        yield "第一步：打开预算管理。"
        raise LLMServiceError("测试替身：流式中途失败")


async def test_sse_midstream_failure_is_saved_as_partial(monkeypatch, client):
    """中途异常：前端收到错误事件，数据库只保存带“回答未完成”标记的前缀。"""
    fake = ExplodingStreamLLM()
    _enable_fake_llm(monkeypatch, fake)
    token = await token_for("customer_a")
    session_id = await _new_session(client, token)
    events = await _collect_events(client, token, session_id, QUESTION)

    error = next(event for event in events if event["type"] == "error")
    assert error["partial"] is True
    result = next(event for event in events if event["type"] == "result")
    assert result["partial"] is True
    assert result["citations"] == []
    saved = next(event for event in events if event["type"] == "saved")
    assert saved["stream_status"] == "partial"

    messages = await client.get(
        f"/api/chat/sessions/{session_id}/messages", headers=auth(token)
    )
    assistant = [m for m in messages.json()["items"] if m["role"] == "assistant"][-1]
    assert assistant["meta"]["stream_status"] == "partial"
    assert "回答未完成" in assistant["content"]
    assert assistant["citations"] == []


# ── 写操作幂等：重复请求不产生重复副作用 ────────────────────────────────


async def _first_ticket_id(client, token: str) -> str:
    response = await client.get("/api/tickets", headers=auth(token))
    assert response.status_code == 200
    items = response.json()["items"]
    assert items, "演示数据应包含 customer_a 的工单"
    return items[0]["id"]


async def test_ticket_message_idempotency_key(client):
    token = await token_for("customer_a")
    ticket_id = await _first_ticket_id(client, token)
    key = f"test-message-{uuid.uuid4().hex}"
    content = f"重复发送测试消息 {key}"
    headers = {**auth(token), "Idempotency-Key": key}

    first = await client.post(
        f"/api/tickets/{ticket_id}/messages", headers=headers, json={"content": content}
    )
    second = await client.post(
        f"/api/tickets/{ticket_id}/messages", headers=headers, json={"content": content}
    )
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert first.json()["id"] == second.json()["id"]

    detail = await client.get(f"/api/tickets/{ticket_id}", headers=auth(token))
    contents = [message["content"] for message in detail.json()["messages"]]
    assert contents.count(content) == 1, "同一幂等键只能写入一条消息"


async def test_ticket_supplement_idempotency_key(client):
    token = await token_for("customer_a")
    ticket_id = await _first_ticket_id(client, token)
    key = f"test-supplement-{uuid.uuid4().hex}"
    marker = f"补充标记 {key}"
    headers = {**auth(token), "Idempotency-Key": key}

    first = await client.post(
        f"/api/tickets/{ticket_id}/supplement",
        headers=headers,
        json={"description": marker},
    )
    second = await client.post(
        f"/api/tickets/{ticket_id}/supplement",
        headers=headers,
        json={"description": marker},
    )
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text

    detail = await client.get(f"/api/tickets/{ticket_id}", headers=auth(token))
    assert detail.json()["description"].count(marker) == 1, "重复补充不能重复追加描述"
