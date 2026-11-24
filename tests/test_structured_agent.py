# tests/test_structured_agent.py
# 覆盖：模型结构化输出的白名单/枚举校验，以及"有草稿时模型不能改写意图"的确定性约束。
# 模型客户端均为测试替身。

from __future__ import annotations

import uuid

from app.agent import nodes, structured
from app.agent.nodes import AgentRuntime
from app.agent.state import AgentState
from app.agent.structured import IntentDecision, TicketFieldSuggestion
from app.db import AsyncSessionLocal
from app.models import TicketDraft
from tests.helpers import ctx_for


class FakeJSONClient:
    configured = True

    def __init__(self, payloads: list[dict]):
        self.payloads = list(payloads)
        self.calls = 0

    async def complete_json(self, messages):
        self.calls += 1
        if not self.payloads:
            raise AssertionError("不应再调用模型")
        return self.payloads.pop(0)


def _patch_client(monkeypatch, fake: FakeJSONClient) -> None:
    monkeypatch.setattr(structured, "get_llm_client", lambda: fake)


async def test_model_cannot_emit_confirm_or_cancel(monkeypatch):
    """confirm/cancel 不在模型意图白名单里，校验失败即回退规则。"""
    fake = FakeJSONClient([{"intent": "confirm", "confidence": 0.99}])
    _patch_client(monkeypatch, fake)

    decision = await structured.classify_intent_with_model("确认创建")
    assert decision is None
    assert fake.calls == 1


async def test_model_low_confidence_is_ignored(monkeypatch):
    fake = FakeJSONClient([{"intent": "human", "confidence": 0.2}])
    _patch_client(monkeypatch, fake)

    assert await structured.classify_intent_with_model("帮我看看这个") is None


async def test_model_intent_valid_value_is_accepted(monkeypatch):
    fake = FakeJSONClient([{"intent": "issue", "confidence": 0.9}])
    _patch_client(monkeypatch, fake)

    decision = await structured.classify_intent_with_model("财务复核那边说预算流水对不上")
    assert decision == IntentDecision(intent="issue", confidence=0.9)


async def test_invalid_category_is_rejected(monkeypatch):
    """模型返回枚举外的类别/优先级时直接拒绝，不能写进草稿。"""
    fake = FakeJSONClient(
        [{"category": "hacker", "priority": "high", "confidence": 0.99}]
    )
    _patch_client(monkeypatch, fake)

    assert await structured.suggest_ticket_fields("随便描述") is None


async def test_model_cannot_override_intent_when_pending_draft(monkeypatch):
    """存在待确认草稿时，模型不得把"不要提交"改成可执行意图。"""
    fake = FakeJSONClient([{"intent": "human", "confidence": 1.0}])
    _patch_client(monkeypatch, fake)

    async def _unexpected(text: str):
        raise AssertionError("有草稿时不应调用模型改写意图")

    monkeypatch.setattr(nodes, "classify_intent_with_model", _unexpected)

    ctx = await ctx_for("customer_a")
    runtime = AgentRuntime(db=None, user=ctx, session_id="test-session")  # type: ignore[arg-type]
    state: AgentState = {
        "user_message": "不要提交",
        "pending_draft": {"id": "draft-x", "version": 1},
    }
    result = await nodes.classify(runtime, state)

    assert result["intent"] == "cancel", "否定表达必须由规则判定为取消"
    assert result["intent_source"] == "rule"
    assert fake.calls == 0, "有草稿时不允许模型改写意图"


async def test_handle_human_uses_model_fields_only_as_draft_suggestion(monkeypatch):
    """规则无把握时模型只建议类别/优先级，草稿仍由后端创建、等待用户确认。"""
    fake = FakeJSONClient([{"category": "budget", "priority": "high", "confidence": 0.9}])
    # handle_human 直接引用了 suggest_ticket_fields，替换这个符号
    async def fake_suggest(description: str):
        fake.calls += 1
        return TicketFieldSuggestion(category="budget", priority="high", confidence=0.9)

    monkeypatch.setattr(nodes, "suggest_ticket_fields", fake_suggest)

    marker = uuid.uuid4().hex[:8]
    ctx = await ctx_for("customer_a")
    async with AsyncSessionLocal() as db:
        runtime = AgentRuntime(db=db, user=ctx, session_id=f"structured-test-{marker}")
        state: AgentState = {
            "user_message": f"财务那边复核完之后页面显示不太对 {marker}",
            "facts": {
                "product_version": "v2.2",
                "description": f"财务那边复核完之后页面显示不太对 {marker}，需要帮忙看看",
            },
        }
        result = await nodes.handle_human(runtime, state)
        await db.commit()

        draft = await db.get(TicketDraft, result["draft"]["id"])

    assert result["draft"]["category"] == "budget"
    assert result["draft"]["priority"] == "high"
    assert result["draft"]["status"] == "draft", "模型只建议草稿，不能直接创建工单"
    assert draft is not None and draft.user_id == ctx.user_id
    assert fake.calls == 1
