# tests/test_multi_turn.py
# 多轮追问：历史只用于理解指代，不作为检索证据；话题切换不携带旧版本。

from __future__ import annotations

from app.services import qa as qa_service
from app.services.qa import _build_messages, _normalize_history, build_retrieval_query
from app.services.retrieval import RetrievedChunk
from tests.helpers import auth, token_for

QUESTION = "v2.2 报销单保存草稿后为什么部门可用预算被冻结？"


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        title="预算编制与额度控制说明",
        doc_type="faq",
        doc_version="1",
        product_version="v2.2",
        section="草稿释放",
        chunk_index=0,
        content="在预算管理 → 草稿占用清理中执行释放，或删除未提交的报销草稿。",
        score=0.9,
        dense_score=0.8,
        lexical_score=0.7,
    )


def test_anaphora_followup_uses_last_user_message_without_old_version():
    history = [
        {"role": "user", "content": QUESTION},
        {"role": "assistant", "content": "请先删除或撤回报销草稿 [1]。"},
    ]
    query = build_retrieval_query("那旧版本呢？", history)

    assert "报销单保存草稿后为什么部门可用预算被冻结" in query
    assert "v2.2" not in query and "2.2" not in query, "旧版本追问不应被重新钉回当前版本"
    assert "撤回报销草稿" not in query, "助手历史回答不能当成检索上下文"


def test_second_step_followup_keeps_previous_user_question():
    history = [{"role": "user", "content": "v2.3 报销超预算怎么生成追加单？"}]
    query = build_retrieval_query("第二个步骤怎么操作？", history)

    assert "报销超预算怎么生成追加单" in query
    assert "第二个步骤" in query


def test_topic_switch_does_not_carry_old_question_or_version():
    history = [
        {"role": "user", "content": QUESTION},
        {"role": "assistant", "content": "请先删除或撤回报销草稿 [1]。"},
    ]
    question = "差旅报销单怎么反审核？"
    assert build_retrieval_query(question, history) == question


def test_long_new_question_is_not_merged_with_history():
    history = [{"role": "user", "content": QUESTION}]
    question = "我在差旅报销单里点了反审核以后提示存在付款批次，应该怎么处理？"
    assert build_retrieval_query(question, history) == question


def test_history_is_trimmed_and_explicitly_not_evidence():
    history = [
        {"role": "user", "content": "旧" * 500},
        {"role": "assistant", "content": "答" * 500},
    ]
    trimmed = _normalize_history(history)
    assert trimmed and all(len(content) <= 400 for _, content in trimmed)

    messages = _build_messages("那旧版本呢？", [_chunk()], "v2.2", trimmed)
    system_prompt, user_prompt = messages[0][1], messages[-1][1]
    assert [role for role, _ in messages[1:-1]] == ["human", "ai"]
    assert "资料片段" in system_prompt
    assert "【历史对话（仅用于理解指代，不作为事实依据）】" in user_prompt
    assert "助手：" in user_prompt
    assert "【资料片段】" in user_prompt
    assert "草稿占用清理" in user_prompt


class CapturingLLM:
    """记录模型实际收到的消息，用于验证历史是否真的接入。"""

    configured = True

    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []

    async def complete(self, messages):
        self.calls.append(list(messages))
        return "第一步：删除或撤回报销草稿以释放预算 [1]。"

    async def stream(self, messages):
        self.calls.append(list(messages))
        yield "第一步：删除或撤回报销草稿以释放预算 [1]。"


async def test_second_turn_really_sends_history_to_model(monkeypatch, client):
    fake = CapturingLLM()
    monkeypatch.setattr(qa_service.settings, "llm_mode", "openai")
    monkeypatch.setattr(qa_service.settings, "llm_api_key", "test-key-not-real")
    monkeypatch.setattr(qa_service, "get_llm_client", lambda: fake)

    token = await token_for("customer_a")
    created = await client.post(
        "/api/chat/sessions", headers=auth(token), json={"title": "多轮追问测试"}
    )
    session_id = created.json()["id"]

    first = await client.post(
        f"/api/chat/sessions/{session_id}/messages",
        headers=auth(token),
        json={"content": QUESTION, "product_version": "v2.2"},
    )
    assert first.status_code == 200, first.text
    assert fake.calls, "第一轮有证据时应调用模型"

    second = await client.post(
        f"/api/chat/sessions/{session_id}/messages",
        headers=auth(token),
        json={"content": "那旧版本呢？"},
    )
    assert second.status_code == 200, second.text
    assert len(fake.calls) >= 2, "第二轮有证据时也应调用模型"

    last_call = fake.calls[-1]
    assert [role for role, _ in last_call[1:-1]] == ["human", "ai"]
    user_prompt = last_call[-1][1]
    assert "【历史对话（仅用于理解指代，不作为事实依据）】" in user_prompt
    assert "报销单保存草稿后为什么部门可用预算被冻结" in user_prompt
    assert "助手：" in user_prompt
    assert "【资料片段】" in user_prompt
