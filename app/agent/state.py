# app/agent/state.py
# LangGraph 的状态定义。状态只放可序列化的普通数据，不放 ORM 对象和数据库连接。

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    user_message: str
    intent: str
    # intent 来自规则还是模型建议，便于排查误判；写操作不依赖这个字段
    intent_source: str
    facts: dict[str, Any]
    # 最近若干轮历史，仅用于理解指代；事实依据仍只能来自检索片段
    history: list[dict]
    reply: str
    citations: list[dict]
    answer_mode: str
    # 检索后端、耗时、引用校验等可观测信息，随消息一起持久化
    answer_meta: dict[str, Any]
    draft: dict | None
    ticket: dict | None
    pending_draft: dict | None
    # 会话里记录的“用户上次看到的草稿版本”与数据库不一致时为 True
    pending_draft_stale: bool
    last_ticket_no: str | None
    tool_ok: bool
    error: str
