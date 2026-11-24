# app/agent/nodes.py
# Agent 各节点：识别诉求 → 补充信息 → 检索回答 → 生成草稿 → 明确确认 → 调工具建单。
# 节点内所有权限判断都调用 services，不靠提示词。

from __future__ import annotations

import re
from dataclasses import dataclass

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.intent import (
    TICKET_NO_RE,
    build_title,
    classify_intent,
    guess_category,
    guess_priority,
    merge_facts,
    missing_ticket_fields,
)
from app.agent.state import AgentState
from app.agent.structured import classify_intent_with_model, suggest_ticket_fields
from app.deps import UserContext
from app.logging_config import get_logger
from app.services import tickets as ticket_service
from app.services import tools as tool_service
from app.services.qa import answer_question

logger = get_logger(__name__)

STATUS_LABELS = {
    "pending": "待处理",
    "processing": "处理中",
    "waiting_customer": "待客户确认",
    "closed": "已关闭",
}
CATEGORY_LABELS = {
    "import": "批量导入",
    "budget": "预算控制",
    "invoice": "数电发票",
    "reimbursement": "费用报销",
    "payment": "银企付款",
    "closing": "月结总账",
    "permission": "账号权限",
    "other": "其他",
}
PRIORITY_LABELS = {"high": "高", "medium": "中", "low": "低"}


@dataclass
class AgentRuntime:
    """一次会话调用的运行环境（数据库会话与可信用户身份）。"""

    db: AsyncSession
    user: UserContext
    session_id: str
    request_id: str = "-"


def _labels(ticket: dict) -> str:
    return (
        f"{STATUS_LABELS.get(ticket.get('status'), ticket.get('status'))} / "
        f"{CATEGORY_LABELS.get(ticket.get('category'), ticket.get('category'))} / "
        f"优先级{PRIORITY_LABELS.get(ticket.get('priority'), ticket.get('priority'))}"
    )


async def load_context(state: AgentState) -> dict:
    """状态兜底：调用方可能只传了 user_message。"""
    return {
        "facts": dict(state.get("facts") or {}),
        "citations": state.get("citations") or [],
        "pending_draft": state.get("pending_draft"),
        "pending_draft_stale": bool(state.get("pending_draft_stale")),
        "last_ticket_no": state.get("last_ticket_no"),
    }


async def classify(runtime: AgentRuntime, state: AgentState) -> dict:
    facts = dict(state.get("facts") or {})
    message = state.get("user_message", "")
    intent = classify_intent(
        message,
        has_pending_draft=bool(state.get("pending_draft")),
        last_ticket_no=state.get("last_ticket_no"),
    )
    intent_source = "rule"

    # 规则只对"明确指令"负责：confirm/cancel/edit_draft 与工单号等确定性分支不交给模型，
    # 避免模型把"不要提交"理解成确认。只有规则无法判断的闲聊/自然语言才请模型建议意图。
    if intent == "chitchat" and not state.get("pending_draft"):
        decision = await classify_intent_with_model(message)
        if decision is not None:
            intent = decision.intent.value
            intent_source = "model"

    facts = merge_facts(facts, message, intent)
    logger.info(
        "agent.intent", intent=intent, source=intent_source, user_id=runtime.user.user_id
    )
    return {"intent": intent, "intent_source": intent_source, "facts": facts}


async def answer(runtime: AgentRuntime, state: AgentState) -> dict:
    facts = state.get("facts") or {}
    question = facts.get("description") or state.get("user_message", "")
    version = facts.get("product_version")
    if facts.get("version_unknown"):
        version = None
    result = await answer_question(
        runtime.db,
        runtime.user,
        question,
        product_version=version,
        history=state.get("history") or [],
    )
    return {
        "reply": result.answer,
        "citations": result.citations,
        "answer_mode": result.answer_mode,
        "answer_meta": {
            "answer_mode": result.answer_mode,
            "search_reason": result.search_reason,
            "version_conflict": result.version_conflict,
            "matched_versions": result.matched_versions,
            "retrieval_ms": round(result.retrieval_ms, 1),
            "embedding_backend": result.embedding_backend,
            "retrieval_backend": result.retrieval_backend,
            "rerank_status": result.rerank_status,
            "search_detail": result.search_detail,
            "used_citation_indices": result.used_citation_indices,
            "citation_warnings": result.citation_warnings,
            "partial": result.partial,
        },
    }


def _ask_for_info(facts: dict, missing: list[str]) -> str:
    known = []
    if facts.get("product_version"):
        known.append(f"产品版本：{facts['product_version']}")
    if facts.get("description"):
        known.append(f"问题现象：{facts['description'][:80]}")
    lines = ["我还需要确认以下信息，才能继续处理："]
    lines.extend(f"{i}. {item}" for i, item in enumerate(missing, start=1))
    if known:
        lines.append("\n目前已记录：" + "；".join(known))
    lines.append(
        "\n如果确实不清楚版本，可以回复「不清楚」，我会在工单里标注未确认；"
        "补齐信息后我再给出处理建议或生成工单草稿。"
    )
    return "\n".join(lines)


async def handle_issue(runtime: AgentRuntime, state: AgentState) -> dict:
    facts = state.get("facts") or {}
    missing = missing_ticket_fields(facts)
    # 回答问题和建单的要求不同：回答问题只要求版本，建单才要求报错/现象
    if not facts.get("product_version") and not facts.get("version_unknown"):
        return {"reply": _ask_for_info(facts, missing)}
    result = await answer(runtime, state)
    reply = result["reply"] + (
        "\n\n如果按资料操作后仍未解决，可以直接说「转人工」，我会把已收集的信息整理成工单草稿，"
        "由你确认后才创建。"
    )
    return {**result, "reply": reply}


async def handle_human(runtime: AgentRuntime, state: AgentState) -> dict:
    facts = state.get("facts") or {}
    missing = missing_ticket_fields(facts)
    if missing:
        return {"reply": _ask_for_info(facts, missing)}

    # 已有待确认草稿时复用，避免用户重复说"转人工"产生多份草稿
    if state.get("pending_draft"):
        draft = state["pending_draft"]
        return {
            "draft": draft,
            "reply": (
                "你已经有一份待确认的工单草稿，我没有重复生成。\n\n"
                f"标题：{draft['title']}\n"
                f"产品版本：{draft['product_version']}\n"
                f"问题现象：{draft['description'][:150]}\n\n"
                "确认无误请回复「确认创建」；需要修改可以直接在页面上编辑草稿后再确认。"
            ),
        }

    description = facts.get("description") or state.get("user_message", "")

    # 规则能明确判断时不调用模型；只有类别=其他且优先级=中（规则没把握）才请模型建议。
    # 模型结果只用于草稿字段，非法枚举/低置信度会自动回退，不影响草稿归属与确认流程。
    category = guess_category(description)
    priority = guess_priority(description)
    if category == "other" and priority == "medium":
        suggestion = await suggest_ticket_fields(description)
        if suggestion is not None:
            category = suggestion.category.value
            priority = suggestion.priority.value
            facts["field_suggestion_source"] = "model"

    draft = await ticket_service.get_or_create_draft(
        runtime.db,
        runtime.user,
        session_id=runtime.session_id,
        title=build_title(description),
        description=description,
        product_version=facts.get("product_version") or "未确认",
        error_message=facts.get("error_message") or "",
        tried_steps=facts.get("tried_steps") or "",
        category=category,
        priority=priority,
    )
    payload = ticket_service.serialize_draft(draft)
    reply = (
        "我先整理了一份工单草稿，请核对（此时还没有创建工单）：\n\n"
        f"标题：{payload['title']}\n"
        f"问题类别：{CATEGORY_LABELS.get(payload['category'], payload['category'])}"
        f"｜优先级：{PRIORITY_LABELS.get(payload['priority'], payload['priority'])}\n"
        f"产品版本：{payload['product_version']}\n"
        f"问题描述：{payload['description'][:200]}\n"
        f"报错信息：{payload['error_message'] or '未提供'}\n"
        f"已尝试：{payload['tried_steps'] or '未提供'}\n\n"
        "如需修改，请直接在草稿卡片中编辑；确认无误后回复「确认创建」，我才会写入工单。"
    )
    return {"draft": payload, "pending_draft": payload, "reply": reply}


async def confirm(runtime: AgentRuntime, state: AgentState) -> dict:
    pending = state.get("pending_draft")
    if not pending:
        return {
            "reply": "当前会话没有待确认的工单草稿。请先描述问题并说「转人工」，我整理出草稿后再确认。",
            "tool_ok": False,
        }

    # 用户上次看到的草稿已被修改：旧确认不能提交新内容，必须让他重新核对。
    if state.get("pending_draft_stale"):
        return {
            "reply": "这份草稿在你确认前已经更新过。为避免提交不是你看到的内容，我没有创建工单。"
            "请核对下方最新草稿，确认无误后再回复「确认创建」。",
            "tool_ok": False,
            "draft": pending,
            "pending_draft": pending,
            "pending_draft_stale": False,
        }

    # 没有记录用户看到的版本时不能猜：不读最新版本代填，先展示草稿让用户重新核对。
    seen_version = pending.get("seen_version")
    if seen_version is None:
        return {
            "reply": "当前会话没有记录你看到的草稿版本，为避免提交未经你核对的内容，我没有创建工单。"
            "请核对下方最新草稿，确认无误后再回复「确认创建」。",
            "tool_ok": False,
            "draft": pending,
            "pending_draft": pending,
        }

    result = await tool_service.tool_create_ticket(
        runtime.db,
        runtime.user,
        pending["id"],
        runtime.request_id,
        expected_version=seen_version,
    )
    if not result.ok:
        # 工具失败时必须明确告知"未创建"，不能回答已成功
        return {
            "reply": f"工单创建失败：{result.error}。本次没有创建工单，原有信息仍保留，请稍后重试。",
            "tool_ok": False,
            "error": result.error,
        }

    ticket = result.data["ticket"]
    already = result.data.get("already_created", False)
    prefix = "这张草稿之前已经创建过工单，我没有重复创建。" if already else "工单已创建成功。"
    reply = (
        f"{prefix}\n\n"
        f"工单号：{ticket['ticket_no']}\n"
        f"标题：{ticket['title']}\n"
        f"状态：{STATUS_LABELS.get(ticket['status'], ticket['status'])}\n\n"
        "你可以在「我的工单」查看进度。客服回复后，状态会变为「待客户确认」。"
    )
    # 写操作在返回成功话术之前提交；提交失败按失败处理，避免“嘴上成功、库里没有”。
    try:
        await runtime.db.commit()
    except Exception:
        await runtime.db.rollback()
        return {
            "reply": "工单创建失败：数据库提交异常，本次没有创建工单。请稍后重试。",
            "tool_ok": False,
            "error": "commit_failed",
        }
    return {
        "reply": reply,
        "ticket": ticket,
        "pending_draft": None,
        "pending_draft_stale": False,
        "last_ticket_no": ticket["ticket_no"],
        "tool_ok": True,
        "draft": {**pending, "status": "confirmed"},
    }


async def query_tickets(runtime: AgentRuntime, state: AgentState) -> dict:
    result = await tool_service.tool_list_tickets(runtime.db, runtime.user, limit=10)
    if not result.ok:
        return {"reply": f"查询工单失败：{result.error}。", "tool_ok": False}

    items = result.data["items"]
    if not items:
        return {"reply": "你当前没有可查看的工单。如果需要，可以描述问题后说「转人工」创建工单。"}

    lines = ["你当前可查看的工单："]
    for ticket in items:
        lines.append(
            f"- {ticket['ticket_no']}｜{ticket['title'][:24]}｜{_labels(ticket)}"
        )
    lines.append("\n回复工单号（例如 YZ202601010001）可以查看详情。")
    return {"reply": "\n".join(lines)}


async def ticket_detail(runtime: AgentRuntime, state: AgentState) -> dict:
    match = TICKET_NO_RE.search(state.get("user_message", ""))
    ticket_no = match.group(0) if match else state.get("last_ticket_no")
    if not ticket_no:
        return {"reply": "请提供要查询的工单号，例如 YZ202601010001。"}

    result = await tool_service.tool_get_ticket(runtime.db, runtime.user, ticket_no)
    if not result.ok:
        return {"reply": f"查询工单详情失败：{result.error}。", "tool_ok": False}

    ticket = result.data
    lines = [
        f"工单号：{ticket['ticket_no']}",
        f"标题：{ticket['title']}",
        f"状态：{STATUS_LABELS.get(ticket['status'], ticket['status'])}",
        f"类别/优先级：{CATEGORY_LABELS.get(ticket['category'], ticket['category'])} / "
        f"{PRIORITY_LABELS.get(ticket['priority'], ticket['priority'])}",
        f"产品版本：{ticket['product_version'] or '未提供'}",
        f"问题描述：{ticket['description'][:300]}",
        f"创建时间：{ticket['created_at']}",
    ]
    if ticket.get("messages"):
        lines.append("\n最近回复：")
        for message in ticket["messages"][-3:]:
            lines.append(f"- [{message['author_role']}] {message['content'][:150]}")
    return {"reply": "\n".join(lines), "last_ticket_no": ticket["ticket_no"]}


async def supplement(runtime: AgentRuntime, state: AgentState) -> dict:
    text = state.get("user_message", "")
    match = TICKET_NO_RE.search(text)
    ticket_no = match.group(0) if match else state.get("last_ticket_no")
    if not ticket_no:
        return {"reply": "请提供要补充信息的工单号，例如 YZ202601010001。"}

    content = TICKET_NO_RE.sub("", text)
    content = re_sub_command(content)
    if len(content.strip()) < 2:
        return {"reply": "请说明要补充的信息内容，例如新的报错提示或已尝试的操作。"}

    result = await tool_service.tool_supplement_ticket(runtime.db, runtime.user, ticket_no, content.strip())
    if not result.ok:
        return {"reply": f"补充工单信息失败：{result.error}。", "tool_ok": False}
    return {
        "reply": f"已补充到工单 {ticket_no}，客服会看到这条信息。",
        "last_ticket_no": ticket_no,
        "tool_ok": True,
    }


def re_sub_command(text: str) -> str:
    import re

    return re.sub(r"^(补充|追加)(一下)?(信息|说明|情况)?[:：，, ]*", "", text.strip())


async def cancel(runtime: AgentRuntime, state: AgentState) -> dict:
    pending = state.get("pending_draft")
    if not pending:
        return {"reply": "当前没有待确认的草稿。"}
    try:
        # 与确认走同一套条件更新：确认和取消只有一个能成功
        draft = await ticket_service.cancel_draft(
            runtime.db, runtime.user, pending["id"], expected_version=pending.get("version")
        )
    except HTTPException as exc:
        return {
            "reply": f"取消失败：{exc.detail}。我没有修改草稿，请在页面上核对最新状态。",
            "tool_ok": False,
            "pending_draft": pending,
        }
    return {
        "reply": "已取消这份工单草稿，不会创建工单。如仍需处理，可以重新描述问题并说「转人工」。",
        "pending_draft": None,
        "draft": ticket_service.serialize_draft(draft),
    }


async def edit_draft(runtime: AgentRuntime, state: AgentState) -> dict:
    """修改草稿：只做确定性的字段更新，版本 +1，并要求用户重新确认。"""
    pending = state.get("pending_draft")
    if not pending:
        return {"reply": "当前没有可修改的草稿。请先描述问题并说「转人工」。"}

    text = state.get("user_message", "")
    fields: dict = {}
    if re.search(r"紧急|最高|urgent", text):
        fields["priority"] = "urgent"
    elif re.search(r"优先级.{0,4}高|高优先级|比较急", text):
        fields["priority"] = "high"
    elif re.search(r"优先级.{0,4}低|不急", text):
        fields["priority"] = "low"

    quoted = re.search(r"[「“\"]([^」”\"]{2,80})[」”\"]", text)
    if quoted and "标题" in text:
        fields["title"] = quoted.group(1).strip()
    elif "标题" in text and "改成" in text:
        title = re.split(r"改成|改为", text, maxsplit=1)[-1].strip(" ：:。.")
        if title:
            fields["title"] = title[:200]

    if not fields:
        # 没有明确字段时，把这句话作为补充说明追加到描述，避免“看似改了其实没改”
        base = (pending.get("description") or "").strip()
        fields["description"] = (base + "\n用户修改说明：" + text).strip()[:5000]

    try:
        draft = await ticket_service.update_draft(
            runtime.db,
            runtime.user,
            pending["id"],
            expected_version=pending.get("version"),
            **fields,
        )
    except Exception as exc:
        return {
            "reply": f"草稿修改失败：{exc}。草稿内容没有变化，请刷新后重试。",
            "tool_ok": False,
        }

    payload = ticket_service.serialize_draft(draft)
    return {
        "reply": "草稿已更新（新版本）。请核对下方内容，确认无误后回复「确认创建」；"
        "旧版本的确认不会提交这次修改。",
        "draft": payload,
        "pending_draft": payload,
        "pending_draft_stale": False,
        "tool_ok": True,
    }


async def clarify(runtime: AgentRuntime, state: AgentState) -> dict:
    """意图不明确时只追问，不做任何写操作。"""
    pending = state.get("pending_draft")
    if pending:
        return {
            "reply": "我还不能确定你是要提交、取消还是修改草稿，所以没有创建工单。\n"
            "请明确回复：\n"
            "1. 确认创建：回复「确认创建」；\n"
            "2. 取消：回复「取消」；\n"
            "3. 修改：直接说明要改哪里，例如「把优先级改成高」。",
            "pending_draft": pending,
            "tool_ok": False,
        }
    return {"reply": "请再说明一下你的需求：是查产品操作、排查故障，还是转人工创建工单？"}


async def smalltalk(runtime: AgentRuntime, state: AgentState) -> dict:
    if state.get("intent") == "greeting":
        reply = (
            "你好，我是云账科技智能财税支持助手。我可以：\n"
            "1. 根据费控与数电票规范回答操作及合规问题（附带资料引用）；\n"
            "2. 帮你排查报销、预算、银企付款与月结异常，信息齐全时整理工单草稿；\n"
            "3. 查询你的工单处理进度。\n\n"
            "请描述问题，并尽量提供产品版本（v2.2 / v2.3）和完整报错提示。"
        )
    else:
        reply = (
            "我可以帮你查财税费控规则、排查单据异常、创建或查询支持工单。\n"
            "你可以试着这样问：「v2.2 报销单保存草稿后为什么部门预算被冻结？」或「转人工」。"
        )
    return {"reply": reply}
