# app/agent/structured.py
# 结构化模型理解层：只负责"听懂"和"给建议"，不写数据库、不产生确认。
# 边界：
#   1. 模型可以选择意图（question/issue/human/...），但可选值里没有 confirm/cancel/edit_draft，
#      提交、取消草稿只能由前端按钮或后端严格白名单指令触发；
#   2. 模型可以建议工单类别/优先级，但字段必须通过 Pydantic 枚举校验；
#   3. 超时、低置信度、非法输出都回退到规则结果，不影响主流程可用性。

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, ValidationError

from app.config import get_settings
from app.logging_config import get_logger
from app.services.llm import LLMServiceError, get_llm_client

settings = get_settings()
logger = get_logger(__name__)

# 模型建议意图的最低置信度；低于该值按"没听懂"处理，回退规则
MIN_CONFIDENCE = 0.6


class ModelIntent(str, Enum):
    """模型可建议的意图白名单。故意不包含 confirm/cancel/edit_draft。"""

    question = "question"
    issue = "issue"
    human = "human"
    query_tickets = "query_tickets"
    ticket_detail = "ticket_detail"
    supplement = "supplement"
    chitchat = "chitchat"


class TicketCategory(str, Enum):
    data_import = "import"
    budget = "budget"
    invoice = "invoice"
    reimbursement = "reimbursement"
    payment = "payment"
    closing = "closing"
    permission = "permission"
    other = "other"


class TicketPriority(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class IntentDecision(BaseModel):
    intent: ModelIntent
    confidence: float = Field(ge=0.0, le=1.0)


class TicketFieldSuggestion(BaseModel):
    category: TicketCategory
    priority: TicketPriority
    confidence: float = Field(ge=0.0, le=1.0)


INTENT_SYSTEM_PROMPT = (
    "你是企业财税费控与数电发票 SaaS 智能支持系统的意图分类器。只输出 JSON，不要输出解释或多余文字。\n"
    "可选 intent：\n"
    "- question：产品操作、财税费控规则、流程等咨询；\n"
    "- issue：系统报错、功能异常、额度或数据不一致等故障；\n"
    "- human：明确要求人工支持、报障、创建工单；\n"
    "- query_tickets：查询自己的工单列表或进度；\n"
    "- ticket_detail：查询某一个具体工单；\n"
    "- supplement：给已有工单补充信息；\n"
    "- chitchat：问候、闲聊或无法判断。\n"
    "你没有创建、提交、取消工单的权限，不要输出 confirm/cancel/edit_draft。\n"
    '输出格式：{"intent": "<枚举值>", "confidence": 0.0}'
)

FIELD_SYSTEM_PROMPT = (
    "你是企业财税费控 SaaS 工单的信息整理助手。只输出 JSON，不要输出解释。\n"
    "根据用户描述建议工单类别与优先级，不要编造用户没有提供的事实。\n"
    "category 只能是：import、budget、invoice、reimbursement、payment、closing、permission、other；\n"
    "priority 只能是：high、medium、low。\n"
    '输出格式：{"category": "<枚举值>", "priority": "<枚举值>", "confidence": 0.0}'
)


def structured_available() -> bool:
    """是否具备调用结构化模型的条件（不发起实际请求）。"""
    return get_llm_client().configured


async def classify_intent_with_model(text: str) -> IntentDecision | None:
    """让模型建议意图；失败/低置信度返回 None，由调用方保留规则结果。"""
    if not structured_available():
        return None
    try:
        data = await get_llm_client().complete_json(
            [("system", INTENT_SYSTEM_PROMPT), ("human", (text or "")[:1500])]
        )
        decision = IntentDecision.model_validate(data)
    except (ValidationError, LLMServiceError) as exc:
        logger.warning("agent.structured_intent_invalid", error=type(exc).__name__)
        return None
    except Exception as exc:  # 模型/网络/解析异常都回退规则，不阻塞对话
        logger.warning("agent.structured_intent_failed", error=type(exc).__name__)
        return None

    if decision.confidence < MIN_CONFIDENCE:
        logger.info("agent.structured_intent_low_confidence", confidence=decision.confidence)
        return None
    logger.info(
        "agent.structured_intent",
        intent=decision.intent.value,
        confidence=decision.confidence,
    )
    return decision


async def suggest_ticket_fields(description: str) -> TicketFieldSuggestion | None:
    """建议工单类别/优先级；只做建议，不写库，也不提取版本、报错等事实。"""
    if not structured_available():
        return None
    try:
        data = await get_llm_client().complete_json(
            [("system", FIELD_SYSTEM_PROMPT), ("human", (description or "")[:1500])]
        )
        suggestion = TicketFieldSuggestion.model_validate(data)
    except (ValidationError, LLMServiceError) as exc:
        logger.warning("agent.structured_fields_invalid", error=type(exc).__name__)
        return None
    except Exception as exc:
        logger.warning("agent.structured_fields_failed", error=type(exc).__name__)
        return None

    if suggestion.confidence < MIN_CONFIDENCE:
        logger.info("agent.structured_fields_low_confidence", confidence=suggestion.confidence)
        return None
    logger.info(
        "agent.structured_fields",
        category=suggestion.category.value,
        priority=suggestion.priority.value,
        confidence=suggestion.confidence,
    )
    return suggestion
