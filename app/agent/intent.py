# app/agent/intent.py
# 规则式意图识别与事实抽取。
# 选择规则而不是小模型：演示项目没有可用的分类模型，规则可解释、可测试、零成本。

from __future__ import annotations

import re

TICKET_NO_RE = re.compile(r"YZ\d{12}", re.IGNORECASE)

CATEGORY_RULES: list[tuple[str, str]] = [
    ("import", r"导入|excel|csv|表格|模板|科目编码|项目编码"),
    ("budget", r"预算|额度|超支|冻结|占用|追加"),
    ("invoice", r"发票|数电|验真|查重|红冲|红字|税号|税额|抵扣|票据|ocr|xml|ofd|拍票"),
    ("payment", r"打款|付款|退票|联行号|银企|回单|出纳|转账"),
    ("closing", r"月结|关账|结账|反结账|凭证|总账|会计期间"),
    ("reimbursement", r"报销|差旅|招待|补贴|反审核|单据"),
    ("permission", r"账号|登录|密码|权限|用户|角色|禁用|离职"),
]


# 否定/取消表达：只要出现，就绝不能进入 confirm。
# 先判断取消，再判断疑问和修改，最后才允许严格白名单的确认。
CANCEL_RE = re.compile(
    r"取消|撤销|先不|暂不|不要|不用|别(提交|创建|建单|发)|算了|不建了|不提交|不创建|停止|等等|稍等"
)
QUESTION_RE = re.compile(r"[?？]|怎么|如何|为什么|是否|能否|能不能|可不可以|吗|呢|确认一下|确定一下")
EDIT_RE = re.compile(r"修改|改一下|改成|换成|更换|调整|补充|更新|编辑|标题|描述|优先级|版本")

# 只接受这些“完整等于”的确认指令（去掉客套前缀和句末标点后）。
# “可以”“好”“行”单独出现都视为不明确，必须追问，避免误提交。
CONFIRM_PATTERNS = [
    re.compile(r"确认"),
    re.compile(r"确认(创建|提交|建单|无误)"),
    re.compile(r"确定(创建|提交|建单)"),
    re.compile(r"就按这个(创建|提交|建单)?"),
    re.compile(r"同意(创建|提交|建单)?"),
    re.compile(r"(创建|提交|建单)吧"),
    re.compile(r"可以(创建|提交|建单)"),
    re.compile(r"没问题[，,]?(创建|提交|建单)吧?"),
]


def _normalize_confirm_text(text: str) -> str:
    value = re.sub(r"^(好的|好|嗯|行|可以)[，,。.!！\s]*", "", text.strip())
    return re.sub(r"[。.!！~～\s]+$", "", value)


def is_explicit_confirm(text: str) -> bool:
    """只有严格白名单里的完整指令才算确认；包含否定/疑问/修改词一律不算。"""
    raw = (text or "").strip()
    if not raw:
        return False
    if CANCEL_RE.search(raw) or QUESTION_RE.search(raw) or EDIT_RE.search(raw):
        return False
    normalized = _normalize_confirm_text(raw)
    return any(pattern.fullmatch(normalized) for pattern in CONFIRM_PATTERNS)


def classify_intent(text: str, *, has_pending_draft: bool, last_ticket_no: str | None) -> str:
    """返回 greeting/chitchat/confirm/cancel/edit_draft/clarify/query_tickets/ticket_detail/supplement/human/issue/question。"""
    t = (text or "").strip()
    if not t:
        return "chitchat"

    if re.fullmatch(r"(你好|您好|hi|hello|嗨|在吗)[!！。.~ ]*", t, re.IGNORECASE):
        return "greeting"

    # 有草稿时：取消 > 疑问 > 修改 > 严格确认 > 不明确的肯定
    if has_pending_draft and CANCEL_RE.search(t):
        return "cancel"
    if has_pending_draft and QUESTION_RE.search(t):
        return "clarify"
    if has_pending_draft and EDIT_RE.search(t):
        return "edit_draft"
    if has_pending_draft and is_explicit_confirm(t):
        return "confirm"
    if has_pending_draft and re.fullmatch(r"(好的?|嗯|行|可以|没问题)[，,。.!！\s]*", t):
        return "clarify"
    # 提到确认/提交但不是白名单指令：只追问，不写库
    if has_pending_draft and re.search(r"确认|确定|提交|创建|建单", t):
        return "clarify"

    if TICKET_NO_RE.search(t):
        if re.search(r"补充|追加", t):
            return "supplement"
        return "ticket_detail"

    if re.search(r"我的工单|查询工单|工单进度|工单状态|处理到哪|工单列表|查一下工单|我提交的工单", t):
        return "query_tickets"

    if re.search(r"(补充|追加).{0,8}(信息|说明|材料|情况)|补充说明", t):
        return "supplement"

    if re.search(r"转人工|人工客服|人工处理|提交工单|创建工单|帮我建单|报修|找技术|工程师", t):
        return "human"

    # 账号/登录/权限类问题通常与本企业产品版本无关，按咨询处理，避免强行追问版本
    if re.search(r"登录|账号|密码|权限|用户|角色", t) and re.search(
        r"怎么|如何|为什么|哪里|提示|错误|失败|禁用|无法|不能", t
    ):
        return "question"

    # 明确的提问式请求（如“请给出/请提供/告诉我”）按咨询处理
    if re.search(r"(请|帮我)?(给出|列出|提供|告诉我)", t):
        return "question"

    if re.search(
        r"报错|错误|异常|失败|无法|不能|打不开|卡住|闪退|丢失|对不上|不一致|"
        r"导入不了|保存不了|用不了|没反应|乱码|重复|少了几|多了几|没有变化|没变化|不生效",
        t,
    ):
        return "issue"

    # 自然语言提问常省略句号，这里覆盖"吗/呢/咋回事/是多少"等口语表达。
    # 该分支在 has_pending_draft 的确认判断之后，不会把"确认创建吗"误判成确认。
    if re.search(
        r"怎么|如何|为什么|为啥|在哪|哪里|是否|可以吗|步骤|教程|规则|有什么区别|"
        r"吗|呢|咋回事|怎么回事|咋办|怎么办|是多少|多少|[?？]",
        t,
    ):
        return "question"

    if last_ticket_no and re.search(r"还是|仍然|依然|没解决|不行|没用", t):
        return "issue"

    return "chitchat"


def extract_product_version(text: str) -> str | None:
    """从文本提取 v2.2 / 2.3 形式的产品版本。"""
    match = re.search(r"[vV]?\s*(\d+\.\d+)", text or "")
    if not match:
        return None
    return f"v{match.group(1)}"


def extract_error_message(text: str) -> str | None:
    """提取报错内容：冒号后的文字、引号内容，或包含"报错"的整句。"""
    text = text or ""
    match = re.search(r"(?:报错|错误|提示|异常)(?:信息|内容)?\s*[:：是为]\s*(.{3,200})", text)
    if match:
        return match.group(1).strip()
    quoted = re.search(r"[「“\"](.{3,120})[」”\"]", text)
    if quoted and re.search(r"报错|错误|提示|异常|失败", text):
        return quoted.group(1).strip()
    return None


def extract_tried_steps(text: str) -> str | None:
    if re.search(r"已经|已尝试|试过|重启|重新登录|重新导入|换过|清理过|联系过", text or ""):
        return (text or "").strip()[:500]
    return None


def guess_category(text: str) -> str:
    for category, pattern in CATEGORY_RULES:
        if re.search(pattern, text or "", re.IGNORECASE):
            return category
    return "other"


def guess_priority(text: str) -> str:
    if re.search(r"紧急|急|无法开单|全部|整个公司|数据丢失|停业|客户在等|今天必须", text or ""):
        return "high"
    if re.search(r"咨询|建议|优化|有空|不急", text or ""):
        return "low"
    return "medium"


def build_title(description: str) -> str:
    first_line = re.split(r"[\n。；;]", (description or "").strip())[0]
    return (first_line or "售后问题")[:40]


def merge_facts(facts: dict, message: str, intent: str) -> dict:
    """把本轮用户消息中的事实合并进会话事实，缺失字段不会被编造。"""
    facts = dict(facts or {})
    text = (message or "").strip()
    if not text:
        return facts

    version = extract_product_version(text)
    if version:
        facts["product_version"] = version
        facts.pop("version_unknown", None)
    elif re.search(r"不清楚|不知道|忘了|不确定|没注意", text) and not facts.get("product_version"):
        facts["version_unknown"] = True

    error_message = extract_error_message(text)
    if error_message:
        facts["error_message"] = error_message

    tried = extract_tried_steps(text)
    if tried and re.search(r"已经|已尝试|试过|重启|重新|换过|清理过", tried):
        facts["tried_steps"] = tried

    if intent in ("issue", "human", "question"):
        old = facts.get("description", "")
        if text not in old:
            facts["description"] = (old + "\n" + text).strip()[:2000]

    return facts


def missing_ticket_fields(facts: dict) -> list[str]:
    """创建工单前必须补齐的信息。unknown 视为用户已明确表示不知道，不再拦截。"""
    missing: list[str] = []
    if not facts.get("product_version") and not facts.get("version_unknown"):
        missing.append("产品版本（例如 v2.2 或 v2.3）")
    if not facts.get("error_message") and len(facts.get("description", "")) < 12:
        missing.append("完整报错提示或具体现象")
    return missing
