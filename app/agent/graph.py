# app/agent/graph.py
# 用 LangGraph 把单个售后 Agent 的状态流程串起来。
# 图很小：一个 Agent + 明确分支，不为了展示技术而拆多 Agent。

from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph

from app.agent import nodes
from app.agent.nodes import AgentRuntime
from app.agent.state import AgentState


def route_by_intent(state: AgentState) -> str:
    mapping = {
        "greeting": "smalltalk",
        "chitchat": "smalltalk",
        "question": "answer",
        "issue": "handle_issue",
        "human": "handle_human",
        "confirm": "confirm",
        "cancel": "cancel",
        "edit_draft": "edit_draft",
        "clarify": "clarify",
        "query_tickets": "query_tickets",
        "ticket_detail": "ticket_detail",
        "supplement": "supplement",
    }
    return mapping.get(state.get("intent", "chitchat"), "smalltalk")


def build_agent_graph(runtime: AgentRuntime):
    """每个请求构建并编译一次图。

    图本身无状态；把数据库会话和用户身份放在 runtime 里注入节点，
    避免把 ORM 对象塞进 LangGraph State（也不利于序列化）。
    """
    graph = StateGraph(AgentState)

    graph.add_node("load_context", nodes.load_context)
    # 用 partial 注入 runtime；partial 仍会被 LangGraph 识别为协程函数并 await。
    # 不能写成 lambda state: nodes.xxx(runtime, state)，否则返回的是协程对象而不是结果。
    graph.add_node("classify", partial(nodes.classify, runtime))
    actions = {
        "answer": nodes.answer,
        "handle_issue": nodes.handle_issue,
        "handle_human": nodes.handle_human,
        "confirm": nodes.confirm,
        "cancel": nodes.cancel,
        "edit_draft": nodes.edit_draft,
        "clarify": nodes.clarify,
        "query_tickets": nodes.query_tickets,
        "ticket_detail": nodes.ticket_detail,
        "supplement": nodes.supplement,
        "smalltalk": nodes.smalltalk,
    }
    for name, handler in actions.items():
        graph.add_node(name, partial(handler, runtime))
        graph.add_edge(name, END)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "classify")
    graph.add_conditional_edges("classify", route_by_intent, list(actions))
    return graph.compile()
