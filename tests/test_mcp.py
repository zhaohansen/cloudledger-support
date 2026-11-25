# tests/test_mcp.py
# 覆盖：MCP 只读端点认证、tools/list 清单、写工具拦截、跨租户知识库隔离、工单跨租户越权拦截。

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.db import AsyncSessionLocal
from app.mcp.client import call_mcp_tool, list_mcp_tools
from app.models import Ticket
from app.security import create_access_token
from tests.helpers import auth, get_user, token_for


async def test_mcp_requires_valid_active_jwt(client):
    """未携带 Token 返回 401；禁用账号即使持有有效 Token 也返回 403。"""
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    unauth = await client.post("/mcp", json=payload)
    assert unauth.status_code == 401

    disabled = await get_user("customer_disabled")
    disabled_token = create_access_token(disabled.id)
    forbidden = await client.post("/mcp", json=payload, headers=auth(disabled_token))
    assert forbidden.status_code == 403


async def test_mcp_initialize_and_tools_list_readonly_only(client):
    """initialize 与 tools/list 正常响应，且只暴露只读查询工具，绝不暴露写库工具。"""
    token = await token_for("customer_a")
    init_resp = await client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        headers=auth(token),
    )
    assert init_resp.status_code == 200
    init_data = init_resp.json()["result"]
    assert init_data["serverInfo"]["name"] == "cloudledger-readonly-mcp"

    tools = await list_mcp_tools("http://testserver", token=token, http_client=client)
    names = {tool["name"] for tool in tools}
    assert names == {"search_tenant_docs", "list_my_tickets", "get_ticket_detail", "web_search"}
    assert "create_ticket" not in names
    assert "supplement_ticket" not in names


async def test_mcp_rejects_write_tool_call(client):
    """尝试通过 MCP 调用写库工具（如 create_ticket）必须被拒绝。"""
    token = await token_for("customer_a")
    with pytest.raises(ValueError, match="只读"):
        await call_mcp_tool(
            "http://testserver",
            "create_ticket",
            {"draft_id": "fake-draft", "expected_version": 1},
            token=token,
            http_client=client,
        )


async def test_mcp_search_tenant_docs_enforces_isolation(client):
    """MCP 知识库检索严格遵循租户隔离，忽略参数中伪造的 tenant_id。"""
    token_a = await token_for("customer_a")
    token_b = await token_for("customer_b")

    res_a = await call_mcp_tool(
        "http://testserver",
        "search_tenant_docs",
        {"query": "星海文化报销管理细则", "tenant_id": "xinghai", "top_k": 5},
        token=token_a,
        http_client=client,
    )
    titles_a = [item["title"] for item in res_a["items"]]
    assert "星海文化报销管理细则" not in titles_a

    res_b = await call_mcp_tool(
        "http://testserver",
        "search_tenant_docs",
        {"query": "星海文化报销管理细则", "top_k": 5},
        token=token_b,
        http_client=client,
    )
    titles_b = [item["title"] for item in res_b["items"]]
    assert "星海文化报销管理细则" in titles_b
    assert "华辰智造费控内控制度" not in titles_b


async def test_mcp_ticket_tools_enforce_visibility(client):
    """MCP 工单列表与详情查询严格隔离其他租户工单。"""
    token_a = await token_for("customer_a")
    token_b = await token_for("customer_b")

    list_a = await call_mcp_tool(
        "http://testserver",
        "list_my_tickets",
        {"limit": 10},
        token=token_a,
        http_client=client,
    )
    assert list_a["count"] >= 1
    ticket_no_a = list_a["items"][0]["ticket_no"]

    detail_a = await call_mcp_tool(
        "http://testserver",
        "get_ticket_detail",
        {"ticket_no": ticket_no_a},
        token=token_a,
        http_client=client,
    )
    assert detail_a["ticket_no"] == ticket_no_a
    assert "messages" in detail_a and "events" in detail_a

    # 租户 B 尝试通过 MCP 查询租户 A 的工单，必须被拒绝
    with pytest.raises(ValueError):
        await call_mcp_tool(
            "http://testserver",
            "get_ticket_detail",
            {"ticket_no": ticket_no_a},
            token=token_b,
            http_client=client,
        )

    # 确认数据库中确实存在多张演示工单，且租户 B 只能看到自己的
    async with AsyncSessionLocal() as db:
        total_tickets = len((await db.execute(select(Ticket))).scalars().all())
    list_b = await call_mcp_tool(
        "http://testserver",
        "list_my_tickets",
        {"limit": 10},
        token=token_b,
        http_client=client,
    )
    assert total_tickets >= 2
    assert all(item["ticket_no"] != ticket_no_a for item in list_b["items"])


async def test_mcp_web_search_unconfigured_and_mocked(client, monkeypatch):
    """未配置搜索源时显式返回 unconfigured；配置外部 MCP/API 时正常返回结构化结果并支持异常降级。"""
    from app.services import web_search as ws_mod

    token = await token_for("customer_a")

    # 1. 默认未配置搜索源：显式降级，不编造结果
    unconf = await call_mcp_tool(
        "http://testserver",
        "web_search",
        {"query": "数电发票红冲规定", "max_results": 2},
        token=token,
        http_client=client,
    )
    assert unconf["status"] == "unconfigured"
    assert unconf["items"] == []

    # 2. 模拟外部 Web Search MCP Server 正常返回
    class _StubSettings:
        web_search_mcp_url = "http://mock-web-mcp:8001"
        web_search_api_key = ""
        web_search_timeout_seconds = 3.0

    async def _fake_call_mcp(server_url, tool_name, arguments, **kwargs):
        assert tool_name == "web_search"
        return [
            {
                "title": "国家税务总局数电发票红冲操作指引",
                "url": "https://example.gov.cn/e-invoice",
                "snippet": "开票方或受票方可在电子税务局发起红字发票信息确认单。",
            }
        ]

    monkeypatch.setattr(ws_mod, "get_settings", lambda: _StubSettings())
    monkeypatch.setattr(ws_mod, "call_mcp_tool", _fake_call_mcp)

    ok_res = await call_mcp_tool(
        "http://testserver",
        "web_search",
        {"query": "数电发票红冲规定", "max_results": 2},
        token=token,
        http_client=client,
    )
    assert ok_res["status"] == "ok"
    assert ok_res["source"] == "mcp"
    assert len(ok_res["items"]) == 1
    assert ok_res["items"][0]["title"] == "国家税务总局数电发票红冲操作指引"

    # 3. 模拟外部服务超时/异常：静默降级为 error + 空列表，不崩接口
    async def _failing_call_mcp(*args, **kwargs):
        raise TimeoutError("mock timeout")

    monkeypatch.setattr(ws_mod, "call_mcp_tool", _failing_call_mcp)
    err_res = await call_mcp_tool(
        "http://testserver",
        "web_search",
        {"query": "数电发票红冲规定"},
        token=token,
        http_client=client,
    )
    assert err_res["status"] == "error"
    assert err_res["items"] == []
