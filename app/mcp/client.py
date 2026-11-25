# app/mcp/client.py
# 轻量异步 MCP 客户端（JSON-RPC 2.0 over HTTP），支持外部 HTTP 调用与测试环境注入 AsyncClient。

from __future__ import annotations

import json
from typing import Any

import httpx


def _build_headers(token: str = "") -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def _post_jsonrpc(
    server_url: str,
    payload: dict[str, Any],
    *,
    token: str = "",
    timeout: float = 30.0,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    url = f"{server_url.rstrip('/')}/mcp"
    headers = _build_headers(token)
    if http_client is not None:
        resp = await http_client.post(url, json=payload, headers=headers)
    else:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            resp = await client.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    return resp.json()


async def list_mcp_tools(
    server_url: str,
    *,
    token: str = "",
    timeout: float = 10.0,
    http_client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    """查询 MCP Server 注册的所有只读工具清单。"""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {},
    }
    data = await _post_jsonrpc(
        server_url,
        payload,
        token=token,
        timeout=timeout,
        http_client=http_client,
    )
    if "error" in data:
        raise ValueError(
            f"MCP tools/list error: {data['error'].get('message', str(data['error']))}"
        )
    return data.get("result", {}).get("tools", [])


async def call_mcp_tool(
    server_url: str,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    token: str = "",
    timeout: float = 30.0,
    http_client: httpx.AsyncClient | None = None,
) -> Any:
    """调用 MCP Server 上注册的只读工具，并自动解包 TextContent 中的 JSON 数据。"""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
    }
    data = await _post_jsonrpc(
        server_url,
        payload,
        token=token,
        timeout=timeout,
        http_client=http_client,
    )
    if "error" in data:
        raise ValueError(
            f"MCP tool '{tool_name}' error: {data['error'].get('message', str(data['error']))}"
        )

    content = data.get("result", {}).get("content", [])
    if not content:
        return []

    items = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text", "")
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            parsed = text
        if isinstance(parsed, list):
            return parsed
        items.append(parsed)

    if len(items) == 1:
        return items[0]
    return items if items else content
