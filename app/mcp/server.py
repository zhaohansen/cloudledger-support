# app/mcp/server.py
# 只读型 MCP Server（JSON-RPC 2.0 over HTTP，挂载于 POST /mcp）。
# 安全边界：
# 1. 所有请求强制经由 get_current_user 校验 Bearer JWT 并回库检查账号激活状态；
# 2. 忽略参数中传入的任何 tenant_id / user_id，身份只取自服务端 UserContext；
# 3. 仅暴露只读查询工具（知识库检索、工单列表、工单详情），严禁暴露建单/修改/删除等写操作。

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app import __version__
from app.db import get_db
from app.deps import UserContext, get_current_user
from app.logging_config import get_logger
from app.services.retrieval import search_knowledge
from app.services.tools import tool_get_ticket, tool_list_tickets
from app.services.web_search import search_public_web

logger = get_logger(__name__)
router = APIRouter(tags=["mcp"])

MCP_PROTOCOL_VERSION = "2024-11-05"

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "search_tenant_docs",
        "description": "按当前登录用户的租户权限与可选产品版本检索企业财税费控知识库（只读）。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索问题或关键词"},
                "product_version": {
                    "type": "string",
                    "description": "可选产品版本号，如 v2.2 / v2.3",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回条数上限（1-10，默认 5）",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_my_tickets",
        "description": "查询当前登录用户有权查看的工单列表，严格遵守多租户与角色可见性隔离（只读）。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status_filter": {
                    "type": "string",
                    "description": "可选工单状态筛选：pending / processing / waiting_customer / closed",
                },
                "limit": {
                    "type": "integer",
                    "description": "返回条数上限（1-50，默认 10）",
                    "minimum": 1,
                    "maximum": 50,
                },
            },
        },
    },
    {
        "name": "get_ticket_detail",
        "description": "按工单号查询工单详情及流转记录，强制校验当前用户权限（只读）。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ticket_no": {"type": "string", "description": "工单号，例如 YZ202611010001"},
            },
            "required": ["ticket_no"],
        },
    },
    {
        "name": "web_search",
        "description": "检索公开互联网财税政策与数电发票公开口径（只读，不传输租户私有数据；未配置搜索源时显式降级）。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "公开政策检索关键词"},
                "max_results": {
                    "type": "integer",
                    "description": "返回条数上限（1-5，默认 3）",
                    "minimum": 1,
                    "maximum": 5,
                },
            },
            "required": ["query"],
        },
    },
]

_ALLOWED_TOOL_NAMES = {tool["name"] for tool in TOOL_DEFINITIONS}


def _jsonrpc_ok(req_id: Any, result: dict[str, Any]) -> JSONResponse:
    return JSONResponse(
        content={
            "jsonrpc": "2.0",
            "id": req_id,
            "result": result,
        }
    )


def _jsonrpc_error(req_id: Any, code: int, message: str) -> JSONResponse:
    return JSONResponse(
        content={
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": code,
                "message": message,
            },
        }
    )


def _wrap_text_result(payload: Any) -> dict[str, Any]:
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(payload, ensure_ascii=False),
            }
        ],
        "isError": False,
    }


async def _handle_tool_call(
    db: AsyncSession,
    user: UserContext,
    tool_name: str,
    arguments: dict[str, Any],
) -> tuple[dict[str, Any] | None, tuple[int, str] | None]:
    if tool_name not in _ALLOWED_TOOL_NAMES:
        return None, (
            -32601,
            f"工具 '{tool_name}' 未注册或不允许调用（MCP 仅开放只读查询工具）",
        )

    if tool_name == "search_tenant_docs":
        query = str(arguments.get("query") or "").strip()
        if not query:
            return None, (-32602, "缺少必填参数 query")
        product_version = arguments.get("product_version")
        if product_version is not None:
            product_version = str(product_version).strip() or None
        raw_top_k = arguments.get("top_k", 5)
        try:
            top_k = max(1, min(int(raw_top_k), 10))
        except (TypeError, ValueError):
            return None, (-32602, "参数 top_k 必须是整数")

        outcome = await search_knowledge(
            db,
            user,
            query,
            product_version=product_version,
            top_k=top_k,
        )
        return (
            {
                "reason": outcome.reason,
                "requested_version": outcome.requested_version,
                "matched_versions": outcome.matched_versions,
                "version_conflict": outcome.version_conflict,
                "embedding_backend": outcome.embedding_backend,
                "retrieval_backend": outcome.retrieval_backend,
                "items": [
                    chunk.to_citation(idx)
                    for idx, chunk in enumerate(outcome.items, start=1)
                ],
            },
            None,
        )

    if tool_name == "list_my_tickets":
        status_filter = arguments.get("status_filter")
        if status_filter is not None:
            status_filter = str(status_filter).strip() or None
        raw_limit = arguments.get("limit", 10)
        try:
            limit = max(1, min(int(raw_limit), 50))
        except (TypeError, ValueError):
            return None, (-32602, "参数 limit 必须是整数")

        result = await tool_list_tickets(
            db,
            user,
            status_filter=status_filter,
            limit=limit,
        )
        if not result.ok:
            return None, (-32000, result.error or "查询工单列表失败")
        return result.data, None

    if tool_name == "get_ticket_detail":
        ticket_no = str(arguments.get("ticket_no") or "").strip()
        if not ticket_no:
            return None, (-32602, "缺少必填参数 ticket_no")
        result = await tool_get_ticket(db, user, ticket_no)
        if not result.ok:
            return None, (-32003, result.error or "无权查看或工单不存在")
        return result.data, None

    if tool_name == "web_search":
        query = str(arguments.get("query") or "").strip()
        if not query:
            return None, (-32602, "缺少必填参数 query")
        raw_max = arguments.get("max_results", 3)
        try:
            max_results = max(1, min(int(raw_max), 5))
        except (TypeError, ValueError):
            return None, (-32602, "参数 max_results 必须是整数")
        data = await search_public_web(query, max_results=max_results)
        return data, None

    return None, (-32601, f"未知工具: {tool_name}")


@router.post("/mcp")
async def mcp_endpoint(
    request: Request,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """MCP JSON-RPC 2.0 只读工具端点。"""
    try:
        body = await request.json()
    except Exception:
        return _jsonrpc_error(None, -32700, "无效的 JSON 请求体")

    if not isinstance(body, dict) or body.get("jsonrpc") != "2.0":
        return _jsonrpc_error(body.get("id") if isinstance(body, dict) else None, -32600, "仅支持 JSON-RPC 2.0 请求")

    req_id = body.get("id")
    method = body.get("method")
    params = body.get("params") or {}
    if not isinstance(params, dict):
        return _jsonrpc_error(req_id, -32602, "params 必须是对象")

    if method == "initialize":
        return _jsonrpc_ok(
            req_id,
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "cloudledger-readonly-mcp",
                    "version": __version__,
                },
            },
        )

    if method == "tools/list":
        logger.info("mcp.tools_list", user_id=user.user_id, role=user.role)
        return _jsonrpc_ok(req_id, {"tools": TOOL_DEFINITIONS})

    if method == "tools/call":
        tool_name = str(params.get("name") or "").strip()
        arguments = params.get("arguments") or {}
        if not tool_name:
            return _jsonrpc_error(req_id, -32602, "缺少工具名 params.name")
        if not isinstance(arguments, dict):
            return _jsonrpc_error(req_id, -32602, "params.arguments 必须是对象")

        data, err = await _handle_tool_call(db, user, tool_name, arguments)
        if err is not None:
            code, msg = err
            logger.warning(
                "mcp.tool_call_rejected",
                tool=tool_name,
                user_id=user.user_id,
                code=code,
            )
            return _jsonrpc_error(req_id, code, msg)

        logger.info(
            "mcp.tool_call_ok",
            tool=tool_name,
            user_id=user.user_id,
            tenant_id=user.tenant_id,
        )
        return _jsonrpc_ok(req_id, _wrap_text_result(data))

    return _jsonrpc_error(req_id, -32601, f"不支持的 MCP 方法: {method}")
