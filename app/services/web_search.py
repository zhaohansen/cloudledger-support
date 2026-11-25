# app/services/web_search.py
# 公开财税政策联网搜索服务（只读，支持外部 MCP Server 转发或 Tavily REST API）。
# 安全边界：仅用于公开财税/数电发票政策检索，严禁将租户私有内控制度内容发往外网；
# 未配置搜索源或外部服务超时/异常时，显式返回可解释状态，不阻断主流程、不伪造结果。

from __future__ import annotations

from typing import Any

import httpx

from app.config import get_settings
from app.logging_config import get_logger
from app.mcp.client import call_mcp_tool

logger = get_logger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def _normalize_items(raw_items: Any, max_results: int) -> list[dict[str, str]]:
    if not isinstance(raw_items, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in raw_items[:max_results]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        snippet = str(item.get("snippet") or item.get("content") or "").strip()
        if title or url or snippet:
            normalized.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet[:400],
                }
            )
    return normalized


async def search_public_web(
    query: str,
    *,
    max_results: int = 3,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """查询公开互联网财税政策信息，支持 MCP 转发与 Tavily API 两种通道。"""
    clean_query = (query or "").strip()
    if not clean_query:
        return {
            "status": "invalid_query",
            "source": "none",
            "items": [],
            "detail": "检索关键词不能为空",
        }

    limit = max(1, min(int(max_results), 5))
    settings = get_settings()

    # 通道 1：转发至独立部署的 Web Search MCP Server
    if settings.web_search_mcp_url:
        try:
            raw = await call_mcp_tool(
                server_url=settings.web_search_mcp_url,
                tool_name="web_search",
                arguments={"query": clean_query, "max_results": limit},
                timeout=settings.web_search_timeout_seconds,
                http_client=http_client,
            )
            items = _normalize_items(raw, limit)
            logger.info("web_search.mcp_ok", count=len(items))
            return {
                "status": "ok",
                "source": "mcp",
                "items": items,
                "detail": "",
            }
        except Exception as exc:
            logger.warning("web_search.mcp_failed", error=type(exc).__name__)
            return {
                "status": "error",
                "source": "mcp",
                "items": [],
                "detail": f"外部联网搜索 MCP 服务暂不可用：{type(exc).__name__}",
            }

    # 通道 2：直连 Tavily Search API
    if settings.web_search_api_key:
        payload = {
            "api_key": settings.web_search_api_key,
            "query": clean_query,
            "max_results": limit,
            "search_depth": "basic",
        }
        try:
            if http_client is not None:
                resp = await http_client.post(TAVILY_SEARCH_URL, json=payload)
            else:
                async with httpx.AsyncClient(
                    timeout=settings.web_search_timeout_seconds
                ) as client:
                    resp = await client.post(TAVILY_SEARCH_URL, json=payload)
            resp.raise_for_status()
            body = resp.json()
            items = _normalize_items(body.get("results", []), limit)
            logger.info("web_search.tavily_ok", count=len(items))
            return {
                "status": "ok",
                "source": "tavily",
                "items": items,
                "detail": "",
            }
        except Exception as exc:
            logger.warning("web_search.tavily_failed", error=type(exc).__name__)
            return {
                "status": "error",
                "source": "tavily",
                "items": [],
                "detail": f"联网搜索接口暂不可用：{type(exc).__name__}",
            }

    return {
        "status": "unconfigured",
        "source": "none",
        "items": [],
        "detail": "未配置 WEB_SEARCH_MCP_URL 或 WEB_SEARCH_API_KEY，当前仅使用本地知识库",
    }
