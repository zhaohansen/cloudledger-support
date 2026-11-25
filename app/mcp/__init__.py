# app/mcp/__init__.py
# 只读型 MCP 协议接入层（JSON-RPC 2.0）：仅暴露租户隔离的只读查询工具，不开放写库权限。

from app.mcp.client import call_mcp_tool, list_mcp_tools
from app.mcp.server import router as mcp_router

__all__ = ["mcp_router", "call_mcp_tool", "list_mcp_tools"]
