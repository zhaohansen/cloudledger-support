# app/api/router.py
# 汇总所有业务路由，便于 main.py 一次性挂载。

from fastapi import APIRouter

from app.api import admin, auth, chat, knowledge, tickets
from app.mcp.server import router as mcp_router

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/api/auth", tags=["auth"])
api_router.include_router(chat.router)
api_router.include_router(tickets.router)
api_router.include_router(knowledge.router)
api_router.include_router(admin.router)
api_router.include_router(mcp_router)
