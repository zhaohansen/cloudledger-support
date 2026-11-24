# app/main.py
# 应用入口：日志、数据库初始化、request_id 中间件、异常处理、静态页面与健康检查。

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.api.router import api_router
from app.config import get_settings
from app.db import dispose_db, get_db, init_db
from app.logging_config import configure_logging, get_logger, request_id_var

settings = get_settings()
logger = get_logger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging()
    await init_db()
    logger.info(
        "app.started",
        env=settings.app_env,
        database="sqlite" if settings.is_sqlite else "postgresql",
        llm_mode=settings.effective_llm_mode,
    )
    yield
    await dispose_db()


app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description="企业财税费控与数电发票 SaaS 智能支持与受控工单系统",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """为每个请求生成/透传 request_id，并记录耗时与状态码。"""
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
    request.state.request_id = rid
    token = request_id_var.set(rid)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.error(
            "http.unhandled_error",
            method=request.method,
            path=request.url.path,
            error=type(exc).__name__,
        )
        raise
    finally:
        request_id_var.reset(token)

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    response.headers["X-Request-ID"] = rid
    logger.info(
        "http.request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        elapsed_ms=elapsed_ms,
        request_id=rid,
    )
    return response


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "request_id": request_id_var.get("-")},
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # 不把内部堆栈返回给客户端；request_id 便于对照服务端日志
    rid = request.state.request_id
    logger.exception("http.internal_error", path=request.url.path, error=type(exc).__name__, request_id=rid)
    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误，请稍后重试", "request_id": rid},
        headers={"X-Request-ID": rid},
    )


app.include_router(api_router)


@app.get("/health", tags=["system"])
async def health(db: AsyncSession = Depends(get_db)):
    db_ok = True
    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        db_ok = False
    return {
        "status": "ok" if db_ok else "degraded",
        "version": __version__,
        "database": "sqlite" if settings.is_sqlite else "postgresql",
        "database_ok": db_ok,
        "llm_mode": settings.effective_llm_mode,
        "llm_configured": bool(settings.llm_api_key),
        # 只反映配置；模型文件/Milvus 是否真的可用需通过检索接口或校验脚本确认
        "embedding_backend": settings.embedding_backend,
        "retrieval_backend": settings.retrieval_backend,
        "rerank_enabled": settings.rerank_enabled,
    }


# ── 静态前端 ────────────────────────────────────────────────
if (FRONTEND_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/login")


def _page(name: str):
    return FileResponse(FRONTEND_DIR / name)


@app.get("/login", include_in_schema=False)
async def login_page():
    return _page("login.html")


@app.get("/chat", include_in_schema=False)
async def chat_page():
    return _page("chat.html")


@app.get("/tickets", include_in_schema=False)
async def tickets_page():
    return _page("tickets.html")


@app.get("/agent", include_in_schema=False)
async def agent_page():
    return _page("agent.html")


@app.get("/admin", include_in_schema=False)
async def admin_page():
    return _page("admin.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return JSONResponse(status_code=204, content=None)
