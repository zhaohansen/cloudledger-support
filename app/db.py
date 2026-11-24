# app/db.py
# 数据库连接与会话管理。默认 SQLite，可通过 DATABASE_URL 切换 PostgreSQL。

from __future__ import annotations

from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import get_settings
from app.models import Base

settings = get_settings()

# SQLite 文件所在目录必须先存在，否则连接时报 "unable to open database file"
if settings.is_sqlite:
    _sqlite_prefix = "sqlite+aiosqlite:///"
    if settings.database_url.startswith(_sqlite_prefix):
        _db_file = settings.database_url[len(_sqlite_prefix):]
        if _db_file and _db_file != ":memory:":
            Path(_db_file).parent.mkdir(parents=True, exist_ok=True)

# SQLite 使用 NullPool：每次会话新建连接，避免测试/多事件循环下复用旧连接；
# PostgreSQL 使用连接池并开启 pre_ping。
_engine_options = {"echo": False}
if settings.is_sqlite:
    _engine_options["poolclass"] = NullPool
else:
    _engine_options["pool_pre_ping"] = True

# SQLite 需要 check_same_thread=False 才能跨线程/异步使用
_connect_args = {"check_same_thread": False} if settings.is_sqlite else {}
engine = create_async_engine(
    settings.database_url,
    connect_args=_connect_args,
    **_engine_options,
)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


if settings.is_sqlite:
    # SQLite 默认不启用外键约束；busy_timeout 减少并发读写锁冲突（连接级设置）
    @event.listens_for(engine.sync_engine, "connect")
    def _configure_sqlite(dbapi_connection, _connection_record):  # pragma: no cover
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖：每个请求一个会话，正常结束提交，异常回滚。"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """建表并执行幂等迁移；新库与已有库都走同一条路径。"""
    from sqlalchemy import text
    from app.migrations import apply_migrations

    async with engine.begin() as conn:
        if settings.is_sqlite:
            await conn.execute(text("PRAGMA journal_mode=WAL"))
        await conn.run_sync(Base.metadata.create_all)
        await apply_migrations(conn)


async def dispose_db() -> None:
    await engine.dispose()
