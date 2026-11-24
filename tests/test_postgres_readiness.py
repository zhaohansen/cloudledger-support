# tests/test_postgres_readiness.py
# PostgreSQL 就绪检查：
#   - 驱动是否安装、能否创建异步引擎（不连接）；
#   - 模型 DDL 能否被 PostgreSQL 方言编译，关键约束是否齐全；
#   - 设置 TEST_POSTGRES_URL 时才运行真实连接/事务测试，否则明确跳过。
#
# 说明：本机 5432 端口未运行 PostgreSQL，因此真实事务行为默认不在此处验证，
# 不能把"DDL 能编译"当成"已通过 PostgreSQL 集成测试"。

from __future__ import annotations

import os

import pytest
from sqlalchemy import text
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.schema import CreateIndex, CreateTable

from app.migrations import apply_migrations
from app.models import Base


def test_postgres_ddl_compiles_with_expected_constraints():
    dialect = postgresql.dialect()
    ddl = "\n".join(str(CreateTable(table).compile(dialect=dialect)) for table in Base.metadata.sorted_tables)
    indexes = "\n".join(
        str(CreateIndex(index).compile(dialect=dialect))
        for table in Base.metadata.sorted_tables
        for index in table.indexes
    )

    assert "CREATE TABLE tickets" in ddl
    assert "ticket_no VARCHAR(32) NOT NULL" in ddl
    assert "UNIQUE (request_key)" in ddl, "工单消息幂等键必须有唯一约束"
    assert "UNIQUE (chunk_id, backend)" in ddl, "同一分块同一后端只能有一条向量"
    assert "version INTEGER NOT NULL" in ddl, "草稿版本不能为空"
    assert "ON DELETE CASCADE" in ddl
    assert "uq_ticket_messages_request_key" in ddl
    assert "ix_ticket_messages_ticket_created" in indexes


async def test_asyncpg_driver_can_create_engine_without_connecting():
    pytest.importorskip("asyncpg", reason="未安装 asyncpg 驱动")
    engine = create_async_engine(
        "postgresql+asyncpg://demo:demo@127.0.0.1:5432/cloudledger",
        pool_pre_ping=True,
    )
    try:
        assert engine.dialect.name == "postgresql"
        assert engine.dialect.driver == "asyncpg"
    finally:
        await engine.dispose()


@pytest.mark.skipif(
    not os.getenv("TEST_POSTGRES_URL"),
    reason="未设置 TEST_POSTGRES_URL，PostgreSQL 真实连接与事务测试未运行",
)
async def test_postgres_live_migration_and_transaction():
    """只有显式提供测试库时才执行；不会清空生产库，使用独立 schema。"""
    url = os.environ["TEST_POSTGRES_URL"]
    engine = create_async_engine(url)
    schema = "cloudledger_eval_check"
    try:
        async with engine.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
            await conn.execute(text(f'SET search_path TO "{schema}"'))
            await conn.run_sync(Base.metadata.create_all)
            await apply_migrations(conn)
            await apply_migrations(conn)  # 幂等
            count = (
                await conn.execute(text("SELECT COUNT(*) FROM schema_migrations"))
            ).scalar_one()
            assert count == 3

            # 事务回滚：写操作失败后不应留下半成品
            await conn.execute(
                text(
                    "INSERT INTO ticket_sequences (key, value) VALUES ('smoke', 1)"
                )
            )
            # 重复主键在 savepoint 内失败：捕获后外层事务必须仍可继续使用
            from sqlalchemy.exc import IntegrityError

            with pytest.raises(IntegrityError):
                async with conn.begin_nested():
                    await conn.execute(
                        text("INSERT INTO ticket_sequences (key, value) VALUES ('smoke', 2)")
                    )
            rows = (
                await conn.execute(
                    text("SELECT COUNT(*) FROM ticket_sequences WHERE key = 'smoke'")
                )
            ).scalar_one()
            assert rows == 1, "savepoint 回滚后不应留下重复行"
    finally:
        async with engine.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await engine.dispose()


@pytest.mark.skipif(
    not os.getenv("TEST_POSTGRES_URL"),
    reason="未设置 TEST_POSTGRES_URL，PostgreSQL 真实并发测试未运行",
)
async def test_postgres_live_confirm_version_and_concurrency():
    """在独立 schema 中验证条件更新、并发确认、编辑/取消竞争和事务回滚。

    只创建并删除 cloudledger_pg_app 这个隔离 schema，不触碰业务表和数据。
    """
    import asyncio
    import uuid

    from fastapi import HTTPException
    from sqlalchemy import func, select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.deps import UserContext
    from app.models import Tenant, Ticket, TicketEvent, User
    from app.services.tickets import (
        ToolError,
        cancel_draft,
        create_ticket_from_draft,
        get_or_create_draft,
        update_draft,
    )

    url = os.environ["TEST_POSTGRES_URL"]
    schema = "cloudledger_pg_app"
    admin = create_async_engine(url)
    # search_path 只放隔离 schema：如果带上 public，create_all 会把业务库里的
    # 同名表当成“已存在”而跳过，随后外键解析到 public 的表，类型对不上。
    engine = create_async_engine(
        url, connect_args={"server_settings": {"search_path": schema}}
    )
    async with admin.begin() as conn:
        await conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await apply_migrations(conn)
            await apply_migrations(conn)  # 重复迁移必须安全

        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as db:
            tenant = Tenant(name="PG 并发验证租户", code=f"pg-{uuid.uuid4().hex[:8]}")
            db.add(tenant)
            await db.flush()
            user = User(
                tenant_id=tenant.id,
                username=f"pg_{uuid.uuid4().hex[:8]}",
                display_name="PG 测试用户",
                password_hash="not-used",
                role="customer",
            )
            db.add(user)
            await db.flush()
            await db.commit()
            ctx = UserContext(
                user_id=user.id,
                username=user.username,
                display_name=user.display_name,
                role=user.role,
                tenant_id=tenant.id,
                tenant_name=tenant.name,
            )

        async def make_draft(description: str) -> tuple[str, int]:
            async with session_factory() as db:
                draft = await get_or_create_draft(
                    db,
                    ctx,
                    session_id=None,
                    title="PG 并发草稿",
                    description=description,
                    product_version="v2.3",
                    error_message="PG 验证",
                    tried_steps="PG 验证",
                    category="other",
                    priority="medium",
                )
                await db.commit()
                return draft.id, draft.version

        async def confirm_worker(draft_id: str, version: int):
            async with session_factory() as db:
                try:
                    ticket, already = await create_ticket_from_draft(
                        db, ctx, draft_id, version
                    )
                    await db.commit()
                    return ("ok", ticket.ticket_no, already)
                except ToolError as exc:
                    await db.rollback()
                    return ("tool_error", exc.code, None)
                except Exception as exc:  # 数据库异常必须暴露，不能算通过
                    await db.rollback()
                    return ("exception", type(exc).__name__, None)

        async def ticket_count(draft_id: str) -> int:
            async with session_factory() as db:
                return int(
                    await db.scalar(
                        select(func.count())
                        .select_from(Ticket)
                        .where(Ticket.draft_id == draft_id)
                    )
                    or 0
                )

        # 1) 同一草稿并发确认：只能生成一张工单，两边拿到同一个单号
        draft_id, version = await make_draft("PG 同草稿并发确认 " + uuid.uuid4().hex)
        results = await asyncio.gather(
            confirm_worker(draft_id, version), confirm_worker(draft_id, version)
        )
        assert all(item[0] == "ok" for item in results), results
        assert len({item[1] for item in results}) == 1, results
        assert sorted(item[2] for item in results) == [False, True], results
        assert await ticket_count(draft_id) == 1

        # 2) 不同草稿并发：都能成功，且工单号不冲突
        draft_a, version_a = await make_draft("PG 不同草稿 A " + uuid.uuid4().hex)
        draft_b, version_b = await make_draft("PG 不同草稿 B " + uuid.uuid4().hex)
        result_a, result_b = await asyncio.gather(
            confirm_worker(draft_a, version_a), confirm_worker(draft_b, version_b)
        )
        assert result_a[0] == "ok" and result_b[0] == "ok", (result_a, result_b)
        assert result_a[1] != result_b[1]

        # 3) 编辑提交后，旧版本确认必须失败，且不会写入新内容
        draft_c, version_c = await make_draft("PG 编辑确认竞争 " + uuid.uuid4().hex)
        async with session_factory() as db:
            await update_draft(
                db, ctx, draft_c, expected_version=version_c, description="PG 修改后的内容"
            )
            await db.commit()
        async with session_factory() as db:
            with pytest.raises(ToolError) as excinfo:
                await create_ticket_from_draft(db, ctx, draft_c, version_c)
            await db.rollback()
        assert excinfo.value.code == "draft_version_conflict"
        assert await ticket_count(draft_c) == 0
        result_c = await confirm_worker(draft_c, version_c + 1)
        assert result_c[0] == "ok", result_c

        # 4) 两个编辑基于同一版本并发：只能一个成功，另一个收到冲突
        draft_d, version_d = await make_draft("PG 双编辑竞争 " + uuid.uuid4().hex)

        async def edit_worker(text: str):
            async with session_factory() as db:
                try:
                    await update_draft(db, ctx, draft_d, expected_version=version_d, description=text)
                    await db.commit()
                    return "ok"
                except HTTPException:
                    await db.rollback()
                    return "conflict"
                except Exception as exc:
                    await db.rollback()
                    return type(exc).__name__

        edit_results = await asyncio.gather(edit_worker("PG 编辑 A"), edit_worker("PG 编辑 B"))
        assert sorted(edit_results) == ["conflict", "ok"], edit_results

        # 5) 取消与确认并发：只有一个生效，工单/草稿状态一致
        draft_e, version_e = await make_draft("PG 取消确认竞争 " + uuid.uuid4().hex)

        async def cancel_worker():
            async with session_factory() as db:
                try:
                    await cancel_draft(db, ctx, draft_e, expected_version=version_e)
                    await db.commit()
                    return "cancelled"
                except HTTPException:
                    await db.rollback()
                    return "blocked"

        cancel_result, confirm_result = await asyncio.gather(
            cancel_worker(), confirm_worker(draft_e, version_e)
        )
        async with session_factory() as db:
            from app.models import TicketDraft

            draft_row = await db.get(TicketDraft, draft_e)
        count_e = await ticket_count(draft_e)
        if cancel_result == "cancelled":
            assert confirm_result[0] == "tool_error", confirm_result
            assert draft_row.status == "cancelled" and count_e == 0
        else:
            assert confirm_result[0] == "ok", confirm_result
            assert draft_row.status == "confirmed" and count_e == 1

        # 6) 事务回滚：工单、事件、草稿状态都不落库
        draft_f, version_f = await make_draft("PG 事务回滚 " + uuid.uuid4().hex)
        async with session_factory() as db:
            ticket, already = await create_ticket_from_draft(db, ctx, draft_f, version_f)
            assert ticket.ticket_no and already is False
            await db.rollback()
        assert await ticket_count(draft_f) == 0
        async with session_factory() as db:
            draft_row = await db.get(TicketDraft, draft_f)
            event_count = await db.scalar(
                select(func.count()).select_from(TicketEvent).where(TicketEvent.ticket_id == ticket.id)
            )
        assert draft_row.status == "draft"
        assert event_count == 0
    finally:
        async with admin.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await engine.dispose()
        await admin.dispose()
