# tests/test_migrations.py
# 验证轻量版本化迁移：旧库补列/补表/补索引后数据不丢，并且重复执行幂等。

from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine

from app.migrations import apply_migrations


async def test_migrations_preserve_data_and_are_idempotent(tmp_path):
    db_path = tmp_path / "legacy.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path.as_posix()}")

    # 模拟旧版本数据库：缺少 ticket_drafts.version、ticket_messages.request_key
    async with engine.begin() as conn:
        await conn.execute(
            text("CREATE TABLE ticket_drafts (id VARCHAR(32) PRIMARY KEY, title VARCHAR(200))")
        )
        await conn.execute(
            text("INSERT INTO ticket_drafts (id, title) VALUES ('d-legacy', '旧草稿不能丢')")
        )
        await conn.execute(
            text("CREATE TABLE ticket_messages (id VARCHAR(32) PRIMARY KEY, content TEXT)")
        )
        await conn.execute(
            text("INSERT INTO ticket_messages (id, content) VALUES ('m-legacy', '旧消息不能丢')")
        )
        await conn.execute(text("CREATE TABLE knowledge_chunks (id VARCHAR(32) PRIMARY KEY)"))
        await apply_migrations(conn)

    # 重复执行不能报错，也不能重复写迁移记录
    async with engine.begin() as conn:
        await apply_migrations(conn)

        draft = (
            await conn.execute(
                text("SELECT title, version FROM ticket_drafts WHERE id = 'd-legacy'")
            )
        ).one()
        assert draft[0] == "旧草稿不能丢"
        assert draft[1] == 1, "旧数据应补上默认版本 1"

        message = (
            await conn.execute(
                text("SELECT content, request_key FROM ticket_messages WHERE id = 'm-legacy'")
            )
        ).one()
        assert message[0] == "旧消息不能丢"
        assert message[1] is None

        migration_count = (
            await conn.execute(text("SELECT COUNT(*) FROM schema_migrations"))
        ).scalar_one()
        assert migration_count == 3

        tables = await conn.run_sync(lambda sync_conn: inspect(sync_conn).get_table_names())
        draft_columns = await conn.run_sync(
            lambda sync_conn: {column["name"] for column in inspect(sync_conn).get_columns("ticket_drafts")}
        )
        message_columns = await conn.run_sync(
            lambda sync_conn: {column["name"] for column in inspect(sync_conn).get_columns("ticket_messages")}
        )
        assert "knowledge_chunk_embeddings" in tables
        assert "version" in draft_columns
        assert "request_key" in message_columns

    await engine.dispose()
