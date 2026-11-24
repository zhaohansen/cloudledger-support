# app/migrations.py
# 轻量版本化迁移：只做幂等的结构升级，保留已有数据。
# 说明：新库由 create_all 建全表；已有库在这里补列/补表/补索引，并记录 schema_migrations。
# 为什么暂时不用 Alembic：本项目是单服务、单数据库的演示工程，迁移只有少量增量；
# 轻量迁移更容易在 SQLite / PostgreSQL 上保持同一套可读步骤。若后续表结构频繁变化，
# 可以把这里的迁移按版本号平移成 Alembic revision，历史记录字段已经兼容。

from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncConnection

MIGRATIONS: list[tuple[int, str]] = [
    (1, "20261116_ticket_draft_version"),
    (2, "20261116_chunk_embedding_index"),
    (3, "20261116_ticket_message_request_key"),
]


async def apply_migrations(conn: AsyncConnection) -> None:
    await conn.execute(
        text(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "id INTEGER PRIMARY KEY, "
            "name VARCHAR(200) NOT NULL UNIQUE, "
            "applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
    )

    # 迁移 1：ticket_drafts.version，用于确认时校验草稿版本
    if not await conn.run_sync(_column_exists, "ticket_drafts", "version"):
        await conn.execute(
            text("ALTER TABLE ticket_drafts ADD COLUMN version INTEGER NOT NULL DEFAULT 1")
        )
    await _record(conn, 1)

    # 迁移 2：分块向量表。新库由 create_all 创建；这里只登记版本，
    # 并确保旧库在 create_all 之后也具备该表（幂等）。
    if not await conn.run_sync(_table_exists, "knowledge_chunk_embeddings"):
        await conn.execute(
            text(
                "CREATE TABLE knowledge_chunk_embeddings ("
                "id VARCHAR(32) PRIMARY KEY, "
                "chunk_id VARCHAR(32) NOT NULL REFERENCES knowledge_chunks(id) ON DELETE CASCADE, "
                "backend VARCHAR(32) NOT NULL, "
                "dim INTEGER NOT NULL DEFAULT 0, "
                "dense JSON NOT NULL, "
                "sparse JSON NOT NULL, "
                "created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "CONSTRAINT uq_chunk_embedding_backend UNIQUE (chunk_id, backend))"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_chunk_embeddings_backend "
                "ON knowledge_chunk_embeddings (backend)"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_knowledge_chunk_embeddings_chunk_id "
                "ON knowledge_chunk_embeddings (chunk_id)"
            )
        )
    await _record(conn, 2)

    # 迁移 3：工单消息幂等键，防止客户端重试重复写入回复/补充
    if not await conn.run_sync(_column_exists, "ticket_messages", "request_key"):
        await conn.execute(
            text("ALTER TABLE ticket_messages ADD COLUMN request_key VARCHAR(128)")
        )
    if not await conn.run_sync(_unique_index_exists, "ticket_messages", "uq_ticket_messages_request_key"):
        await conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_ticket_messages_request_key "
                "ON ticket_messages (request_key)"
            )
        )
    await _record(conn, 3)


async def _record(conn: AsyncConnection, migration_id: int) -> None:
    name = next(name for item_id, name in MIGRATIONS if item_id == migration_id)
    # 显式 id + ON CONFLICT 保证重复启动不会写重复记录
    await conn.execute(
        text(
            "INSERT INTO schema_migrations (id, name) VALUES (:id, :name) "
            "ON CONFLICT (id) DO NOTHING"
        ),
        {"id": migration_id, "name": name},
    )


def _table_exists(sync_conn, table_name: str) -> bool:
    return table_name in inspect(sync_conn).get_table_names()


def _column_exists(sync_conn, table_name: str, column_name: str) -> bool:
    inspector = inspect(sync_conn)
    if table_name not in inspector.get_table_names():
        return False
    return column_name in {column["name"] for column in inspector.get_columns(table_name)}


def _unique_index_exists(sync_conn, table_name: str, index_name: str) -> bool:
    inspector = inspect(sync_conn)
    if table_name not in inspector.get_table_names():
        return False
    names = {index["name"] for index in inspector.get_indexes(table_name) if index.get("unique")}
    names |= {item["name"] for item in inspector.get_unique_constraints(table_name)}
    return index_name in names
