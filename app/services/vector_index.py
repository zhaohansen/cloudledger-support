# app/services/vector_index.py
# 分块向量的持久化与读取。
# 本地模式把向量存业务库（knowledge_chunk_embeddings），按 backend 分开；
# Milvus 模式把同一份向量同步到向量库，检索时以业务库为准做权限与版本复核。

from __future__ import annotations

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import KnowledgeChunkEmbedding
from app.services.embeddings import EmbeddedText


def sparse_to_json(sparse: dict[int, float] | None) -> list[list[float]]:
    return [[int(token), float(weight)] for token, weight in (sparse or {}).items()]


def sparse_from_json(items: list | None) -> dict[int, float]:
    result: dict[int, float] = {}
    for pair in items or []:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            result[int(pair[0])] = float(pair[1])
    return result


async def store_chunk_embeddings(
    db: AsyncSession,
    chunk_vectors: list[tuple[str, EmbeddedText]],
    backend: str,
) -> None:
    """写入分块向量。调用方负责保证 chunk_id 属于同一事务中新写入的分块。"""
    for chunk_id, vector in chunk_vectors:
        db.add(
            KnowledgeChunkEmbedding(
                chunk_id=chunk_id,
                backend=backend,
                dim=len(vector.dense),
                dense=vector.dense,
                sparse=sparse_to_json(vector.sparse),
            )
        )
    await db.flush()


async def load_chunk_embeddings(
    db: AsyncSession,
    chunk_ids: list[str],
    backend: str,
) -> dict[str, EmbeddedText]:
    if not chunk_ids:
        return {}
    rows = (
        await db.execute(
            select(KnowledgeChunkEmbedding).where(
                KnowledgeChunkEmbedding.chunk_id.in_(chunk_ids),
                KnowledgeChunkEmbedding.backend == backend,
            )
        )
    ).scalars().all()
    return {
        row.chunk_id: EmbeddedText(
            dense=list(row.dense or []),
            sparse=sparse_from_json(row.sparse),
        )
        for row in rows
    }


async def delete_chunk_embeddings(
    db: AsyncSession, chunk_ids: list[str], backend: str | None = None
) -> None:
    if not chunk_ids:
        return
    stmt = delete(KnowledgeChunkEmbedding).where(KnowledgeChunkEmbedding.chunk_id.in_(chunk_ids))
    if backend:
        stmt = stmt.where(KnowledgeChunkEmbedding.backend == backend)
    await db.execute(stmt)


async def index_stats(db: AsyncSession, backend: str) -> dict:
    """返回当前后端的索引覆盖情况，供健康检查与评测报告使用。"""
    chunk_count = await db.scalar(select(func.count()).select_from(KnowledgeChunkEmbedding))
    vector_count = await db.scalar(
        select(func.count())
        .select_from(KnowledgeChunkEmbedding)
        .where(KnowledgeChunkEmbedding.backend == backend)
    )
    return {
        "backend": backend,
        "vector_rows": int(vector_count or 0),
        "all_vector_rows": int(chunk_count or 0),
    }
