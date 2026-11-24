# scripts/reindex_embeddings.py
# 为已有知识库重建向量索引（切换 Embedding 后端或首次启用 BGE-M3 时使用）。
# 用法：
#   python scripts/reindex_embeddings.py --backend bge_m3 --index local
#   python scripts/reindex_embeddings.py --backend bge_m3 --index milvus
# 安全：只重建向量行，不修改文档内容；不会删除业务资料。

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="重建知识库向量索引")
    parser.add_argument("--backend", choices=["hash", "bge_m3"], default=None)
    parser.add_argument("--index", choices=["local", "milvus"], default=None)
    parser.add_argument("--document-id", default=None, help="只重建指定文档")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


async def main() -> int:
    from sqlalchemy import delete, select

    from app.config import get_settings
    from app.db import AsyncSessionLocal, init_db
    from app.logging_config import get_logger
    from app.models import KnowledgeChunk, KnowledgeDocument, KnowledgeChunkEmbedding
    from app.services.embeddings import EmbeddingBackendUnavailable, get_embedding_backend
    from app.services.milvus_index import MilvusUnavailable, get_milvus_index
    from app.services.vector_index import store_chunk_embeddings

    logger = get_logger("reindex")
    settings = get_settings()
    backend = get_embedding_backend()
    print(f"实际后端: embedding={backend.name} retrieval={settings.retrieval_backend}")

    if settings.retrieval_backend == "milvus" and backend.name != "bge_m3":
        print("错误：milvus 需要 bge_m3 后端", file=sys.stderr)
        return 2

    await init_db()
    total_chunks = 0
    async with AsyncSessionLocal() as db:
        stmt = select(KnowledgeDocument).where(KnowledgeDocument.status == "ready")
        if args.document_id:
            stmt = stmt.where(KnowledgeDocument.id == args.document_id)
        documents = (await db.scalars(stmt)).all()
        if not documents:
            print("没有可重建的 ready 文档")
            return 0

        index = get_milvus_index(backend) if settings.retrieval_backend == "milvus" else None
        for document in documents:
            chunks = (
                await db.scalars(
                    select(KnowledgeChunk)
                    .where(KnowledgeChunk.document_id == document.id)
                    .order_by(KnowledgeChunk.chunk_index.asc())
                )
            ).all()
            if not chunks:
                continue
            chunk_ids = [chunk.id for chunk in chunks]
            await db.execute(
                delete(KnowledgeChunkEmbedding).where(
                    KnowledgeChunkEmbedding.chunk_id.in_(chunk_ids),
                    KnowledgeChunkEmbedding.backend == backend.name,
                )
            )

            for start in range(0, len(chunks), args.batch_size):
                batch = chunks[start : start + args.batch_size]
                try:
                    vectors = await asyncio.to_thread(
                        backend.encode_documents, [chunk.content for chunk in batch]
                    )
                except EmbeddingBackendUnavailable as exc:
                    print(f"向量化失败（文档 {document.id}）：{exc}", file=sys.stderr)
                    return 3
                await store_chunk_embeddings(
                    db,
                    [(chunk.id, vector) for chunk, vector in zip(batch, vectors)],
                    backend.name,
                )
                if index is not None:
                    rows = [
                        {
                            "chunk_id": chunk.id,
                            "document_id": document.id,
                            "tenant_id": document.tenant_id or "",
                            "access_scope": document.access_scope,
                            "product_version": document.product_version,
                            "dense": vector.dense,
                            "sparse": vector.sparse or {},
                        }
                        for chunk, vector in zip(batch, vectors)
                    ]
                    try:
                        await asyncio.to_thread(index.upsert, rows)
                    except MilvusUnavailable as exc:
                        print(f"Milvus 写入失败（文档 {document.id}）：{exc}", file=sys.stderr)
                        return 4
            if index is not None:
                # 分块 ID 不变，upsert 会覆盖同一主键；数据库中已不存在的旧向量
                # 会在检索时被业务库复核过滤，不会进入模型上下文。
                pass

            await db.commit()
            total_chunks += len(chunks)
            print(f"已重建 {document.title}（{len(chunks)} 块）")

    print(f"完成：{len(documents)} 个文档，{total_chunks} 个分块，backend={backend.name}")
    logger.info("reindex.done", documents=len(documents), chunks=total_chunks)
    return 0


if __name__ == "__main__":
    args = parse_args()
    if args.backend:
        os.environ["EMBEDDING_BACKEND"] = args.backend
    if args.index:
        os.environ["RETRIEVAL_BACKEND"] = args.index
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(130)
