# tests/test_knowledge.py
# 覆盖：跨租户知识库隔离、文档更新/删除后旧内容不可检索。

from __future__ import annotations

import uuid
from types import SimpleNamespace

from sqlalchemy import event, select

from app.db import AsyncSessionLocal, engine
from app.models import KnowledgeChunk, KnowledgeDocument, new_id
from app.services import retrieval
from app.services.embeddings import EmbeddedText
from app.services.retrieval import search_knowledge
from app.services.vector_index import store_chunk_embeddings
from tests.helpers import auth, ctx_for, token_for


async def test_knowledge_cross_tenant_isolation():
    """7. 客户只能检索到公共资料和本租户私有资料，不能跨租户。"""
    async with AsyncSessionLocal() as db:
        ctx_a = await ctx_for("customer_a")
        outcome_a = await search_knowledge(db, ctx_a, "苏州研发中心 月结 拍照 OCR", top_k=8)
        titles_a = [item.title for item in outcome_a.items]
        assert any("华辰智造费控内控制度" in title for title in titles_a)
        assert all("星海文化报销管理细则" not in title for title in titles_a)

        ctx_b = await ctx_for("customer_b")
        outcome_b = await search_knowledge(db, ctx_b, "苏州研发中心 月结 拍照 OCR", top_k=8)
        titles_b = [item.title for item in outcome_b.items]
        assert all("华辰智造费控内控制度" not in title for title in titles_b)


async def test_document_update_and_delete_removes_old_content(client):
    """8. 文档更新后旧内容不再命中；删除后新内容也不再命中。"""
    admin_token = await token_for("admin")
    marker = "临时标记" + uuid.uuid4().hex[:10]
    new_marker = "更新标记" + uuid.uuid4().hex[:10]

    content = f"# 临时测试资料\n\n这是一份用于测试的模拟文档，唯一标记：{marker}。".encode("utf-8")
    created = await client.post(
        "/api/knowledge/documents",
        headers=auth(admin_token),
        files={"file": ("tmp-doc.md", content, "text/markdown")},
        data={
            "title": f"临时测试资料 {marker}",
            "doc_type": "faq",
            "version": "test",
            "product_version": "all",
            "access_scope": "public",
        },
    )
    assert created.status_code == 200, created.text
    document = created.json()
    assert document["status"] == "ready"
    document_id = document["id"]

    hit = await client.post(
        "/api/knowledge/search", headers=auth(admin_token), json={"query": marker}
    )
    assert hit.status_code == 200
    assert any(marker in item["content"] or marker in item.get("snippet", "") for item in hit.json()["items"])

    # 更新为全新内容，旧分块应被替换
    new_content = f"# 更新后的临时资料\n\n内容已经完全替换，新标记：{new_marker}。".encode("utf-8")
    updated = await client.put(
        f"/api/knowledge/documents/{document_id}",
        headers=auth(admin_token),
        files={"file": ("tmp-doc.md", new_content, "text/markdown")},
        data={"title": "更新后的临时资料"},
    )
    assert updated.status_code == 200, updated.text

    old_hit = await client.post(
        "/api/knowledge/search", headers=auth(admin_token), json={"query": marker}
    )
    assert all(marker not in item["content"] for item in old_hit.json()["items"])

    new_hit = await client.post(
        "/api/knowledge/search", headers=auth(admin_token), json={"query": new_marker}
    )
    assert any(new_marker in item["content"] for item in new_hit.json()["items"])

    deleted = await client.delete(f"/api/knowledge/documents/{document_id}", headers=auth(admin_token))
    assert deleted.status_code == 200

    after_delete = await client.post(
        "/api/knowledge/search", headers=auth(admin_token), json={"query": new_marker}
    )
    assert all(new_marker not in item["content"] for item in after_delete.json()["items"])


async def test_milvus_fetches_only_recalled_chunks_and_rechecks_access(monkeypatch):
    ctx = await ctx_for("customer_a")
    vector = EmbeddedText(dense=[1.0, 0.0], sparse={1: 1.0})
    statements = []

    async with AsyncSessionLocal() as db:
        chunk_ids = []
        for condition in (
            KnowledgeDocument.product_version == "v2.2",
            KnowledgeDocument.product_version == "v2.3",
            KnowledgeDocument.title.contains("星海文化报销管理细则"),
        ):
            chunk_id = await db.scalar(
                select(KnowledgeChunk.id)
                .join(KnowledgeDocument)
                .where(condition)
                .limit(1)
            )
            assert chunk_id
            chunk_ids.append(chunk_id)
        allowed, wrong_version, other_tenant = chunk_ids
        await store_chunk_embeddings(db, [(chunk_id, vector) for chunk_id in chunk_ids], "bge_m3")
        await db.flush()

        def recall(dense, sparse, tenants, version, top_k):
            assert tenants == [ctx.tenant_id]
            assert version == "v2.2"
            return [
                {"chunk_id": chunk_id}
                for chunk_id in (other_tenant, wrong_version, new_id(), allowed)
            ]

        backend = SimpleNamespace(name="bge_m3", encode_query=lambda _: vector)
        monkeypatch.setattr(retrieval.settings, "retrieval_backend", "milvus")
        monkeypatch.setattr(retrieval.settings, "rerank_enabled", False)
        monkeypatch.setattr(retrieval, "get_embedding_backend", lambda: backend)
        monkeypatch.setattr(retrieval, "get_milvus_index", lambda _: SimpleNamespace(search=recall))

        def record_query(conn, cursor, statement, parameters, context, executemany):
            if "knowledge_chunks.content" in statement:
                statements.append(statement)

        event.listen(engine.sync_engine, "before_cursor_execute", record_query)
        try:
            outcome = await search_knowledge(db, ctx, "预算", product_version="v2.2")
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", record_query)
            await db.rollback()

    assert outcome.reason == "ok"
    assert [item.chunk_id for item in outcome.items] == [allowed]
    assert len(statements) == 1
    assert "knowledge_chunks.id IN (" in statements[0]


async def test_milvus_metadata_update_reuses_vectors_and_failure_preserves_file(client, monkeypatch):
    from pathlib import Path

    from app.services import knowledge

    headers = auth(await token_for("admin"))
    original = ("# Test document\n\nOriginal data " + uuid.uuid4().hex).encode()
    created = await client.post(
        "/api/knowledge/documents",
        headers=headers,
        files={"file": ("original.md", original, "text/markdown")},
        data={"title": "Milvus metadata test", "product_version": "v2.2"},
    )
    assert created.status_code == 200, created.text
    document_id = created.json()["id"]
    url = f"/api/knowledge/documents/{document_id}"
    async with AsyncSessionLocal() as db:
        document = await db.get(KnowledgeDocument, document_id)
        original_path = Path(document.stored_path)
        old_ids = set((await db.scalars(
            select(KnowledgeChunk.id).where(KnowledgeChunk.document_id == document_id)
        )).all())

    indexed = {}
    cleaned = []
    def sync_milvus(document, pairs):
        indexed.update({
            chunk.id: (document.product_version, vector)
            for chunk, vector in pairs
        })

    def cleanup_milvus(ids):
        cleaned.extend(ids)
        for chunk_id in ids:
            indexed.pop(chunk_id, None)

    with monkeypatch.context() as patch:
        patch.setattr(knowledge.settings, "retrieval_backend", "milvus")
        patch.setattr(knowledge, "_sync_milvus", sync_milvus)
        patch.setattr(knowledge, "get_milvus_index", lambda backend: SimpleNamespace(delete_chunks=cleanup_milvus))
        updated = await client.put(url, headers=headers, data={"product_version": "v2.3"})
        assert updated.status_code == 200, updated.text
        assert updated.json()["product_version"] == "v2.3"
        assert old_ids <= set(cleaned)
        replacement_ids = set(indexed)
        assert replacement_ids and not replacement_ids.intersection(old_ids)
        assert all(version == "v2.3" and vector.dense for version, vector in indexed.values())

        def fail_sync(document, pairs):
            sync_milvus(document, pairs)
            raise knowledge.VectorIndexError("index write failed")

        patch.setattr(knowledge, "_sync_milvus", fail_sync)
        failed = await client.put(
            url, headers=headers, files={"file": ("new.md", b"# New\n\nReplacement data", "text/markdown")}
        )
        assert failed.status_code == 503, failed.text
        assert set(indexed) == replacement_ids
        assert original_path.read_bytes() == original
        assert not list(original_path.parent.glob(f"{document_id}-*"))
        async with AsyncSessionLocal() as db:
            saved = await db.get(KnowledgeDocument, document_id)
            assert saved.product_version == "v2.3"
            assert saved.stored_path == str(original_path)
            assert set((await db.scalars(
                select(KnowledgeChunk.id).where(KnowledgeChunk.document_id == document_id)
            )).all()) == replacement_ids

    deleted = await client.delete(url, headers=headers)
    assert deleted.status_code == 200, deleted.text


def test_long_paragraph_has_no_duplicate_tail():
    from app.services.text_utils import chunk_text

    text = "甲" * 1520
    chunks = chunk_text(text, size=600, overlap=80)
    assert [len(chunk) for chunk in chunks] == [600, 600, 480]

    text = "乙" * 1120
    assert chunk_text(text) == [text[:600], text[520:]]
