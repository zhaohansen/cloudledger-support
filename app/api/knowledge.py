# app/api/knowledge.py
# 知识库管理 API（仅管理员）：上传、查询状态、更新、删除、检索调试。

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.uploads import read_upload
from app.db import get_db
from app.deps import UserContext, require_roles
from app.models import KnowledgeChunk, KnowledgeDocument
from app.services import knowledge as knowledge_service
from app.services.knowledge import DocumentParseError, DuplicateDocumentError
from app.services.retrieval import search_knowledge

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    product_version: str | None = Field(None, max_length=32)


@router.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    doc_type: str = Form("manual"),
    version: str = Form("1.0"),
    product_version: str = Form("all"),
    access_scope: str = Form("public"),
    tenant_id: str | None = Form(None),
    user: UserContext = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    content = await read_upload(file)
    try:
        document = await knowledge_service.ingest_document(
            db,
            user,
            title=title,
            filename=file.filename or "document.md",
            content=content,
            doc_type=doc_type,
            version=version,
            product_version=product_version,
            access_scope=access_scope,
            tenant_id=tenant_id,
        )
    except DuplicateDocumentError as exc:
        raise HTTPException(status_code=409, detail=f"相同内容的文档已存在，id={exc.document_id}")
    except DocumentParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # 解析失败也返回 200 + status=failed，管理员能在列表里看到原因
    return knowledge_service.serialize_document(document)


@router.get("/documents")
async def list_documents(
    status_filter: str | None = None,
    doc_type: str | None = None,
    tenant_id: str | None = None,
    user: UserContext = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    documents = await knowledge_service.list_documents(
        db, status_filter=status_filter, doc_type=doc_type, tenant_id=tenant_id
    )
    return {"items": [knowledge_service.serialize_document(d) for d in documents]}


@router.get("/documents/{document_id}")
async def get_document(
    document_id: str,
    user: UserContext = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    document = await db.get(KnowledgeDocument, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    data = knowledge_service.serialize_document(document)
    chunks = (
        await db.execute(
            select(KnowledgeChunk)
            .where(KnowledgeChunk.document_id == document_id)
            .order_by(KnowledgeChunk.chunk_index.asc())
            .limit(20)
        )
    ).scalars().all()
    data["chunks"] = [
        {
            "id": c.id,
            "chunk_index": c.chunk_index,
            "section": c.section,
            "content": c.content[:300],
        }
        for c in chunks
    ]
    return data


@router.put("/documents/{document_id}")
async def update_document(
    document_id: str,
    file: UploadFile | None = File(None),
    title: str | None = Form(None),
    doc_type: str | None = Form(None),
    version: str | None = Form(None),
    product_version: str | None = Form(None),
    access_scope: str | None = Form(None),
    tenant_id: str | None = Form(None),
    user: UserContext = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    content = await read_upload(file) if file is not None else None
    try:
        document = await knowledge_service.update_document(
            db,
            user,
            document_id,
            title=title,
            doc_type=doc_type,
            version=version,
            product_version=product_version,
            access_scope=access_scope,
            # Form(None) 无法区分"不传"和"显式清空"，这里只支持修改为有效租户
            tenant_id=tenant_id,
            filename=file.filename if file else None,
            content=content,
        )
    except DuplicateDocumentError as exc:
        raise HTTPException(status_code=409, detail=f"相同内容的文档已存在，id={exc.document_id}")
    return knowledge_service.serialize_document(document)


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    user: UserContext = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    await knowledge_service.delete_document(db, user, document_id)
    return {"ok": True, "message": "文档及其向量已删除"}


@router.post("/search")
async def search(
    req: SearchRequest,
    user: UserContext = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    """管理员的检索调试接口，返回分块、分数与来源，便于核对引用。"""
    outcome = await search_knowledge(
        db, user, req.query, product_version=req.product_version, top_k=8
    )
    return {
        "reason": outcome.reason,
        "requested_version": outcome.requested_version,
        "matched_versions": outcome.matched_versions,
        "version_conflict": outcome.version_conflict,
        # 实际后端（而不是配置里的期望值），方便确认是否真的走了语义检索
        "embedding_backend": outcome.embedding_backend,
        "retrieval_backend": outcome.retrieval_backend,
        "rerank_status": outcome.rerank_status,
        "degraded": outcome.degraded,
        "detail": outcome.detail,
        "items": [
            {
                **item.to_citation(index + 1),
                "dense_score": round(item.dense_score, 4),
                "lexical_score": round(item.lexical_score, 4),
                "sparse_score": round(item.sparse_score, 4),
                "score_kind": item.score_kind,
                "content": item.content,
            }
            for index, item in enumerate(outcome.items)
        ],
    }
