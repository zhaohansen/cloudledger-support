# app/services/knowledge.py
# 知识文档的导入、解析、切分、更新、删除。
# 设计取舍：演示项目把向量存进业务库（JSON 字段），避免依赖 Milvus；
# 检索函数在 retrieval.py 中先按权限过滤，再计算相似度。

from __future__ import annotations

import asyncio
import hashlib
import re
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.deps import UserContext
from app.logging_config import get_logger
from app.models import KnowledgeChunk, KnowledgeDocument, Tenant, new_id, utcnow
from app.services.embeddings import (
    EmbeddingBackendUnavailable,
    EmbeddedText,
    get_embedding_backend,
)
from app.services.milvus_index import MilvusUnavailable, get_milvus_index
from app.services.text_utils import (
    chunk_text,
    embed_text,
    split_markdown_sections,
    split_plain_sections,
)
from app.services.vector_index import load_chunk_embeddings, store_chunk_embeddings

settings = get_settings()
logger = get_logger(__name__)

ALLOWED_DOC_TYPES = {"manual", "release", "faq", "private"}
ALLOWED_SCOPES = {"public", "tenant"}


class DocumentParseError(Exception):
    """文件无法解析（格式错误、扫描版 PDF、空文本等）。"""


class VectorIndexError(Exception):
    """向量化或向量库写入失败。明确报错，不静默退回演示检索。"""


class DuplicateDocumentError(Exception):
    def __init__(self, document_id: str):
        super().__init__("相同内容的文档已存在")
        self.document_id = document_id


def _validate_retrieval_config() -> None:
    if settings.retrieval_backend == "milvus" and settings.embedding_backend != "bge_m3":
        raise VectorIndexError(
            "RETRIEVAL_BACKEND=milvus 需要 EMBEDDING_BACKEND=bge_m3（Milvus 需要 dense + sparse 向量）"
        )


async def _embed_texts(texts: list[str]) -> tuple[list[EmbeddedText], str]:
    """统一定义“导入时向量化”的入口，失败时抛出可解释错误。"""
    backend = get_embedding_backend()
    try:
        vectors = await asyncio.to_thread(backend.encode_documents, texts)
    except EmbeddingBackendUnavailable as exc:
        raise VectorIndexError(str(exc)) from exc
    return vectors, backend.name


def _sync_milvus(document: KnowledgeDocument, pairs: list[tuple[KnowledgeChunk, EmbeddedText]]) -> None:
    """把新分块写入 Milvus。失败抛出 VectorIndexError，由上层回滚事务。"""
    backend = get_embedding_backend()
    index = get_milvus_index(backend)
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
        for chunk, vector in pairs
    ]
    try:
        index.upsert(rows)
    except MilvusUnavailable as exc:
        raise VectorIndexError(str(exc)) from exc


async def _add_chunks(
    db: AsyncSession,
    document: KnowledgeDocument,
    chunk_specs: list[tuple[str, str]],
) -> list[tuple[KnowledgeChunk, EmbeddedText]]:
    """向量化后写入分块与索引；向量化失败时不会留下半截分块。"""
    texts = [text for _, text in chunk_specs]
    vectors, backend_name = await _embed_texts(texts)
    return await _add_chunks_with_vectors(db, document, chunk_specs, vectors, backend_name)


async def _add_chunks_with_vectors(
    db: AsyncSession,
    document: KnowledgeDocument,
    chunk_specs: list[tuple[str, str]],
    vectors: list[EmbeddedText],
    backend_name: str,
    *,
    sync_milvus: bool = True,
) -> list[tuple[KnowledgeChunk, EmbeddedText]]:
    pairs: list[tuple[KnowledgeChunk, EmbeddedText]] = []
    for index, ((section, text), vector) in enumerate(zip(chunk_specs, vectors)):
        chunk = KnowledgeChunk(
            id=new_id(),
            document_id=document.id,
            chunk_index=index,
            section=section[:200],
            content=text,
            # 兼容字段只保存 hash 演示向量；真实向量在 knowledge_chunk_embeddings
            embedding=embed_text(text, settings.embedding_dim) if backend_name == "hash" else [],
        )
        db.add(chunk)
        pairs.append((chunk, vector))

    await db.flush()
    await store_chunk_embeddings(
        db, [(chunk.id, vector) for chunk, vector in pairs], backend_name
    )
    if sync_milvus and settings.retrieval_backend == "milvus":
        await asyncio.to_thread(_sync_milvus, document, pairs)
    return pairs


def _normalize_product_version(value: str) -> str:
    value = (value or "").strip().lower()
    if value in ("", "all", "通用", "全版本"):
        return "all"
    if re.fullmatch(r"\d+\.\d+", value):
        return f"v{value}"
    return value


def parse_document(filename: str, content: bytes) -> list[tuple[str, str]]:
    """把上传文件解析成 [(章节, 文本)]。

    PDF 用 PyMuPDF 提取文本；若提取不到足够文字，明确提示可能是扫描版（不做 OCR）。
    """
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:  # pragma: no cover
            raise DocumentParseError("服务器未安装 PyMuPDF，无法解析 PDF") from exc

        try:
            with fitz.open(stream=content, filetype="pdf") as pdf:
                pages = [page.get_text("text") for page in pdf]
        except Exception as exc:
            raise DocumentParseError(f"PDF 解析失败：{exc}") from exc

        text = "\n\n".join(pages).strip()
        if len(re.sub(r"\s", "", text)) < 30:
            raise DocumentParseError(
                "PDF 中没有提取到可用文字，可能是扫描版 PDF。当前版本暂不支持 OCR，"
                "请上传可复制文字的 PDF 或转换为 Markdown/TXT 后重试。"
            )
        return split_plain_sections(text)

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("gb18030")
        except UnicodeDecodeError as exc:
            raise DocumentParseError("文件编码无法识别，请使用 UTF-8 编码") from exc

    if not text.strip():
        raise DocumentParseError("文件内容为空")
    if suffix in (".md", ".markdown"):
        return split_markdown_sections(text)
    return split_plain_sections(text)


def build_chunks(sections: list[tuple[str, str]]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for section, text in sections:
        if not text.strip():
            continue
        for piece in chunk_text(text, settings.chunk_size, settings.chunk_overlap):
            result.append((section, piece))
    return result


def _document_chunks(filename: str, content: bytes) -> list[tuple[str, str]]:
    chunks = build_chunks(parse_document(filename, content))
    if not chunks:
        raise DocumentParseError("未提取到有效文本")
    return chunks


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _validate_meta(*, doc_type: str, access_scope: str, tenant_id: str | None) -> None:
    if doc_type not in ALLOWED_DOC_TYPES:
        raise HTTPException(status_code=422, detail=f"doc_type 必须是 {sorted(ALLOWED_DOC_TYPES)} 之一")
    if access_scope not in ALLOWED_SCOPES:
        raise HTTPException(status_code=422, detail="access_scope 必须是 public 或 tenant")
    if access_scope == "tenant" and not tenant_id:
        raise HTTPException(status_code=422, detail="私有资料必须指定 tenant_id")


async def _assert_tenant_exists(db: AsyncSession, tenant_id: str | None) -> None:
    if tenant_id and await db.get(Tenant, tenant_id) is None:
        raise HTTPException(status_code=422, detail="tenant_id 不存在")


async def ingest_document(
    db: AsyncSession,
    user: UserContext,
    *,
    title: str,
    filename: str,
    content: bytes,
    doc_type: str = "manual",
    version: str = "1.0",
    product_version: str = "all",
    access_scope: str = "public",
    tenant_id: str | None = None,
) -> KnowledgeDocument:
    """导入一份文档。解析失败不抛 500，而是记录 failed + 错误原因，便于管理页查看。"""
    filename = Path(filename).name  # 去掉客户端可能携带的路径
    suffix = Path(filename).suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise HTTPException(
            status_code=422,
            detail=f"不支持的文件类型 {suffix}，仅支持 {', '.join(settings.allowed_extensions)}",
        )
    if not title.strip():
        raise HTTPException(status_code=422, detail="文档标题不能为空")

    access_scope = access_scope.strip().lower()
    tenant_id = tenant_id or None
    _validate_meta(doc_type=doc_type, access_scope=access_scope, tenant_id=tenant_id)
    await _assert_tenant_exists(db, tenant_id)
    if access_scope == "public":
        tenant_id = None

    try:
        _validate_retrieval_config()
    except VectorIndexError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    checksum = _sha256(content)
    existing = await db.scalar(select(KnowledgeDocument).where(KnowledgeDocument.checksum == checksum))
    if existing:
        raise DuplicateDocumentError(existing.id)

    document = KnowledgeDocument(
        title=title.strip(),
        doc_type=doc_type,
        version=version or "1.0",
        product_version=_normalize_product_version(product_version),
        access_scope=access_scope,
        tenant_id=tenant_id,
        filename=filename,
        checksum=checksum,
        status="processing",
        created_by=user.user_id,
        updated_by=user.user_id,
    )
    db.add(document)
    await db.flush()  # 先拿到 id

    try:
        chunks = _document_chunks(filename, content)
        await _add_chunks(db, document, chunks)
        document.chunk_count = len(chunks)
        document.status = "ready"

        # 解析与向量化都成功后再落盘，避免留下无法对应文档的脏文件
        stored_path = settings.upload_path / f"{document.id}{suffix}"
        stored_path.write_bytes(content)
        document.stored_path = str(stored_path)
    except VectorIndexError as exc:
        # 向量化/向量库失败：事务回滚，不留下“有元数据但无索引”的半成品
        raise HTTPException(
            status_code=503,
            detail=f"文档已解析，但向量索引构建失败：{exc}。本次导入未生效，请检查向量后端后重试。",
        ) from exc
    except DocumentParseError as exc:
        document.status = "failed"
        document.error_message = str(exc)
    except Exception as exc:  # 兜底：记录原因，不把内部堆栈暴露给前端
        document.status = "failed"
        document.error_message = f"导入失败：{exc.__class__.__name__}"

    await db.flush()
    return document


async def update_document(
    db: AsyncSession,
    user: UserContext,
    document_id: str,
    *,
    title: str | None = None,
    doc_type: str | None = None,
    version: str | None = None,
    product_version: str | None = None,
    access_scope: str | None = None,
    tenant_id: str | None = None,
    filename: str | None = None,
    content: bytes | None = None,
) -> KnowledgeDocument:
    """更新文档。若提供新文件，先完整解析成功，再替换旧块；失败则保持原内容。"""
    document = await db.get(KnowledgeDocument, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="文档不存在")

    original_updated_at = document.updated_at
    new_title = title.strip() if title is not None else document.title
    if not new_title:
        raise HTTPException(status_code=422, detail="文档标题不能为空")
    new_product_version = (
        _normalize_product_version(product_version)
        if product_version is not None else document.product_version
    )
    new_doc_type = doc_type or document.doc_type
    new_scope = (access_scope or document.access_scope).lower()
    new_tenant = tenant_id if tenant_id is not None else document.tenant_id
    if new_scope == "public":
        new_tenant = None
    _validate_meta(doc_type=new_doc_type, access_scope=new_scope, tenant_id=new_tenant)
    await _assert_tenant_exists(db, new_tenant)

    new_chunks: list[tuple[str, str]] = []
    new_vectors: list[EmbeddedText] = []
    new_backend = ""
    new_checksum = document.checksum
    new_filename = document.filename
    new_suffix = Path(document.filename).suffix.lower()

    if content is not None:
        new_filename = Path(filename or document.filename).name
        new_suffix = Path(new_filename).suffix.lower()
        if new_suffix not in settings.allowed_extensions:
            raise HTTPException(status_code=422, detail="不支持的文件类型")

        checksum = _sha256(content)
        duplicated = await db.scalar(
            select(KnowledgeDocument).where(
                KnowledgeDocument.checksum == checksum, KnowledgeDocument.id != document.id
            )
        )
        if duplicated:
            raise DuplicateDocumentError(duplicated.id)

        # 解析失败直接抛给路由，文档保持原样，不会出现"旧块已删、新块没写"的半成品
        try:
            new_chunks = _document_chunks(new_filename, content)
        except DocumentParseError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # 先完成向量化，再删旧块；向量化失败时旧资料仍然可用
        try:
            new_vectors, new_backend = await _embed_texts([text for _, text in new_chunks])
        except VectorIndexError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"新内容向量化失败：{exc}。文档保持原样，未做任何替换。",
            ) from exc
        new_checksum = checksum

    use_milvus = settings.retrieval_backend == "milvus"
    reindex_metadata = use_milvus and (
        new_product_version != document.product_version
        or new_scope != document.access_scope
        or new_tenant != document.tenant_id
    )

    # 先占用当前文档版本，避免并发更新覆盖彼此的文件指针和分块。
    claimed = await db.scalar(
        update(KnowledgeDocument)
        .where(
            KnowledgeDocument.id == document_id,
            KnowledgeDocument.updated_at == original_updated_at,
        )
        .values(updated_at=utcnow())
        .returning(KnowledgeDocument.id)
        .execution_options(synchronize_session=False)
    )
    if claimed is None:
        raise HTTPException(status_code=409, detail="文档已被修改，请刷新后重试")
    await db.refresh(document)

    old_path = Path(document.stored_path) if document.stored_path else None
    staged_path = None
    old_chunk_ids: list[str] = []
    new_pairs: list[tuple[KnowledgeChunk, EmbeddedText]] = []
    try:
        if content is not None or reindex_metadata:
            old_chunks = list((await db.scalars(
                select(KnowledgeChunk)
                .where(KnowledgeChunk.document_id == document_id)
                .order_by(KnowledgeChunk.chunk_index)
            )).all())
            old_chunk_ids = [chunk.id for chunk in old_chunks]
            if content is None and old_chunks:
                new_backend = settings.embedding_backend
                stored = await load_chunk_embeddings(db, old_chunk_ids, new_backend)
                if len(stored) != len(old_chunks):
                    raise VectorIndexError("文档缺少向量，请重建索引后再修改筛选字段")
                new_chunks = [(chunk.section, chunk.content) for chunk in old_chunks]
                new_vectors = [stored[chunk_id] for chunk_id in old_chunk_ids]

        if content is not None:
            staged_path = settings.upload_path / f"{document_id}-{new_id()}{new_suffix}"
            await asyncio.to_thread(staged_path.write_bytes, content)
            document.stored_path = str(staged_path)
            document.filename = new_filename
            document.checksum = new_checksum
            document.status = "ready"
            document.error_message = ""

        document.title = new_title
        document.doc_type = new_doc_type
        document.version = version or document.version
        document.product_version = new_product_version
        document.access_scope = new_scope
        document.tenant_id = new_tenant
        document.updated_by = user.user_id

        if new_chunks:
            await db.execute(delete(KnowledgeChunk).where(KnowledgeChunk.document_id == document_id))
            new_pairs = await _add_chunks_with_vectors(
                db, document, new_chunks, new_vectors, new_backend, sync_milvus=False
            )
            document.chunk_count = len(new_pairs)
            if use_milvus:
                await asyncio.to_thread(_sync_milvus, document, new_pairs)
        await db.commit()
    except Exception as exc:
        await db.rollback()
        new_chunk_ids = [chunk.id for chunk, _ in new_pairs]
        await _cleanup_document_resources(staged_path, new_chunk_ids)
        if isinstance(exc, (VectorIndexError, EmbeddingBackendUnavailable, MilvusUnavailable, OSError)):
            logger.warning("knowledge.update_failed", document_id=document_id, error=str(exc)[:200])
            raise HTTPException(status_code=503, detail="文档更新未完成，请检查存储或索引后重试") from exc
        raise

    await _cleanup_document_resources(old_path if staged_path else None, old_chunk_ids)
    return document


async def _cleanup_document_resources(path: Path | None, chunk_ids: list[str]) -> None:
    if settings.retrieval_backend == "milvus" and chunk_ids:
        for attempt in range(2):
            try:
                backend = get_embedding_backend()
                await asyncio.to_thread(get_milvus_index(backend).delete_chunks, chunk_ids)
                break
            except (MilvusUnavailable, EmbeddingBackendUnavailable) as exc:
                if attempt == 1:
                    logger.warning("knowledge.index_cleanup_failed", error=str(exc)[:200])
    if path is not None:
        try:
            await asyncio.to_thread(path.unlink, missing_ok=True)
        except OSError as exc:
            logger.warning("knowledge.file_cleanup_failed", path=str(path), error=str(exc)[:200])


async def delete_document(db: AsyncSession, user: UserContext, document_id: str) -> None:
    document = await db.get(KnowledgeDocument, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="文档不存在")

    stored_path = Path(document.stored_path) if document.stored_path else None
    chunk_ids = list((await db.scalars(
        select(KnowledgeChunk.id).where(KnowledgeChunk.document_id == document_id)
    )).all())
    removed = await db.scalar(
        delete(KnowledgeDocument)
        .where(
            KnowledgeDocument.id == document_id,
            KnowledgeDocument.updated_at == document.updated_at,
        )
        .returning(KnowledgeDocument.id)
    )
    if removed is None:
        raise HTTPException(status_code=409, detail="文档已被修改，请刷新后重试")
    await db.commit()
    await _cleanup_document_resources(stored_path, chunk_ids)


async def list_documents(
    db: AsyncSession,
    *,
    status_filter: str | None = None,
    doc_type: str | None = None,
    tenant_id: str | None = None,
) -> list[KnowledgeDocument]:
    stmt = select(KnowledgeDocument).order_by(KnowledgeDocument.updated_at.desc())
    if status_filter:
        stmt = stmt.where(KnowledgeDocument.status == status_filter)
    if doc_type:
        stmt = stmt.where(KnowledgeDocument.doc_type == doc_type)
    if tenant_id:
        stmt = stmt.where(KnowledgeDocument.tenant_id == tenant_id)
    return list((await db.execute(stmt)).scalars().all())


def serialize_document(doc: KnowledgeDocument) -> dict:
    return {
        "id": doc.id,
        "title": doc.title,
        "doc_type": doc.doc_type,
        "version": doc.version,
        "product_version": doc.product_version,
        "access_scope": doc.access_scope,
        "tenant_id": doc.tenant_id,
        "filename": doc.filename,
        "status": doc.status,
        "error_message": doc.error_message,
        "chunk_count": doc.chunk_count,
        "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
    }
