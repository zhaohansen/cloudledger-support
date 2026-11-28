# app/services/retrieval.py
# 知识检索：权限过滤 → 版本过滤 → 向量/词法打分 → 可选重排。
# 重要：
# 1. 过滤在打分之前完成，不会把其他租户的私有资料取回后再交给模型判断；
# 2. hash 后端是明确标注的演示模式，不是语义 Embedding；
# 3. bge_m3 / milvus 后端失败时返回可解释错误，绝不静默退回 hash 并声称语义检索。

from __future__ import annotations

import asyncio
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from sqlalchemy import Select, and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.deps import UserContext
from app.logging_config import get_logger
from app.models import KnowledgeChunk, KnowledgeChunkEmbedding, KnowledgeDocument
from app.services.embeddings import (
    EmbeddingBackendUnavailable,
    EmbeddedText,
    get_embedding_backend,
)
from app.services.milvus_index import MilvusUnavailable, get_milvus_index
from app.services.permissions import allowed_kb_tenant_ids
from app.services.text_utils import (
    content_tokens,
    cosine,
    make_snippet,
    tokenize,
)
from app.services.vector_index import load_chunk_embeddings

settings = get_settings()
logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    title: str
    doc_type: str
    doc_version: str
    product_version: str
    section: str
    chunk_index: int
    content: str
    score: float
    dense_score: float
    lexical_score: float
    sparse_score: float = 0.0
    score_kind: str = ""

    def to_citation(self, index: int) -> dict:
        return {
            "index": index,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "title": self.title,
            "section": self.section,
            "chunk_index": self.chunk_index,
            "doc_version": self.doc_version,
            "product_version": self.product_version,
            "snippet": make_snippet(self.content),
            "score": round(self.score, 4),
        }


@dataclass
class SearchOutcome:
    items: list[RetrievedChunk] = field(default_factory=list)
    # ok / no_match / no_version_match / index_missing / backend_error
    reason: str = "ok"
    requested_version: str | None = None
    matched_versions: list[str] = field(default_factory=list)
    version_conflict: bool = False
    # 实际使用的后端，评测报告与调试接口必须展示，不能只写配置文件里的期望值
    embedding_backend: str = "hash"
    retrieval_backend: str = "local"
    rerank_status: str = "disabled"  # disabled / used / failed
    detail: str = ""
    degraded: bool = False


async def search_knowledge(
    db: AsyncSession,
    user: UserContext,
    query: str,
    *,
    product_version: str | None = None,
    top_k: int | None = None,
    tenant_scope: str | None = None,
) -> SearchOutcome:
    query = (query or "").strip()
    top_k = top_k or settings.retrieval_top_k
    requested = _normalize_version(product_version)

    if not query:
        return SearchOutcome(reason="no_match", requested_version=requested)

    try:
        backend = get_embedding_backend()
    except EmbeddingBackendUnavailable as exc:
        return SearchOutcome(
            reason="backend_error",
            requested_version=requested,
            embedding_backend=settings.embedding_backend,
            retrieval_backend=settings.retrieval_backend,
            detail=str(exc),
        )

    allowed_tenants = await allowed_kb_tenant_ids(db, user)
    if tenant_scope is not None:
        # 工单回复仅使用该工单租户的私有资料和公共资料。
        allowed_tenants = [tenant_id for tenant_id in allowed_tenants if tenant_id == tenant_scope]

    # 明确指定版本时，先确认知识库是否真的有该版本资料。
    # 只有"全版本通用"资料而没有该版本专属资料时，直接返回版本不匹配，
    # 避免把通用资料当成 v9.9 这类不存在版本的专属答案。
    if requested:
        known_versions = await _known_versions(db, allowed_tenants)
        if known_versions and requested not in known_versions:
            return SearchOutcome(
                reason="no_version_match",
                requested_version=requested,
                matched_versions=known_versions,
                embedding_backend=backend.name,
                retrieval_backend=settings.retrieval_backend,
                detail=f"知识库现有产品版本：{', '.join(known_versions)}",
            )

    candidates = _candidate_query(allowed_tenants, requested)
    if settings.retrieval_backend == "milvus":
        # 只检查是否存在可用分块；正文在 Milvus 召回后按命中 ID 加载。
        has_candidate = await db.scalar(
            candidates.with_only_columns(KnowledgeChunk.id).limit(1)
        )
        if has_candidate:
            return await _search_with_milvus(
                db, query, candidates, allowed_tenants, requested, top_k, backend
            )
    else:
        rows = list((await db.execute(candidates)).all())
        if rows:
            return await _search_locally(db, query, rows, requested, top_k, backend)
    return SearchOutcome(
        reason=await _empty_reason(db, allowed_tenants, requested),
        requested_version=requested,
        embedding_backend=backend.name,
        retrieval_backend=settings.retrieval_backend,
    )


# ── 候选集与权限过滤 ─────────────────────────────────────────


def _visible_documents(allowed_tenants: list[str]):
    """公共资料和授权租户的私有资料都必须已经入库。"""
    return and_(
        KnowledgeDocument.status == "ready",
        or_(
            KnowledgeDocument.access_scope == "public",
            and_(
                KnowledgeDocument.access_scope == "tenant",
                KnowledgeDocument.tenant_id.in_(allowed_tenants),
            ),
        ),
    )


def _matching_version(requested: str):
    return KnowledgeDocument.product_version.in_((requested, "all"))


def _candidate_query(
    allowed_tenants: list[str],
    requested: str | None,
) -> Select[tuple[KnowledgeChunk, KnowledgeDocument]]:
    query = (
        select(KnowledgeChunk, KnowledgeDocument)
        .join(KnowledgeDocument, KnowledgeChunk.document_id == KnowledgeDocument.id)
        .where(_visible_documents(allowed_tenants))
    )
    if requested:
        query = query.where(_matching_version(requested))
    return query


async def _known_versions(db: AsyncSession, allowed_tenants: list[str]) -> list[str]:
    """列出用户能访问的具体产品版本。"""
    query = (
        select(KnowledgeDocument.product_version)
        .where(_visible_documents(allowed_tenants), KnowledgeDocument.product_version != "all")
        .distinct()
    )
    versions = await db.scalars(query)
    return sorted(version for version in versions if version)


async def _empty_reason(
    db: AsyncSession,
    allowed_tenants: list[str],
    requested: str | None,
) -> str:
    """有对应版本的资料时，告诉用户只是没有搜到相关内容。"""
    if not requested:
        return "no_match"
    matching_document = await db.scalar(
        select(KnowledgeDocument.id)
        .where(_visible_documents(allowed_tenants), _matching_version(requested))
        .limit(1)
    )
    return "no_match" if matching_document else "no_version_match"


# ── 本地向量后端 ─────────────────────────────────────────────


async def _search_locally(
    db: AsyncSession,
    query: str,
    rows: list[tuple[KnowledgeChunk, KnowledgeDocument]],
    requested: str | None,
    top_k: int,
    backend,
) -> SearchOutcome:
    chunk_ids = [chunk.id for chunk, _ in rows]
    stored = await load_chunk_embeddings(db, chunk_ids, backend.name)

    if backend.name == "hash":
        vectors = {
            chunk.id: stored.get(chunk.id) or EmbeddedText(dense=list(chunk.embedding or []))
            for chunk, _ in rows
        }
    else:
        missing = [chunk_id for chunk_id in chunk_ids if chunk_id not in stored]
        if len(missing) == len(chunk_ids):
            return SearchOutcome(
                reason="index_missing",
                requested_version=requested,
                embedding_backend=backend.name,
                retrieval_backend="local",
                detail=(
                    f"知识库还没有 {backend.name} 向量索引（缺少 {len(missing)} 个分块的向量）。"
                    "请执行 scripts/reindex_embeddings.py 重建；不会退回 hash 演示检索。"
                ),
            )
        vectors = stored
        if missing:
            logger.warning(
                "retrieval.partial_index",
                backend=backend.name,
                missing=len(missing),
                total=len(chunk_ids),
            )

    try:
        query_vector = await _encode_query(backend, query)
    except EmbeddingBackendUnavailable as exc:
        return SearchOutcome(
            reason="backend_error",
            requested_version=requested,
            embedding_backend=backend.name,
            retrieval_backend="local",
            detail=str(exc),
        )

    if backend.name == "hash":
        items, degraded = _score_hash(query, query_vector, rows, vectors)
        rerank_status = "disabled"
        detail = "哈希向量 + IDF 词法覆盖（演示模式，不是语义 Embedding）"
        items.sort(key=lambda item: item.score, reverse=True)
    else:
        items = _score_semantic(query_vector, rows, vectors)
        items.sort(key=lambda item: item.score, reverse=True)
        rerank_status, rerank_detail = await _apply_rerank(query, items)
        detail = "BGE-M3 dense 余弦 + sparse lexical_weights 加权覆盖"
        if rerank_detail:
            detail += f"；{rerank_detail}"
        degraded = bool(rerank_detail)

    items = items[:top_k]
    if not items:
        return SearchOutcome(
            reason="no_match",
            requested_version=requested,
            embedding_backend=backend.name,
            retrieval_backend="local",
            detail=detail,
            degraded=degraded,
        )

    matched_versions = sorted({item.product_version for item in items if item.product_version != "all"})
    return SearchOutcome(
        items=items,
        reason="ok",
        requested_version=requested,
        matched_versions=matched_versions,
        version_conflict=not requested and len(matched_versions) > 1,
        embedding_backend=backend.name,
        retrieval_backend="local",
        rerank_status=rerank_status,
        detail=detail,
        degraded=degraded,
    )


def _score_hash(
    query: str,
    query_vector: EmbeddedText,
    rows: list[tuple[KnowledgeChunk, KnowledgeDocument]],
    vectors: dict[str, EmbeddedText],
) -> tuple[list[RetrievedChunk], bool]:
    """原有演示算法原样保留：哈希余弦 + IDF 加权词法覆盖。

    这不是语义检索，只作为零依赖离线模式；文档、报告与接口都会明确标注 backend=hash。
    """
    query_tokens = content_tokens(query) or tokenize(query)
    query_token_set = set(query_tokens)

    prepared: list[tuple[KnowledgeChunk, KnowledgeDocument, set[str], set[str], set[str]]] = []
    document_frequency: Counter[str] = Counter()
    for chunk, doc in rows:
        content_set = set(tokenize(chunk.content))
        section_set = set(tokenize(chunk.section or ""))
        title_set = set(tokenize(doc.title))
        prepared.append((chunk, doc, content_set, section_set, title_set))
        document_frequency.update(content_set)

    total_docs = max(len(prepared), 1)
    idf = {
        token: math.log((total_docs + 1) / (document_frequency.get(token, 0) + 1)) + 1.0
        for token in query_token_set
    }
    total_query_weight = sum(idf.values()) or 1.0

    def weighted_coverage(token_sets: set[str]) -> float:
        matched = sum(weight for token, weight in idf.items() if token in token_sets)
        return matched / total_query_weight

    scored: list[RetrievedChunk] = []
    for chunk, doc, content_set, section_set, title_set in prepared:
        vector = vectors.get(chunk.id)
        dense = cosine(query_vector.dense, vector.dense if vector else [])
        content_cov = weighted_coverage(content_set)
        section_cov = weighted_coverage(section_set)
        title_cov = weighted_coverage(title_set)
        # 章节标题往往比正文更能说明主题（例如“四、已知问题”），
        # 因此章节/标题覆盖也作为独立信号参与取最大值。
        lexical = max(
            content_cov,
            section_cov,
            title_cov,
            0.65 * content_cov + 0.35 * section_cov,
            0.60 * content_cov + 0.40 * title_cov,
        )
        # 纯向量相似度容易把无关内容带进来；至少要有两个内容词命中，
        # 或者章节标题高度匹配（例如问题本身就是章节名），否则视为证据不足。
        matched_terms = [token for token in query_token_set if token in (content_set | section_set | title_set)]
        if len(matched_terms) < 2 and section_cov < 0.5:
            continue
        if lexical < 0.2:
            continue
        # 哈希向量噪声大，词法占主导；向量只作为补充信号
        score = 0.35 * dense + 0.65 * lexical
        if score < settings.retrieval_threshold:
            continue
        scored.append(
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=doc.id,
                title=doc.title,
                doc_type=doc.doc_type,
                doc_version=doc.version,
                product_version=doc.product_version,
                section=chunk.section or "正文",
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                score=score,
                dense_score=dense,
                lexical_score=lexical,
                score_kind="hash+idf_lexical",
            )
        )
    return scored, False


def _score_semantic(
    query_vector: EmbeddedText,
    rows: list[tuple[KnowledgeChunk, KnowledgeDocument]],
    vectors: dict[str, EmbeddedText],
) -> list[RetrievedChunk]:
    """BGE-M3 混合打分：dense 余弦 + sparse 词权重覆盖率。

    公式：score = w_dense * cosine(q_dense, d_dense) + w_sparse * (Σ min(q_w, d_w) / Σ q_w)
    其中 sparse 权重来自 BGE-M3 的 lexical_weights，不是 BM25，也不是 RRF 融合。
    """
    q_sparse = query_vector.sparse or {}
    q_total = sum(q_sparse.values()) or 0.0
    dense_weight = settings.hybrid_dense_weight
    sparse_weight = settings.hybrid_sparse_weight
    total_weight = dense_weight + sparse_weight or 1.0

    scored: list[RetrievedChunk] = []
    for chunk, doc in rows:
        vector = vectors.get(chunk.id)
        if vector is None:
            continue
        dense = cosine(query_vector.dense, vector.dense)
        sparse = 0.0
        if q_total and vector.sparse:
            overlap = sum(min(weight, vector.sparse.get(token, 0.0)) for token, weight in q_sparse.items())
            sparse = overlap / q_total
        score = (dense_weight * dense + sparse_weight * sparse) / total_weight

        # 真实语义后端也保留最低门槛，避免把弱相关片段当成证据。
        # dense/sparse 任一明显偏离、或融合分过低，都视为证据不足。
        if dense < settings.semantic_dense_threshold and sparse < settings.semantic_sparse_threshold:
            continue
        if score < settings.semantic_score_threshold:
            continue
        scored.append(
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=doc.id,
                title=doc.title,
                doc_type=doc.doc_type,
                doc_version=doc.version,
                product_version=doc.product_version,
                section=chunk.section or "正文",
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                score=score,
                dense_score=dense,
                lexical_score=0.0,
                sparse_score=sparse,
                score_kind="bge_m3_dense_sparse",
            )
        )
    return scored


async def _apply_rerank(query: str, items: list[RetrievedChunk]) -> tuple[str, str]:
    """可选重排。重排失败只降级排序信号，并明确记录失败原因。"""
    if not settings.rerank_enabled:
        return "disabled", ""
    candidates = items[: max(settings.rerank_candidate_k, 1)]
    if not candidates:
        return "disabled", ""
    try:
        from app.services.reranker import get_reranker

        reranker = get_reranker()
        scores = reranker.rerank(
            query, [item.content for item in candidates], top_k=len(candidates)
        )
    except Exception as exc:  # 可选组件失败不影响主检索，但会明确标注
        logger.warning("retrieval.rerank_failed", error=str(exc)[:200])
        return "failed", f"重排不可用（{str(exc)[:80]}），已按融合分数排序"

    ranked = []
    for item, score in zip(candidates, scores):
        item.score = float(score)
        item.score_kind = "bge_reranker"
        ranked.append(item)
    ranked.sort(key=lambda item: item.score, reverse=True)
    items[:] = ranked + items[len(candidates):]
    return "used", ""

    # 说明：重排后的 score 是 CrossEncoder 相关性概率，与融合分数不可直接比较


async def _encode_query(backend, query: str):
    return await asyncio.to_thread(backend.encode_query, query)


# ── Milvus 后端 ──────────────────────────────────────────────


async def _search_with_milvus(
    db: AsyncSession,
    query: str,
    candidates: Select[tuple[KnowledgeChunk, KnowledgeDocument]],
    allowed_tenants: list[str],
    requested: str | None,
    top_k: int,
    backend,
) -> SearchOutcome:
    """Milvus 负责召回；回到业务库复核权限/版本，形成纵深防御。"""
    try:
        query_vector = await _encode_query(backend, query)
        index = get_milvus_index(backend)
        hits = await asyncio.to_thread(
            index.search,
            query_vector.dense,
            query_vector.sparse or {},
            allowed_tenants,
            requested,
            top_k,
        )
    except (MilvusUnavailable, EmbeddingBackendUnavailable) as exc:
        return SearchOutcome(
            reason="backend_error",
            requested_version=requested,
            embedding_backend=backend.name,
            retrieval_backend="milvus",
            detail=str(exc),
        )

    if not hits:
        stats_rows = await db.scalar(
            select(func.count())
            .select_from(KnowledgeChunkEmbedding)
            .where(KnowledgeChunkEmbedding.backend == backend.name)
        )
        return SearchOutcome(
            reason="no_match" if (stats_rows or 0) else "index_missing",
            requested_version=requested,
            embedding_backend=backend.name,
            retrieval_backend="milvus",
            detail="Milvus 未返回候选" if stats_rows else "业务库中还没有对应后端的向量索引",
        )

    hit_ids = {hit["chunk_id"] for hit in hits if hit.get("chunk_id")}
    rows = await db.execute(candidates.where(KnowledgeChunk.id.in_(hit_ids)))
    by_id = {chunk.id: (chunk, doc) for chunk, doc in rows}
    # Milvus 的 WeightedRanker 分数被压缩到较窄区间，不适合直接当证据门槛。
    # 召回后回到业务库取同一份 BGE-M3 向量重算融合分，保证 local/milvus 两个后端
    # 使用同一套阈值口径，也顺便补齐 dense/sparse 明细。
    stored = await load_chunk_embeddings(db, list(by_id), backend.name)
    q_sparse = query_vector.sparse or {}
    q_total = sum(q_sparse.values()) or 0.0
    dense_weight = settings.hybrid_dense_weight
    sparse_weight = settings.hybrid_sparse_weight
    total_weight = dense_weight + sparse_weight or 1.0

    items: list[RetrievedChunk] = []
    dropped = 0
    for hit in hits:
        pair = by_id.get(hit.get("chunk_id"))
        if pair is None:
            # 向量库里的旧数据/越权数据在这里被丢弃，不会进入模型上下文
            continue
        chunk, doc = pair
        vector = stored.get(chunk.id)
        if vector is None:
            dropped += 1
            continue
        dense = cosine(query_vector.dense, vector.dense)
        sparse = 0.0
        if q_total and vector.sparse:
            overlap = sum(min(weight, vector.sparse.get(token, 0.0)) for token, weight in q_sparse.items())
            sparse = overlap / q_total
        score = (dense_weight * dense + sparse_weight * sparse) / total_weight
        if score < settings.semantic_score_threshold:
            dropped += 1
            continue
        items.append(
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=doc.id,
                title=doc.title,
                doc_type=doc.doc_type,
                doc_version=doc.version,
                product_version=doc.product_version,
                section=chunk.section or "正文",
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                score=score,
                dense_score=dense,
                lexical_score=0.0,
                sparse_score=sparse,
                score_kind="milvus_recall+bge_m3_rescore",
            )
        )
    items.sort(key=lambda item: item.score, reverse=True)

    rerank_status, rerank_detail = await _apply_rerank(query, items)
    detail = "Milvus dense COSINE + sparse IP 召回，BGE-M3 融合分复核"
    if dropped:
        detail += f"；{dropped} 条候选低于证据门槛被丢弃"
    if rerank_detail:
        detail += f"；{rerank_detail}"
    degraded = bool(rerank_detail)

    if not items:
        return SearchOutcome(
            reason="no_match",
            requested_version=requested,
            embedding_backend=backend.name,
            retrieval_backend="milvus",
            rerank_status=rerank_status,
            detail=detail,
            degraded=degraded,
        )

    matched_versions = sorted({item.product_version for item in items if item.product_version != "all"})
    return SearchOutcome(
        items=items[:top_k],
        reason="ok",
        requested_version=requested,
        matched_versions=matched_versions,
        version_conflict=not requested and len(matched_versions) > 1,
        embedding_backend=backend.name,
        retrieval_backend="milvus",
        rerank_status=rerank_status,
        detail=detail,
        degraded=bool(rerank_detail),
    )


# ── 工具函数 ─────────────────────────────────────────────────


def _normalize_version(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip().lower()
    if value in ("all", "通用", "全版本"):
        return None
    if value.replace("v", "").replace(".", "").isdigit():
        return value if value.startswith("v") else f"v{value}"
    return value


def citations_from_items(items: list[RetrievedChunk]) -> list[dict]:
    return [item.to_citation(i + 1) for i, item in enumerate(items)]


def invalid_citation_numbers(answer: str, citation_count: int) -> list[int]:
    """找出回答里引用了不存在的编号（用于校验模型是否编造来源）。"""
    numbers = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    return sorted(n for n in numbers if n < 1 or n > citation_count)
