# app/services/milvus_index.py
# Milvus 混合检索适配器（dense COSINE + sparse IP，WeightedRanker 加权融合）。
# 注意：这是可选的“向量库后端”，默认不启用；Milvus 不可用时会明确报错，不静默退回本地哈希。
# 权限过滤在检索请求的 expr 中完成（检索前过滤），返回结果还会在业务库复核一次。

from __future__ import annotations

import threading
from typing import Any

from app.config import get_settings
from app.logging_config import get_logger
from app.services.embeddings import EmbeddingBackend

logger = get_logger(__name__)
settings = get_settings()


class MilvusUnavailable(RuntimeError):
    """Milvus 依赖缺失或服务不可用。"""


def _quote(value: str) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_filter_expression(
    allowed_tenant_ids: list[str],
    product_version: str | None,
) -> str:
    """构造 Milvus bool 表达式：公共资料或已授权租户资料，并限制产品版本。"""
    clauses = ['access_scope == "public"']
    if allowed_tenant_ids:
        tenants = ", ".join(_quote(item) for item in allowed_tenant_ids)
        clauses.append(f'(access_scope == "tenant" and tenant_id in [{tenants}])')
    expression = "(" + " or ".join(clauses) + ")"
    if product_version:
        expression += (
            f' and (product_version == {_quote(product_version)} or product_version == "all")'
        )
    return expression


class MilvusVectorIndex:
    """Milvus 客户端封装。首次使用时连接并确保集合存在。"""

    def __init__(self, backend: EmbeddingBackend):
        if backend.name != "bge_m3":
            raise MilvusUnavailable("Milvus 混合检索只支持 BGE-M3 后端（需要 dense + sparse 向量）")
        self.backend = backend
        self.collection = settings.milvus_collection
        self._client = None
        self._lock = threading.Lock()
        self._error: str | None = None

    # ── 连接与集合 ────────────────────────────────────────
    def _get_client(self):
        if self._client is not None:
            return self._client
        if self._error:
            raise MilvusUnavailable(self._error)
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            self._error = (
                "未安装 pymilvus，无法使用 RETRIEVAL_BACKEND=milvus。"
                "请安装 pymilvus，或改用 local。"
            )
            raise MilvusUnavailable(self._error) from exc

        try:
            client = MilvusClient(uri=settings.milvus_uri, timeout=settings.milvus_timeout_seconds)
            client.list_collections()
        except Exception as exc:
            self._error = f"Milvus 连接失败（{settings.milvus_uri}）：{exc.__class__.__name__}"
            raise MilvusUnavailable(self._error) from exc
        self._client = client
        return client

    def ensure_collection(self) -> None:
        client = self._get_client()
        if client.has_collection(self.collection):
            client.load_collection(self.collection)
            return
        with self._lock:
            if client.has_collection(self.collection):
                return
            from pymilvus import DataType

            schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=96)
            schema.add_field("chunk_id", DataType.VARCHAR, max_length=64)
            schema.add_field("document_id", DataType.VARCHAR, max_length=64)
            schema.add_field("tenant_id", DataType.VARCHAR, max_length=64)
            schema.add_field("access_scope", DataType.VARCHAR, max_length=16)
            schema.add_field("product_version", DataType.VARCHAR, max_length=32)
            schema.add_field("dense", DataType.FLOAT_VECTOR, dim=self.backend.dim)
            schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)

            index_params = client.prepare_index_params()
            index_params.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="COSINE")
            index_params.add_index(
                field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="IP"
            )
            client.create_collection(
                collection_name=self.collection, schema=schema, index_params=index_params
            )
            client.load_collection(self.collection)
            logger.info("milvus.collection_created", collection=self.collection)

    # ── 写入 / 删除 ───────────────────────────────────────
    def upsert(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        self.ensure_collection()
        client = self._get_client()
        payload = []
        for row in rows:
            sparse = row.get("sparse") or {}
            payload.append(
                {
                    "id": f"{self.backend.name}:{row['chunk_id']}",
                    "chunk_id": row["chunk_id"],
                    "document_id": row["document_id"],
                    "tenant_id": row.get("tenant_id") or "",
                    "access_scope": row.get("access_scope") or "public",
                    "product_version": row.get("product_version") or "all",
                    "dense": row["dense"],
                    "sparse": {int(k): float(v) for k, v in sparse.items()},
                }
            )
        try:
            client.upsert(collection_name=self.collection, data=payload)
        except Exception as exc:
            raise MilvusUnavailable(f"Milvus 写入失败：{exc.__class__.__name__}: {exc}") from exc
        return len(payload)

    def delete_document(self, document_id: str) -> None:
        self.ensure_collection()
        client = self._get_client()
        try:
            client.delete(
                collection_name=self.collection, filter=f"document_id == {_quote(document_id)}"
            )
        except Exception as exc:
            # 删除失败不阻断业务：检索结果会回到业务库复核，已删除分块不会进入模型上下文
            logger.warning("milvus.delete_failed", document_id=document_id, error=str(exc)[:200])

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """只删除指定分块，避免文档更新时误删刚写入的新向量。"""
        if not chunk_ids:
            return
        self.ensure_collection()
        client = self._get_client()
        values = ", ".join(_quote(chunk_id) for chunk_id in chunk_ids)
        try:
            client.delete(collection_name=self.collection, filter=f"chunk_id in [{values}]")
        except Exception as exc:
            raise MilvusUnavailable(f"Milvus 分块清理失败：{exc}") from exc

    # ── 检索 ─────────────────────────────────────────────
    def search(
        self,
        dense: list[float],
        sparse: dict[int, float],
        allowed_tenant_ids: list[str],
        product_version: str | None,
        top_k: int,
        candidate_k: int = 20,
    ) -> list[dict]:
        self.ensure_collection()
        client = self._get_client()
        from pymilvus import AnnSearchRequest, WeightedRanker

        expression = build_filter_expression(allowed_tenant_ids, product_version)
        requests = [
            AnnSearchRequest(
                data=[dense],
                anns_field="dense",
                param={"metric_type": "COSINE"},
                limit=candidate_k,
                expr=expression,
            )
        ]
        if sparse:
            requests.append(
                AnnSearchRequest(
                    data=[{int(k): float(v) for k, v in sparse.items()}],
                    anns_field="sparse",
                    param={"metric_type": "IP"},
                    limit=candidate_k,
                    expr=expression,
                )
            )
        ranker = WeightedRanker(
            settings.hybrid_dense_weight,
            settings.hybrid_sparse_weight if len(requests) == 2 else 1.0,
        )
        try:
            results = client.hybrid_search(
                collection_name=self.collection,
                reqs=requests,
                ranker=ranker,
                limit=top_k,
                output_fields=["chunk_id", "document_id"],
            )
        except Exception as exc:
            raise MilvusUnavailable(f"Milvus 检索失败：{exc.__class__.__name__}: {exc}") from exc

        items: list[dict] = []
        for hits in results:
            for hit in hits:
                entity: dict[str, Any] = hit.get("entity") or {}
                items.append(
                    {
                        "chunk_id": entity.get("chunk_id") or hit.get("id"),
                        "document_id": entity.get("document_id", ""),
                        "distance": float(hit.get("distance", 0.0)),
                    }
                )
        return items

    def health(self) -> dict:
        try:
            self.ensure_collection()
            return {
                "backend": "milvus",
                "available": True,
                "uri": settings.milvus_uri,
                "collection": self.collection,
                "detail": "",
            }
        except MilvusUnavailable as exc:
            return {
                "backend": "milvus",
                "available": False,
                "uri": settings.milvus_uri,
                "collection": self.collection,
                "detail": str(exc),
            }


_INDEXES: dict[str, MilvusVectorIndex] = {}


def get_milvus_index(backend: EmbeddingBackend) -> MilvusVectorIndex:
    key = f"{settings.milvus_uri}|{settings.milvus_collection}|{backend.name}"
    if key not in _INDEXES:
        _INDEXES[key] = MilvusVectorIndex(backend)
    return _INDEXES[key]
