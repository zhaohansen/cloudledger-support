# scripts/verify_semantic_backend.py
# 真实语义后端验证脚本（不会写入业务库）：
#   1. 加载本地 BGE-M3，验证 dense 相似度排序与 sparse 输出；
#   2. 在临时 Milvus 集合中写入/检索/删除，验证混合检索与租户过滤。
# 需要具备 torch / FlagEmbedding / pymilvus 的环境，例如 conda edu_agent。
# 用法：
#   python scripts/verify_semantic_backend.py
#   python scripts/verify_semantic_backend.py --skip-milvus

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("EMBEDDING_BACKEND", "bge_m3")
os.environ.setdefault("RETRIEVAL_BACKEND", "milvus")
os.environ.setdefault("BGE_M3_MAX_LENGTH", "512")
os.environ.setdefault("MILVUS_COLLECTION", f"cloudledger_verify_{uuid.uuid4().hex[:8]}")

DOCS = [
    {
        "chunk_id": "verify-1",
        "text": "v2.2 版本中，保存报销草稿后系统会预占冻结部门可用预算，防止多人同时填单导致超支。",
        "tenant_id": "tenant_a",
        "access_scope": "tenant",
        "product_version": "v2.2",
    },
    {
        "chunk_id": "verify-2",
        "text": "如需修改已复核通过的差旅报销单，需在未打款且未生成凭证前提下由财务复核岗点击“反审核”。",
        "tenant_id": "",
        "access_scope": "public",
        "product_version": "v2.3",
    },
    {
        "chunk_id": "verify-3",
        "text": "银企直连打款退票后，原报销单锁定的部门预算不会自动释放，需修正收款卡重发或冲销关闭。",
        "tenant_id": "",
        "access_scope": "public",
        "product_version": "all",
    },
    {
        "chunk_id": "verify-4",
        "text": "tenant_b 的内部差旅住宿标准：一线城市上限 650 元/晚，仅限该租户内部使用。",
        "tenant_id": "tenant_b",
        "access_scope": "tenant",
        "product_version": "all",
    },
]

QUERIES = [
    ("报销单保存草稿以后为什么可用预算变少了", "verify-1"),
    ("已经财务复核过的报销单怎么退回修改", "verify-2"),
    ("银行打款退票了预算怎么处理", "verify-3"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-milvus", action="store_true")
    args = parser.parse_args()

    from app.config import get_settings
    from app.services.embeddings import get_embedding_backend
    from app.services.text_utils import cosine

    settings = get_settings()
    backend = get_embedding_backend("bge_m3")
    print(f"[配置] embedding={backend.name} dim={backend.dim} milvus={settings.milvus_uri}")
    print(f"[模型] {backend.health().get('model_path')}")

    started = time.perf_counter()
    vectors = backend.encode_documents([doc["text"] for doc in DOCS])
    elapsed = time.perf_counter() - started
    assert all(len(vector.dense) == 1024 for vector in vectors), "dense 维度不是 1024"
    assert all(vector.sparse for vector in vectors), "sparse 输出为空"
    print(f"[PASS] BGE-M3 编码 4 条文本，耗时 {elapsed:.1f}s，dense=1024，sparse 非空")

    failures = 0
    for query, expected in QUERIES:
        query_vector = backend.encode_query(query)
        scores = [cosine(query_vector.dense, vector.dense) for vector in vectors]
        ranked = sorted(range(len(DOCS)), key=lambda i: scores[i], reverse=True)
        top = DOCS[ranked[0]]["chunk_id"]
        ok = top == expected
        failures += 0 if ok else 1
        pairs = ", ".join(
            f"{DOCS[i]['chunk_id']}={scores[i]:.3f}" for i in ranked[:3]
        )
        print(f"[{'PASS' if ok else 'FAIL'}] {query} -> top1={top} ({pairs})")

    if args.skip_milvus:
        return 1 if failures else 0

    from pymilvus import MilvusClient

    from app.services.milvus_index import (
        MilvusUnavailable,
        build_filter_expression,
        get_milvus_index,
    )

    expression = build_filter_expression(["tenant_a"], "v2.2")
    assert 'tenant_id in ["tenant_a"]' in expression and 'product_version == "v2.2"' in expression
    print("[PASS] 权限/版本过滤表达式构造正确")

    index = get_milvus_index(backend)
    client = MilvusClient(uri=settings.milvus_uri, timeout=5)
    collection = settings.milvus_collection
    try:
        rows = [
            {
                "chunk_id": doc["chunk_id"],
                "document_id": doc["chunk_id"].split("-")[0],
                "tenant_id": doc["tenant_id"],
                "access_scope": doc["access_scope"],
                "product_version": doc["product_version"],
                "dense": vector.dense,
                "sparse": vector.sparse or {},
            }
            for doc, vector in zip(DOCS, vectors)
        ]
        written = index.upsert(rows)
        time.sleep(1)  # 等待 Milvus 可见性
        print(f"[PASS] Milvus 写入 {written} 条到临时集合 {collection}")

        query_vector = backend.encode_query(QUERIES[0][0])
        hits = index.search(
            query_vector.dense,
            query_vector.sparse or {},
            allowed_tenant_ids=["tenant_a"],
            product_version="v2.2",
            top_k=5,
        )
        ids = [hit["chunk_id"] for hit in hits]
        print(f"[信息] tenant_a + v2.2 命中：{ids}")
        assert "verify-2" not in ids, "跨版本资料不应命中"
        assert "verify-4" not in ids, "其他租户私有资料泄漏进检索结果"
        if ids:
            assert ids[0] == "verify-1", f"期望 verify-1 排第一，实际 {ids}"
            print("[PASS] Milvus 混合检索 top1 正确，且未泄漏其他租户/版本资料")
        else:
            print("[FAIL] Milvus 未返回结果")
            failures += 1

        # 删除后不可再检索到（验证索引同步）
        index.delete_document("verify")
        time.sleep(1)
        hits = index.search(
            query_vector.dense,
            query_vector.sparse or {},
            allowed_tenant_ids=["tenant_a"],
            product_version=None,
            top_k=5,
        )
        remaining = [hit["chunk_id"] for hit in hits]
        if "verify-1" in remaining:
            print(f"[FAIL] 删除后仍能检索到：{remaining}")
            failures += 1
        else:
            print("[PASS] 删除文档向量后不再命中")
    except MilvusUnavailable as exc:
        print(f"[FAIL] Milvus 不可用：{exc}")
        failures += 1
    finally:
        try:
            client.drop_collection(collection)
            print(f"[清理] 已删除临时集合 {collection}")
        except Exception as exc:
            print(f"[清理失败] {exc}")
        client.close()

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
