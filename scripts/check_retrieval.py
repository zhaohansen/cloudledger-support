# scripts/check_retrieval.py
# 检索后端自检：用指定用户身份跑一次真实检索，打印实际后端、命中分块与分数。
# 用法：
#   python scripts/check_retrieval.py --username customer_a --version v2.2 --query "报销草稿保存后预算被冻结"

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--username", default="customer_a")
    parser.add_argument("--version", default=None)
    parser.add_argument("--backend", choices=["hash", "bge_m3"], default=None)
    parser.add_argument("--index", choices=["local", "milvus"], default=None)
    args = parser.parse_args()
    if args.backend:
        os.environ["EMBEDDING_BACKEND"] = args.backend
    if args.index:
        os.environ["RETRIEVAL_BACKEND"] = args.index
    return asyncio.run(_run(args))


async def _run(args) -> int:
    from sqlalchemy import select

    from app.db import AsyncSessionLocal, init_db
    from app.deps import UserContext
    from app.models import Tenant, User
    from app.services.retrieval import search_knowledge

    await init_db()
    async with AsyncSessionLocal() as db:
        user = await db.scalar(select(User).where(User.username == args.username))
        if user is None:
            print(f"用户不存在：{args.username}", file=sys.stderr)
            return 2
        tenant = await db.get(Tenant, user.tenant_id)
        ctx = UserContext(
            user_id=user.id,
            username=user.username,
            display_name=user.display_name,
            role=user.role,
            tenant_id=user.tenant_id,
            tenant_name=tenant.name if tenant else "",
        )
        outcome = await search_knowledge(
            db, ctx, args.query, product_version=args.version, top_k=5
        )

    print(
        f"reason={outcome.reason} embedding={outcome.embedding_backend} "
        f"retrieval={outcome.retrieval_backend} rerank={outcome.rerank_status} "
        f"conflict={outcome.version_conflict}"
    )
    print(f"detail={outcome.detail}")
    if not outcome.items:
        print("没有命中分块")
        return 1 if outcome.reason != "no_match" else 0
    for index, item in enumerate(outcome.items, start=1):
        print(
            f"[{index}] {item.title} / {item.section} / {item.product_version} "
            f"score={item.score:.4f} dense={item.dense_score:.4f} "
            f"sparse={item.sparse_score:.4f} lexical={item.lexical_score:.4f} "
            f"kind={item.score_kind}"
        )
        print(f"    {item.content[:120].replace(chr(10), ' ')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
