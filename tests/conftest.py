# tests/conftest.py
# 测试环境：独立的临时 SQLite 数据库 + 固定测试密钥，避免污染本地演示数据。
# 注意：环境变量必须在导入 app.* 之前设置。

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

TEST_ROOT = Path(tempfile.mkdtemp(prefix="cloudledger_test_"))
os.environ["APP_ENV"] = "test"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{(TEST_ROOT / 'test.db').as_posix()}"
os.environ["JWT_SECRET_KEY"] = "test-only-secret-key-not-for-production"
os.environ["JWT_EXPIRE_MINUTES"] = "60"
os.environ["LLM_MODE"] = "extractive"
os.environ["EMBEDDING_BACKEND"] = "hash"
os.environ["RETRIEVAL_BACKEND"] = "local"
os.environ["DATA_DIR"] = str(TEST_ROOT / "data")

import pytest  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def prepare_database():
    """建表并导入演示数据（租户、账号、13 份资料、2 张工单）。"""
    from app.db import dispose_db, init_db
    from scripts.seed_demo import main as seed_main

    async def _run():
        await init_db()
        await seed_main()
        await dispose_db()

    asyncio.run(_run())
    yield


@pytest.fixture
async def client():
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as async_client:
        yield async_client
