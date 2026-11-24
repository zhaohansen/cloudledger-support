# scripts/init_db.py
# 初始化数据库结构。默认不会删除已有数据；需要重建时显式传 --reset。
#
# 用法：
#   python scripts/init_db.py            # 创建缺失的表
#   python scripts/init_db.py --reset    # 删除 SQLite 数据库文件后重建（仅本地演示）
#   python scripts/init_db.py --demo     # 初始化后导入演示数据

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# 允许直接以 `python scripts/init_db.py` 运行
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _sqlite_file_path(database_url: str) -> Path | None:
    prefix = "sqlite+aiosqlite:///"
    if not database_url.startswith(prefix):
        return None
    return Path(database_url[len(prefix):])


async def _run(reset: bool, demo: bool) -> None:
    from app.config import get_settings
    from app.db import dispose_db, init_db

    settings = get_settings()

    if reset:
        db_file = _sqlite_file_path(settings.database_url)
        if db_file is None:
            raise SystemExit("--reset 只支持本地 SQLite，请手动处理 PostgreSQL 数据")
        # 先释放可能存在的连接，再删除文件
        await dispose_db()
        db_file.unlink(missing_ok=True)
        print(f"[init_db] 已删除旧数据库：{db_file}")

    await init_db()
    print(f"[init_db] 数据表已就绪：{settings.database_url}")
    await dispose_db()

    if demo:
        from scripts.seed_demo import main as seed_main

        await seed_main()


def main() -> None:
    parser = argparse.ArgumentParser(description="初始化云账客服系统数据库")
    parser.add_argument("--reset", action="store_true", help="删除本地 SQLite 文件后重建")
    parser.add_argument("--demo", action="store_true", help="初始化后导入演示租户、账号、资料和工单")
    args = parser.parse_args()
    asyncio.run(_run(reset=args.reset, demo=args.demo))


if __name__ == "__main__":
    main()
