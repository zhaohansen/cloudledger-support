# 运行方式：
#   1. 复制 .env.example 为 .env（可选，默认 SQLite 可直接运行）
#   2. .venv\Scripts\python.exe scripts\init_db.py
#   3. .venv\Scripts\python.exe scripts\seed_demo.py
#   4. .venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# 或直接执行：.venv\Scripts\python.exe run.py
import uvicorn

from app.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    print(f"登录页 http://{settings.app_host}:{settings.app_port}/login", flush=True)
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
    )
