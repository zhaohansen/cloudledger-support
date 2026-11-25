# app/config.py
# 全项目唯一配置中心：所有可调参数集中在这里，通过环境变量或 .env 覆盖。
# 注意：本文件不保存任何真实密钥；未配置 JWT 密钥时会自动生成到 data/.jwt_secret。

from __future__ import annotations

import os
import secrets
from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """应用配置。字段名即环境变量名（大小写不敏感）。"""

    # ── 应用 ──────────────────────────────────────────────
    app_name: str = "云账科技 企业财税费控支持与工单系统"
    app_env: str = "local"
    app_debug: bool = False
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    log_level: str = "INFO"

    # ── 数据库 ────────────────────────────────────────────
    # 默认 SQLite，开箱即用；配置 DATABASE_URL 可切换 PostgreSQL（本次未实测）。
    database_url: str = f"sqlite+aiosqlite:///{(BASE_DIR / 'data' / 'app.db').as_posix()}"

    # ── 认证 ──────────────────────────────────────────────
    # 留空时启动会生成随机密钥存入 data/.jwt_secret，避免把密钥写死在代码里。
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 480

    # ── 大模型 ────────────────────────────────────────────
    # auto：有 Key 用 OpenAI 兼容接口，没有则走摘录式回答（不编造）
    # extractive：强制摘录式；openai：强制调用模型（失败会降级并标注）
    llm_mode: str = "auto"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    llm_timeout_seconds: int = 30
    llm_temperature: float = 0.1

    # ── 知识库 ────────────────────────────────────────────
    data_dir: str = str(BASE_DIR / "data")
    upload_max_mb: int = 5
    chunk_size: int = 600
    chunk_overlap: int = 80
    retrieval_top_k: int = 5
    retrieval_threshold: float = 0.2
    # 允许上传的扩展名；扫描版 PDF 暂不支持，会在导入状态里明确提示
    allowed_extensions: tuple[str, ...] = (".md", ".markdown", ".txt", ".pdf")

    # ── 检索后端 ──────────────────────────────────────────
    # embedding_backend=hash  ：512 维哈希 + 词法覆盖，零依赖演示模式，不是语义 Embedding
    # embedding_backend=bge_m3：本地 BGE-M3（1024 维 dense + sparse lexical_weights），真实语义检索
    embedding_backend: str = "hash"
    embedding_dim: int = 512  # 仅 hash 后端使用
    bge_m3_model_path: str = ""
    bge_m3_batch_size: int = 8
    bge_m3_max_length: int = 1024
    # local=业务库存向量；milvus=使用 Milvus 混合检索（需要先启动服务并重建索引）
    retrieval_backend: str = "local"
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_collection: str = "cloudledger_knowledge"
    milvus_timeout_seconds: int = 5
    # BGE-M3 混合检索：dense 余弦 + sparse 词权重覆盖的加权和，不是 BM25/RRF
    hybrid_dense_weight: float = 0.65
    hybrid_sparse_weight: float = 0.35
    semantic_dense_threshold: float = 0.35
    semantic_sparse_threshold: float = 0.12
    # 混合相关度下限：低于该值视为证据不足，避免语义检索对任何问题都"凑"出结果。
    # 该值来自实际分数分布的保守起点，更换 Embedding/重排后应重新校准。
    semantic_score_threshold: float = 0.42
    # 可选重排：默认关闭，只有显式开启且模型可用时才生效；失败会在结果里标注
    rerank_enabled: bool = False
    rerank_model_path: str = ""
    rerank_candidate_k: int = 10
    rerank_top_k: int = 5

    # ── 会话 ──────────────────────────────────────────────
    chat_history_limit: int = 8

    # ── MCP / 公开联网搜索（可选，默认关闭） ───────────────
    # web_search_mcp_url：外部 Web Search MCP Server 地址（如 http://127.0.0.1:8001）
    # web_search_api_key：可选 Tavily 等公开搜索 API Key；均留空时显式返回 unconfigured，绝不伪造结果
    web_search_mcp_url: str = ""
    web_search_api_key: str = ""
    web_search_timeout_seconds: float = 8.0

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @model_validator(mode="after")
    def _fallback_env_keys(self) -> "Settings":
        if not self.llm_api_key and "deepseek.com" in self.llm_base_url.lower():
            self.llm_api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not self.web_search_api_key:
            self.web_search_api_key = os.getenv("TAVILY_API_KEY", "").strip()
        return self

    # ── 派生属性 ──────────────────────────────────────────
    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def upload_path(self) -> Path:
        p = self.data_path / "uploads"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def llm_available(self) -> bool:
        """是否具备真实模型调用条件（仅判断配置，不代表调用一定成功）。"""
        if self.llm_mode == "extractive":
            return False
        return bool(self.llm_api_key)

    @property
    def effective_llm_mode(self) -> str:
        if self.llm_mode == "openai" and not self.llm_api_key:
            return "extractive"
        if self.llm_mode == "auto":
            return "openai" if self.llm_api_key else "extractive"
        return self.llm_mode

    def resolve_jwt_secret(self) -> str:
        """返回 JWT 密钥：优先环境变量，其次本地随机文件，最后新建。

        这样既不把密钥写进仓库，也保证重启后已签发 Token 仍然有效。
        """
        if self.jwt_secret_key:
            return self.jwt_secret_key
        secret_file = self.data_path / ".jwt_secret"
        if secret_file.exists():
            return secret_file.read_text(encoding="utf-8").strip()
        value = secrets.token_urlsafe(48)
        secret_file.write_text(value, encoding="utf-8")
        return value


@lru_cache()
def get_settings() -> Settings:
    """全局单例配置，避免重复读取 .env。"""
    return Settings()
