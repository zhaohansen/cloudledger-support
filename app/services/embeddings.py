# app/services/embeddings.py
# Embedding 后端抽象：hash（演示）与 bge_m3（真实语义）。
# 关键约束：真实后端不可用时必须抛错，不能静默退回 hash 后继续声称使用了语义检索。

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.config import BASE_DIR, get_settings
from app.logging_config import get_logger
from app.services.text_utils import embed_text

logger = get_logger(__name__)
settings = get_settings()


class EmbeddingBackendUnavailable(RuntimeError):
    """真实 Embedding 后端缺少依赖、模型文件或加载失败。"""


@dataclass
class EmbeddedText:
    dense: list[float]
    sparse: dict[int, float] | None = None


class EmbeddingBackend(Protocol):
    name: str
    dim: int

    def encode_documents(self, texts: list[str]) -> list[EmbeddedText]: ...

    def encode_query(self, text: str) -> EmbeddedText: ...

    def health(self) -> dict: ...


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


class HashEmbeddingBackend:
    """轻量演示模式：哈希向量 + 词法覆盖。不是语义 Embedding，展示时必须如实标注。"""

    name = "hash"
    dim = settings.embedding_dim

    def encode_documents(self, texts: list[str]) -> list[EmbeddedText]:
        return [EmbeddedText(dense=embed_text(text, self.dim)) for text in texts]

    def encode_query(self, text: str) -> EmbeddedText:
        return EmbeddedText(dense=embed_text(text, self.dim))

    def health(self) -> dict:
        return {
            "name": self.name,
            "mode": "demo",
            "available": True,
            "dim": self.dim,
            "detail": "哈希向量 + 词法匹配，仅用于离线演示，不代表语义检索能力",
        }


def resolve_bge_m3_path() -> Path | None:
    """按配置、项目内目录、同工作区原项目目录的顺序查找本地 BGE-M3 权重。"""
    candidates: list[Path] = []
    if settings.bge_m3_model_path:
        candidates.append(Path(settings.bge_m3_model_path))
    candidates.extend(
        [
            BASE_DIR / "models" / "embedding" / "bge-m3",
            BASE_DIR.parent / "EduAgent完整代码" / "backend" / "models" / "embedding" / "bge-m3",
        ]
    )
    for path in candidates:
        if path.is_dir() and any(path.glob("*.safetensors")):
            return path
        if path.is_dir() and any(path.glob("pytorch_model*.bin")):
            return path
    return None


class BGEM3EmbeddingBackend:
    """BGE-M3：一次推理输出 1024 维 dense 与 sparse lexical_weights。

    模型只在首次编码时加载，避免应用启动被 2GB 权重拖慢；
    加载失败会抛出 EmbeddingBackendUnavailable，由调用方明确报错。
    """

    name = "bge_m3"
    dim = 1024

    def __init__(self) -> None:
        self._model = None
        self._load_error: str | None = None

    def _load(self):
        if self._model is not None:
            return self._model
        if self._load_error:
            raise EmbeddingBackendUnavailable(self._load_error)

        model_path = resolve_bge_m3_path()
        if model_path is None:
            self._load_error = (
                "未找到本地 BGE-M3 模型目录。请设置 BGE_M3_MODEL_PATH，"
                "或把权重放到 models/embedding/bge-m3。"
            )
            raise EmbeddingBackendUnavailable(self._load_error)

        try:
            # 兼容 FlagEmbedding 1.3.x 对 transformers 内部函数的依赖
            import importlib.util as _ilu

            from transformers.utils import import_utils as _tf_iu

            if not hasattr(_tf_iu, "is_torch_fx_available"):
                _tf_iu.is_torch_fx_available = (
                    lambda: _ilu.find_spec("torch.fx") is not None
                )

            import torch
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:  # 依赖缺失是可解释的阻塞，不降级
            self._load_error = (
                "缺少 BGE-M3 运行依赖（FlagEmbedding/torch）。"
                "请在具备这些依赖的环境中运行，或改用 EMBEDDING_BACKEND=hash 演示模式。"
            )
            raise EmbeddingBackendUnavailable(self._load_error) from exc

        logger.info("embedding.bge_m3.loading", model_path=str(model_path))
        self._model = BGEM3FlagModel(
            model_name_or_path=str(model_path),
            use_fp16=torch.cuda.is_available(),
        )
        logger.info("embedding.bge_m3.loaded", device="cuda" if torch.cuda.is_available() else "cpu")
        return self._model

    def _encode(self, texts: list[str], batch_size: int) -> list[EmbeddedText]:
        model = self._load()
        try:
            output = model.encode(
                texts,
                batch_size=batch_size,
                max_length=settings.bge_m3_max_length,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
        except Exception as exc:  # 模型推理失败同样不能静默降级
            self._load_error = f"BGE-M3 编码失败：{exc.__class__.__name__}: {exc}"
            raise EmbeddingBackendUnavailable(self._load_error) from exc

        dense_vectors = output["dense_vecs"].tolist()
        sparse_vectors = [
            {int(key): float(value) for key, value in weights.items()}
            for weights in output["lexical_weights"]
        ]
        return [
            EmbeddedText(dense=_normalize(dense), sparse=sparse)
            for dense, sparse in zip(dense_vectors, sparse_vectors)
        ]

    def encode_documents(self, texts: list[str]) -> list[EmbeddedText]:
        if not texts:
            return []
        return self._encode(texts, settings.bge_m3_batch_size)

    def encode_query(self, text: str) -> EmbeddedText:
        return self._encode([text], 1)[0]

    def health(self) -> dict:
        model_path = resolve_bge_m3_path()
        dependencies = all(
            importlib.util.find_spec(name) is not None
            for name in ("torch", "FlagEmbedding", "transformers")
        )
        return {
            "name": self.name,
            "mode": "semantic",
            "available": bool(model_path and dependencies),
            "dim": self.dim,
            "model_path": str(model_path) if model_path else None,
            "dependencies_available": dependencies,
            "loaded": self._model is not None,
            "detail": self._load_error or "",
        }


_BACKENDS: dict[str, EmbeddingBackend] = {}


def get_embedding_backend(name: str | None = None) -> EmbeddingBackend:
    """按配置返回后端单例；未知名称直接报错，避免拼写错误被悄悄当成 hash。"""
    backend_name = (name or settings.embedding_backend or "hash").strip().lower()
    if backend_name not in ("hash", "bge_m3"):
        raise EmbeddingBackendUnavailable(
            f"未知 EMBEDDING_BACKEND={backend_name}，可选 hash / bge_m3"
        )
    if backend_name not in _BACKENDS:
        _BACKENDS[backend_name] = (
            HashEmbeddingBackend() if backend_name == "hash" else BGEM3EmbeddingBackend()
        )
    return _BACKENDS[backend_name]


def reset_embedding_backend_cache() -> None:
    """测试用：切换配置后清空单例，避免拿到旧后端。"""
    _BACKENDS.clear()


def embedding_runtime_status() -> dict:
    """健康检查与评测报告用：只检查可用性，不触发大模型加载。"""
    try:
        backend = get_embedding_backend()
        return {
            "backend": backend.name,
            "retrieval_backend": settings.retrieval_backend,
            **backend.health(),
        }
    except EmbeddingBackendUnavailable as exc:
        return {
            "backend": settings.embedding_backend,
            "retrieval_backend": settings.retrieval_backend,
            "available": False,
            "detail": str(exc),
        }
