# app/services/reranker.py
# 可选 BGE-Reranker 精排。默认关闭；只有显式配置且模型可用时才加载。
# 失败处理：由 retrieval 记录 "重排不可用，已按融合分数排序"，不冒充重排结果。

from __future__ import annotations

import threading
from pathlib import Path

from app.config import BASE_DIR, get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()

RERANK_MAX_INPUT_CHARS = 512


class RerankerUnavailable(RuntimeError):
    pass


def resolve_reranker_path() -> Path | None:
    candidates: list[Path] = []
    if settings.rerank_model_path:
        candidates.append(Path(settings.rerank_model_path))
    candidates.extend(
        [
            BASE_DIR / "models" / "reranker" / "bge-reranker-large",
            BASE_DIR / "models" / "reranker" / "bge-reranker-v2-m3",
        ]
    )
    for path in candidates:
        if path.is_dir() and any(path.glob("*.safetensors")):
            return path
    return None


class BGEReranker:
    def __init__(self) -> None:
        self._model = None
        self._error: str | None = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return self._model
        if self._error:
            raise RerankerUnavailable(self._error)

        model_path = resolve_reranker_path()
        if model_path is None:
            self._error = "未找到 BGE-Reranker 模型目录，请配置 RERANK_MODEL_PATH"
            raise RerankerUnavailable(self._error)
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            self._error = "缺少 sentence_transformers/torch，无法启用重排"
            raise RerankerUnavailable(self._error) from exc

        with self._lock:
            if self._model is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info("reranker.loading", model_path=str(model_path), device=device)
                self._model = CrossEncoder(str(model_path), device=device, max_length=512)
                logger.info("reranker.loaded")
        return self._model

    def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[float]:
        """返回与 documents 顺序一一对应的相关性分数（0~1 概率）。"""
        model = self._load()
        pairs = [(query, (document or "")[:RERANK_MAX_INPUT_CHARS]) for document in documents]
        try:
            scores = model.predict(pairs).tolist()
        except Exception as exc:
            self._error = f"重排推理失败：{exc.__class__.__name__}"
            raise RerankerUnavailable(self._error) from exc
        return [float(score) for score in scores[: top_k or len(scores)]]

    def health(self) -> dict:
        model_path = resolve_reranker_path()
        return {
            "enabled": settings.rerank_enabled,
            "available": bool(model_path),
            "model_path": str(model_path) if model_path else None,
            "loaded": self._model is not None,
            "detail": self._error or "",
        }


_RERANKER: BGEReranker | None = None


def get_reranker() -> BGEReranker:
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = BGEReranker()
    return _RERANKER
