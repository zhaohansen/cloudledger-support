# app/services/llm.py
# 模型客户端工厂：集中管理模型配置、超时、有限重试与日志。
# 不打印密钥；未配置凭证时由上层明确标注为摘录/降级，而不是假装调用成功。

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator

from app.config import get_settings
from app.logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)


class LLMServiceError(RuntimeError):
    """模型服务不可用、返回异常或解析失败。"""


class LLMTimeout(LLMServiceError):
    """模型调用超时。"""


class LLMClient:
    def __init__(self) -> None:
        self._models: dict[bool, object] = {}

    @property
    def configured(self) -> bool:
        return settings.llm_available

    def _model(self, streaming: bool):
        if streaming not in self._models:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as exc:
                raise LLMServiceError("未安装 langchain-openai，无法调用大模型") from exc

            self._models[streaming] = ChatOpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                timeout=settings.llm_timeout_seconds,
                max_retries=0,  # 重试由本类控制，避免与 SDK 重试叠加
                streaming=streaming,
            )
        return self._models[streaming]

    async def complete(self, messages: list[tuple[str, str]]) -> str:
        """非流式调用。超时不重试；其他瞬时错误最多重试 1 次。"""
        if not self.configured:
            raise LLMServiceError("未配置 LLM_API_KEY，无法调用模型")

        last_error: Exception | None = None
        for attempt in range(2):
            started = time.perf_counter()
            try:
                async with asyncio.timeout(settings.llm_timeout_seconds):
                    result = await self._model(False).ainvoke(messages)
                elapsed = time.perf_counter() - started
                logger.info("llm.complete", attempt=attempt + 1, ms=round(elapsed * 1000, 1))
                return str(result.content).strip()
            except (TimeoutError, asyncio.TimeoutError) as exc:
                logger.warning("llm.timeout", attempt=attempt + 1)
                raise LLMTimeout(f"模型响应超过 {settings.llm_timeout_seconds} 秒") from exc
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "llm.error", attempt=attempt + 1, error=exc.__class__.__name__
                )
                if attempt == 0:
                    await asyncio.sleep(0.4)
        raise LLMServiceError(
            f"模型调用失败：{last_error.__class__.__name__ if last_error else 'unknown'}"
        )

    async def stream(self, messages: list[tuple[str, str]]) -> AsyncIterator[str]:
        """真实流式：逐块转发模型输出；仅在首 token 之前允许重试 1 次，避免瞬时连接抖动。"""
        if not self.configured:
            raise LLMServiceError("未配置 LLM_API_KEY，无法调用模型")
        started = time.perf_counter()
        first_token_ms: float | None = None
        try:
            for attempt in range(2):
                try:
                    async for chunk in self._model(True).astream(messages):
                        content = getattr(chunk, "content", "")
                        if isinstance(content, list):
                            content = "".join(
                                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                                for part in content
                            )
                        if not content:
                            continue
                        if first_token_ms is None:
                            first_token_ms = (time.perf_counter() - started) * 1000
                        yield str(content)
                    return
                except Exception as exc:
                    if first_token_ms is None and attempt == 0:
                        logger.warning(
                            "llm.stream_retry", attempt=attempt + 1, error=exc.__class__.__name__
                        )
                        await asyncio.sleep(0.4)
                        continue
                    raise
        finally:
            logger.info(
                "llm.stream",
                first_token_ms=round(first_token_ms or 0, 1),
                total_ms=round((time.perf_counter() - started) * 1000, 1),
            )

    async def complete_json(self, messages: list[tuple[str, str]]) -> dict:
        """要求模型输出 JSON 并解析。用于 Agent 的结构化理解，不做写操作。"""
        raw = await self.complete(messages)
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMServiceError("模型输出不是合法 JSON") from exc
        if not isinstance(data, dict):
            raise LLMServiceError("模型输出的 JSON 不是对象")
        return data


_CLIENT: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = LLMClient()
    return _CLIENT


def llm_status() -> dict:
    return {
        "mode": settings.effective_llm_mode,
        "configured_mode": settings.llm_mode,
        "model": settings.llm_model,
        "configured": settings.llm_available,
        "timeout_seconds": settings.llm_timeout_seconds,
    }
