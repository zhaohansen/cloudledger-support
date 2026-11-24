# app/logging_config.py
# 日志配置：每条日志自动带 request_id，便于把一次请求的 API、工具、检索串起来。

from __future__ import annotations

import contextvars
import logging
import sys

from app.config import get_settings

# 请求级上下文变量：中间件写入，日志格式器读取
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get("-")
        return True


class _Logger:
    """结构化日志包装：logger.info("event", user_id=...) 输出 key=value。"""

    def __init__(self, name: str):
        self._log = logging.getLogger(name)

    @staticmethod
    def _fmt(event: str, kw: dict) -> str:
        if not kw:
            return event
        return event + " | " + " ".join(f"{k}={v!r}" for k, v in kw.items())

    def debug(self, event: str, **kw) -> None:
        self._log.debug(self._fmt(event, kw))

    def info(self, event: str, **kw) -> None:
        self._log.info(self._fmt(event, kw))

    def warning(self, event: str, **kw) -> None:
        self._log.warning(self._fmt(event, kw))

    def error(self, event: str, **kw) -> None:
        exc_info = kw.pop("exc_info", True)
        self._log.error(self._fmt(event, kw), exc_info=exc_info)

    def exception(self, event: str, **kw) -> None:
        self._log.exception(self._fmt(event, kw))


def configure_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] [rid=%(request_id)s] %(name)s: %(message)s")
    )
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # 降低第三方库噪音
    for name in ("httpx", "httpcore", "sqlalchemy.engine", "sqlalchemy.pool", "uvicorn.access"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> _Logger:
    return _Logger(name)
