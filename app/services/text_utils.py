# app/services/text_utils.py
# 文本处理小工具：分词、哈希向量、余弦相似度、切分。
# 本机没有可用的嵌入模型，使用「字符 bigram + 英文词」的哈希向量作为本地可验证方案。
# 它不等同于语义嵌入，文档和简历中都应如实说明。

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter

try:
    import jieba

    jieba.setLogLevel(logging.WARNING)
except Exception:  # pragma: no cover - 未安装时退回 bigram
    jieba = None

_CJK = r"\u4e00-\u9fff"


def tokenize(text: str) -> list[str]:
    """中文优先用 jieba 分词，英文/数字按词；未安装 jieba 时退回两字 bigram。"""
    text = (text or "").lower()
    tokens: list[str] = []
    tokens.extend(re.findall(r"[a-z0-9_]+(?:\.[0-9]+)?", text))
    for segment in re.findall(f"[{_CJK}]+", text):
        if jieba is not None:
            tokens.extend(word for word in jieba.lcut(segment) if word.strip())
        elif len(segment) == 1:
            tokens.append(segment)
        else:
            tokens.extend(segment[i : i + 2] for i in range(len(segment) - 1))
    return tokens


# 查询侧停用词：只去掉疑问/客套等无信息量的词，减少对检索排序的干扰。
# 文档入库不做停用词过滤，避免影响已有向量。
_QUERY_STOPWORDS = frozenset(
    {
        # 中文单字
        "的", "了", "吗", "呢", "啊", "吧", "是", "在", "和", "与", "或", "及", "我", "你", "他",
        "它", "们", "有", "无", "请", "帮", "给", "出", "去", "来", "用", "做", "会", "能", "要",
        # 疑问/客套词
        "怎么", "怎样", "如何", "什么", "哪些", "哪个", "哪里", "为何", "为什么", "是否", "可以",
        "能否", "还能", "应该", "需要", "请问", "请给", "给出", "提供", "帮我", "告诉", "一下",
        "这个", "那个", "就是", "还是", "有没有", "没有", "多少", "多久", "现在", "一下",
        # 高频动作词：单独命中时不足以证明内容相关
        "支持", "导出", "进行", "使用",
        # 英文停用词
        "how", "what", "why", "is", "are", "can", "the", "a", "an", "to", "of", "for", "and",
        "or", "in", "on", "with", "please", "do", "does", "i", "you",
    }
)


def content_tokens(text: str) -> list[str]:
    """抽取查询中的内容词，用于词法覆盖率计算；过滤单字虚词。"""
    return [
        token
        for token in tokenize(text)
        if token not in _QUERY_STOPWORDS and (len(token) > 1 or re.fullmatch(r"[a-z0-9.]+", token))
    ]


def embed_text(text: str, dim: int = 512) -> list[float]:
    """基于哈希的向量化：同一 token 固定映射到同一维度。

    优点是零依赖、确定性、可离线运行；缺点是无法理解同义词。
    """
    counts = Counter(tokenize(text))
    vec = [0.0] * dim
    for token, count in counts.items():
        digest = hashlib.md5(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[index] += sign * (1.0 + math.log(count))
    norm = math.sqrt(sum(v * v for v in vec))
    if norm:
        vec = [v / norm for v in vec]
    return vec


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def lexical_coverage(query_tokens: list[str], doc_tokens: list[str]) -> float:
    """查询词在文档中的覆盖率，范围 0~1。比 IDF 简单，演示场景足够。"""
    if not query_tokens:
        return 0.0
    q = set(query_tokens)
    d = set(doc_tokens)
    return len(q & d) / len(q)


def split_markdown_sections(text: str) -> list[tuple[str, str]]:
    """按 Markdown 标题分节，返回 [(章节标题, 正文)]。"""
    sections: list[tuple[str, str]] = []
    current_title = "正文"
    buffer: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if match:
            if "".join(buffer).strip():
                sections.append((current_title, "\n".join(buffer).strip()))
            current_title = match.group(2).strip()
            buffer = []
        else:
            buffer.append(line)
    if "".join(buffer).strip():
        sections.append((current_title, "\n".join(buffer).strip()))
    return sections or [("正文", text.strip())]


def split_plain_sections(text: str) -> list[tuple[str, str]]:
    """纯文本按单节处理。"""
    return [("正文", text.strip())] if text.strip() else []


def chunk_text(text: str, size: int = 600, overlap: int = 80) -> list[str]:
    """按段落优先切分，超长段落再硬切；相邻块保留少量重叠。"""
    if not 0 <= overlap < size:
        raise ValueError("overlap must be non-negative and smaller than size")
    text = text.strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for para in paragraphs:
        if len(para) > size:
            flush()
            start = 0
            while start < len(para):
                chunks.append(para[start : start + size].strip())
                if start + size >= len(para):
                    break
                start += size - overlap
            continue
        if len(current) + len(para) + 1 <= size:
            current = f"{current}\n{para}" if current else para
        else:
            flush()
            current = para
    flush()

    if len(chunks) >= 2 and len(chunks[-1]) < 30 and len(chunks[-2]) + 1 + len(chunks[-1]) <= size:
        chunks[-2] = chunks[-2] + "\n" + chunks[-1]
        chunks.pop()
    return chunks


def make_snippet(content: str, limit: int = 160) -> str:
    content = re.sub(r"\s+", " ", content).strip()
    return content[:limit] + ("…" if len(content) > limit else "")
