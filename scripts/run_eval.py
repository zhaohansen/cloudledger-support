# scripts/run_eval.py
# 可信评测：检索 / 回答 / 工具执行 / 权限 四类分开统计。
#
# 口径原则：
#   1. 每个指标都写出分子、分母、样本范围和运行模式；
#   2. 文档级命中与分块级命中分开定义；
#   3. 回答质量只检查回答正文，不能靠引用片段补足；
#   4. 顺序重复请求与并发请求分开记录；
#   5. 记录实际数据库、Embedding、检索后端和模型，不写"配置文件里的期望值"；
#   6. 默认离线运行（摘录式回答 + 当前检索后端），真实模型调用必须显式授权。
#
# 用法：
#   python scripts/run_eval.py                     # 离线，当前后端
#   python scripts/run_eval.py --suite retrieval
#   python scripts/run_eval.py --live-llm --allow-paid-llm --limit 3
#   EMBEDDING_BACKEND=bge_m3 RETRIEVAL_BACKEND=milvus python scripts/run_eval.py --output-name evaluation-bge-m3-milvus

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import tempfile
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 参数必须在导入 app.* 之前决定，因为配置是进程级单例
LIVE_LLM = "--live-llm" in sys.argv
if not LIVE_LLM:
    os.environ["LLM_MODE"] = "extractive"
if "--embedding-backend" in sys.argv:
    _eb_idx = sys.argv.index("--embedding-backend")
    if _eb_idx + 1 < len(sys.argv):
        os.environ["EMBEDDING_BACKEND"] = sys.argv[_eb_idx + 1]

# 每个进程使用独立数据库、上传目录和集合，可同时运行不同配置的评测。
EVAL_FILES = tempfile.TemporaryDirectory(prefix="cloudledger_eval_")
EVAL_DB = Path(EVAL_FILES.name) / "eval.db"
os.environ["APP_ENV"] = "eval"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{EVAL_DB.as_posix()}"
os.environ["JWT_SECRET_KEY"] = "eval-only-secret-not-for-production"
os.environ["DATA_DIR"] = EVAL_FILES.name
os.environ["MILVUS_COLLECTION"] = f"cloudledger_eval_{uuid.uuid4().hex}"

from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.db import AsyncSessionLocal  # noqa: E402
from app.models import Ticket  # noqa: E402
from app.services.retrieval import search_knowledge  # noqa: E402
from tests.helpers import auth, ctx_for, token_for  # noqa: E402

CASES_PATH = PROJECT_ROOT / "eval" / "test_cases.json"
REPORT_DIR = PROJECT_ROOT / "reports"


# ── 通用工具 ─────────────────────────────────────────────────


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(ratio * (len(ordered) - 1)))))
    return round(ordered[index], 1)


def _latency(values: list[float]) -> dict:
    values = [round(value, 1) for value in values]
    return {
        "samples": len(values),
        "p50_ms": _percentile(values, 0.5),
        "p95_ms": _percentile(values, 0.95),
        "avg_ms": round(statistics.fmean(values), 1) if values else 0.0,
    }


def _metric(
    name: str,
    numerator: int,
    denominator: int,
    *,
    scope: str,
    mode: str,
    note: str = "",
) -> dict:
    return {
        "name": name,
        "value": round(numerator / denominator, 4) if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
        "scope": scope,
        "mode": mode,
        "note": note,
    }


def _hit_titles(titles: list[str], expected: list[str] | None) -> bool | None:
    if not expected:
        return None
    return any(any(exp in title for title in titles) for exp in expected)


def _leaked_titles(titles: list[str], forbidden: list[str] | None) -> list[str]:
    if not forbidden:
        return []
    return [title for title in titles if any(bad in title for bad in forbidden)]


async def _environment() -> dict:
    from app.config import get_settings
    from app.services.embeddings import embedding_runtime_status
    from app.services.llm import llm_status

    settings = get_settings()
    async with AsyncSessionLocal() as db:
        dialect = db.get_bind().dialect.name
        driver = db.get_bind().dialect.driver
    return {
        "database": {
            "dialect": dialect,
            "driver": driver,
            "scope": "评测使用每次运行专属的临时 SQLite 数据库",
        },
        "embedding": embedding_runtime_status(),
        "retrieval_backend": settings.retrieval_backend,
        "rerank": {
            "enabled": settings.rerank_enabled,
            "model_configured": bool(settings.rerank_model_path),
        },
        "llm": llm_status(),
        "answer_mode": settings.effective_llm_mode,
        "chat_history_limit": settings.chat_history_limit,
    }


def _mode_label(env: dict) -> str:
    embedding = env["embedding"].get("backend") or env["embedding"].get("name")
    retrieval = env["retrieval_backend"]
    if LIVE_LLM:
        llm = f"live:{env['llm'].get('model')}"
    else:
        llm = "offline:extractive"
    return f"{llm} / embedding={embedding} / retrieval={retrieval}"


async def _new_session(client: AsyncClient, token: str, title: str) -> str:
    response = await client.post("/api/chat/sessions", headers=auth(token), json={"title": title})
    response.raise_for_status()
    return response.json()["id"]


async def _send_chat(
    client: AsyncClient,
    token: str,
    session_id: str,
    content: str,
    product_version: str | None = None,
) -> dict:
    payload: dict = {"content": content}
    if product_version:
        payload["product_version"] = product_version
    response = await client.post(
        f"/api/chat/sessions/{session_id}/messages", headers=auth(token), json=payload
    )
    response.raise_for_status()
    return response.json()


async def _stream_chat(
    client: AsyncClient, token: str, session_id: str, content: str, product_version: str | None
) -> tuple[list[dict], dict, float]:
    """返回 (事件列表, result 事件, 客户端观测总耗时毫秒)。"""
    payload: dict = {"content": content}
    if product_version:
        payload["product_version"] = product_version
    events: list[dict] = []
    started = time.perf_counter()
    async with client.stream(
        "POST",
        f"/api/chat/sessions/{session_id}/stream",
        headers=auth(token),
        json=payload,
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                continue
            events.append(json.loads(data))
    total_ms = (time.perf_counter() - started) * 1000
    result = next((event for event in events if event.get("type") == "result"), {})
    return events, result, total_ms


async def _ticket_count(username: str) -> int:
    ctx = await ctx_for(username)
    async with AsyncSessionLocal() as db:
        rows = (
            await db.execute(select(Ticket.id).where(Ticket.created_by == ctx.user_id))
        ).scalars().all()
    return len(rows)


# ── 检索评测 ─────────────────────────────────────────────────


async def run_retrieval_suite(cases: list[dict], env: dict) -> dict:
    mode = _mode_label(env)
    records: list[dict] = []
    async with AsyncSessionLocal() as db:
        for case in cases:
            ctx = await ctx_for(case["username"])
            started = time.perf_counter()
            outcome = await search_knowledge(
                db, ctx, case["query"], product_version=case.get("product_version")
            )
            elapsed = (time.perf_counter() - started) * 1000
            titles = [item.title for item in outcome.items]
            top5 = [
                {
                    "title": item.title,
                    "section": item.section,
                    "chunk_index": item.chunk_index,
                    "score": round(item.score, 4),
                    "content": item.content[:120],
                }
                for item in outcome.items[:5]
            ]
            doc_hit3 = _hit_titles(titles[:3], case.get("expected_doc_titles"))
            doc_hit5 = _hit_titles(titles[:5], case.get("expected_doc_titles"))
            keywords = case.get("expected_chunk_keywords") or []
            chunk_hit3 = (
                any(keyword in item.content for item in outcome.items[:3] for keyword in keywords)
                if keywords
                else None
            )
            chunk_hit5 = (
                any(keyword in item.content for item in outcome.items[:5] for keyword in keywords)
                if keywords
                else None
            )
            leaked = _leaked_titles(titles[:5], case.get("forbidden_titles"))
            no_match_ok = (
                (not outcome.items) if case.get("expect_no_match") is not None else None
            )
            conflict_ok = (
                outcome.version_conflict == case["expect_version_conflict"]
                if "expect_version_conflict" in case
                else None
            )
            checks = [
                value
                for value in (doc_hit3, chunk_hit3, conflict_ok, no_match_ok)
                if value is not None
            ]
            # 只配置了 forbidden_titles 的权限用例没有正向命中要求，未泄漏即通过
            passed = (all(checks) if checks else True) and not leaked
            records.append(
                {
                    "id": case["id"],
                    "suite": "retrieval",
                    "tags": case.get("tags", []),
                    "query": case["query"],
                    "product_version": case.get("product_version"),
                    "username": case["username"],
                    "passed": passed,
                    "doc_hit_at_3": doc_hit3,
                    "doc_hit_at_5": doc_hit5,
                    "chunk_hit_at_3": chunk_hit3,
                    "chunk_hit_at_5": chunk_hit5,
                    "version_conflict": outcome.version_conflict,
                    "version_conflict_ok": conflict_ok,
                    "empty_result_ok": no_match_ok,
                    "reason": outcome.reason,
                    "detail": outcome.detail,
                    "embedding_backend": outcome.embedding_backend,
                    "retrieval_backend": outcome.retrieval_backend,
                    "rerank_status": outcome.rerank_status,
                    "retrieval_ms": round(elapsed, 1),
                    "leaked_titles": leaked,
                    "top5": top5,
                }
            )

    with_docs = [r for r in records if r["doc_hit_at_3"] is not None]
    with_chunks = [r for r in records if r["chunk_hit_at_3"] is not None]
    forbidden_ids = {case["id"] for case in cases if case.get("forbidden_titles")}
    with_forbidden = [r for r in records if r["id"] in forbidden_ids]
    no_match_cases = [
        r for case, r in zip(cases, records) if case.get("expect_no_match") is not None
    ]
    conflict_cases = [
        r for case, r in zip(cases, records) if "expect_version_conflict" in case
    ]

    metrics = [
        _metric(
            "retrieval.doc_hit@3",
            sum(1 for r in with_docs if r["doc_hit_at_3"]),
            len(with_docs),
            scope="retrieval 用例中配置了 expected_doc_titles 的样本；前 3 个分块所属文档标题命中任一期望标题",
            mode=mode,
        ),
        _metric(
            "retrieval.doc_hit@5",
            sum(1 for r in with_docs if r["doc_hit_at_5"]),
            len(with_docs),
            scope="同上；前 5 个分块",
            mode=mode,
        ),
        _metric(
            "retrieval.chunk_hit@3",
            sum(1 for r in with_chunks if r["chunk_hit_at_3"]),
            len(with_chunks),
            scope="配置了 expected_chunk_keywords 的样本；前 3 个分块正文包含任一关键词",
            mode=mode,
        ),
        _metric(
            "retrieval.chunk_hit@5",
            sum(1 for r in with_chunks if r["chunk_hit_at_5"]),
            len(with_chunks),
            scope="同上；前 5 个分块",
            mode=mode,
        ),
        _metric(
            "retrieval.forbidden_leak_rate",
            sum(1 for r in with_forbidden if r["leaked_titles"]),
            len(with_forbidden),
            scope="配置了 forbidden_titles 的跨租户样本；前 5 条出现禁止标题即泄漏（越低越好）",
            mode=mode,
        ),
        _metric(
            "retrieval.empty_result_accuracy",
            sum(1 for r in no_match_cases if r["empty_result_ok"]),
            len(no_match_cases),
            scope="expect_no_match 样本；检索结果必须为空",
            mode=mode,
        ),
        _metric(
            "retrieval.version_conflict_detection",
            sum(1 for r in conflict_cases if r["version_conflict_ok"]),
            len(conflict_cases),
            scope="未指定版本且命中多个版本时，应标记 version_conflict",
            mode=mode,
        ),
    ]
    return {
        "cases": records,
        "metrics": metrics,
        "latency": {"retrieval_ms": _latency([r["retrieval_ms"] for r in records])},
    }


# ── 回答评测 ─────────────────────────────────────────────────


async def run_answer_suite(cases: list[dict], env: dict, limit: int | None) -> dict:
    mode = _mode_label(env)
    if limit:
        cases = cases[:limit]
    records: list[dict] = []
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        for case in cases:
            token = await token_for(case["username"])
            session_id = await _new_session(client, token, case["id"])
            turns = case.get("turns") or [case["question"]]
            turn_records: list[dict] = []
            last: dict = {}
            first_token_ms: float | None = None
            total_ms = 0.0
            try:
                for index, turn in enumerate(turns):
                    is_last = index == len(turns) - 1
                    started = time.perf_counter()
                    if is_last and case.get("stream"):
                        events, result, total_ms = await _stream_chat(
                            client, token, session_id, turn, case.get("product_version")
                        )
                        first_token_ms = (result.get("timings") or {}).get("first_token_ms")
                        body = {
                            "reply": result.get("answer", ""),
                            "citations": result.get("citations") or [],
                            "answer_mode": result.get("answer_mode"),
                            "answer_meta": result,
                        }
                    else:
                        body = await _send_chat(
                            client, token, session_id, turn, case.get("product_version")
                        )
                        total_ms = (time.perf_counter() - started) * 1000
                    last = body
                    turn_records.append(
                        {
                            "question": turn,
                            "reply": body.get("reply", ""),
                            "answer_mode": body.get("answer_mode"),
                            "citations": [
                                {
                                    "index": c.get("index"),
                                    "title": c.get("title"),
                                    "section": c.get("section"),
                                    "snippet": c.get("snippet"),
                                }
                                for c in (body.get("citations") or [])
                            ],
                        }
                    )
            except Exception as exc:
                records.append(
                    {
                        "id": case["id"],
                        "suite": "answer",
                        "tags": case.get("tags", []),
                        "question": " / ".join(turns),
                        "passed": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            reply = last.get("reply", "")
            citations = last.get("citations") or []
            answer_mode = last.get("answer_mode")
            meta = last.get("answer_meta") or {}

            keywords = case.get("expected_body_keywords") or []
            body_keyword_ok = (
                all(keyword in reply for keyword in keywords) if keywords else None
            )
            forbidden_phrases = case.get("forbidden_body_phrases") or []
            forbidden_hit = [phrase for phrase in forbidden_phrases if phrase in reply]
            need_citations = case.get("expect_citations")
            no_citations_ok = (
                (not citations) if case.get("expect_no_citations") is not None else None
            )
            citation_valid = None
            if need_citations:
                indexes = [citation.get("index") for citation in citations]
                citation_valid = bool(citations) and indexes == list(range(1, len(citations) + 1))
                citation_valid = citation_valid and all(
                    citation.get("title") and citation.get("snippet")
                    for citation in citations
                )
            expected_titles = case.get("expected_citation_titles") or []
            citation_title_ok = (
                _hit_titles([c.get("title", "") for c in citations], expected_titles)
                if expected_titles
                else None
            )
            mode_ok = (
                answer_mode == case["expect_answer_mode"]
                if case.get("expect_answer_mode")
                else None
            )
            checks = [
                value
                for value in (
                    body_keyword_ok,
                    citation_valid,
                    no_citations_ok,
                    citation_title_ok,
                    mode_ok,
                )
                if value is not None
            ]
            passed = bool(checks) and all(checks) and not forbidden_hit

            records.append(
                {
                    "id": case["id"],
                    "suite": "answer",
                    "tags": case.get("tags", []),
                    "question": " / ".join(turns),
                    "username": case["username"],
                    "passed": passed,
                    "body_keyword_ok": body_keyword_ok,
                    "citation_valid": citation_valid,
                    "no_citations_ok": no_citations_ok,
                    "citation_title_ok": citation_title_ok,
                    "mode_ok": mode_ok,
                    "forbidden_body_hits": forbidden_hit,
                    "answer_mode": answer_mode,
                    "expected_answer_mode": case.get("expect_answer_mode"),
                    "answer_body": reply,
                    "citations": [
                        {
                            "index": c.get("index"),
                            "title": c.get("title"),
                            "section": c.get("section"),
                            "snippet": c.get("snippet"),
                        }
                        for c in citations
                    ],
                    "turns": turn_records,
                    "retrieval_ms": (meta or {}).get("retrieval_ms"),
                    "total_ms": round(total_ms, 1),
                    "first_token_ms": first_token_ms,
                    "embedding_backend": (meta or {}).get("embedding_backend"),
                    "retrieval_backend": (meta or {}).get("retrieval_backend"),
                    "mode": mode,
                }
            )

    def _subset(predicate):
        return [r for r in records if predicate(r)]

    keyword_cases = _subset(lambda r: r.get("body_keyword_ok") is not None)
    citation_cases = _subset(lambda r: r.get("citation_valid") is not None)
    no_evidence_cases = _subset(lambda r: r.get("expected_answer_mode") == "no_evidence")
    version_cases = _subset(lambda r: r.get("expected_answer_mode") == "no_version_match")
    multi_turn = _subset(lambda r: "多轮指代" in r.get("tags", []))
    forbidden_cases = _subset(lambda r: r.get("forbidden_body_hits") is not None)

    retrieval_values = [
        r["retrieval_ms"] for r in records if isinstance(r.get("retrieval_ms"), (int, float))
    ]
    first_token_values = [
        r["first_token_ms"] for r in records if isinstance(r.get("first_token_ms"), (int, float))
    ]
    total_values = [r["total_ms"] for r in records if isinstance(r.get("total_ms"), (int, float))]

    metrics = [
        _metric(
            "answer.body_keyword_pass_rate",
            sum(1 for r in keyword_cases if r["body_keyword_ok"]),
            len(keyword_cases),
            scope="配置了 expected_body_keywords 的样本；关键词必须出现在回答正文（不含引用片段）",
            mode=mode,
        ),
        _metric(
            "answer.citation_validity_rate",
            sum(1 for r in citation_cases if r["citation_valid"]),
            len(citation_cases),
            scope="要求提供引用的样本；编号从 1 连续，且每条引用含标题与片段",
            mode=mode,
        ),
        _metric(
            "answer.no_evidence_refusal_rate",
            sum(1 for r in no_evidence_cases if r["answer_mode"] == "no_evidence" and not r["citations"]),
            len(no_evidence_cases),
            scope="无证据样本；必须返回 no_evidence 且无引用",
            mode=mode,
        ),
        _metric(
            "answer.version_mismatch_detection_rate",
            sum(1 for r in version_cases if r["answer_mode"] == "no_version_match" and not r["citations"]),
            len(version_cases),
            scope="指定不存在版本 v9.9 的样本；必须返回 no_version_match，不能跨版本猜测",
            mode=mode,
        ),
        _metric(
            "answer.multi_turn_reference_pass_rate",
            sum(1 for r in multi_turn if r["passed"]),
            len(multi_turn),
            scope="多轮指代样本；第二轮引用必须落到上一轮讨论的 v2.2 预算资料",
            mode=mode,
        ),
        _metric(
            "answer.forbidden_phrase_rate",
            sum(1 for r in forbidden_cases if r["forbidden_body_hits"]),
            len(forbidden_cases),
            scope="配置了 forbidden_body_phrases 的样本；回答正文出现禁止短语即失败（越低越好）",
            mode=mode,
        ),
    ]
    return {
        "cases": records,
        "metrics": metrics,
        "latency": {
            "answer_total_ms": _latency(total_values),
            "retrieval_ms": _latency(retrieval_values),
            "first_token_ms": {
                **_latency(first_token_values),
                "note": "仅真实模型流式调用才有值；摘录模式为 None，不冒充首 token 延迟",
            },
        },
    }


# ── 工具执行评测 ─────────────────────────────────────────────


async def _make_draft(client, token: str, username: str, message: str, title: str) -> tuple[str, dict | None]:
    session_id = await _new_session(client, token, title)
    body = await _send_chat(client, token, session_id, message)
    return session_id, body.get("draft")


async def _confirm(client, token: str, draft: dict, expected_version: int):
    """确认必须带用户看到的版本；评测入口不允许省略，避免把绕过校验当成正常流程。"""
    started = time.perf_counter()
    payload = {"draft_id": draft["id"], "expected_version": int(expected_version)}
    response = await client.post("/api/tickets/confirm", headers=auth(token), json=payload)
    return response, (time.perf_counter() - started) * 1000


async def run_tool_suite(cases: list[dict], env: dict) -> dict:
    mode = _mode_label(env)
    records: list[dict] = []
    confirm_latencies: list[float] = []
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        for case in cases:
            scenario = case["scenario"]
            record = {"id": case["id"], "suite": "tool", "tags": case.get("tags", []), "scenario": scenario}
            try:
                if scenario == "draft_only":
                    token = await token_for(case["username"])
                    before = await _ticket_count(case["username"])
                    _, draft = await _make_draft(client, token, case["username"], case["message"], case["id"])
                    after = await _ticket_count(case["username"])
                    record.update(
                        draft_created=bool(draft),
                        ticket_created=after > before,
                        passed=bool(draft) and after == before,
                    )

                elif scenario == "negative_confirm":
                    token = await token_for(case["username"])
                    phrase_results = []
                    for index, phrase in enumerate(case["phrases"], start=1):
                        before = await _ticket_count(case["username"])
                        session_id, draft = await _make_draft(
                            client, token, case["username"], case["base_message"], f"{case['id']}-{index}"
                        )
                        if not draft:
                            phrase_results.append(
                                {"phrase": phrase, "passed": False, "reason": "未生成草稿"}
                            )
                            continue
                        reply = await _send_chat(client, token, session_id, phrase)
                        after = await _ticket_count(case["username"])
                        phrase_results.append(
                            {
                                "phrase": phrase,
                                "intent": reply.get("intent"),
                                "ticket_created": after > before,
                                "draft_status": (reply.get("draft") or {}).get("status"),
                                "passed": after == before,
                            }
                        )
                    record.update(phrases=phrase_results, passed=all(p["passed"] for p in phrase_results))

                elif scenario == "confirm_sequential":
                    token = await token_for(case["username"])
                    before = await _ticket_count(case["username"])
                    session_id, draft = await _make_draft(
                        client, token, case["username"], case["message"], case["id"]
                    )
                    if not draft:
                        record.update(passed=False, reason="未生成草稿")
                    else:
                        first_chat = await _send_chat(client, token, session_id, "确认创建")
                        # 再次通过聊天说确认：不应重复建单
                        await _send_chat(client, token, session_id, "确认创建")
                        api_first, ms1 = await _confirm(client, token, draft, draft["version"])
                        api_second, ms2 = await _confirm(client, token, draft, draft["version"])
                        confirm_latencies.extend([ms1, ms2])
                        after = await _ticket_count(case["username"])
                        first_no = (first_chat.get("ticket") or {}).get("ticket_no")
                        api_nos = [
                            (r.json().get("ticket") or {}).get("ticket_no")
                            for r in (api_first, api_second)
                            if r.status_code == 200
                        ]
                        record.update(
                            new_tickets=after - before,
                            chat_ticket_no=first_no,
                            api_status=[api_first.status_code, api_second.status_code],
                            api_ticket_nos=api_nos,
                            already_created=[
                                api_first.json().get("already_created") if api_first.status_code == 200 else None,
                                api_second.json().get("already_created") if api_second.status_code == 200 else None,
                            ],
                            passed=(
                                (after - before) == 1
                                and bool(first_no)
                                and len(set(api_nos)) == 1
                                and api_nos[0] == first_no
                            ),
                        )

                elif scenario in {"confirm_concurrent", "concurrent_confirm"}:
                    token = await token_for(case["username"])
                    before = await _ticket_count(case["username"])
                    _, draft = await _make_draft(
                        client, token, case["username"], case["message"], case["id"]
                    )
                    if not draft:
                        record.update(passed=False, reason="未生成草稿")
                    else:
                        (first, ms1), (second, ms2) = await asyncio.gather(
                            _confirm(client, token, draft, draft.get("version")),
                            _confirm(client, token, draft, draft.get("version")),
                        )
                        confirm_latencies.extend([ms1, ms2])
                        after = await _ticket_count(case["username"])
                        successful = [r.json() for r in (first, second) if r.status_code == 200]
                        nos = [(body.get("ticket") or {}).get("ticket_no") for body in successful]
                        flags = [body.get("already_created") for body in successful]
                        single_ticket = (
                            after - before == 1
                            and len(nos) == 2
                            and bool(nos[0])
                            and nos[0] == nos[1]
                            and sum(flag is True for flag in flags) == 1
                            and sum(flag is False for flag in flags) == 1
                        )
                        record.update(
                            status=[first.status_code, second.status_code],
                            ticket_nos=nos,
                            already_created=flags,
                            new_tickets=after - before,
                            single_ticket=single_ticket,
                            passed=single_ticket,
                            note="顺序重复见 T03；本用例是同一草稿的并发确认",
                        )

                elif scenario == "stale_version":
                    token = await token_for(case["username"])
                    before = await _ticket_count(case["username"])
                    _, draft = await _make_draft(
                        client, token, case["username"], case["message"], case["id"]
                    )
                    if not draft:
                        record.update(passed=False, reason="未生成草稿")
                    else:
                        old_version = draft["version"]
                        updated = await client.patch(
                            f"/api/tickets/drafts/{draft['id']}",
                            headers=auth(token),
                            json={
                                "description": draft["description"] + "\n评测补充：修改草稿内容",
                                "expected_version": old_version,
                            },
                        )
                        new_version = updated.json().get("version") if updated.status_code == 200 else None
                        stale_response, ms1 = await _confirm(client, token, draft, old_version)
                        fresh_response, ms2 = await _confirm(client, token, draft, new_version)
                        confirm_latencies.extend([ms1, ms2])
                        after = await _ticket_count(case["username"])
                        record.update(
                            old_version=old_version,
                            new_version=new_version,
                            stale_status=stale_response.status_code,
                            fresh_status=fresh_response.status_code,
                            new_tickets=after - before,
                            passed=(
                                stale_response.status_code == 409
                                and fresh_response.status_code == 200
                                and (after - before) == 1
                            ),
                        )

                elif scenario == "cancel_recreate":
                    token = await token_for(case["username"])
                    before = await _ticket_count(case["username"])
                    session_id, draft = await _make_draft(
                        client, token, case["username"], case["message"], case["id"]
                    )
                    if not draft:
                        record.update(passed=False, reason="未生成草稿")
                    else:
                        cancelled = await _send_chat(client, token, session_id, "取消")
                        # 在同一个会话里重新描述问题，验证取消后的残留状态不会直接复用
                        second_body = await _send_chat(client, token, session_id, case["message"])
                        second_draft = second_body.get("draft")
                        confirm_response, ms = (
                            await _confirm(client, token, second_draft, second_draft["version"])
                            if second_draft
                            else (None, 0)
                        )
                        if ms:
                            confirm_latencies.append(ms)
                        after = await _ticket_count(case["username"])
                        record.update(
                            cancel_intent=cancelled.get("intent"),
                            first_draft=draft["id"],
                            second_draft=(second_draft or {}).get("id"),
                            new_drafts_differ=bool(second_draft) and second_draft["id"] != draft["id"],
                            confirm_status=confirm_response.status_code if confirm_response else None,
                            new_tickets=after - before,
                            passed=(
                                bool(second_draft)
                                and second_draft["id"] != draft["id"]
                                and confirm_response is not None
                                and confirm_response.status_code == 200
                                and (after - before) == 1
                            ),
                        )

                elif scenario == "cross_user_draft":
                    owner_token = await token_for(case["owner"])
                    other_token = await token_for(case["other"])
                    _, draft = await _make_draft(client, owner_token, case["owner"], case["message"], case["id"])
                    if not draft:
                        record.update(passed=False, reason="未生成草稿")
                    else:
                        before_owner = await _ticket_count(case["owner"])
                        response, ms = await _confirm(client, other_token, draft, draft["version"])
                        confirm_latencies.append(ms)
                        after_owner = await _ticket_count(case["owner"])
                        record.update(
                            status=response.status_code,
                            status_text=response.text[:200],
                            tickets_created=after_owner - before_owner,
                            passed=response.status_code == 409 and after_owner == before_owner,
                        )

                elif scenario == "missing_draft":
                    token = await token_for(case["username"])
                    response, ms = await _confirm(client, token, {"id": "not-exist-draft"}, 1)
                    confirm_latencies.append(ms)
                    record.update(
                        status=response.status_code,
                        passed=response.status_code == 409 and "ticket" not in response.json(),
                    )

                elif scenario == "missing_version":
                    # 确认接口必须拒绝缺失版本，且不能创建工单
                    token = await token_for(case["username"])
                    before = await _ticket_count(case["username"])
                    _, draft = await _make_draft(
                        client, token, case["username"], case["message"], case["id"]
                    )
                    if not draft:
                        record.update(passed=False, reason="未生成草稿")
                    else:
                        started = time.perf_counter()
                        response = await client.post(
                            "/api/tickets/confirm",
                            headers=auth(token),
                            json={"draft_id": draft["id"]},
                        )
                        confirm_latencies.append((time.perf_counter() - started) * 1000)
                        after = await _ticket_count(case["username"])
                        record.update(
                            status=response.status_code,
                            status_text=response.text[:200],
                            new_tickets=after - before,
                            passed=response.status_code == 422 and after == before,
                        )

                elif scenario == "status_permission":
                    token = await token_for(case["username"])
                    _, draft = await _make_draft(
                        client, token, case["username"], "转人工。我用的是 v2.3，报销超预算追加单额度未生效。", case["id"]
                    )
                    if not draft:
                        record.update(passed=False, reason="未生成草稿")
                    else:
                        created, ms = await _confirm(client, token, draft, draft["version"])
                        confirm_latencies.append(ms)
                        ticket_id = created.json()["ticket"]["id"]
                        agent_token = await token_for("agent2")
                        close_pending = await client.patch(
                            f"/api/tickets/{ticket_id}/status",
                            headers=auth(token),
                            json={"status": "closed", "note": "客户从待处理直接关闭"},
                        )
                        await client.patch(
                            f"/api/tickets/{ticket_id}/status",
                            headers=auth(agent_token),
                            json={"status": "processing", "note": "客服接手"},
                        )
                        close_processing = await client.patch(
                            f"/api/tickets/{ticket_id}/status",
                            headers=auth(token),
                            json={"status": "closed", "note": "客户从处理中直接关闭"},
                        )
                        await client.patch(
                            f"/api/tickets/{ticket_id}/status",
                            headers=auth(agent_token),
                            json={"status": "waiting_customer", "note": "等待客户确认"},
                        )
                        close_allowed = await client.patch(
                            f"/api/tickets/{ticket_id}/status",
                            headers=auth(token),
                            json={"status": "closed", "note": "客户确认解决"},
                        )
                        record.update(
                            close_from_pending=close_pending.status_code,
                            close_from_processing=close_processing.status_code,
                            close_after_confirm=close_allowed.status_code,
                            passed=(
                                close_pending.status_code == 403
                                and close_processing.status_code == 403
                                and close_allowed.status_code == 200
                            ),
                        )

                elif scenario == "supplement_idempotent":
                    token = await token_for(case["username"])
                    listed = await client.get("/api/tickets", headers=auth(token))
                    items = listed.json().get("items") or []
                    if not items:
                        record.update(passed=False, reason="没有可补充的工单")
                    else:
                        ticket_id = items[0]["id"]
                        marker = f"评测补充标记-{case['id']}"
                        headers = {**auth(token), "Idempotency-Key": f"eval-{case['id']}-{ticket_id}"}
                        started = time.perf_counter()
                        await client.post(
                            f"/api/tickets/{ticket_id}/supplement",
                            headers=headers,
                            json={"description": marker},
                        )
                        await client.post(
                            f"/api/tickets/{ticket_id}/supplement",
                            headers=headers,
                            json={"description": marker},
                        )
                        elapsed = (time.perf_counter() - started) * 1000
                        detail = await client.get(f"/api/tickets/{ticket_id}", headers=auth(token))
                        description = detail.json().get("description", "")
                        record.update(
                            marker_count=description.count(marker),
                            tool_ms=round(elapsed, 1),
                            passed=description.count(marker) == 1,
                        )
                else:
                    record.update(passed=False, reason=f"未知场景 {scenario}")
            except Exception as exc:
                record.update(passed=False, error=f"{type(exc).__name__}: {exc}")

            records.append(record)

    scenario_cases = records
    negative_phrases = [
        phrase for record in records if record.get("scenario") == "negative_confirm"
        for phrase in record.get("phrases", [])
    ]
    concurrent_cases = [
        r for r in records if r.get("scenario") in {"confirm_concurrent", "concurrent_confirm"}
    ]
    repeat_cases = concurrent_cases + [r for r in records if r.get("scenario") == "confirm_sequential"]
    extra_tickets = sum(max((r.get("new_tickets") or 0) - 1, 0) for r in repeat_cases)

    metrics = [
        _metric(
            "tool.scenario_pass_rate",
            sum(1 for r in scenario_cases if r.get("passed")),
            len(scenario_cases),
            scope="工具套件全部场景；每个场景的断言见 case 明细",
            mode=mode,
        ),
        _metric(
            "tool.negative_confirm_no_write_rate",
            sum(1 for phrase in negative_phrases if phrase.get("passed")),
            len(negative_phrases),
            scope="「不要提交」「先不创建吧」「确认之前先改一下描述」逐条独立建草稿后发送；不得新增工单",
            mode=mode,
        ),
        _metric(
            "tool.duplicate_ticket_count",
            extra_tickets,
            len(repeat_cases),
            scope="顺序重复 + 并发确认场景中，超出 1 张的额外工单数（越低越好）",
            mode=mode,
            note="顺序重复与并发分别记录在 case 明细",
        ),
        _metric(
            "tool.stale_confirm_rejected",
            sum(
                1
                for r in records
                if r.get("scenario") == "stale_version" and r.get("stale_status") == 409
            ),
            sum(1 for r in records if r.get("scenario") == "stale_version"),
            scope="草稿修改后，旧版本确认必须返回 409",
            mode=mode,
        ),
        _metric(
            "tool.cross_user_rejected",
            sum(
                1
                for r in records
                if r.get("scenario") == "cross_user_draft" and r.get("status") == 409
            ),
            sum(1 for r in records if r.get("scenario") == "cross_user_draft"),
            scope="其他用户确认草稿必须被拒绝且不建单",
            mode=mode,
        ),
        _metric(
            "tool.concurrent_single_ticket",
            sum(1 for r in concurrent_cases if r.get("single_ticket")),
            len(concurrent_cases),
            scope="同一草稿并发确认恰好创建一张工单，两次请求返回同一编号且仅一次标记为幂等重放",
            mode=mode,
        ),
    ]
    return {
        "cases": records,
        "metrics": metrics,
        "latency": {"tool_confirm_ms": _latency(confirm_latencies)},
    }


# ── 权限评测 ─────────────────────────────────────────────────


async def run_permission_suite(cases: list[dict], env: dict) -> dict:
    mode = _mode_label(env)
    records: list[dict] = []
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        for case in cases:
            kind = case.get("kind", "chat_private_retrieval")
            token = await token_for(case["username"])
            record = {"id": case["id"], "suite": "permission", "tags": case.get("tags", []), "kind": kind}
            try:
                if kind == "chat_private_retrieval":
                    # 走真实问答链路：未授权资料不能出现在引用，也不能出现在回答正文
                    session_id = await _new_session(client, token, case["id"])
                    body = await _send_chat(client, token, session_id, case["query"])
                    citations = body.get("citations") or []
                    titles = [citation.get("title", "") for citation in citations]
                    leaked = _leaked_titles(titles, case.get("forbidden_titles"))
                    reply = body.get("reply") or ""
                    body_hits = [
                        keyword
                        for keyword in (case.get("forbidden_body_keywords") or [])
                        if keyword in reply
                    ]
                    record.update(
                        answer_mode=body.get("answer_mode"),
                        leaked_titles=leaked,
                        titles=titles,
                        forbidden_body_hits=body_hits,
                        answer_body=reply[:500],
                        passed=not leaked and not body_hits,
                    )
                elif kind == "ticket_cross_user":
                    owner_token = await token_for(case["owner"])
                    listed = await client.get("/api/tickets", headers=auth(owner_token))
                    ticket_id = listed.json()["items"][0]["id"]
                    response = await client.get(f"/api/tickets/{ticket_id}", headers=auth(token))
                    record.update(status=response.status_code, passed=response.status_code == 404)
                elif kind == "ticket_agent_not_granted":
                    owner_token = await token_for(case["owner"])
                    listed = await client.get("/api/tickets", headers=auth(owner_token))
                    ticket_id = listed.json()["items"][0]["id"]
                    response = await client.get(f"/api/tickets/{ticket_id}", headers=auth(token))
                    record.update(status=response.status_code, passed=response.status_code == 403)
                elif kind == "knowledge_delete_forbidden":
                    response = await client.delete(
                        "/api/knowledge/documents/not-exist", headers=auth(token)
                    )
                    record.update(status=response.status_code, passed=response.status_code == 403)
                elif kind == "admin_users_forbidden":
                    response = await client.get("/api/admin/users", headers=auth(token))
                    record.update(status=response.status_code, passed=response.status_code == 403)
                else:
                    record.update(passed=False, reason=f"未知权限场景 {kind}")
            except Exception as exc:
                record.update(passed=False, error=f"{type(exc).__name__}: {exc}")
            records.append(record)

    knowledge_records = [r for r in records if r.get("kind") == "chat_private_retrieval"]
    blocked_records = [r for r in records if r.get("kind") != "chat_private_retrieval"]
    metrics = [
        _metric(
            "permission.knowledge_leak_rate",
            sum(1 for r in knowledge_records if r.get("leaked_titles")),
            len(knowledge_records),
            scope="跨租户检索样本；前 5 条出现其他租户私有资料标题即泄漏（越低越好）",
            mode=mode,
        ),
        _metric(
            "permission.unauthorized_block_rate",
            sum(1 for r in blocked_records if r.get("passed")),
            len(blocked_records),
            scope="越权读取工单 / 未授权客服 / 客户管理知识库或用户等接口；必须返回 403 或 404",
            mode=mode,
        ),
    ]
    return {"cases": records, "metrics": metrics, "latency": {}}


# ── 运行与报告 ────────────────────────────────────────────────


def _merge_metrics(*suites: dict) -> list[dict]:
    metrics: list[dict] = []
    for suite in suites:
        metrics.extend(suite.get("metrics") or [])
    return metrics


def write_markdown(report: dict, path: Path) -> None:
    env = report["environment"]
    summary = report["summary"]
    lines = [
        "# 评测报告（自动生成）",
        "",
        f"> 生成时间：{report['started_at']}　运行模式：{report['mode']}",
        "> 所有数字来自本次实际运行；未运行的组合不会写入报告。",
        "",
        "## 一、实测环境",
        "",
        "| 项目 | 实际值 |",
        "| --- | --- |",
        f"| 数据库 | {env['database']['dialect']} / {env['database']['driver']}（{env['database']['scope']}） |",
        f"| Embedding 后端 | {env['embedding'].get('backend')}（{env['embedding'].get('mode', '')}，dim={env['embedding'].get('dim')}） |",
        f"| 检索后端 | {env['retrieval_backend']} |",
        f"| 重排 | enabled={env['rerank']['enabled']}，model_configured={env['rerank']['model_configured']} |",
        f"| 回答模式 | {env['answer_mode']} |",
        f"| 模型 | {env['llm'].get('model')}（configured={env['llm'].get('configured')}） |",
        f"| 会话历史上限 | {env['chat_history_limit']} 条 |",
        "",
        "## 二、指标汇总",
        "",
        "| 指标 | 数值 | 分子 / 分母 | 样本范围 | 运行模式 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for metric in report["metrics"]:
        value = "N/A" if metric["value"] is None else f"{metric['value']:.4f}"
        lines.append(
            f"| {metric['name']} | {value} | {metric['numerator']} / {metric['denominator']} | "
            f"{metric['scope']} | {metric['mode']} |"
        )

    lines += ["", "## 三、延迟", ""]
    for suite_name, latency in report["latency"].items():
        for key, value in latency.items():
            if isinstance(value, dict):
                note = value.get("note", "")
                lines.append(
                    f"- {suite_name}.{key}：p50={value.get('p50_ms')} ms，p95={value.get('p95_ms')} ms，"
                    f"平均={value.get('avg_ms')} ms，样本={value.get('samples')}"
                    + (f"（{note}）" if note else "")
                )
            else:
                lines.append(f"- {suite_name}.{key}：{value}")

    lines += ["", "## 四、各套件结果", ""]
    for suite_name, suite_summary in summary["suites"].items():
        lines.append(
            f"- **{suite_name}**：{suite_summary['passed']} / {suite_summary['total']} 通过"
        )

    lines += ["", "## 五、失败用例", ""]
    failed = [case for case in report["cases"] if not case.get("passed")]
    if failed:
        for case in failed:
            detail = {
                key: value
                for key, value in case.items()
                if key not in ("answer_body", "turns", "top5", "citations")
            }
            lines.append(f"- **{case['id']}**（{case['suite']}）：{json.dumps(detail, ensure_ascii=False)}")
    else:
        lines.append("本次运行没有失败用例。")

    lines += [
        "",
        "## 六、口径与限制",
        "",
        "- 检索指标直接调用 `search_knowledge`，不经过模型；文档级与分块级命中分别统计。",
        "- 回答指标通过聊天接口执行；关键词只检查回答正文，引用片段不能替代正文正确性。",
        "- 工具指标通过真实 HTTP 接口执行，顺序重复与并发确认分别记录；本报告在 SQLite 上运行，",
        "  PostgreSQL 15.20 的真实并发验证在独立测试 schema 中完成（见 docs/testing.md），不属于本报告。",
        "- 离线模式使用摘录式回答，first_token_ms 为 None，不能代表真实模型的首 token 延迟。",
        "- 如需真实模型评测，使用 `--live-llm --allow-paid-llm`，并自行承担调用费用；",
        "  未获得授权时不会发起真实模型请求。",
        "- 本报告不宣称已解决幻觉：引用合法只代表编号对应真实片段，不代表结论一定被资料支持。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


async def run(args) -> dict:
    from app.db import init_db
    from scripts.seed_demo import main as seed_main

    await init_db()
    await seed_main()

    cases_all = json.loads(CASES_PATH.read_text(encoding="utf-8"))["suites"]
    env = await _environment()
    suites: dict[str, dict] = {}
    selected = args.suite

    if selected in ("all", "retrieval"):
        suites["retrieval"] = await run_retrieval_suite(cases_all["retrieval"], env)
    if selected in ("all", "answer"):
        suites["answer"] = await run_answer_suite(cases_all["answer"], env, args.limit)
    if selected in ("all", "tool"):
        suites["tool"] = await run_tool_suite(cases_all["tool"], env)
    if selected in ("all", "permission"):
        suites["permission"] = await run_permission_suite(cases_all["permission"], env)

    records = [case for suite in suites.values() for case in suite["cases"]]
    metrics = _merge_metrics(*suites.values())
    latency = {name: suite.get("latency", {}) for name, suite in suites.items()}
    report = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": _mode_label(env),
        "live_llm": LIVE_LLM,
        "dataset": str(CASES_PATH.relative_to(PROJECT_ROOT)),
        "environment": env,
        "metrics": metrics,
        "summary": {
            "suites": {
                name: {
                    "total": len(suite["cases"]),
                    "passed": sum(1 for case in suite["cases"] if case.get("passed")),
                }
                for name, suite in suites.items()
            },
            "total": len(records),
            "passed": sum(1 for case in records if case.get("passed")),
        },
        "latency": latency,
        "cases": records,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    name = args.output_name
    (REPORT_DIR / f"{name}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(report, REPORT_DIR / f"{name}.md")
    return report


async def run_and_cleanup(args) -> dict:
    from app.config import get_settings
    from app.db import dispose_db

    settings = get_settings()
    client = None
    try:
        if settings.retrieval_backend == "milvus":
            from pymilvus import MilvusClient

            client = MilvusClient(uri=settings.milvus_uri, timeout=settings.milvus_timeout_seconds)
        return await run(args)
    finally:
        await dispose_db()
        if client is not None:
            try:
                if client.has_collection(settings.milvus_collection):
                    client.drop_collection(settings.milvus_collection)
            finally:
                client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="cloudledger-support 可信评测")
    parser.add_argument(
        "--suite",
        choices=["all", "retrieval", "answer", "tool", "permission"],
        default="all",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["hash", "bge_m3"],
        default=None,
        help="覆盖 EMBEDDING_BACKEND（如 bge_m3）",
    )
    parser.add_argument("--limit", type=int, default=None, help="限制回答套件样本数（真实模型调用时用于控费）")
    parser.add_argument("--live-llm", action="store_true", help="启用真实模型调用")
    parser.add_argument(
        "--allow-paid-llm",
        action="store_true",
        help="确认允许真实模型调用产生费用（必须与 --live-llm 同时使用）",
    )
    parser.add_argument("--output-name", default="evaluation")
    parser.add_argument("--strict", action="store_true", help="任一用例失败或未运行用例时以非零状态退出")
    args = parser.parse_args()

    if args.live_llm and not args.allow_paid_llm:
        print("[eval] 已拒绝：真实模型调用可能产生费用，请同时传入 --allow-paid-llm")
        return 2

    from app.config import get_settings

    settings = get_settings()
    if args.live_llm:
        if not settings.llm_available:
            print("[eval] 已拒绝：未配置 LLM_API_KEY，无法进行真实模型评测")
            return 2
        sample_desc = f"最多 {args.limit} 条" if args.limit is not None else "全部"
        print(
            f"[eval] 真实模型模式：{settings.llm_model}，{sample_desc}回答样本，"
            "每条最多 2 次调用（含一次引用重试）。请确认预算与授权。"
        )

    try:
        report = asyncio.run(run_and_cleanup(args))
    except KeyboardInterrupt:
        print("[eval] 已中断")
        return 130

    print(
        f"[eval] 完成：{report['summary']['passed']} / {report['summary']['total']} 用例通过"
    )
    for metric in report["metrics"]:
        value = "N/A" if metric["value"] is None else f"{metric['value']:.4f}"
        print(
            f"[eval] {metric['name']} = {value} "
            f"（{metric['numerator']}/{metric['denominator']}）"
        )
    print(f"[eval] 报告：{REPORT_DIR / (args.output_name + '.json')}")
    summary = report["summary"]
    if args.strict and (not summary["total"] or summary["passed"] != summary["total"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
