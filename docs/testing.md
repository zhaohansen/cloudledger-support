# 测试与评测说明

2026-11-24 在 Windows + Python 3.11 环境下完成最新回归验证：
- **自动化测试（pytest）**：共收集 **80 项**，默认 SQLite 模式 **78 passed，2 skipped**（跳过项为需显式传入 `TEST_POSTGRES_URL` 的 2 项真实 PostgreSQL 迁移与并发事务测试）。
- **四套件离线回归评测**：40 条财税费控与数电发票独立样例 **40/40 通过**（其中受控工单工具套件 12/12、多租户权限套件 6/6）。
- **DeepSeek 真实模型与 SSE 流式验证**：通过系统 `DEEPSEEK_API_KEY` 对接 `deepseek-chat` 实测回答套件（含多轮指代、无证据拒答、不存在版本拦截及 2 条真实 SSE 流式回答），流式首 Token 延迟（TTFT）实测 `p50 = 233.5 ms`（`avg = 356.6 ms`）。

---

## 1. 运行方式

```powershell
cd cloudledger-support
.\.venv\Scripts\Activate.ps1

# 1. 自动化回归测试（使用独立临时 SQLite，不污染 data/app.db）
python -m pytest -q --disable-warnings

# 2. 接入本地 PostgreSQL 隔离 schema 跑全量事务与并发竞争测试（测试结束自动清理 schema）
$env:TEST_POSTGRES_URL="postgresql+asyncpg://<user>:<password>@127.0.0.1:5433/eduagent"
python -m pytest -q --disable-warnings

# 3. 离线四套件评测（摘录模式，零模型调用成本）
python scripts/run_eval.py --suite all --output-name evaluation-hash-local
python scripts/run_eval.py --suite tool --strict --output-name evaluation-tool
python scripts/run_eval.py --suite permission --strict --output-name evaluation-permission

# 4. 真实大模型与流式问答评测（需配置 LLM_API_KEY 或 DEEPSEEK_API_KEY）
python scripts/run_eval.py --suite answer --live-llm --allow-paid-llm --output-name evaluation-live-deepseek
```

---

## 2. 自动化测试矩阵（80 项）

| 测试文件 | 覆盖重点 |
| --- | --- |
| [tests/test_auth_permissions.py](../tests/test_auth_permissions.py) | 登录正误、伪造/过期 Token、离职禁用账号即时拦截、管理员接口鉴权、同租户跨员工与跨租户工单隔离、财务顾问授权范围 |
| [tests/test_confirm_safety.py](../tests/test_confirm_safety.py) | 明确确认、明确取消、否定句/疑问句拦截、编辑草稿、旧版本确认拦截、重复点击幂等、已取消草稿拒绝、跨用户确认拒绝 |
| [tests/test_confirm_version_and_races.py](../tests/test_confirm_version_and_races.py) | 缺版本/null/0/负数/字符串/小数的 `422` 校验矩阵；服务层兜底拒绝；编辑 vs 确认、取消 vs 确认、双编辑并发竞争；缺 `seen_version` 禁止代填确认；事务异常回滚 |
| [tests/test_tickets.py](../tests/test_tickets.py) | 未确认不落库、重复确认单工单保证、工具异常返回 409、角色状态机合法/非法流转 |
| [tests/test_knowledge.py](../tests/test_knowledge.py) | 华辰智造与星海文化私有制度隔离、文档更新/删除后旧分块失效、Milvus 召回候选回业务库复核与元数据失败回滚 |
| [tests/test_chat.py](../tests/test_chat.py) | 无证据拒答不编造、引用连续且版本受控、跨用户与租户变更会话隔离、普通与流式消息 `Idempotency-Key` 幂等重放 |
| [tests/test_qa_stream.py](../tests/test_qa_stream.py) | 合法引用保留、编造越界引用 `[99]` 剔除降级、模型超时降级、SSE `astream` 逐块转发、流式中途异常 `partial` 保存、工单消息/补充幂等 |
| [tests/test_multi_turn.py](../tests/test_multi_turn.py) | 「那旧版本呢」去旧版本号改写、「第二个步骤」指代补全、切换话题（如「差旅报销单怎么反审核？」）不串上下文、历史仅作指代不作事实依据 |
| [tests/test_structured_agent.py](../tests/test_structured_agent.py) | 模型输出 `confirm/cancel` 被枚举白名单拒绝、低置信度忽略、非法分类枚举拒绝、存在待确认草稿时禁止模型改写意图 |
| [tests/test_migrations.py](../tests/test_migrations.py) | 旧表结构缺列/缺表时执行 `schema_migrations` 平滑升级并保留历史数据 |
| [tests/test_postgres_readiness.py](../tests/test_postgres_readiness.py) | PostgreSQL DDL/唯一约束、asyncpg 引擎、隔离 schema 真实迁移与 Savepoint 回滚、真实并发确认与编辑/取消竞争 |

---

## 3. 独立评测套件设计（40 条）

样例定义见 [eval/test_cases.json](../eval/test_cases.json)，按四类能力独立统计分子与分母：

1. **retrieval（12 条）**：覆盖同义改写、口语问法、`v2.2/v2.3` 预算冻结规则冲突检测、指定版本过滤、无关技术问题空结果校验，以及华辰智造 / 星海文化跨租户私有制度零泄露校验（`forbidden_leak_rate = 0/2`）。
2. **answer（10 条）**：检查回答正文关键词（不靠引用片段充数）、引用编号从 1 连续合法、无证据返回 `no_evidence`（`2/2`）、不存在版本 `v9.9` 返回 `no_version_match`（`1/1`）、多轮指代追问（`1/1`）及 SSE 流式输出（A09、A10）。
3. **tool（12 条）**：通过真实 ASGI HTTP 接口检验草稿仅暂存不建单、3 类否定/修改短语零误建单、顺序重复点击与双请求并发确认仅建 1 张工单且返回一致 `already_created` 标记、草稿修改后旧版本确认返回 `409`、缺 `expected_version` 返回 `422`、取消后重建 `:r2` 草稿、跨用户确认拦截与补充幂等。
4. **permission（6 条）**：检验聊天接口跨租户私有制度零泄露（`0/2`），以及跨企业读取工单、未获授权财务顾问读取工单、普通客户删除知识库或访问管理接口的拦截率（`4/4`）。

---

## 4. 关键专项验证记录

- **DeepSeek 真实模型与流式实测**：在配置 `DEEPSEEK_API_KEY`（模型 `deepseek-chat`）下运行回答评测，有依据样本均按检索证据分步作答并附合法 `[n]` 引用；无依据与 `v9.9` 样本在检索阶段即短路拒答（不消耗额外生成 Token、不产生幻觉）；SSE 流式接口真实消费 `astream` 输出，实测首 Token 延迟 `p50 = 233.5 ms`、完整回答耗时 `p50 = 1410.3 ms`。
- **PostgreSQL 15.20 隔离 Schema 并发验证**：已在本地 PostgreSQL 15.20（端口 5433）独立 schema 中真实验证重复迁移、外键与唯一约束、Savepoint 内唯一冲突回查、`ticket_sequences` 并发取号不撞号，以及同草稿并发确认/编辑/取消竞争的原子裁决。
- **Milvus + BGE-M3 混合检索验证**：已在独立临时 Collection 验证 1024 维稠密 + 稀疏向量写入、`access_scope + tenant_id + product_version` 表达式预筛、召回后回业务库二次权限复核与重算融合分，以及文档删除后的向量同步清理。

> 口径边界：40 条评测样本属于固定规则与场景回归，用于防止权限越界、确认误触和检索退化，不等同于真实线上流量的业务准确率或生产容量压测。
