# CloudLedger · 企业财税费控与数电发票受控工单系统

面向企业财务共享与费控 SaaS 场景的多租户智能支持与受控工单工作流系统。围绕**全面数字化的电子发票（数电发票）入池验真、部门与专项预算控制、差旅报销审批反审核、银企直连打款退票及月度关账异常**五类高频财务内控场景，串联**多租户权限隔离问答**与**人机协同（Human-in-the-Loop）受控建单闭环**。

- **业务背景与痛点**：企业财务制度具有强租户私密性（各集团差旅标准、核算中心编码、月结关账日互不相同），同时数电票冲销、反结账、预算占用清理等工单操作对误触发零容忍——既不能跨租户泄露内控制度，也不能让大模型未经申请人核对就直接写库建单。
- **受控工单工作流（LangGraph）**：规则路由负责 `confirm / cancel / edit_draft` 等确定性写指令，大模型仅限在枚举白名单内建议咨询意图与草稿分类；确认建单强制绑定用户所见的 `expected_version`，通过数据库条件更新（`UPDATE ... WHERE status='draft' AND version=:expected`）、两层唯一约束与 `ticket_sequences` 原子取号防御并发重复提交与旧版本覆盖。
- **多租户权限与版本感知检索**：支持公共财税规范与租户私有内控制度隔离检索，支持本地 BGE-M3（1024 维稠密 + 稀疏词权重）+ Milvus 混合召回后回业务库二次复核权限与产品版本（`v2.2` / `v2.3`），并提供零外部依赖的离线回归模式。

[架构与核心流程](docs/architecture.md) · [数据模型与状态机](docs/data-model.md) · [API 说明](docs/api.md) · [测试与评测](docs/testing.md) · [工程边界与限制](docs/limitations.md)

---

## 1. 核心工程设计

### 1.1 多租户 RBAC 与即时权限吊销
- **三层角色模型**：企业员工（`customer`）/ 授权财务顾问（`agent`）/ 平台管理员（`admin`），基于 `tenants`、`users`、`agent_tenant_grants` 实现租户隔离与跨企业顾问按需授权。
- **无状态 Token + 逐请求查库复核**：JWT 仅携带 `sub`（用户 ID）与 `exp`，不在载荷中固化角色与租户；每次请求回库校验 `is_active` 与最新授权关系，员工离职禁用或撤销顾问授权后**下一秒请求立即拦截**。
- **全链路统一可见性裁剪**：知识库检索（公共规范 vs 租户私有内控制度）、会话历史、工单列表/详情/回复/附件统一复用 `permissions.py` 服务端断言。

### 1.2 受控工单 Agent 与确认安全协议
- **非自主写库 Agent**：LangGraph 单图编排 `load_context → classify → 确定性业务节点 → finalize`。模型结构化输出枚举中**刻意排除 `confirm / cancel / edit_draft`**，杜绝提示词注入或语义误判直接触发建单。
- **严格确认白名单**：存在待确认草稿时，取消、疑问、修改意图优先级高于确认（例如「不要提交」「先不创建吧」「确认之前先改一下描述」均不会触发建单），仅接受明确确认指令或前端确认按钮。
- **乐观锁版本绑定（`expected_version`）**：草稿每次修改版本号递增；确认接口在 Pydantic 层与服务层双重强校验严格正整数 `expected_version`，缺失/非法返回 `422`，落后于当前版本返回 `409`，服务端绝不读取最新版本代填确认。

### 1.3 两层幂等、Savepoint 事务恢复与并发控制
- **两层唯一约束**：`ticket_drafts.idempotency_key` 唯一防止同会话重复生成草稿（取消后重发自动派生 `:rN` 后缀）；`tickets.draft_id` 唯一保证同一草稿最多生成一张正式工单。
- **原子工单取号**：通过 `ticket_sequences` 表的行级 `INSERT ... ON CONFLICT DO UPDATE ... RETURNING` 分配 `YZ+年月+序号` 单号，避免 `MAX+1` 并发撞号。
- **Savepoint 冲突恢复**：在 `db.begin_nested()` 内部执行 `add + flush`，捕获并发唯一键冲突后安全回查已创建的工单并标记 `already_created=true`，不破坏外层事务。
- **会话与消息幂等**：普通问答、SSE 流式问答、工单回复与补充均支持 `Idempotency-Key` 重放保护。

### 1.4 知识检索、引用校验与真实流式输出
- **双后端抽象**：`bge_m3`（1024 维 dense + sparse 混合检索，支持业务库 `local` 或 `milvus` 召回后回查复核并统一重算融合分）与 `hash`（512 维哈希 + jieba/IDF 词法覆盖，专用于无 GPU/模型权重环境的快速离线回归与 CI）。
- **可解释错误出口**：明确区分 `no_evidence`（证据不足拒答）、`no_version_match`（指定版本无资料）、`index_missing`（向量索引未构建）与 `retrieval_error`（检索服务异常），不将后端故障伪装成“没有相关制度”。
- **引用编号校验与降级**：校验回答中的 `[n]` 是否落在本次检索片段范围内；无有效引用时自动提示重试一次，仍不合规或调用超时则退回资料摘录并显式标注 `llm_uncited` / `llm_degraded`。
- **真实 SSE 流式输出**：`LLMClient.stream` 消费 OpenAI 兼容接口（默认对接 DeepSeek `deepseek-chat`，自动识别 `LLM_API_KEY` 或系统环境变量 `DEEPSEEK_API_KEY`）的 `astream` 逐块推送 `token` 事件；客户端断连立即中止生成，未完成文本标记 `stream_status=partial` 且不挂载引用列表。

### 1.5 只读 MCP 工具接入层（JSON-RPC 2.0）
- **最小权限原则**：在 `POST /mcp` 暴露标准 MCP（`initialize` / `tools/list` / `tools/call`）只读工具：`search_tenant_docs`（租户知识库检索）、`list_my_tickets`（工单列表）、`get_ticket_detail`（工单详情）与 `web_search`（公开财税政策联网搜索，支持 `WEB_SEARCH_MCP_URL` 转发或 `TAVILY_API_KEY` / `WEB_SEARCH_API_KEY` 接入）。
- **与受控写库边界一致**：所有 `/mcp` 调用强制校验 Bearer JWT 并回库核验账号状态，忽略参数中传入的伪造 `tenant_id`；**刻意不向 MCP 开放任何建单/修改/删除写库工具**，写操作仍必须经由 LangGraph 人机确认与乐观锁校验。
- **可解释联网降级**：未配置联网搜索源或外部服务超时异常时，显式返回 `unconfigured` / `error` 状态与空列表，绝不伪造外部搜索结果，也不向外网传输任何租户私有文档。

---

## 2. 技术栈与目录结构

| 层级 | 技术选型 |
| --- | --- |
| Web 与异步框架 | FastAPI 0.117 + Uvicorn |
| ORM 与数据库 | SQLAlchemy 2.0 Async + SQLite（`aiosqlite`，默认开箱即用）/ PostgreSQL 15（`asyncpg`） |
| 工作流与工具协议 | LangGraph 1.0（单图状态机、确定性条件路由）+ MCP 只读接入层（JSON-RPC 2.0 `tools/list` / `tools/call` + 联网搜索扩展） |
| 大模型接入 | `langchain-openai`（OpenAI 兼容协议，支持 DeepSeek API 真流式与 JSON 结构化建议） |
| 检索与重排 | FlagEmbedding `BGE-M3`、Milvus（`pymilvus` `WeightedRanker`）、可选 `CrossEncoder` 重排、`jieba` |
| 认证与安全 | `python-jose`（HS256 JWT）+ `passlib/bcrypt`（线程池异步校验） |
| 前端工作台 | 原生 HTML5 / CSS3 / JavaScript（登录、SSE 问答与草稿卡、客户工单、顾问工作台、管理后台） |
| 测试与 CI | `pytest` + `pytest-asyncio` + `httpx.ASGITransport`、GitHub Actions |

```text
cloudledger-support/
├── app/
│   ├── main.py                  # FastAPI 入口、request_id 中间件、异常与静态路由
│   ├── config.py                # 集中配置中心（支持 .env 与 DEEPSEEK_API_KEY 自动回退）
│   ├── db.py / migrations.py    # 异步引擎与保留数据的轻量版本化迁移（schema_migrations）
│   ├── models.py                # 14 张业务与向量数据表定义
│   ├── security.py / deps.py    # bcrypt、JWT 与逐请求数据库身份复核
│   ├── agent/                   # LangGraph 状态、节点、规则意图与 Pydantic 白名单结构化建议
│   ├── api/                     # auth / chat / tickets / knowledge / admin 路由
│   ├── mcp/                     # 只读型 MCP Server（POST /mcp）与异步 MCP Client
│   └── services/                # 权限、知识切分同步、BGE-M3/Milvus 检索、问答流式、工单事务、联网搜索
├── demo_docs/                   # 13 份财税费控与数电发票规范（覆盖 v2.2 / v2.3 / 租户私有内控制度）
├── frontend/                    # 5 个角色视图页面与公共交互脚本
├── scripts/                     # 数据库初始化、演示种子、评测脚本、向量重建与语义后端自检
├── eval/test_cases.json         # 40 条独立评测集（检索 12 / 回答 10 / 工具 12 / 权限 6）
├── tests/                       # 87 项自动化回归测试（覆盖鉴权、MCP 只读/隔离、乐观锁竞争、幂等、SSE 与 PG 事务）
└── docs/                        # 架构、数据模型、API、测试报告与工程边界说明
```

---

## 3. 快速启动

环境要求：Python 3.11+。

```powershell
cd cloudledger-support
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# 初始化 SQLite 表结构并导入财税费控演示数据（已有数据默认保留）
python scripts/init_db.py --demo

# 启动服务（若系统已配置 DEEPSEEK_API_KEY 或在 .env 配置 LLM_API_KEY，自动启用大模型流式回答）
python run.py
```

默认访问地址：<http://127.0.0.1:8000/login>（若 `.env` 配置了 `APP_PORT`，以终端输出端口为准；健康检查 `/health`，接口文档 `/docs`）。

### 本地演示账号

| 角色 | 账号 | 密码 | 所属/授权租户 | 权限范围说明 |
| --- | --- | --- | --- | --- |
| 管理员 | `admin` | `Admin@123456` | 云账科技（平台方） | 知识库维护、用户启停、顾问授权、全量工单 |
| 财务顾问 | `agent1` | `Agent@123456` | 平台方（授权：华辰智造） | 仅可处理华辰智造工单及检索其私有费控内控制度 |
| 财务顾问 | `agent2` | `Agent@123456` | 平台方（授权：星海文化） | 仅可处理星海文化工单及检索其私有报销管理细则 |
| 企业会计 | `customer_a` | `Customer@123456` | 华辰智造 | 主演示账号，可查公共规范 + 华辰私有制度、提交本人单据工单 |
| 企业出纳 | `customer_a2` | `Customer@123456` | 华辰智造 | 验证同租户不同员工之间的会话与工单隔离 |
| 企业财务 | `customer_b` | `Customer@123456` | 星海文化 | 验证跨租户知识库与工单强隔离 |
| 离职账号 | `customer_disabled` | `Disabled@123456` | 华辰智造 | 已禁用账号，登录及历史 Token 请求均被即时拦截 |

> 以上账号与口令仅用于本地功能演示，严禁用于任何生产环境。

### 3 分钟核心链路验证路径
1. 使用 `customer_a` 登录，提问「`v2.2 报销单保存草稿后为什么部门可用预算被冻结？`」，展开来源卡片核对版本依据；
2. 追问「`那旧版本呢`」或切换 `v2.3`，观察多轮指代改写与跨版本规则差异提示；
3. 发送「`转人工。v2.3 报销超预算生成追加单后额度仍未生效，已刷新页面`」，系统生成待确认草稿但**不写工单表**；
4. 发送「`不要提交`」验证否定拦截，或修改草稿描述使版本号递增后点击「确认并创建工单」，验证 `expected_version` 乐观锁绑定与 `YZ` 工单号生成；
5. 切换 `agent1` 登录顾问工作台（仅见华辰智造工单，不可见星海文化工单），推进状态至「待客户确认」并回复处理方案；
6. 切回 `customer_a` 在「我的工单」确认已解决，查看完整状态机流转留痕。

---

## 4. 可选：启用 BGE-M3 + Milvus 混合检索

```powershell
$env:EMBEDDING_BACKEND="bge_m3"
$env:BGE_M3_MODEL_PATH="<本地 BGE-M3 权重目录>"
$env:RETRIEVAL_BACKEND="milvus"
$env:MILVUS_URI="http://127.0.0.1:19530"

python scripts/verify_semantic_backend.py
python scripts/reindex_embeddings.py --backend bge_m3 --index milvus
python run.py
```

- 切换 Embedding 后端后需执行 `reindex_embeddings.py` 重建对应后端向量；
- 真实语义后端或 Milvus 不可用时显式返回错误状态，**绝不静默降级为哈希模式冒充语义检索**。

---

## 5. 自动化测试与可复现评测

### 5.1 回归测试（pytest）

```powershell
.\.venv\Scripts\python.exe -m pytest -q --disable-warnings
```

- **SQLite 默认回归（2026-11-26 实测）**：共收集 **87 项测试，85 passed，2 skipped**（跳过项为需显式提供 `TEST_POSTGRES_URL` 的 2 项真实 PostgreSQL 并发/迁移测试；此前已在本地 PostgreSQL 15.20 隔离 schema 实测通过）。
- 覆盖范围：JWT 即时吊销、跨租户/跨用户越权拦截、否定/疑问确认防误触、`expected_version` 422/409 矩阵、编辑/取消/确认并发竞争、Savepoint 唯一冲突恢复、知识更新删除同步、引用越界剔除及 SSE 中途断流 `partial` 保存。

### 5.2 四套件评测（40 条独立样本）

```powershell
# 离线零成本回归（摘录回答）
.\.venv\Scripts\python.exe scripts/run_eval.py --suite all --output-name evaluation-hash-local

# 真实大模型 + SSE 流式评测（需配置 LLM_API_KEY 或 DEEPSEEK_API_KEY）
.\.venv\Scripts\python.exe scripts/run_eval.py --suite answer --live-llm --allow-paid-llm --output-name evaluation-live-deepseek
```

- **离线全套件回归（2026-11-24）**：40/40 通过，其中受控工单工具场景 **12/12**（否定短语零误建单、顺序重复与并发确认零重复工单、旧版本确认 409 拦截）、多租户权限拦截 **6/6**（私有内控制度零跨租户泄露）。
- **DeepSeek 真实生成与 SSE 流式验证（2026-11-24，`deepseek-chat`）**：验证真实模型问答、多轮指代、无证据拒答、不存在版本 `v9.9` 拦截及真实 `astream` 逐块流式转发，流式样本实测首 Token 延迟（TTFT）`p50 = 233.5 ms`（均值 `356.6 ms`），回答正文违禁短语率 `0/10`。
- GitHub Actions 工作流（[.github/workflows/checks.yml](.github/workflows/checks.yml)）在每次推送时自动执行 pytest 与工单/权限严格门禁检查。

---

## 6. 工程边界说明

- 本项目为围绕企业财税费控与数电发票真实业务规则构建的本地工程实践系统，内置租户（华辰智造、星海文化）、账号与单据均为本地构造数据，未对接真实税务数字账户专线或银行生产前置机。
- 引用校验负责保证 `[n]` 落在当次检索证据编号内并在违规时降级，不等同于句子级自然语言推理（NLI）事实核验。
- PostgreSQL 与 Milvus 已完成单机真实功能、隔离与并发竞争验证，尚未经过生产级高并发压测与多节点故障切换演练。详见 [已知限制](docs/limitations.md)。
