# 架构与核心流程

> 本文说明系统分层、关键请求链路、Agent 图和检索流程，以及主要设计取舍。
> 配套阅读：[数据模型与权限](data-model.md)、[API](api.md)、[测试与评测](testing.md)。

## 1. 总体架构

```text
浏览器（原生 HTML/CSS/JS）
  ├── /login     登录
  ├── /chat      客户问答（SSE + 引用 + 工单草稿卡）
  ├── /tickets   客户工单
  ├── /agent     客服工作台
  └── /admin     管理员后台
        │  fetch + Bearer Token
        ▼
FastAPI（app/main.py）
  ├── request_id 中间件 / 统一异常
  ├── /api/auth      登录、当前用户
  ├── /api/chat      会话、消息、SSE
  ├── /api/tickets   草稿、确认、工单、回复、状态、附件
  ├── /api/knowledge 文档上传/更新/删除/检索（admin）
  └── /api/admin     租户、用户、客服授权（admin）
        │
        ├── 服务层 app/services/
        │     ├── permissions.py  统一权限判断
        │     ├── knowledge.py    解析、切分、入库、索引同步
        │     ├── embeddings.py   hash（演示）/ BGE-M3（真实语义）
        │     ├── vector_index.py 业务库向量存取
        │     ├── milvus_index.py Milvus 混合检索适配器
        │     ├── reranker.py     可选 CrossEncoder 精排
        │     ├── retrieval.py    权限/版本过滤 + 证据门槛
        │     ├── qa.py           回答生成、引用校验、流式输出
        │     ├── llm.py          模型客户端（超时/重试/结构化）
        │     ├── tickets.py      工单状态机、幂等、事务、取号
        │     └── tools.py        Agent 可调用工具
        │
        ├── LangGraph（app/agent/）
        │     load_context → classify → 确定性分支
        │       answer / handle_issue / handle_human / confirm / cancel /
        │       edit_draft / query_tickets / ticket_detail / supplement / clarify
        │
        └── SQLAlchemy async → SQLite（默认）/ PostgreSQL（15.20 隔离 schema 真实集成验证）
```

设计原则：

- **单体优先**：一个 FastAPI 进程 + 一个数据库，降低本地运行和讲解成本。
- **服务端为准**：角色、租户、工单可见性全部在后端判断；前端只做展示。
- **先过滤后检索**：知识检索先在 SQL 层按访问范围与版本过滤，再打分。
- **写操作可追溯**：工单状态变化写 `ticket_events`，回复写 `ticket_messages`。

## 2. 目录与职责

| 路径 | 职责 |
| --- | --- |
| [app/main.py](../app/main.py) | 应用生命周期、中间件、异常处理、健康检查、静态页面 |
| [app/config.py](../app/config.py) | 集中配置，读取 `.env`，提供默认值 |
| [app/db.py](../app/db.py) | 异步引擎；SQLite 用 NullPool，PostgreSQL 用连接池 |
| [app/models.py](../app/models.py) | 14 张表（含向量表、工单序号表） |
| [app/migrations.py](../app/migrations.py) | 轻量版本化迁移，保留已有数据 |
| [app/security.py](../app/security.py) | bcrypt 密码哈希、JWT 签发/解析 |
| [app/deps.py](../app/deps.py) | 从 Bearer Token 恢复 `UserContext`；角色依赖 |
| [app/logging_config.py](../app/logging_config.py) | request_id contextvar、日志过滤器 |
| [app/agent/graph.py](../app/agent/graph.py) | LangGraph 单图与确定性路由 |
| [app/agent/nodes.py](../app/agent/nodes.py) | 各节点逻辑 |
| [app/agent/intent.py](../app/agent/intent.py) | 诉求分类与版本/信息抽取 |
| [app/agent/structured.py](../app/agent/structured.py) | 模型意图/字段建议，Pydantic 白名单校验 |
| [app/services/permissions.py](../app/services/permissions.py) | 客服授权、租户范围、工单可见性 |
| [app/services/knowledge.py](../app/services/knowledge.py) | 文档解析、切分、入库、索引同步 |
| [app/services/embeddings.py](../app/services/embeddings.py) | hash / BGE-M3 后端抽象与健康状态 |
| [app/services/vector_index.py](../app/services/vector_index.py) | 业务库向量存取与统计 |
| [app/services/milvus_index.py](../app/services/milvus_index.py) | Milvus 建表、写入、删除、混合检索 |
| [app/services/reranker.py](../app/services/reranker.py) | 可选 CrossEncoder 精排与状态 |
| [app/services/retrieval.py](../app/services/retrieval.py) | 检索打分、证据门槛与版本处理 |
| [app/services/qa.py](../app/services/qa.py) | 摘录式/LLM 回答、引用校验、流式 |
| [app/services/llm.py](../app/services/llm.py) | 模型调用、超时/重试、JSON 输出 |
| [app/services/tickets.py](../app/services/tickets.py) | 草稿、幂等创建、事务、状态机、消息、附件 |
| [app/services/tools.py](../app/services/tools.py) | Agent 工具封装，统一返回 `ToolResult` |
| [app/api/](../app/api/) | HTTP 路由层，只做参数校验与调用服务 |
| [frontend/](../frontend/) | 5 个页面与公共脚本 |
| [scripts/](../scripts/) | 初始化、演示数据、评测、索重建、后端验证 |
| [tests/](../tests/) | 80 项自动化测试（仅 SQLite 时 78 passed / 2 skipped；接入 PostgreSQL 隔离 schema 可跑完整库事务测试） |
| [eval/test_cases.json](../eval/test_cases.json) | 40 条独立评测样例（不参与调参） |

## 3. 登录与身份链路

```text
POST /api/auth/login
  → 查 users（username）
  → bcrypt 校验（线程池，避免阻塞事件循环）
  → is_active 检查
  → 签发 JWT（sub=user_id，exp）
  → 返回 token + 角色 + 租户（仅用于前端展示）

后续请求 Authorization: Bearer <token>
  → decode JWT
  → 每次查库恢复用户
  → 校验 is_active
  → 组装 UserContext（role / tenant_id / tenant_name）
  → 路由依赖 require_roles(...)
  → 服务层再做租户/工单级校验
```

关键点：

- JWT **不携带角色和租户**，避免管理员改角色后旧 Token 仍保留旧权限。
- 禁用账号后，下一次请求即被拒绝，不需要等 Token 过期。
- 任何接口都不接受请求参数指定 `user_id` / `role` / `tenant_id` 作为身份来源。

## 4. 知识库导入链路

```text
管理员上传（md / txt / pdf）
  → 大小限制（默认 5MB）与扩展名白名单
  → 计算 checksum（重复文件直接拒绝）
  → 创建 knowledge_documents，status=processing
  → 解析：
       md/txt：按 Markdown 标题分节
       PDF：PyMuPDF 提取文本；扫描版（文本 < 30 字）明确报错
  → 切分：600 字 / 80 重叠，保留章节标题
  → 选择 Embedding 后端：
       hash   → 512 维哈希演示向量（不是语义 Embedding）
       bge_m3 → 1024 维 dense + sparse lexical_weights
  → 写 knowledge_chunks 与 knowledge_chunk_embeddings（按 backend 分行）
  → RETRIEVAL_BACKEND=milvus 时同步 upsert 到 Milvus
  → 更新 status=ready、chunk_count
失败：status=failed，error_message 记录原因，可在后台看到
```

更新或删除文档时：

- 更新：先解析、切分、向量化，全部成功后才在同一事务里删旧块、写新块和向量；
  任一步失败回滚，**旧资料仍然可用**，不会出现“旧块已删、新块没写”的半成品。
- Milvus 旧 chunk_id 在新向量写完后才清理；清理失败只记日志，检索回业务库时找不到对应分块会被丢弃。
- 删除：数据库级联删除 chunks 与向量行，再删 Milvus 中该文档的全部向量。
- 重复导入：checksum 唯一，重复文件直接拒绝，避免同一资料生成多份索引。

## 5. 检索链路

```text
query + 可选 product_version
  → 计算当前用户可访问的租户集合（公共资料 + 授权租户私有资料）
  → SQL 过滤 status=ready + access_scope + tenant_id
  → 若指定版本：
       先检查知识库中是否真实存在该版本（排除 all）
       不存在 → 直接返回 no_version_match，不用通用资料顶替
       存在   → 只保留 product_version=指定版本 或 all
  → 按 EMBEDDING_BACKEND / RETRIEVAL_BACKEND 选择打分方式：

     hash + local（演示）
       dense   = 哈希向量余弦
       lexical = jieba + IDF 的内容词/标题/章节覆盖
       score   = 0.35*dense + 0.65*lexical，且要求内容词命中或章节覆盖达标

     bge_m3 + local（真实语义）
       从 knowledge_chunk_embeddings 读取同一 backend 的 dense/sparse
       score = (dense_weight*余弦 + sparse_weight*词权重覆盖) / 权重和

     bge_m3 + milvus
       expr 在 Milvus 内先过滤 access_scope / tenant_id / product_version
       dense COSINE + sparse IP，WeightedRanker 融合召回
       按召回的 chunk_id 查询业务库，复核 ready、租户可见性和版本
       加载通过复核的原始 BGE-M3 向量并重算融合分

  → 证据门槛：semantic_score_threshold（默认 0.42）；低于门槛视为证据不足
  → 可选重排：RERANK_ENABLED=true 且模型可用时执行 CrossEncoder；失败标注 failed
  → 排序取 top_k（默认 5）
  → 返回 items / no_match / no_version_match / index_missing / retrieval_error
```

为什么哈希模式词法占主导：本地哈希向量不是语义 embedding，容易把无关文本拉近；在无模型 Key 的离线环境里，jieba + IDF 的词法覆盖更可靠。代价是近义表达召回有限，见[已知限制](limitations.md)。

为什么 Milvus 要二次打分：`WeightedRanker` 会把分数压到较窄区间（实测无关查询约 0.48–0.65，相关查询约 0.74–0.79），统一阈值不可靠。召回后回业务库用原始 BGE-M3 向量重算，才能让 `local` 与 `milvus` 使用同一套证据口径。

Milvus 路径在召回前只做分块存在性检查，最多取一个 ID。召回后按候选 ID 加载正文和元数据，业务库返回的正文行数受召回候选数约束。已删除、已不可见或版本不匹配的分块不会进入重排或回答上下文。

版本规则：

- 明确指定 `v2.2`：只检索 `v2.2` 与 `all`，不会回退到 `v2.3`；
- 未指定版本且同时命中多个版本：标记 `version_conflict`，回答中提示存在版本差异；
- 指定版本但知识库没有该版本：明确告知版本不匹配，不拿其他版本顶替。

## 6. 问答链路

```text
POST /api/chat/sessions/{id}/stream（SSE）或 /messages（普通 JSON）
  → 校验会话归属（user_id + tenant_id 必须一致）
  → 读取最近 N 条用户原话（默认 8 条），不拼上一轮 AI 结论
  → 组装检索 query：
       普通追问：拼接用户原话
       “那旧版本呢”：去掉上一问的版本号，避免把 v2.3 带进 v2.2 的追问
  → 检索（见第 5 节）
  → 无结果时按原因区分：
       no_evidence      证据不足，说明不足并建议补充信息/转人工
       no_version_match 指定版本不存在，列出知识库实际版本
       index_missing    向量索引缺失，提示重建索引
       retrieval_error  后端异常，明确说服务异常，不说“没有资料”
  → 有证据：
       extractive（默认）：按检索片段组织答案，引用编号 [n]
       llm               ：把证据作为资料传给模型，要求只依据资料回答
       llm_uncited       ：模型两次都没有给出有效引用，丢弃模型回答，退回摘录
       llm_degraded      ：超时/网络/缺 Key，退回摘录并附加降级说明
  → 校验引用编号：超出实际引用数的编号会被删除并记录 warning
  → 保存 assistant 消息（citations_json / meta_json）
  → SSE 事件：
       start → meta → stage → retrieval → citations →
       token...（真实模型 astream 片段）→ result → saved → done
       异常：warning / error
```

流式边界：

- 只有 `LLM_MODE=openai` 且模型可用时才消费真实 token 流；摘录模式一次性产生文本，不冒充逐 token。
- 客户端断开后停止继续消费，不无限生成。
- 部分回答保存时 `meta.stream_status=partial`，并**不保存引用列表**，避免把未完成的文本当成有证据的完整回答。
- 流式过程只生成回答，不执行任何未经确认的写操作。
- 已通过测试替身覆盖超时、断连、中途异常 `partial` 保存与越界引用剔除，并通过 `deepseek-chat` 真实 API 验证 SSE `astream` 逐 token 转发与首 token 延迟统计。

## 7. 工单 Agent（LangGraph，受控工作流）

单图结构：

```text
load_context（读取会话、待确认草稿、最近工单号）
  → classify（规则优先；必要时模型只做意图建议）
      ├── question      → answer（检索 + 回答）
      ├── issue         → handle_issue
      │        ├── 缺产品版本 → 追问（本轮结束）
      │        └── 有版本 → answer + 提示“未解决可转人工”
      ├── human         → handle_human
      │        ├── 缺报错/现象 → 追问
      │        ├── 已有待确认草稿 → 复用，不重复建草稿
      │        └── 信息齐全 → 生成 ticket_drafts 草稿（不写 tickets）
      ├── confirm       → confirm（校验草稿归属 + expected_version，再调用创建工具）
      ├── cancel        → cancel（标记草稿 cancelled，清空会话待确认状态）
      ├── edit_draft    → edit_draft（更新字段，草稿 version + 1）
      ├── query_tickets → query_tickets（只查当前用户可见范围）
      ├── ticket_detail → ticket_detail
      ├── supplement    → supplement（内容摘要幂等键，防重复补充）
      ├── clarify       → clarify（不确定时追问，不执行写操作）
      └── greeting/chitchat → smalltalk
  → finalize → END
```

关键约束：

- 缺产品版本、操作路径、报错信息时追问，不编造；
- 草稿只写 `ticket_drafts`，不写 `tickets`；
- 只有 `POST /api/tickets/confirm` 或 Agent 的 `confirm` 节点（且意图由严格规则判定）才调用创建工具；
- 工具返回失败时，回答必须说失败，不能改口说「已创建」；
- 创建成功后返回真实工单号，并写入会话上下文；
- 模型只能建议意图和草稿字段，**可选意图枚举里没有 `confirm / cancel / edit_draft`**；
- 存在待确认草稿时，模型完全不能改写意图，避免把“不要提交”理解成确认。

`runtime`（db、user、session_id）通过 `functools.partial` 注入节点。不要用 lambda 返回协程，否则 LangGraph 不会 await，会出现 `'coroutine' object is not subscriptable`。

项目定位：这是**受控工作流 + 规则路由 + 有限模型理解**，不是让模型自主规划、自由选择工具的 Agent。

## 8. 工单幂等、取号与事务

五道防线：

1. `expected_version` 必填：请求层与服务层都拒绝缺失/非法版本，调用方不能绕过；
2. 条件 `UPDATE ... WHERE status='draft' AND version=:expected` 原子认领草稿，
   编辑、取消、确认共用这一模式，竞争失败方得到 409；
3. `ticket_drafts.idempotency_key` 唯一：同一用户 + 会话 + 规范化描述只生成一个草稿；
   草稿取消后重新发起会生成 `:r2`、`:r3` 后缀的新键，不会被旧键卡住；
4. `tickets.draft_id` 唯一：同一草稿最多创建一张工单；
5. 工单号由 `ticket_sequences` 行级 UPSERT … RETURNING 原子分配，不靠 `MAX(ticket_no)+1`；
   创建逻辑在 savepoint 内 `add + flush`，捕获 `IntegrityError` 后查回已有工单；
   事务提交失败不会向用户返回“创建成功”，认领失败会回滚草稿状态。

顺序重复与并发是两件事：

- 顺序重复确认：第二次请求查到 `tickets.draft_id` 已有记录，直接返回同一张工单（`already_created=true`）；
- 并发确认：两个请求同时进入，条件 UPDATE 只允许一个认领成功，另一个按最新状态返回冲突或已有工单；
- 不使用进程内锁冒充跨进程并发保护，最终防线是数据库条件更新与唯一约束。

状态流转：

```text
pending（待处理）
  ├── 客服/管理员 → processing（处理中）
  └── 管理员 → closed（已关闭）

processing
  ├── 客服/管理员 → waiting_customer（待客户确认）
  └── 管理员 → pending

waiting_customer
  ├── 客户/客服/管理员 → closed（客户确认解决 / 客服关闭）
  └── 客服/管理员 → processing（客户反馈未解决）

closed
  └── 管理员 → processing（重新打开）
```

每次变化写 `ticket_events`（操作人、动作、前后状态、备注），客户与客服都能在详情里看到流转记录。

## 9. 权限模型

| 角色 | 数据范围 | 典型操作 |
| --- | --- | --- |
| 客户 customer | 自己创建的工单；本租户可访问的公共/私有资料 | 问答、建草稿、确认建单、查看/回复自己的工单、确认解决 |
| 客服 agent | 被 `agent_tenant_grants` 授权的租户工单；平台资料 + 授权租户私有资料 | 查看、回复、推进状态、附件 |
| 管理员 admin | 全部租户、全部工单、知识库、用户与授权 | 知识库管理、用户管理、授权、任意状态流转 |

实现要点：

- 客户过滤条件：`created_by == user.user_id`；
- 客服过滤条件：`tenant_id in (授权租户)`；
- 管理员不过滤；
- 工单详情、回复、状态、附件、补充全部复用同一可见性判断；
- 知识检索的租户过滤与工单过滤分开实现，但都从 `UserContext` 出发。

## 10. 可观测性与错误处理

- 每个请求生成 `request_id`，写入日志、异常响应和响应头 `X-Request-ID`；
- 日志使用 `key=value` 结构化格式，敏感字段不记录；
- 业务错误使用 `ToolError` / `HTTPException`，统一返回 `{"detail": ...}`；
- 未捕获异常返回 500，带 `request_id`，不返回堆栈；
- 模型调用有可配置超时（默认 30 秒）和有限重试；超时/失败退回摘录并标注 `llm_degraded`，
  不伪装成模型成功，也不把服务异常说成“没有资料”；
- 健康检查 `/health` 返回数据库状态、当前回答模式，以及当前配置的 Embedding / 检索后端；
  它只反映配置与数据库连通性，不代表模型或 Milvus 已实际可用。

## 11. 前端结构

| 页面 | 文件 | 说明 |
| --- | --- | --- |
| 登录 | [frontend/login.html](../frontend/login.html) | 演示账号快速填入，错误提示 |
| 客户问答 | [frontend/chat.html](../frontend/chat.html) | 会话列表、SSE、引用折叠、可编辑草稿卡 |
| 客户工单 | [frontend/tickets.html](../frontend/tickets.html) | 统计、筛选、详情、回复、附件、确认解决 |
| 客服工作台 | [frontend/agent.html](../frontend/agent.html) | 授权范围队列、筛选、状态流转、回复 |
| 管理后台 | [frontend/admin.html](../frontend/admin.html) | 知识库、用户、授权、租户 |
| 公共脚本 | [frontend/assets/common.js](../frontend/assets/common.js) | Token、请求封装、标签、导航、提示 |

前端只使用原生 API，不需要 Node、npm 或构建步骤。角色不匹配时页面会跳转，但真正的拦截仍在后端。
