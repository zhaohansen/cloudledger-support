# 数据表、角色权限与状态说明

> 对应代码：[app/models.py](../app/models.py)。数据库默认 SQLite，表名与字段可直接迁移到 PostgreSQL。

## 1. 实体关系

```text
tenants 1 ── n users
tenants 1 ── n knowledge_documents（私有资料）
tenants 1 ── n tickets
users   1 ── n agent_tenant_grants（客服授权）
users   1 ── n chat_sessions 1 ── n chat_messages
users   1 ── n ticket_drafts 1 ── 0..1 tickets
tickets 1 ── n ticket_messages
tickets 1 ── n ticket_events
tickets 1 ── n ticket_attachments
knowledge_documents 1 ── n knowledge_chunks
knowledge_chunks 1 ── n knowledge_chunk_embeddings（按 backend 分行）
ticket_sequences（按年月维护工单号序号）
```

## 2. 数据表

### 2.1 tenants（租户）

| 字段 | 类型 | 约束 / 说明 |
| --- | --- | --- |
| id | string(32) | 主键 |
| name | string(128) | 企业名称 |
| code | string(64) | 唯一，如 `cloudledger` / `huachen` / `xinghai` |
| is_platform | bool | 是否平台方租户 |
| created_at | datetime | 创建时间 |

### 2.2 users（用户）

| 字段 | 类型 | 约束 / 说明 |
| --- | --- | --- |
| id | string(32) | 主键 |
| tenant_id | string(32) | 外键 tenants，删除限制 |
| username | string(64) | 唯一，登录名 |
| display_name | string(64) | 展示名 |
| email | string(128) | 邮箱 |
| password_hash | string(255) | bcrypt 哈希，绝不存明文 |
| role | string(16) | `customer` / `agent` / `admin` |
| is_active | bool | 禁用后登录与已登录请求都被拒绝 |
| created_at / updated_at | datetime | 时间戳 |

### 2.3 agent_tenant_grants（客服租户授权）

| 字段 | 类型 | 约束 / 说明 |
| --- | --- | --- |
| id | string(32) | 主键 |
| agent_user_id | string(32) | 外键 users |
| tenant_id | string(32) | 外键 tenants |
| created_at | datetime | 创建时间 |

唯一约束：`(agent_user_id, tenant_id)`。客服只能访问授权租户的工单和私有资料。

### 2.4 knowledge_documents（知识文档）

| 字段 | 类型 | 约束 / 说明 |
| --- | --- | --- |
| id | string(32) | 主键 |
| title | string(200) | 标题 |
| doc_type | string(16) | `manual` / `release` / `faq` / `troubleshooting` 等 |
| version | string(32) | 文档版本 |
| product_version | string(32) | `v2.2` / `v2.3` / `all` |
| access_scope | string(16) | `public` / `tenant` |
| tenant_id | string(32) | 私有资料所属租户，可空 |
| filename | string(255) | 原始文件名 |
| stored_path | string(500) | 本地存储路径 |
| checksum | string(64) | 唯一，阻止重复导入 |
| status | string(16) | `processing` / `ready` / `failed` |
| error_message | text | 解析失败原因 |
| chunk_count | int | 分块数 |
| created_by / updated_by | string(32) | 操作人 |
| created_at / updated_at | datetime | 时间戳 |

### 2.5 knowledge_chunks（知识分块）

| 字段 | 类型 | 约束 / 说明 |
| --- | --- | --- |
| id | string(32) | 主键 |
| document_id | string(32) | 外键 knowledge_documents，级联删除 |
| chunk_index | int | 分块序号 |
| section | string(200) | 章节标题 |
| content | text | 分块正文 |
| embedding | JSON | 兼容字段：仅 hash 演示模式写 512 维向量；BGE-M3 模式为空 |
| created_at | datetime | 创建时间 |

### 2.5.1 knowledge_chunk_embeddings（分块向量）

| 字段 | 类型 | 约束 / 说明 |
| --- | --- | --- |
| id | string(32) | 主键 |
| chunk_id | string(32) | 外键 knowledge_chunks，级联删除 |
| backend | string(32) | `hash` / `bge_m3` |
| dense | JSON | 向量数组：hash=512 维，bge_m3=1024 维 |
| sparse | JSON | BGE-M3 lexical weights，存为 `[[token_id, weight], ...]`；hash 为空 |
| dim | int | 向量维度 |
| created_at | datetime | 创建时间 |

唯一约束：`(chunk_id, backend)`。同一分块可以同时保留不同后端的向量，切换后端后必须重建索引；检索只读取当前配置的 `backend`。

### 2.6 chat_sessions（会话）

| 字段 | 类型 | 约束 / 说明 |
| --- | --- | --- |
| id | string(32) | 主键 |
| user_id | string(32) | 外键 users |
| tenant_id | string(32) | 会话所属租户 |
| title | string(120) | 会话标题 |
| context_json | text | Agent 恢复用上下文（facts / pending_draft） |
| created_at / updated_at | datetime | 时间戳 |

### 2.7 chat_messages（会话消息）

| 字段 | 类型 | 约束 / 说明 |
| --- | --- | --- |
| id | string(32) | 主键 |
| session_id | string(32) | 外键 chat_sessions |
| role | string(16) | `user` / `assistant` |
| content | text | 消息正文 |
| citations_json | JSON | 本次回答实际使用的引用 |
| meta_json | JSON | answer_mode、intent、draft_id、tool_ok 等 |
| created_at | datetime | 创建时间 |

### 2.8 ticket_drafts（工单草稿）

| 字段 | 类型 | 约束 / 说明 |
| --- | --- | --- |
| id | string(32) | 主键 |
| user_id | string(32) | 外键 users |
| tenant_id | string(32) | 草稿所属租户 |
| session_id | string(32) | 来源会话，可空 |
| idempotency_key | string(200) | 唯一，防重复草稿 |
| title | string(200) | 标题 |
| description | text | 问题描述 |
| product_version | string(32) | 产品版本 |
| error_message | text | 报错信息 |
| tried_steps | text | 已尝试步骤 |
| category | string(32) | 分类 |
| priority | string(16) | `low` / `medium` / `high` |
| version | int | 草稿版本，默认 1；每次编辑 +1 |
| status | string(16) | `draft` / `confirmed` / `cancelled` |
| created_at / updated_at | datetime | 时间戳 |

`version` 用于确认绑定：确认请求必须带 `expected_version`（缺失或非法直接 422/服务层拒绝），与当前版本不一致返回 409；服务端不会读取最新版本替用户确认。
草稿取消后保留记录（`status=cancelled`），不能再次确认；重新发起会创建新草稿。

### 2.9 tickets（工单）

| 字段 | 类型 | 约束 / 说明 |
| --- | --- | --- |
| id | string(32) | 主键 |
| ticket_no | string(32) | 唯一，如 `YZ202611160001` |
| draft_id | string(32) | 唯一，外键 ticket_drafts |
| tenant_id | string(32) | 所属租户 |
| created_by | string(32) | 创建客户 |
| assigned_to | string(32) | 处理客服，可空 |
| title | string(200) | 标题 |
| description | text | 描述 |
| product_version | string(32) | 版本 |
| error_message | text | 报错 |
| tried_steps | text | 已尝试步骤 |
| category | string(32) | 分类 |
| priority | string(16) | 优先级 |
| status | string(24) | `pending` / `processing` / `waiting_customer` / `closed` |
| created_at / updated_at / closed_at | datetime | 时间戳 |

`draft_id` 唯一是第二层幂等：同一草稿最多生成一张工单。工单号由
`ticket_sequences` 原子分配，不在应用层做 `MAX(ticket_no)+1`。

### 2.9.1 ticket_sequences（工单号序号）

| 字段 | 类型 | 约束 / 说明 |
| --- | --- | --- |
| period | string(8) | 序号周期，如 `202611` |
| next_value | int | 下一个可用序号 |
| updated_at | datetime | 更新时间 |

主键 `period`。创建工单时执行 `INSERT ... ON CONFLICT DO UPDATE ... RETURNING`，
由数据库保证同一周期内不重复；不同草稿并发提交不会因为编号冲突破坏事务。

### 2.10 ticket_messages（工单沟通）

| 字段 | 类型 | 约束 / 说明 |
| --- | --- | --- |
| id | string(32) | 主键 |
| ticket_id | string(32) | 外键 tickets，级联删除 |
| author_id | string(32) | 作者 |
| author_role | string(16) | 作者角色 |
| content | text | 内容 |
| message_type | string(16) | `reply` / `supplement` / `system` |
| request_key | string(128) | 可空；唯一约束 `uq_ticket_messages_request_key`，防重复发送/重复补充 |
| created_at | datetime | 创建时间 |

### 2.11 ticket_events（流转留痕）

| 字段 | 类型 | 约束 / 说明 |
| --- | --- | --- |
| id | string(32) | 主键 |
| ticket_id | string(32) | 外键 tickets，级联删除 |
| actor_id | string(32) | 操作人 |
| action | string(32) | 动作，如 `status_update` / `customer_confirm` |
| from_status | string(24) | 原状态 |
| to_status | string(24) | 新状态 |
| note | text | 备注 |
| created_at | datetime | 创建时间 |

### 2.12 ticket_attachments（附件）

| 字段 | 类型 | 约束 / 说明 |
| --- | --- | --- |
| id | string(32) | 主键 |
| ticket_id | string(32) | 外键 tickets，级联删除 |
| filename | string(255) | 原始文件名 |
| stored_path | string(500) | 本地存储路径 |
| content_type | string(128) | MIME |
| size | int | 字节数 |
| uploaded_by | string(32) | 上传人 |
| created_at | datetime | 创建时间 |

附件扩展名白名单：`.png .jpg .jpeg .gif .webp .pdf .txt .log .csv`；大小沿用上传限制。

## 3. 角色与权限矩阵

| 操作 | 客户 | 客服 | 管理员 |
| --- | --- | --- | --- |
| 登录 / 查看自己 | ✅ | ✅ | ✅ |
| 创建会话、提问 | ✅ | ✅ | ✅ |
| 检索公共资料 | ✅ | ✅ | ✅ |
| 检索租户私有资料 | 仅本租户 | 仅授权租户 | 全部 |
| 生成工单草稿 | ✅ | ✅ | ✅ |
| 确认创建工单 | ✅（仅自己草稿） | ✅（仅自己草稿） | ✅ |
| 查看工单列表 | 仅自己创建 | 授权租户 | 全部 |
| 查看工单详情/消息/附件 | 仅自己创建 | 授权租户 | 全部 |
| 回复工单 | 仅自己创建 | 授权租户 | 全部 |
| 推进状态 | 仅待客户确认时确认/反馈 | 授权租户，按状态机 | 全部，按状态机 |
| 上传附件 | 仅自己工单 | 授权租户 | 全部 |
| 知识库上传/更新/删除/检索调试 | ❌ | ❌ | ✅ |
| 用户创建/禁用/改角色/重置密码 | ❌ | ❌ | ✅ |
| 客服授权 | ❌ | ❌ | ✅ |
| 租户列表 | ❌ | ❌ | ✅ |

后端实现位置：[app/services/permissions.py](../app/services/permissions.py)、[app/deps.py](../app/deps.py)、各 API 路由依赖。

## 4. 工单状态机

```text
pending ──客服/管理员──▶ processing ──客服/管理员──▶ waiting_customer
   │                        │  ▲                         │
   │                        │  └──── 客服/管理员 ────────┘
   │                        │        （客户反馈未解决）
   │                        └──── 管理员 ──▶ pending
   │
   └──管理员──▶ closed ◀──客户/客服/管理员── waiting_customer
                   │
                   └──管理员──▶ processing
```

角色允许的流转（代码：[app/services/tickets.py](../app/services/tickets.py)）：

| 当前状态 | 客户可转 | 客服可转 | 管理员可转 |
| --- | --- | --- | --- |
| pending | 无 | processing | processing / closed |
| processing | 无 | waiting_customer | waiting_customer / pending |
| waiting_customer | closed / processing | closed / processing | closed / processing |
| closed | 无 | 无 | processing |

同时满足「全局状态机允许」与「角色允许」才能流转；成功写入 `ticket_events`。

## 5. 其他枚举

| 枚举 | 取值 |
| --- | --- |
| 用户角色 | `customer` / `agent` / `admin` |
| 文档状态 | `processing` / `ready` / `failed` |
| 资料范围 | `public` / `tenant` |
| 草稿状态 | `draft` / `confirmed` / `cancelled` |
| 工单状态 | `pending` / `processing` / `waiting_customer` / `closed` |
| 工单消息类型 | `reply` / `supplement` / `system` |
| 优先级 | `low` / `medium` / `high` |
| 分类 | `budget` / `invoice` / `reimbursement` / `payment` / `closing` / `import` / `permission` / `other` |
| Embedding 后端 | `hash`（演示） / `bge_m3`（真实语义） |
| 检索后端 | `local`（业务库向量） / `milvus`（混合检索） |
| 回答模式 | `extractive` / `llm` / `llm_degraded` / `llm_uncited` / `no_evidence` / `no_version_match` / `index_missing` / `retrieval_error` |
