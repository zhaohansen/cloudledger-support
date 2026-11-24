# API 说明与示例

> 基础地址：`http://127.0.0.1:8000`
> 在线文档：`/docs`（Swagger UI）
> 所有业务接口都需要 `Authorization: Bearer <access_token>`。

## 1. 通用约定

### 认证

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
```

Token 只包含用户 ID 和过期时间。角色、租户、账号状态每次请求从数据库读取。

### 错误响应

```json
{ "detail": "用户名或密码错误" }
```

| 状态码 | 含义 |
| --- | --- |
| 401 | 未登录、Token 无效或过期、密码错误 |
| 403 | 已登录但无权限，或账号被禁用、状态流转不允许 |
| 404 | 资源不存在，或对方无权看到（避免泄露存在性） |
| 409 | 冲突：重复内容、工具失败、草稿状态不允许 |
| 413 | 上传文件超过大小限制 |
| 422 | 参数校验失败、文件为空或类型不允许 |
| 500 | 未捕获异常，响应含 `request_id` |

每个响应头都有 `X-Request-ID`，报错时可用它查日志。

### 健康检查

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "version": "1.0.0",
  "database": "sqlite",
  "database_ok": true,
  "llm_mode": "extractive",
  "llm_configured": false,
  "embedding_backend": "hash",
  "retrieval_backend": "local",
  "rerank_enabled": false
}
```

`embedding_backend` / `retrieval_backend` 只反映配置，不代表模型文件或 Milvus 已实际可用；
真实可用性需通过 `/api/knowledge/search` 或 `scripts/verify_semantic_backend.py` 验证。

## 2. 认证

### POST /api/auth/login

```bash
curl -X POST http://127.0.0.1:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"customer_a","password":"Customer@123456"}'
```

```json
{
  "access_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 28800,
  "user_id": "…",
  "username": "customer_a",
  "display_name": "华辰智造-王会计",
  "role": "customer",
  "tenant_id": "…",
  "tenant_name": "华辰智造"
}
```

### GET /api/auth/me

返回当前用户与租户信息，用于前端刷新身份。

## 3. 会话与问答

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/chat/sessions` | 新建会话，body: `{"title": "新会话"}` |
| GET | `/api/chat/sessions` | 当前用户会话列表 |
| GET | `/api/chat/sessions/{id}/messages` | 会话消息；非本人返回 404 |
| POST | `/api/chat/sessions/{id}/messages` | 非流式发送，返回完整结果 |
| POST | `/api/chat/sessions/{id}/stream` | SSE 流式发送 |

请求体：

```json
{ "content": "v2.2 报销单保存草稿后为什么部门可用预算被冻结？", "product_version": "v2.2" }
```

`product_version` 可空；指定后会限制检索版本。

### 非流式示例

```bash
curl -X POST http://127.0.0.1:8000/api/chat/sessions/$SESSION_ID/messages \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"content":"v2.2 报销单保存草稿后为什么部门可用预算被冻结？","product_version":"v2.2"}'
```

返回字段：

| 字段 | 说明 |
| --- | --- |
| reply | 回答正文，引用编号形如 `[1]` |
| citations | 本次实际检索到的引用列表 |
| intent | 分类结果 |
| draft | 工单草稿（没有则为 null） |
| ticket | 已创建工单（没有则为 null） |
| answer_mode | `extractive` / `llm` / `llm_degraded` / `llm_uncited` / `no_evidence` / `no_version_match` / `index_missing` / `retrieval_error` |
| tool_ok | 工具是否成功 |
| answer_meta | 实际后端、降级原因、检索耗时等排查信息 |
| intent_source | `rule` 或 `model`，表示意图来源；写操作不依赖该字段 |

### SSE 示例

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat/sessions/$SESSION_ID/stream \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"content":"Excel 导入提示模板版本不匹配"}'
```

事件顺序：

```text
data: {"type":"start","session_id":"..."}
data: {"type":"meta","session":{...},"intent":"question","intent_source":"rule"}
data: {"type":"stage","stage":"retrieval"}
data: {"type":"retrieval","reason":"ok","embedding_backend":"bge_m3","retrieval_backend":"milvus"}
data: {"type":"citations","citations":[...]}
data: {"type":"token","content":"根据..."}
data: {"type":"result","message_id":"...","citations":[...],"draft":null,"ticket":null,"answer_mode":"llm","tool_ok":true}
data: {"type":"saved","message_id":"..."}
data: [DONE]
```

异常事件：`{"type":"warning",...}` / `{"type":"error","detail":"..."}`。

说明：

- 配置真实模型（`LLM_API_KEY` 或 `DEEPSEEK_API_KEY`）时，`token` 来自模型 `astream`，是实际逐块流式输出；摘录模式一次性产生文本。
- 客户端断开后服务端停止生成；已产生的部分回答保存为 `meta.stream_status=partial`，不保存引用列表。
- 已通过可控替身验证异常分支（超时、断流、越界引用），并通过 `deepseek-chat` 验证真实 SSE 流式输出。

## 4. 工单

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/tickets` | 工单列表，可带 `status` / `category` |
| GET | `/api/tickets/counts` | 按状态统计 |
| GET | `/api/tickets/drafts/{draft_id}` | 获取草稿（仅本人） |
| PATCH | `/api/tickets/drafts/{draft_id}` | 修改草稿字段 |
| POST | `/api/tickets/confirm` | 确认草稿并创建工单（幂等） |
| GET | `/api/tickets/{ticket_id}` | 详情（含消息、事件、附件） |
| PATCH | `/api/tickets/{ticket_id}/status` | 状态流转 |
| POST | `/api/tickets/{ticket_id}/messages` | 回复 |
| POST | `/api/tickets/{ticket_id}/supplement` | 客户补充描述/报错/步骤 |
| POST | `/api/tickets/{ticket_id}/attachments` | 上传附件（multipart） |
| GET | `/api/tickets/{ticket_id}/attachments/{attachment_id}` | 下载附件 |

### 修改草稿

```bash
curl -X PATCH http://127.0.0.1:8000/api/tickets/drafts/$DRAFT_ID \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title":"v2.2 报销草稿冻结预算","priority":"high","tried_steps":"已核对报销单草稿状态","expected_version":2}'
```

- 建议始终带 `expected_version`；服务端用条件 `UPDATE ... WHERE status='draft' AND version=:expected` 更新，版本不一致返回 **409**，不会覆盖别人的修改。
- 内容真正变化后 `version + 1`；旧版本上发出的确认随之失效。

### 确认创建（幂等 + 版本绑定）

```bash
curl -X POST http://127.0.0.1:8000/api/tickets/confirm \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"draft_id\":\"$DRAFT_ID\",\"expected_version\":3}"
```

```json
{
  "ticket": {
    "id": "…",
    "ticket_no": "YZ202611240003",
    "status": "pending",
    "title": "v2.2 报销草稿冻结预算",
    "tenant_id": "…"
  },
  "already_created": false
}
```

- `expected_version` **必填**（严格整数）。缺失、`null`、`0`、负数、数字字符串、小数都返回 **422**；服务层同样拒绝，Agent 工具和脚本不能绕过。
- `expected_version` 必须等于草稿当前 `version`；不一致返回 **409**，不会用新内容提交旧确认。草稿已确认后如果版本号不匹配，也返回 409。
- 草稿已取消、不属于当前用户或不存在，返回 404/409；已取消草稿不能再确认。
- 同一草稿、同一版本重复确认返回同一张工单，`already_created=true`；首次确认是 `false`。这是“同一次操作重复提交”的幂等，不是用旧版本确认新内容。
- 并发确认同一草稿最多生成一张工单：服务层用条件 UPDATE 原子认领草稿，`tickets.draft_id` 唯一约束是最终防线。
- 编辑与确认、取消与确认同时发生时只有一个生效，失败方收到 409；工单内容对应用户确认的那一版草稿。
- 确认、创建工单、写状态事件在同一事务中提交；提交失败返回 500，不会报告创建成功。
- 该接口只接受结构化请求体，不解析自然语言；否定句/疑问句不会触发创建。取消草稿通过聊天中的明确取消指令调用服务层，同样做条件更新。

### 状态流转

```bash
curl -X PATCH http://127.0.0.1:8000/api/tickets/$TICKET_ID/status \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"status":"processing","note":"客服已接手"}'
```

非法流转返回 403；每次成功流转写 `ticket_events`。

### 回复与附件

回复和补充支持 `Idempotency-Key` 请求头；不传时服务端按内容摘要派生键，重复提交相同内容不会重复写入。

```bash
curl -X POST http://127.0.0.1:8000/api/tickets/$TICKET_ID/messages \
  -H "Authorization: Bearer $TOKEN" \
  -H "Idempotency-Key: 3f8c2a..." \
  -H "Content-Type: application/json" \
  -d '{"content":"请提供报错截图和操作菜单路径"}'
```

```bash
curl -X POST http://127.0.0.1:8000/api/tickets/$TICKET_ID/attachments \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@./error.png"
```

## 5. 知识库（仅管理员）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/knowledge/documents` | 上传并解析（multipart） |
| GET | `/api/knowledge/documents` | 文档列表与导入状态 |
| GET | `/api/knowledge/documents/{id}` | 文档详情 + 前 20 个分块 |
| PUT | `/api/knowledge/documents/{id}` | 更新元数据或替换文件 |
| DELETE | `/api/knowledge/documents/{id}` | 删除文档与旧向量 |
| POST | `/api/knowledge/search` | 检索调试，返回分数与来源 |

上传示例：

```bash
curl -X POST http://127.0.0.1:8000/api/knowledge/documents \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -F "file=@./demo_docs/11-华辰智造费控内控制度.md" \
  -F "title=华辰智造费控内控制度" \
  -F "doc_type=private" \
  -F "version=1.0" \
  -F "product_version=all" \
  -F "access_scope=tenant" \
  -F "tenant_id=$HUACHEN_TENANT_ID"
```

检索调试：

```bash
curl -X POST http://127.0.0.1:8000/api/knowledge/search \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"v2.2 报销草稿 冻结预算","product_version":"v2.2"}'
```

返回：

- `reason`：`ok` / `no_match` / `no_version_match` / `index_missing` / `retrieval_error`；
- `embedding_backend` / `retrieval_backend`：本次实际使用的后端；
- `degraded` / `detail`：失败或降级原因；
- `items`：每条含 `score`、`dense_score`、`sparse_score`（hash 模式为 `lexical_score`）、
  `document_title`、`chunk_id`、`product_version` 等，可定位到实际文档与分块。

真实后端不可用时返回明确错误，不会静默改用哈希检索。

## 6. 管理后台（仅管理员）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/admin/tenants` | 租户列表 |
| GET | `/api/admin/users` | 用户列表 |
| POST | `/api/admin/users` | 创建用户 |
| PATCH | `/api/admin/users/{id}` | 修改资料/角色/租户/状态/密码 |
| GET | `/api/admin/agent-grants` | 客服授权列表 |
| POST | `/api/admin/agent-grants` | 新增授权 |
| DELETE | `/api/admin/agent-grants/{id}` | 移除授权 |

创建用户：

```bash
curl -X POST http://127.0.0.1:8000/api/admin/users \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"username":"customer_c","password":"Demo@123456","display_name":"新客户","role":"customer","tenant_id":"'$YUNSHANG_TENANT_ID'"}'
```

禁用账号：

```bash
curl -X PATCH http://127.0.0.1:8000/api/admin/users/$USER_ID \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"is_active":false}'
```

禁用后该用户的登录与已签发 Token 都会被拒绝。

新增客服授权：

```bash
curl -X POST http://127.0.0.1:8000/api/admin/agent-grants \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"agent_user_id":"'$AGENT_ID'","tenant_id":"'$YUNSHANG_TENANT_ID'"}'
```

## 7. 页面路由

| 路径 | 页面 |
| --- | --- |
| `/login` | 登录 |
| `/chat` | 客户问答（customer / agent / admin） |
| `/tickets` | 我的工单（customer / agent / admin） |
| `/agent` | 客服工作台（agent / admin） |
| `/admin` | 管理后台（admin） |

页面跳转只影响体验；越权请求仍会被后端拒绝。
