// frontend/assets/common.js
// 公共工具：登录态、API 请求封装、导航、状态文案、HTML 转义与小提示。
// 说明：前端隐藏按钮只是体验层，真正的权限校验全部在后端；前端不保存密码。

const TOKEN_KEY = "cloudledger_token";
const USER_KEY = "cloudledger_user";

const STATUS_LABELS = {
  pending: "待处理",
  processing: "处理中",
  waiting_customer: "待客户确认",
  closed: "已关闭",
};

const CATEGORY_LABELS = {
  budget: "预算控制",
  invoice: "数电发票",
  reimbursement: "费用报销",
  payment: "银企付款",
  closing: "月结总账",
  import: "批量导入",
  permission: "账号权限",
  other: "其他",
};

const PRIORITY_LABELS = { high: "高", medium: "中", low: "低" };
const ROLE_LABELS = { customer: "客户", agent: "财务顾问", admin: "管理员" };

const DEMO_ACCOUNTS = [
  { username: "customer_a", password: "Customer@123456", label: "客户 · 华辰智造" },
  { username: "customer_b", password: "Customer@123456", label: "客户 · 星海文化" },
  { username: "agent1", password: "Agent@123456", label: "顾问 · 华辰智造" },
  { username: "agent2", password: "Agent@123456", label: "顾问 · 星海文化" },
  { username: "admin", password: "Admin@123456", label: "管理员" },
];

function getToken() {
  return localStorage.getItem(TOKEN_KEY) || "";
}

function getUser() {
  try {
    const user = JSON.parse(localStorage.getItem(USER_KEY) || "null");
    if (user && user.expires_at && user.expires_at <= Date.now()) {
      clearAuth();
      return null;
    }
    return user;
  } catch (_err) {
    return null;
  }
}

function saveAuth(data) {
  localStorage.setItem(TOKEN_KEY, data.access_token);
  localStorage.setItem(
    USER_KEY,
    JSON.stringify({
      user_id: data.user_id,
      username: data.username,
      display_name: data.display_name,
      role: data.role,
      tenant_id: data.tenant_id,
      tenant_name: data.tenant_name,
      expires_at: Date.now() + (data.expires_in || 0) * 1000,
    })
  );
}

function clearAuth() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

function homeFor(role) {
  if (role === "admin") return "/admin";
  if (role === "agent") return "/agent";
  return "/tickets";
}

function requireAuth(allowedRoles) {
  const token = getToken();
  const user = getUser();
  if (!token || !user) {
    location.href = "/login";
    return null;
  }
  if (allowedRoles && !allowedRoles.includes(user.role)) {
    location.href = homeFor(user.role);
    return null;
  }
  return user;
}

async function api(path, options = {}) {
  const headers = Object.assign({}, options.headers || {});
  const token = getToken();
  if (token) headers["Authorization"] = "Bearer " + token;

  let body = options.body;
  if (body && !(body instanceof FormData) && typeof body !== "string") {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(body);
  }

  let response;
  try {
    response = await fetch("/api" + path, Object.assign({}, options, { headers, body }));
  } catch (err) {
    throw new Error("网络请求失败，请检查服务是否已启动");
  }

  if (response.status === 401 && !options.skipAuthRedirect) {
    clearAuth();
    location.href = "/login";
    throw new Error("登录已失效，请重新登录");
  }

  let data = null;
  const type = response.headers.get("content-type") || "";
  if (type.includes("application/json")) {
    data = await response.json().catch(() => null);
  }
  if (!response.ok) {
    const detail = (data && (data.detail || data.message)) || "请求失败（HTTP " + response.status + "）";
    const message = Array.isArray(detail)
      ? detail.map(item => item.msg).filter(Boolean).join("，")
      : detail;
    const error = new Error(message || "请求失败");
    error.status = response.status;
    error.requestId = response.headers.get("X-Request-ID") || (data && data.request_id);
    throw error;
  }
  return data;
}

function escapeHtml(value) {
  return String(value == null ? "" : value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function fmtTime(iso) {
  if (!iso) return "-";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const pad = (n) => String(n).padStart(2, "0");
  return (
    date.getFullYear() +
    "-" + pad(date.getMonth() + 1) +
    "-" + pad(date.getDate()) +
    " " + pad(date.getHours()) +
    ":" + pad(date.getMinutes())
  );
}

function statusBadge(status) {
  return '<span class="badge ' + escapeHtml(status) + '">' + escapeHtml(STATUS_LABELS[status] || status) + "</span>";
}

function priorityBadge(priority) {
  return (
    '<span class="badge priority-' + escapeHtml(priority) + '">优先级 ' +
    escapeHtml(PRIORITY_LABELS[priority] || priority) + "</span>"
  );
}

function renderTicketStats(counts) {
  const items = [
    { key: "", label: "全部工单", value: counts.total || 0 },
    { key: "pending", label: "待处理", value: counts.pending || 0 },
    { key: "processing", label: "处理中", value: counts.processing || 0 },
    { key: "waiting_customer", label: "待客户确认", value: counts.waiting_customer || 0 },
    { key: "closed", label: "已关闭", value: counts.closed || 0 },
  ];
  const active = el("status-filter").value;
  el("stat-row").innerHTML = items.map(function (item) {
    return '<button type="button" class="stat ' + (active === item.key ? "active" : "") + '" data-status="' + item.key +
      '"><span class="stat-label"><i class="stat-dot" aria-hidden="true"></i>' + item.label +
      '</span><strong class="value">' + item.value + "</strong></button>";
  }).join("");
}

function syncTicketStatActive(status) {
  const row = el("stat-row");
  if (!row) return;
  row.querySelectorAll(".stat").forEach(function (node) {
    node.classList.toggle("active", (node.dataset.status || "") === (status || ""));
  });
}

function ticketProgressHtml(status) {
  const steps = [
    { key: "pending", label: "待处理" },
    { key: "processing", label: "处理中" },
    { key: "waiting_customer", label: "待确认" },
    { key: "closed", label: "已关闭" },
  ];
  const currentIdx = Math.max(0, steps.findIndex(function (s) { return s.key === status; }));
  return '<ol class="ticket-stepper" aria-label="处理进度">' + steps.map(function (step, idx) {
    const cls = idx < currentIdx ? "is-done" : idx === currentIdx ? "is-current" : "";
    return '<li class="' + cls + '"><span class="step-num">' + (idx < currentIdx ? "✓" : (idx + 1)) +
      '</span><span class="step-text">' + step.label + "</span></li>";
  }).join("") + "</ol>";
}

function setupTicketConversation(detail, dialog, user, refresh) {
  dialog.innerHTML = (
    '<div class="ticket-chat-shell">' +
    '<header class="ticket-chat-header"><div><h2 id="ticket-chat-title">工单对话</h2>' +
    '<p class="ticket-chat-subtitle"></p></div>' +
    '<div class="ticket-chat-controls">' +
    '<button type="button" class="btn secondary sm" data-chat-refresh>刷新消息</button>' +
    '<button type="button" class="btn ghost sm" data-chat-close aria-label="关闭工单对话">关闭</button>' +
    '</div></header>' +
    '<div class="ticket-chat-log" aria-label="沟通记录" tabindex="0"></div>' +
    '<form class="ticket-chat-compose">' +
    '<label for="ticket-chat-message">回复</label>' +
    '<textarea class="textarea" id="ticket-chat-message" rows="2" maxlength="5000" required></textarea>' +
    '<div class="ticket-chat-tools">' +
    '<label class="ticket-chat-file">添加附件<input type="file" aria-label="添加附件"></label>' +
    '<div class="ticket-chat-controls"><button type="button" class="btn secondary" data-ai-draft hidden>AI 起草</button>' +
    '<button type="submit" class="btn">发送回复</button></div>' +
    '</div>' +
    '<section class="ticket-chat-draft" aria-label="AI 回复草稿" hidden></section>' +
    '</form>' +
    '<p class="ticket-chat-feedback" role="status" hidden></p>' +
    '</div>'
  );

  const log = dialog.querySelector(".ticket-chat-log");
  const form = dialog.querySelector("form");
  const input = form.querySelector("textarea");
  const fileInput = form.querySelector('input[type="file"]');
  const send = form.querySelector('button[type="submit"]');
  const feedback = dialog.querySelector(".ticket-chat-feedback");
  const refreshButton = dialog.querySelector("[data-chat-refresh]");
  const closeButton = dialog.querySelector("[data-chat-close]");
  const draftButton = dialog.querySelector("[data-ai-draft]");
  const preview = dialog.querySelector(".ticket-chat-draft");
  let draftText = "";
  let ticket;
  let busy = false;

  function announce(message, isError = false) {
    feedback.textContent = message;
    feedback.hidden = !message;
    feedback.classList.toggle("is-error", isError);
  }

  function syncButtons() {
    send.disabled = busy;
    fileInput.disabled = busy;
    refreshButton.disabled = busy;
    draftButton.disabled = busy;
  }

  function latest() {
    log.scrollTop = log.scrollHeight;
  }

  function update(next) {
    const changed = ticket?.id !== next.id;
    const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
    const scroll = log.scrollTop;
    ticket = next;
    if (changed) {
      dialog.close();
      input.value = "";
      fileInput.value = "";
      preview.hidden = true;
      draftText = "";
      announce("");
    }

    dialog.querySelector(".ticket-chat-subtitle").textContent =
      [ticket.ticket_no, STATUS_LABELS[ticket.status], ticket.title].filter(Boolean).join(" · ");
    form.hidden = ticket.status === "closed" && user?.role !== "admin";
    draftButton.hidden = ticket.status === "closed" || !["agent", "admin"].includes(user?.role);
    if (draftButton.hidden) preview.hidden = true;
    input.placeholder = user?.role === "customer"
      ? "补充问题细节，或告诉顾问你试过哪些步骤…"
      : "给客户回复排查步骤、处理方案或需要补充的信息…";

    let day = "";
    log.innerHTML = (ticket.messages || []).map(function (message) {
      const isMine = message.author_id === user?.user_id;
      const role = ROLE_LABELS[message.author_role] || "系统";
      const time = fmtTime(message.created_at);
      const hasDate = /^\d{4}-\d{2}-\d{2} /.test(time);
      const newDay = hasDate ? time.slice(0, 10) : "";
      const divider = newDay && day !== newDay
        ? '<div class="ticket-chat-date">' + escapeHtml(newDay) + '</div>' : "";
      if (newDay) day = newDay;
      const label = (isMine ? "我" : role) + (message.message_type === "supplement" ? " · 补充信息" : "");
      return divider + '<div class="ticket-chat-message' + (isMine ? " is-mine" : "") + '">' +
        '<div class="ticket-chat-avatar" aria-hidden="true">' + escapeHtml(role.slice(0, 1)) + '</div>' +
        '<div class="ticket-chat-message-body"><div class="ticket-chat-meta">' + escapeHtml(label) +
        ' <time>' + escapeHtml(hasDate ? time.slice(11) : time) + '</time></div>' +
        '<div class="ticket-chat-bubble">' + escapeHtml(message.content) + '</div></div></div>';
    }).join("") || '<p class="ticket-chat-empty">暂无沟通记录</p>';

    if (changed || atBottom) latest();
    else log.scrollTop = scroll;
  }

  async function perform(work) {
    if (!ticket || busy) return;
    busy = true;
    syncButtons();
    announce("");
    const id = ticket.id;
    try {
      await work(id);
    } catch (error) {
      if (ticket?.id === id) announce(error.message, true);
    } finally {
      busy = false;
      syncButtons();
    }
  }

  detail.addEventListener("click", function (event) {
    if (!event.target.closest("[data-open-conversation]") || !ticket) return;
    dialog.showModal();
    latest();
  });
  closeButton.addEventListener("click", () => dialog.close());
  dialog.addEventListener("close", () => detail.querySelector("[data-open-conversation]")?.focus());
  refreshButton.addEventListener("click", () => perform(async function (id) {
    await refresh(false);
    if (ticket?.id === id) announce("已刷新消息");
  }));
  draftButton.addEventListener("click", () => perform(async function (id) {
    announce("正在查找资料并准备草稿…");
    const result = await api("/tickets/" + encodeURIComponent(id) + "/reply-draft", { method: "POST" });
    if (ticket?.id !== id) return;
    announce("");
    const citations = result.citations || [];
    preview.hidden = false;
    draftText = "";
    if (!result.draft) {
      preview.textContent = result.message || "当前没有足够的资料，请手动回复。";
      return;
    }
    const formatCitation = (item) => "[" + item.index + "] " + item.title + (item.section ? " · " + item.section : "");
    draftText = result.draft + (citations.length ? "\n\n参考资料\n" + citations.map(formatCitation).join("\n") : "");
    preview.innerHTML = '<div class="ticket-chat-draft-head"><strong>' +
      (result.answer_mode === "llm" ? "AI 回复草稿" : "资料摘录草稿") +
      '</strong><button type="button" class="btn ghost sm" data-apply-draft>填入回复</button></div>' +
      '<p class="ticket-chat-draft-body">' + escapeHtml(result.draft) + '</p>' +
      '<ul>' + citations.map(item => '<li>' + escapeHtml(formatCitation(item)) +
        (item.snippet ? '<span>' + escapeHtml(item.snippet) + '</span>' : '') + '</li>').join("") + '</ul>';
  }));
  preview.addEventListener("click", function (event) {
    if (!event.target.closest("[data-apply-draft]") || !draftText) return;
    const next = [input.value.trim(), draftText].filter(Boolean).join("\n\n");
    if (next.length > input.maxLength) {
      announce("回复超过 5000 字，请精简后再使用草稿。", true);
      return;
    }
    input.value = next;
    preview.hidden = true;
    announce("草稿已填入，请核对后发送。");
    input.focus();
  });
  form.addEventListener("submit", function (event) {
    event.preventDefault();
    const content = input.value.trim();
    if (!content) {
      announce("请输入回复内容", true);
      return;
    }
    perform(async function (id) {
      await api("/tickets/" + encodeURIComponent(id) + "/messages", {
        method: "POST", body: { content },
      });
      if (ticket?.id === id && input.value.trim() === content) input.value = "";
      let refreshed = true;
      try {
        await refresh(true);
      } catch (_error) {
        refreshed = false;
      }
      if (ticket?.id === id && dialog.open) {
        latest();
        announce(refreshed ? "回复已发送" : "回复已发送。请刷新消息查看最新记录。", !refreshed);
        input.focus();
      }
    });
  });
  fileInput.addEventListener("change", function () {
    const file = fileInput.files[0];
    if (!file) return;
    fileInput.value = "";
    perform(async function (id) {
      const data = new FormData();
      data.append("file", file);
      await api("/tickets/" + encodeURIComponent(id) + "/attachments", { method: "POST", body: data });
      await refresh(true);
      if (ticket?.id === id && dialog.open) announce("已添加附件 " + file.name);
    });
  });
  return { update };
}

function ticketConversationEntry(ticket) {
  const messages = ticket.messages || [];
  const count = messages.length;
  const last = count ? messages[count - 1] : null;
  const preview = last
    ? '<div class="ticket-chat-preview"><b>' + escapeHtml(ROLE_LABELS[last.author_role] || "系统") +
      "</b>：" + escapeHtml((last.content || "").replace(/\s+/g, " ").slice(0, 72)) +
      ((last.content || "").length > 72 ? "…" : "") + "</div>"
    : '<div class="ticket-chat-preview is-empty">暂无沟通记录，点击右侧按钮补充说明或与顾问对话。</div>';
  return '<div class="ticket-chat-entry"><div class="ticket-chat-entry-info">' +
    '<div class="ticket-chat-entry-head"><strong>工单沟通</strong>' +
    '<span class="count-pill">' + (count ? count + " 条消息" : "暂无消息") + "</span></div>" +
    preview + "</div>" +
    '<button type="button" class="btn secondary" data-open-conversation aria-haspopup="dialog">' +
    (ticket.status === "closed" ? "查看对话" : "打开聊天") + "</button></div>";
}


function _formatEventNote(note) {
  if (!note) return "";
  return String(note)
    .replace(/[，,]?\s*(?:draft_version|request_id)=[^，,\s]*/gi, "")
    .replace(/[，,\s]+$/g, "")
    .trim();
}


function ticketEventsHtml(events) {
  if (!events || !events.length) return '<div class="muted small">暂无流转记录</div>';
  return '<ul class="event-list">' + events.map(function (item) {
    const from = item.from_status ? (STATUS_LABELS[item.from_status] || item.from_status) : "";
    const to = item.to_status ? (STATUS_LABELS[item.to_status] || item.to_status) : "";
    const flow = from && to && from !== to ? from + " → " + to : (to || "记录");
    const note = _formatEventNote(item.note || item.action || "");
    return '<li><div class="event-row"><b>' + escapeHtml(flow) + "</b>" +
      '<time class="muted">' + escapeHtml(fmtTime(item.created_at)) + "</time></div>" +
      (note ? '<div class="event-note">' + escapeHtml(note) + "</div>" : "") + "</li>";
  }).join("") + "</ul>";
}

function ticketAttachmentsHtml(ticket) {
  if (!ticket.attachments || !ticket.attachments.length) return '<div class="muted small">暂无附件</div>';
  return '<div class="attach-list">' + ticket.attachments.map(function (file) {
    return '<div class="attach-item"><span class="name">' + escapeHtml(file.filename) + "</span>" +
      '<span class="muted small">' + Math.ceil((file.size || 0) / 1024) + " KB</span>" +
      '<button class="btn ghost sm" data-download="' + escapeHtml(file.id) + '">下载</button></div>';
  }).join("") + "</div>";
}

function categoryLabel(category) {
  return CATEGORY_LABELS[category] || category || "其他";
}

function toast(message, type) {
  let wrap = document.querySelector(".toast-wrap");
  if (!wrap) {
    wrap = document.createElement("div");
    wrap.className = "toast-wrap";
    document.body.appendChild(wrap);
  }
  const item = document.createElement("div");
  item.className = "toast" + (type ? " " + type : "");
  item.textContent = message;
  wrap.appendChild(item);
  setTimeout(() => item.remove(), 3600);
}

function renderNav(active, user) {
  const header = document.getElementById("app-header");
  if (!header) return;
  const links = [];
  if (user.role === "customer" || user.role === "admin" || user.role === "agent") {
    links.push({ href: "/chat", label: "智能问答", key: "chat" });
  }
  if (user.role === "customer") {
    links.push({ href: "/tickets", label: "我的工单", key: "tickets" });
  }
  if (user.role === "agent" || user.role === "admin") {
    links.push({ href: "/agent", label: "工单处理", key: "agent" });
  }
  if (user.role === "admin") {
    links.push({ href: "/admin", label: "管理后台", key: "admin" });
  }
  header.innerHTML =
    '<a class="brand" href="' + homeFor(user.role) + '">' +
      '<span class="brand-mark">云</span><span>云账科技 · 财税费控支持</span>' +
    "</a>" +
    '<nav class="nav-links">' +
      links.map((item) =>
        '<a href="' + item.href + '" class="' + (active === item.key ? "active" : "") + '">' + item.label + "</a>"
      ).join("") +
    "</nav>" +
    '<div class="user-box">' +
      '<div class="user-meta">' +
        '<div class="user-name">' + escapeHtml(user.display_name || user.username) + "</div>" +
        '<div class="user-role">' + escapeHtml(ROLE_LABELS[user.role] || user.role) +
          (user.tenant_name ? " · " + escapeHtml(user.tenant_name) : "") + "</div>" +
      "</div>" +
      '<button class="btn secondary sm" id="logout-btn">退出</button>' +
    "</div>";
  const logoutBtn = document.getElementById("logout-btn");
  if (logoutBtn) {
    logoutBtn.addEventListener("click", () => {
      clearAuth();
      location.href = "/login";
    });
  }
}

function queryParam(name) {
  return new URLSearchParams(location.search).get(name);
}

function el(id) {
  return document.getElementById(id);
}
