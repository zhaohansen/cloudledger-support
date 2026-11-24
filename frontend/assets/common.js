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
    { key: "", label: "全部", value: counts.total || 0 },
    { key: "pending", label: "待处理", value: counts.pending || 0 },
    { key: "processing", label: "处理中", value: counts.processing || 0 },
    { key: "waiting_customer", label: "待客户确认", value: counts.waiting_customer || 0 },
    { key: "closed", label: "已关闭", value: counts.closed || 0 },
  ];
  const active = el("status-filter").value;
  el("stat-row").innerHTML = items.map(function (item) {
    return '<div class="stat ' + (active === item.key ? "active" : "") + '" data-status="' + item.key +
      '"><div class="label">' + item.label + '</div><div class="value">' + item.value + "</div></div>";
  }).join("");
}

function ticketEventsHtml(events) {
  if (!events || !events.length) return '<div class="muted small">暂无流转记录</div>';
  return '<ul class="event-list">' + events.map(function (item) {
    const from = item.from_status ? (STATUS_LABELS[item.from_status] || item.from_status) : "";
    const to = item.to_status ? (STATUS_LABELS[item.to_status] || item.to_status) : "";
    const flow = from && to && from !== to ? from + " → " + to : (to || "记录");
    return "<li><b>" + escapeHtml(flow) + "</b>　" + escapeHtml(item.note || item.action || "") +
      '<div class="muted">' + escapeHtml(fmtTime(item.created_at)) + "</div></li>";
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
