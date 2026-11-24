# tests/test_auth_permissions.py
# 覆盖：登录正误、无效/过期 Token、禁用账号、管理员接口鉴权、工单跨租户/跨用户越权。

from __future__ import annotations

from sqlalchemy import select

from app.db import AsyncSessionLocal
from app.models import Ticket
from app.security import create_access_token
from tests.helpers import auth, get_user, login, token_for


async def _ticket_ids() -> dict[str, str]:
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(select(Ticket))).scalars().all()
    return {ticket.title: ticket.id for ticket in rows}


async def test_login_success_and_wrong_password(client):
    """1. 正确密码可登录；错误密码返回 401。"""
    response = await login(client, "customer_a", "Customer@123456")
    assert response.status_code == 200
    body = response.json()
    assert body["access_token"]
    assert body["role"] == "customer"
    assert body["tenant_name"] == "华辰智造"

    wrong = await login(client, "customer_a", "wrong-password")
    assert wrong.status_code == 401


async def test_invalid_and_expired_token(client):
    """2. 伪造 Token 与过期 Token 都被拒绝。"""
    invalid = await client.get("/api/tickets", headers=auth("not.a.valid.token"))
    assert invalid.status_code == 401

    user = await get_user("customer_a")
    expired = create_access_token(user.id, expires_minutes=-5)
    expired_resp = await client.get("/api/tickets", headers=auth(expired))
    assert expired_resp.status_code == 401


async def test_disabled_account_rejected_immediately(client):
    """3. 禁用账号不能登录；已签发 Token 也立即失效。"""
    login_resp = await login(client, "customer_disabled", "Disabled@123456")
    assert login_resp.status_code == 403

    disabled = await get_user("customer_disabled")
    token = create_access_token(disabled.id)
    response = await client.get("/api/tickets", headers=auth(token))
    assert response.status_code == 403


async def test_admin_endpoints_require_admin(client):
    """4. 客户/客服访问管理接口均被后端拒绝，管理员可访问。"""
    customer_token = await token_for("customer_a")
    agent_token = await token_for("agent1")
    admin_token = await token_for("admin")

    assert (await client.get("/api/admin/users", headers=auth(customer_token))).status_code == 403
    assert (await client.get("/api/admin/users", headers=auth(agent_token))).status_code == 403
    assert (await client.get("/api/admin/users", headers=auth(admin_token))).status_code == 200


async def test_customer_only_sees_own_tickets(client):
    """5. 客户不能查看同租户其他客户、或其他租户的工单。"""
    tickets = await _ticket_ids()
    ticket_a = tickets["v2.2 报销单保存草稿后部门可用预算被冻结"]
    ticket_b = tickets["Excel 导入费用科目提示编码重复"]

    customer_a = await token_for("customer_a")
    customer_a2 = await token_for("customer_a2")

    assert (await client.get(f"/api/tickets/{ticket_a}", headers=auth(customer_a))).status_code == 200
    # 同租户的另一客户也不能看
    assert (await client.get(f"/api/tickets/{ticket_a}", headers=auth(customer_a2))).status_code == 404
    # 跨租户统一 404，避免枚举
    assert (await client.get(f"/api/tickets/{ticket_b}", headers=auth(customer_a))).status_code == 404


async def test_agent_scope_limited_to_grants(client):
    """6. 客服只能查看被授权租户的工单。"""
    tickets = await _ticket_ids()
    ticket_a = tickets["v2.2 报销单保存草稿后部门可用预算被冻结"]
    ticket_b = tickets["Excel 导入费用科目提示编码重复"]

    agent1 = await token_for("agent1")
    agent2 = await token_for("agent2")

    assert (await client.get(f"/api/tickets/{ticket_a}", headers=auth(agent1))).status_code == 200
    assert (await client.get(f"/api/tickets/{ticket_b}", headers=auth(agent1))).status_code == 403
    assert (await client.get(f"/api/tickets/{ticket_b}", headers=auth(agent2))).status_code == 200
    assert (await client.get(f"/api/tickets/{ticket_a}", headers=auth(agent2))).status_code == 403
