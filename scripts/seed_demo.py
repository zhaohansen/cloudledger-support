# scripts/seed_demo.py
# 导入演示数据：3 个租户（平台方 + 两家企业租户）、7 个演示账号、
# 13 份财税费控与数电发票产品资料、2 张演示工单。
# 仅用于本地环境初始化与回归验证，演示口令禁止用于生产环境。
#
# 用法：python scripts/seed_demo.py

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import select  # noqa: E402

from app.db import AsyncSessionLocal, init_db  # noqa: E402
from app.deps import UserContext  # noqa: E402
from app.models import AgentTenantGrant, Tenant, Ticket, User  # noqa: E402
from app.security import hash_password  # noqa: E402
from app.services.knowledge import (  # noqa: E402
    DuplicateDocumentError,
    ingest_document,
)
from app.services.tickets import (  # noqa: E402
    add_message,
    create_ticket_from_draft,
    get_or_create_draft,
    update_status,
)

DEMO_PASSWORD_ADMIN = "Admin@123456"
DEMO_PASSWORD_AGENT = "Agent@123456"
DEMO_PASSWORD_CUSTOMER = "Customer@123456"
DEMO_PASSWORD_DISABLED = "Disabled@123456"

DOCS_DIR = PROJECT_ROOT / "demo_docs"

# (文件名, 标题, 类型, 资料版本, 适用产品版本, 可见范围, 租户 code)
DOCUMENT_PLAN = [
    ("01-数电发票池与验真规范.md", "数电发票池与验真规范", "manual", "1.0", "all", "public", None),
    ("02-差旅与日常费用报销手册.md", "差旅与日常费用报销手册", "manual", "1.0", "all", "public", None),
    ("03-银企直连付款与退票处理手册.md", "银企直连付款与退票处理手册", "manual", "1.0", "all", "public", None),
    ("04-预算编制与额度控制手册-v2.2.md", "预算编制与额度控制手册（v2.2）", "manual", "2.2", "v2.2", "public", None),
    ("05-预算编制与额度控制手册-v2.3.md", "预算编制与额度控制手册（v2.3）", "manual", "2.3", "v2.3", "public", None),
    ("06-Excel批量导入常见问题.md", "Excel 批量导入常见问题", "faq", "1.0", "all", "public", None),
    ("07-角色权限与职责分离规范.md", "角色权限与职责分离规范", "manual", "1.0", "all", "public", None),
    ("08-常见财务异常排查手册.md", "常见财务异常排查手册", "faq", "1.0", "all", "public", None),
    ("09-版本更新说明-v2.3.md", "版本更新说明 v2.3", "release", "2.3", "v2.3", "public", None),
    ("10-版本更新说明-v2.2.md", "版本更新说明 v2.2", "release", "2.2", "v2.2", "public", None),
    ("11-华辰智造费控内控制度.md", "华辰智造费控内控制度", "private", "1.0", "all", "tenant", "huachen"),
    ("12-星海文化报销管理细则.md", "星海文化报销管理细则", "private", "1.0", "all", "tenant", "xinghai"),
    ("13-合规引用与数据安全规范.md", "合规引用与数据安全规范", "faq", "1.0", "all", "public", None),
]


async def _get_or_create_tenant(db, *, name: str, code: str, is_platform: bool = False) -> Tenant:
    tenant = await db.scalar(select(Tenant).where(Tenant.code == code))
    if tenant:
        return tenant
    tenant = Tenant(name=name, code=code, is_platform=is_platform)
    db.add(tenant)
    await db.flush()
    return tenant


async def _get_or_create_user(
    db,
    *,
    tenant: Tenant,
    username: str,
    display_name: str,
    role: str,
    password: str,
    is_active: bool = True,
) -> User:
    user = await db.scalar(select(User).where(User.username == username))
    if user:
        return user
    user = User(
        tenant_id=tenant.id,
        username=username,
        display_name=display_name,
        email=f"{username}@example.invalid",
        password_hash=hash_password(password),
        role=role,
        is_active=is_active,
    )
    db.add(user)
    await db.flush()
    return user


async def _ensure_grant(db, agent_id: str, tenant_id: str) -> None:
    exists = await db.scalar(
        select(AgentTenantGrant).where(
            AgentTenantGrant.agent_user_id == agent_id,
            AgentTenantGrant.tenant_id == tenant_id,
        )
    )
    if not exists:
        db.add(AgentTenantGrant(agent_user_id=agent_id, tenant_id=tenant_id))
        await db.flush()


def _ctx(user: User, tenant: Tenant) -> UserContext:
    return UserContext(
        user_id=user.id,
        username=user.username,
        display_name=user.display_name,
        role=user.role,
        tenant_id=tenant.id,
        tenant_name=tenant.name,
    )


async def _seed_documents(db, admin_ctx: UserContext, tenant_by_code: dict[str, Tenant]) -> int:
    imported = 0
    for filename, title, doc_type, version, product_version, scope, tenant_code in DOCUMENT_PLAN:
        path = DOCS_DIR / filename
        if not path.exists():
            print(f"[seed] 跳过缺失资料：{filename}")
            continue
        content = path.read_bytes()
        try:
            document = await ingest_document(
                db,
                admin_ctx,
                title=title,
                filename=filename,
                content=content,
                doc_type=doc_type,
                version=version,
                product_version=product_version,
                access_scope=scope,
                tenant_id=tenant_by_code[tenant_code].id if tenant_code else None,
            )
            imported += 1
            print(f"[seed] 资料已导入：{title}（{document.chunk_count} 块，状态 {document.status}）")
        except DuplicateDocumentError:
            print(f"[seed] 资料已存在，跳过：{title}")
    return imported


async def _create_demo_ticket(
    db,
    user_ctx: UserContext,
    *,
    title: str,
    description: str,
    product_version: str,
    error_message: str,
    tried_steps: str,
    category: str,
    priority: str,
) -> Ticket | None:
    exists = await db.scalar(
        select(Ticket).where(Ticket.created_by == user_ctx.user_id, Ticket.title == title)
    )
    if exists:
        print(f"[seed] 工单已存在，跳过：{exists.ticket_no}")
        return exists

    draft = await get_or_create_draft(
        db,
        user_ctx,
        session_id=None,
        title=title,
        description=description,
        product_version=product_version,
        error_message=error_message,
        tried_steps=tried_steps,
        category=category,
        priority=priority,
    )
    # 演示数据直接按草稿当前版本创建，模拟“用户看到了这一版并确认”
    ticket, already = await create_ticket_from_draft(db, user_ctx, draft.id, draft.version)
    print(f"[seed] 工单已创建：{ticket.ticket_no}（重复确认={already}）")
    return ticket


async def main() -> None:
    await init_db()
    async with AsyncSessionLocal() as db:
        platform = await _get_or_create_tenant(
            db, name="云账科技（平台方）", code="cloudledger", is_platform=True
        )
        tenant_a = await _get_or_create_tenant(db, name="华辰智造", code="huachen")
        tenant_b = await _get_or_create_tenant(db, name="星海文化", code="xinghai")
        tenant_by_code = {"huachen": tenant_a, "xinghai": tenant_b}

        admin = await _get_or_create_user(
            db,
            tenant=platform,
            username="admin",
            display_name="平台管理员",
            role="admin",
            password=DEMO_PASSWORD_ADMIN,
        )
        agent1 = await _get_or_create_user(
            db,
            tenant=platform,
            username="agent1",
            display_name="财务顾问-张明（华辰智造）",
            role="agent",
            password=DEMO_PASSWORD_AGENT,
        )
        agent2 = await _get_or_create_user(
            db,
            tenant=platform,
            username="agent2",
            display_name="财务顾问-李静（星海文化）",
            role="agent",
            password=DEMO_PASSWORD_AGENT,
        )
        customer_a = await _get_or_create_user(
            db,
            tenant=tenant_a,
            username="customer_a",
            display_name="华辰智造-王会计",
            role="customer",
            password=DEMO_PASSWORD_CUSTOMER,
        )
        await _get_or_create_user(
            db,
            tenant=tenant_a,
            username="customer_a2",
            display_name="华辰智造-赵出纳",
            role="customer",
            password=DEMO_PASSWORD_CUSTOMER,
        )
        customer_b = await _get_or_create_user(
            db,
            tenant=tenant_b,
            username="customer_b",
            display_name="星海文化-陈财务",
            role="customer",
            password=DEMO_PASSWORD_CUSTOMER,
        )
        await _get_or_create_user(
            db,
            tenant=tenant_a,
            username="customer_disabled",
            display_name="华辰智造-离职账号",
            role="customer",
            password=DEMO_PASSWORD_DISABLED,
            is_active=False,
        )

        await _ensure_grant(db, agent1.id, tenant_a.id)
        await _ensure_grant(db, agent2.id, tenant_b.id)
        await db.commit()

        admin_ctx = _ctx(admin, platform)
        agent1_ctx = _ctx(agent1, platform)
        customer_a_ctx = _ctx(customer_a, tenant_a)
        customer_b_ctx = _ctx(customer_b, tenant_b)

        await _seed_documents(db, admin_ctx, tenant_by_code)
        await db.commit()

        ticket_a = await _create_demo_ticket(
            db,
            customer_a_ctx,
            title="v2.2 报销单保存草稿后部门可用预算被冻结",
            description="我们在 v2.2 新建差旅报销单后只保存了草稿，还没有提交审批，预算查询里的部门可用预算就减少了。希望确认原因和释放方式。",
            product_version="v2.2",
            error_message="没有报错弹窗，但部门可用预算额度变少",
            tried_steps="已核对报销单状态仍为草稿",
            category="budget",
            priority="high",
        )
        ticket_b = await _create_demo_ticket(
            db,
            customer_b_ctx,
            title="Excel 导入费用科目提示编码重复",
            description="星海文化导入传媒项目专属费用科目时，系统提示科目编码重复，但表格内肉眼未见重复行。",
            product_version="v2.2",
            error_message="导入失败：第 12 行科目编码重复",
            tried_steps="已重新导出模板并核对编码列",
            category="import",
            priority="medium",
        )
        await db.commit()

        # 演示顾问处理流程：华辰智造的工单推进到待客户确认
        if ticket_a and ticket_a.status == "pending":
            await update_status(db, agent1_ctx, ticket_a, "processing", note="财务顾问已接手")
            await add_message(
                db,
                agent1_ctx,
                ticket_a,
                "已确认：v2.2 保存报销草稿即预占冻结部门可用预算，这是该版本的既定控制规则。"
                "请先删除或撤回未提交草稿以释放额度；升级到 v2.3 后改为提交审批时才冻结。请确认是否解决。",
            )
            await update_status(
                db, agent1_ctx, ticket_a, "waiting_customer", note="已给出处理方案，等待客户确认"
            )
            await db.commit()
            print(f"[seed] 工单 {ticket_a.ticket_no} 已推进到待客户确认")

        if ticket_b:
            print(f"[seed] 工单 {ticket_b.ticket_no} 保持待处理，用于演示顾问授权筛选")

        print("\n[seed] 演示账号（仅本地环境使用，禁止用于生产）：")
        print(f"  管理员    admin            / {DEMO_PASSWORD_ADMIN}")
        print(f"  顾问      agent1（华辰智造）/ {DEMO_PASSWORD_AGENT}")
        print(f"  顾问      agent2（星海文化）/ {DEMO_PASSWORD_AGENT}")
        print(f"  客户      customer_a（华辰智造）/ {DEMO_PASSWORD_CUSTOMER}")
        print(f"  客户      customer_a2（华辰智造）/ {DEMO_PASSWORD_CUSTOMER}")
        print(f"  客户      customer_b（星海文化）/ {DEMO_PASSWORD_CUSTOMER}")
        print(f"  禁用账号  customer_disabled / {DEMO_PASSWORD_DISABLED}")
        print(f"\n[seed] 文档目录：{DOCS_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
