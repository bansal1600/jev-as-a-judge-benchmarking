"""Tools the triage agent can call, plus the tiny fake order database behind them."""
from langchain_core.tools import tool

# Valid taxonomy. The agent must pick from these exact strings.
CATEGORIES = [
    "damaged_goods",
    "late_delivery",
    "billing_issue",
    "return_request",
    "product_question",
    "account_access",
]

TEAMS = ["refund_team", "logistics_team", "billing_team", "tier2_support", "none"]

ORDER_DB = {
    "4471": {"status": "delivered", "date": "2026-09-18", "item": "ceramic vase", "total": "$89.00"},
    "5502": {"status": "in_transit", "date": "est. 2026-09-25", "item": "standing desk", "total": "$412.50"},
    "3310": {"status": "delivered", "date": "2026-08-02", "item": "wireless headphones", "total": "$149.99"},
    "6120": {"status": "cancelled", "date": "2026-09-10", "item": "running shoes", "total": "$76.00"},
}


@tool
def lookup_order(order_id: str) -> str:
    """Look up a customer order by its ID. Returns status, date, item and total."""
    o = ORDER_DB.get(order_id.strip().lstrip("#"))
    if not o:
        return f"No order found with ID {order_id}."
    return (f"Order {order_id}: status={o['status']}, date={o['date']}, "
            f"item={o['item']}, total={o['total']}")


@tool
def categorize(category: str) -> str:
    """Record the category of this support ticket.

    Must be exactly one of: damaged_goods, late_delivery, billing_issue,
    return_request, product_question, account_access.
    """
    c = category.strip().lower()
    if c not in CATEGORIES:
        return f"Invalid category '{category}'. Must be one of: {', '.join(CATEGORIES)}"
    return f"Ticket categorized as {c}."


@tool
def escalate(team: str) -> str:
    """Escalate this ticket to a specialist team.

    Must be exactly one of: refund_team, logistics_team, billing_team,
    tier2_support, none. Use 'none' if no escalation is needed.
    """
    t = team.strip().lower()
    if t not in TEAMS:
        return f"Invalid team '{team}'. Must be one of: {', '.join(TEAMS)}"
    if t == "none":
        return "No escalation required; handled at tier 1."
    return f"Escalated to {t}."


ALL_TOOLS = [lookup_order, categorize, escalate]
