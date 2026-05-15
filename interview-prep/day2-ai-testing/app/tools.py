"""Tool implementations for the mini agent workbench.

These start as small deterministic tools so Day 3 tests can assert tool
selection, arguments, failures, and final-answer grounding.
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ToolResult:
    """Standard response shape returned by every tool."""

    success: bool
    tool_name: str
    data: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "tool_name": self.tool_name,
            "data": self.data,
            "error": self.error,
        }


ALLOWED_OPERATORS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

ALLOWED_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval_expression(expression: str) -> int | float:
    """Evaluate simple arithmetic without using unsafe eval()."""

    def evaluate(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)

        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:
            left = evaluate(node.left)
            right = evaluate(node.right)
            return ALLOWED_OPERATORS[type(node.op)](left, right)

        if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_UNARY_OPERATORS:
            operand = evaluate(node.operand)
            return ALLOWED_UNARY_OPERATORS[type(node.op)](operand)

        raise ValueError(f"Unsupported expression: {expression}")

    parsed = ast.parse(expression, mode="eval")
    return evaluate(parsed)


def calculator_tool(expression: str) -> dict[str, Any]:
    """Calculate a simple arithmetic expression.

    This deterministic tool is useful for testing whether an agent selected
    the right tool and passed the right argument.
    """

    try:
        result = _safe_eval_expression(expression)
        return ToolResult(
            success=True,
            tool_name="calculator",
            data={"expression": expression, "result": result},
        ).to_dict()
    except Exception as error:
        return ToolResult(
            success=False,
            tool_name="calculator",
            data={"expression": expression},
            error=str(error),
        ).to_dict()


CUSTOMERS = {
    "CUST-101": {"name": "Amit Sharma", "plan": "Premium", "status": "active"},
    "CUST-202": {"name": "Neha Singh", "plan": "Basic", "status": "inactive"},
}


def customer_lookup_tool(customer_id: str) -> dict[str, Any]:
    """Look up a customer from a small deterministic in-memory data source."""

    customer = CUSTOMERS.get(customer_id)
    if not customer:
        return ToolResult(
            success=False,
            tool_name="customer_lookup",
            data={"customer_id": customer_id},
            error="Customer not found",
        ).to_dict()

    return ToolResult(
        success=True,
        tool_name="customer_lookup",
        data={"customer_id": customer_id, **customer},
    ).to_dict()


def ticket_creation_tool(title: str, priority: str = "medium") -> dict[str, Any]:
    """Create a deterministic mock ticket."""

    allowed_priorities = {"low", "medium", "high", "critical"}
    if priority not in allowed_priorities:
        return ToolResult(
            success=False,
            tool_name="ticket_creation",
            data={"title": title, "priority": priority},
            error=f"Unsupported priority: {priority}",
        ).to_dict()

    ticket_id = f"TCK-{abs(hash((title, priority))) % 10000:04d}"
    return ToolResult(
        success=True,
        tool_name="ticket_creation",
        data={"ticket_id": ticket_id, "title": title, "priority": priority},
    ).to_dict()


def knowledge_search_tool(query: str) -> dict[str, Any]:
    """Return a deterministic knowledge snippet for simple RAG-style testing."""

    snippets = {
        "password reset": "Users can reset passwords from Settings > Security.",
        "refund policy": "Refunds are processed within 7 business days.",
        "agent tools": "Agents should only call tools allowed by their role.",
    }

    query_lower = query.lower()
    for keyword, answer in snippets.items():
        if keyword in query_lower:
            return ToolResult(
                success=True,
                tool_name="knowledge_search",
                data={"query": query, "context": answer},
            ).to_dict()

    return ToolResult(
        success=False,
        tool_name="knowledge_search",
        data={"query": query},
        error="No matching knowledge snippet found",
    ).to_dict()


TOOL_FUNCTIONS: dict[str, Callable[..., dict[str, Any]]] = {
    "calculator": calculator_tool,
    "customer_lookup": customer_lookup_tool,
    "ticket_creation": ticket_creation_tool,
    "knowledge_search": knowledge_search_tool,
}
