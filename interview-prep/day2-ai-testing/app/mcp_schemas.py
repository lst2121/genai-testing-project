"""MCP-style tool schemas for the mini agent workbench.

These schemas model the kind of tool-registration contracts an AI workbench
must validate before allowing agents to call tools.
"""

from __future__ import annotations

from typing import Any


CALCULATOR_SCHEMA: dict[str, Any] = {
    "name": "calculator",
    "description": "Calculate a simple arithmetic expression.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression, for example '15 * 7'.",
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
}


CUSTOMER_LOOKUP_SCHEMA: dict[str, Any] = {
    "name": "customer_lookup",
    "description": "Look up customer details by customer id.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "customer_id": {
                "type": "string",
                "description": "Customer id, for example 'CUST-101'.",
            }
        },
        "required": ["customer_id"],
        "additionalProperties": False,
    },
}


TICKET_CREATION_SCHEMA: dict[str, Any] = {
    "name": "ticket_creation",
    "description": "Create a support ticket.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short issue title."},
            "priority": {
                "type": "string",
                "enum": ["low", "medium", "high", "critical"],
                "description": "Ticket priority.",
            },
        },
        "required": ["title"],
        "additionalProperties": False,
    },
}


KNOWLEDGE_SEARCH_SCHEMA: dict[str, Any] = {
    "name": "knowledge_search",
    "description": "Search a small deterministic knowledge base.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "User query or search phrase.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


MCP_TOOL_SCHEMAS = {
    "calculator": CALCULATOR_SCHEMA,
    "customer_lookup": CUSTOMER_LOOKUP_SCHEMA,
    "ticket_creation": TICKET_CREATION_SCHEMA,
    "knowledge_search": KNOWLEDGE_SEARCH_SCHEMA,
}


def mcp_schema_to_openai_tool(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert an MCP-style tool schema to OpenAI function-calling format."""

    return {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema["description"],
            "parameters": schema["inputSchema"],
        },
    }


OPENAI_TOOL_SCHEMAS = [
    mcp_schema_to_openai_tool(schema) for schema in MCP_TOOL_SCHEMAS.values()
]
