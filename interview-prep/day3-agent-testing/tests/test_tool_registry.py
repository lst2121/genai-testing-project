# pyright: reportMissingImports=false

import pytest
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[3]
day2_root = project_root / "interview-prep" / "day2-ai-testing"
sys.path.append(str(day2_root))

from app.tool_registry import (
    build_default_registry,
    ToolAlreadyExistsError,
    ToolDisabledError,
    ToolNotFoundError,
    ToolPermissionError,
    ToolSchemaError,
)
from app.mcp_schemas import MCP_TOOL_SCHEMAS
from app.tools import TOOL_FUNCTIONS

@pytest.fixture
def registry():
    return build_default_registry()

def test_execute_calculator_with_valid_args(registry):
    result = registry.execute_tool("calculator", {"expression": "15 * 7"}, role="tester")
    assert result["success"] is True
    assert result["data"]["result"] == 105

def test_admin_can_execute_customer_lookup(registry):
    result = registry.execute_tool(
        "customer_lookup",
        {"customer_id": "CUST-101"},
        role="admin",
    )

    assert result["success"] is True
    assert result["data"]["name"] == "Amit Sharma"
    assert result["data"]["plan"] == "Premium"
    assert result["data"]["status"] == "active"

def test_missing_required_argument_fails(registry):
    with pytest.raises(ToolSchemaError):
        registry.execute_tool("calculator", {}, role="tester")

def test_disabled_tool_cannot_execute(registry):
    registry.disable_tool("calculator")

    with pytest.raises(ToolDisabledError):
        registry.execute_tool("calculator", {"expression": "15 * 7"}, role="tester")

def test_viewer_cannot_execute_customer_lookup(registry):
    with pytest.raises(ToolPermissionError):
        registry.execute_tool("customer_lookup", {"customer_id": "CUST-101"}, role="viewer")

def test_unknown_tool_fails(registry):
    with pytest.raises(ToolNotFoundError):
        registry.execute_tool("delete_all_users", {}, role="admin")

def test_disabled_tool_hidden_from_discovery(registry):
    registry.disable_tool("calculator")
    tools = registry.list_tools(role="tester")
    assert "calculator" not in [tool["name"] for tool in tools]

def test_default_registry_lists_tools_for_tester(registry):
    tools = registry.list_tools(role="tester")
    tool_names = [tool["name"] for tool in tools]
    
    assert "calculator" in tool_names
    assert "customer_lookup" in tool_names
    assert "ticket_creation" in tool_names
    assert "knowledge_search" in tool_names

def test_duplicate_tool_registration_fails(registry):
    with pytest.raises(ToolAlreadyExistsError):
        registry.register_tool(MCP_TOOL_SCHEMAS["calculator"], TOOL_FUNCTIONS["calculator"])

def test_extra_argument_fails(registry):
    with pytest.raises(ToolSchemaError):
        registry.execute_tool("calculator", {"expression": "15 * 7", "extra": "bad"}, role="tester")

def test_disabled_tool_hidden_from_discovery(registry):
    registry.disable_tool("calculator")
    tools = registry.list_tools(role="tester")
    assert "calculator" not in [tool["name"] for tool in tools]
