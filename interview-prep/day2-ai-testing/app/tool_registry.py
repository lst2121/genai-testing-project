"""Tool registry for the mini agent workbench.

This will model how an AI workbench stores tool metadata, schemas,
enabled/disabled state, and execution permissions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable

from .mcp_schemas import MCP_TOOL_SCHEMAS
from .tools import TOOL_FUNCTIONS


class ToolRegistryError(Exception):
    """Base exception for registry failures."""


class ToolAlreadyExistsError(ToolRegistryError):
    """Raised when registering a duplicate tool."""


class ToolNotFoundError(ToolRegistryError):
    """Raised when a requested tool is not registered."""


class ToolDisabledError(ToolRegistryError):
    """Raised when a disabled tool is invoked."""


class ToolPermissionError(ToolRegistryError):
    """Raised when a role is not allowed to invoke a tool."""


class ToolSchemaError(ToolRegistryError):
    """Raised when a schema or argument validation fails."""


@dataclass
class RegisteredTool:
    """Tool metadata stored by the workbench registry."""

    name: str
    description: str
    input_schema: dict[str, Any]
    function: Callable[..., dict[str, Any]]
    enabled: bool = True
    allowed_roles: set[str] = field(default_factory=lambda: {"admin", "tester"})

    def to_descriptor(self) -> dict[str, Any]:
        """Return MCP-style descriptor exposed to clients."""

        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "enabled": self.enabled,
            "allowed_roles": sorted(self.allowed_roles),
        }


class ToolRegistry:
    """In-memory MCP-style tool registry for practice and Day 3 tests."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register_tool(
        self,
        schema: dict[str, Any],
        function: Callable[..., dict[str, Any]],
        *,
        allowed_roles: set[str] | None = None,
        enabled: bool = True,
    ) -> RegisteredTool:
        """Register a tool after validating descriptor shape."""

        self._validate_tool_descriptor(schema)
        name = schema["name"]

        if name in self._tools:
            raise ToolAlreadyExistsError(f"Tool already registered: {name}")

        tool = RegisteredTool(
            name=name,
            description=schema["description"],
            input_schema=schema["inputSchema"],
            function=function,
            enabled=enabled,
            allowed_roles=allowed_roles or {"admin", "tester"},
        )
        self._tools[name] = tool
        return tool

    def update_tool(
        self,
        name: str,
        *,
        description: str | None = None,
        input_schema: dict[str, Any] | None = None,
        allowed_roles: set[str] | None = None,
    ) -> RegisteredTool:
        """Update metadata for a registered tool."""

        tool = self.get_tool(name)
        if description is not None:
            tool.description = description
        if input_schema is not None:
            self._validate_input_schema(input_schema)
            tool.input_schema = input_schema
        if allowed_roles is not None:
            tool.allowed_roles = allowed_roles
        return tool

    def delete_tool(self, name: str) -> None:
        """Delete a registered tool."""

        if name not in self._tools:
            raise ToolNotFoundError(f"Tool not registered: {name}")
        del self._tools[name]

    def enable_tool(self, name: str) -> None:
        self.get_tool(name).enabled = True

    def disable_tool(self, name: str) -> None:
        self.get_tool(name).enabled = False

    def get_tool(self, name: str) -> RegisteredTool:
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool not registered: {name}")
        return self._tools[name]

    def list_tools(
        self,
        *,
        role: str | None = None,
        include_disabled: bool = False,
    ) -> list[dict[str, Any]]:
        """List tool descriptors discoverable by a role."""

        descriptors = []
        for tool in self._tools.values():
            if not include_disabled and not tool.enabled:
                continue
            if role is not None and role not in tool.allowed_roles:
                continue
            descriptors.append(tool.to_descriptor())
        return descriptors

    def execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        role: str = "tester",
    ) -> dict[str, Any]:
        """Validate permissions/schema, then execute the registered function."""

        tool = self.get_tool(name)
        if not tool.enabled:
            raise ToolDisabledError(f"Tool is disabled: {name}")
        if role not in tool.allowed_roles:
            raise ToolPermissionError(f"Role '{role}' cannot execute tool '{name}'")

        self.validate_arguments(name, arguments)
        return tool.function(**arguments)

    def validate_arguments(self, name: str, arguments: dict[str, Any]) -> None:
        """Validate tool arguments using JSON Schema."""

        tool = self.get_tool(name)
        try:
            jsonschema = import_module("jsonschema")
            jsonschema.validate(instance=arguments, schema=tool.input_schema)
        except Exception as error:
            raise ToolSchemaError(str(error)) from error

    def _validate_tool_descriptor(self, schema: dict[str, Any]) -> None:
        required = {"name", "description", "inputSchema"}
        missing = required - set(schema)
        if missing:
            raise ToolSchemaError(f"Tool descriptor missing required fields: {missing}")

        if not isinstance(schema["name"], str) or not schema["name"]:
            raise ToolSchemaError("Tool name must be a non-empty string")
        if not isinstance(schema["description"], str) or not schema["description"]:
            raise ToolSchemaError("Tool description must be a non-empty string")
        self._validate_input_schema(schema["inputSchema"])

    def _validate_input_schema(self, input_schema: dict[str, Any]) -> None:
        if not isinstance(input_schema, dict):
            raise ToolSchemaError("inputSchema must be an object")
        if input_schema.get("type") != "object":
            raise ToolSchemaError("inputSchema.type must be 'object'")
        if "properties" not in input_schema:
            raise ToolSchemaError("inputSchema must contain properties")


def build_default_registry() -> ToolRegistry:
    """Create the default registry used in notebooks and Day 3 tests."""

    registry = ToolRegistry()
    for tool_name, schema in MCP_TOOL_SCHEMAS.items():
        registry.register_tool(
            schema,
            TOOL_FUNCTIONS[tool_name],
            allowed_roles={"admin", "tester"},
        )
    return registry
