"""OpenAI tool-calling agent for the mini workbench.

This module is intentionally separate from ``graph_agent.py``:

- ``graph_agent.py`` gives deterministic graph behavior for CI-friendly tests.
- This file uses real OpenAI function calling so you can see actual model tool
  selection, argument generation, tool execution, and final response synthesis.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

from .mcp_schemas import MCP_TOOL_SCHEMAS, OPENAI_TOOL_SCHEMAS
from .tools import TOOL_FUNCTIONS


@dataclass
class ToolCallRecord:
    """Trace record for one model-requested tool call."""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    validation_error: str | None = None
    result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "validation_error": self.validation_error,
            "result": self.result,
        }


@dataclass
class OpenAIToolAgentResult:
    """Serializable result of one OpenAI tool-calling run."""

    user_input: str
    model: str
    first_model_message: str | None = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    final_answer: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_input": self.user_input,
            "model": self.model,
            "first_model_message": self.first_model_message,
            "tool_calls": [tool_call.to_dict() for tool_call in self.tool_calls],
            "final_answer": self.final_answer,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            },
            "errors": self.errors,
        }


def _load_env(env_path: str | None = None) -> None:
    """Load OPENAI_API_KEY from .env if python-dotenv is installed."""

    try:
        load_dotenv = import_module("dotenv").load_dotenv
    except Exception:
        return

    if env_path:
        load_dotenv(env_path)
        return

    project_root_env = Path(__file__).resolve().parents[3] / ".env"
    if project_root_env.exists():
        load_dotenv(project_root_env)


def _get_openai_client(env_path: str | None = None) -> Any:
    _load_env(env_path)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not available. Add it to .env or set it in the "
            "notebook environment before calling the live OpenAI agent."
        )

    OpenAI = import_module("openai").OpenAI
    return OpenAI()


def _add_usage(result: OpenAIToolAgentResult, response: Any) -> None:
    usage = getattr(response, "usage", None)
    if not usage:
        return

    result.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
    result.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
    result.total_tokens += getattr(usage, "total_tokens", 0) or 0


def _parse_tool_arguments(raw_arguments: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as error:
        raise ValueError(f"Tool arguments are not valid JSON: {error}") from error

    if not isinstance(parsed, dict):
        raise ValueError("Tool arguments must be a JSON object")

    return parsed


def _validate_tool_arguments(tool_name: str, arguments: dict[str, Any]) -> None:
    schema = MCP_TOOL_SCHEMAS[tool_name]["inputSchema"]
    jsonschema = import_module("jsonschema")
    jsonschema.validate(instance=arguments, schema=schema)


def _execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    trace_tools: bool = False,
) -> dict[str, Any]:
    if tool_name not in TOOL_FUNCTIONS:
        return {
            "success": False,
            "tool_name": tool_name,
            "data": arguments,
            "error": f"Unknown tool: {tool_name}",
        }

    if trace_tools:
        from .tracing import traced_execute_tool

        return traced_execute_tool(tool_name, arguments)

    return TOOL_FUNCTIONS[tool_name](**arguments)


def run_openai_tool_agent(
    user_input: str,
    model: str = "gpt-4o-mini",
    env_path: str | None = None,
    trace_tools: bool = False,
) -> OpenAIToolAgentResult:
    """Run a real OpenAI tool-calling flow.

    Flow:
        1. Send user input + tool schemas to OpenAI.
        2. Let the model choose tool name and arguments.
        3. Validate arguments against the MCP-style schema.
        4. Execute local Python tool.
        5. Send tool result back to OpenAI for final answer synthesis.

    This is the live counterpart to the deterministic graph agent.
    """

    client = _get_openai_client(env_path)
    result = OpenAIToolAgentResult(user_input=user_input, model=model)

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a tool-using support workbench agent. Use tools when "
                "needed. Do not invent tool results. If a tool returns an error, "
                "explain the error instead of guessing."
            ),
        },
        {"role": "user", "content": user_input},
    ]

    first_response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=messages,
        tools=OPENAI_TOOL_SCHEMAS,
        tool_choice="auto",
    )
    _add_usage(result, first_response)

    message = first_response.choices[0].message
    result.first_model_message = message.content
    tool_calls = message.tool_calls or []

    if not tool_calls:
        result.final_answer = message.content or ""
        return result

    messages.append(message.model_dump())

    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        record = ToolCallRecord(
            tool_call_id=tool_call.id,
            tool_name=tool_name,
            arguments={},
        )

        try:
            arguments = _parse_tool_arguments(tool_call.function.arguments)
            record.arguments = arguments
            _validate_tool_arguments(tool_name, arguments)
            record.result = _execute_tool(
                tool_name,
                arguments,
                trace_tools=trace_tools,
            )
        except Exception as error:
            record.validation_error = str(error)
            record.result = {
                "success": False,
                "tool_name": tool_name,
                "data": record.arguments,
                "error": str(error),
            }

        result.tool_calls.append(record)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(record.result),
            }
        )

    final_response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=messages,
    )
    _add_usage(result, final_response)
    result.final_answer = final_response.choices[0].message.content or ""

    return result
