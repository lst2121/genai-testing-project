"""Lightweight trace model for agent runs.

Day 2 uses this to understand trace shape; Day 3 uses it to test expected
agent trajectory, tool calls, latency, and failure handling.
"""

from __future__ import annotations

from contextlib import nullcontext
from importlib import import_module
from typing import Any, Callable

from .tools import TOOL_FUNCTIONS


def _get_traceable() -> Callable[..., Callable[[Callable[..., Any]], Callable[..., Any]]]:
    """Return LangSmith's traceable decorator, or a no-op fallback."""

    try:
        return import_module("langsmith").traceable
    except Exception:
        def no_op_traceable(*_args: Any, **_kwargs: Any) -> Callable:
            def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
                return func

            return decorator

        return no_op_traceable


traceable = _get_traceable()


@traceable(name="execute_registered_tool", run_type="tool")
def traced_execute_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a registered tool with a real LangSmith tool span."""

    if tool_name not in TOOL_FUNCTIONS:
        return {
            "success": False,
            "tool_name": tool_name,
            "data": arguments,
            "error": f"Unknown tool: {tool_name}",
        }

    return TOOL_FUNCTIONS[tool_name](**arguments)


@traceable(name="openai_tool_agent_run", run_type="chain")
def traced_openai_tool_agent_run(
    user_input: str,
    model: str = "gpt-4o-mini",
    env_path: str | None = None,
) -> dict[str, Any]:
    """Run the OpenAI tool agent inside a LangSmith trace."""

    from .simple_agent import run_openai_tool_agent

    result = run_openai_tool_agent(
        user_input,
        model=model,
        env_path=env_path,
        trace_tools=True,
    )
    return result.to_dict()


def langsmith_context(
    *,
    project_name: str = "day2-agent-workbench",
    enabled: bool = True,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Create a LangSmith tracing context, or no-op if langsmith is unavailable."""

    try:
        tracing_context = import_module("langsmith").tracing_context
    except Exception:
        return nullcontext()

    return tracing_context(
        enabled=enabled,
        project_name=project_name,
        tags=tags or ["day2", "agent-workbench"],
        metadata=metadata or {},
    )


def run_with_langsmith(
    user_input: str,
    *,
    model: str = "gpt-4o-mini",
    project_name: str = "day2-agent-workbench",
    env_path: str | None = None,
) -> dict[str, Any]:
    """Convenience wrapper for notebooks.

    This sends real traces to LangSmith when LANGSMITH_API_KEY is configured.
    """

    with langsmith_context(
        project_name=project_name,
        metadata={"model": model, "source": "day2-block6"},
    ):
        return traced_openai_tool_agent_run(
            user_input,
            model=model,
            env_path=env_path,
        )
