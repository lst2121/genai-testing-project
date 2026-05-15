"""LangGraph-style agent workflow for Day 2.

This file will evolve into a small graph:
classifier -> direct answer/tool/guardrail -> final response.
"""

from __future__ import annotations

import json
import os
import re
from importlib import import_module
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from .tools import TOOL_FUNCTIONS


Intent = Literal[
    "direct_answer",
    "calculation",
    "customer_lookup",
    "ticket_creation",
    "knowledge_search",
    "blocked",
]

VALID_INTENTS = {
    "direct_answer",
    "calculation",
    "customer_lookup",
    "ticket_creation",
    "knowledge_search",
    "blocked",
}


class AgentState(TypedDict, total=False):
    """Shared state passed between graph nodes.

    This mirrors how LangGraph works: each node reads and updates the state.
    Day 3 tests can assert state transitions and selected paths.
    """

    user_input: str
    intent: Intent
    tool_name: str | None
    tool_args: dict[str, Any]
    tool_result: dict[str, Any] | None
    final_answer: str
    errors: list[str]
    path: list[str]


@dataclass
class GraphRunResult:
    """Serializable result returned by the mini graph agent."""

    user_input: str
    intent: str
    path: list[str]
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_result: dict[str, Any] | None = None
    final_answer: str = ""
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_input": self.user_input,
            "intent": self.intent,
            "path": self.path,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "tool_result": self.tool_result,
            "final_answer": self.final_answer,
            "errors": self.errors,
        }


BLOCKED_PATTERNS = (
    "ignore previous instructions",
    "delete all users",
    "admin password",
    "bypass permissions",
    "call restricted tool",
)


def classify_intent_deterministic(user_input: str) -> Intent:
    """Deterministic classifier used for practice and Day 3 stable tests."""

    text = user_input.lower()

    if any(pattern in text for pattern in BLOCKED_PATTERNS):
        return "blocked"

    if re.search(r"\d+\s*[\+\-\*\/]\s*\d+", text):
        return "calculation"

    if re.search(r"cust-\d+", text):
        return "customer_lookup"

    if "ticket" in text or "bug" in text or "issue" in text:
        return "ticket_creation"

    if "policy" in text or "reset" in text or "knowledge" in text:
        return "knowledge_search"

    return "direct_answer"


def classify_intent_openai(user_input: str, model: str = "gpt-4o-mini") -> Intent:
    """Optional live classifier using OpenAI.

    This keeps API use isolated. If no API key is present or the call fails,
    the deterministic classifier is used as a fallback.
    """

    if not os.getenv("OPENAI_API_KEY"):
        return classify_intent_deterministic(user_input)

    try:
        OpenAI = import_module("openai").OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the user request into exactly one intent: "
                        "direct_answer, calculation, customer_lookup, "
                        "ticket_creation, knowledge_search, blocked. "
                        "Return only the intent string."
                    ),
                },
                {"role": "user", "content": user_input},
            ],
        )
        intent = response.choices[0].message.content.strip()
        if intent in VALID_INTENTS:
            return intent  # type: ignore[return-value]
    except Exception:
        pass

    return classify_intent_deterministic(user_input)


def classifier_node(state: AgentState, use_openai: bool = False) -> AgentState:
    intent = (
        classify_intent_openai(state["user_input"])
        if use_openai
        else classify_intent_deterministic(state["user_input"])
    )
    state["intent"] = intent
    state.setdefault("path", []).append("classifier")
    return state


def route_from_intent(state: AgentState) -> str:
    """Conditional edge function: chooses next node based on state."""

    intent = state["intent"]
    if intent == "blocked":
        return "guardrail"
    if intent == "direct_answer":
        return "direct_answer"
    return "tool_call"


def _extract_calculation_args(user_input: str) -> dict[str, str]:
    match = re.search(r"(\d+(?:\.\d+)?\s*[\+\-\*\/]\s*\d+(?:\.\d+)?)", user_input)
    return {"expression": match.group(1)} if match else {"expression": user_input}


def _extract_customer_args(user_input: str) -> dict[str, str]:
    match = re.search(r"(CUST-\d+)", user_input, flags=re.IGNORECASE)
    return {"customer_id": match.group(1).upper()} if match else {"customer_id": ""}


def _extract_ticket_args(user_input: str) -> dict[str, str]:
    priority = "high" if "high" in user_input.lower() else "medium"
    title = user_input.strip().rstrip(".")
    return {"title": title, "priority": priority}


def _extract_knowledge_args(user_input: str) -> dict[str, str]:
    return {"query": user_input}


def tool_call_node(state: AgentState) -> AgentState:
    """Select and execute a tool based on classified intent."""

    intent = state["intent"]
    tool_name: str
    tool_args: dict[str, Any]

    if intent == "calculation":
        tool_name = "calculator"
        tool_args = _extract_calculation_args(state["user_input"])
    elif intent == "customer_lookup":
        tool_name = "customer_lookup"
        tool_args = _extract_customer_args(state["user_input"])
    elif intent == "ticket_creation":
        tool_name = "ticket_creation"
        tool_args = _extract_ticket_args(state["user_input"])
    elif intent == "knowledge_search":
        tool_name = "knowledge_search"
        tool_args = _extract_knowledge_args(state["user_input"])
    else:
        raise ValueError(f"No tool mapping for intent: {intent}")

    state["tool_name"] = tool_name
    state["tool_args"] = tool_args
    state["tool_result"] = TOOL_FUNCTIONS[tool_name](**tool_args)
    state.setdefault("path", []).append("tool_call")
    return state


def direct_answer_node(state: AgentState) -> AgentState:
    state["final_answer"] = (
        "This looks like a simple informational question. "
        "No external tool was needed for this demo."
    )
    state["tool_name"] = None
    state["tool_args"] = {}
    state["tool_result"] = None
    state.setdefault("path", []).append("direct_answer")
    return state


def guardrail_node(state: AgentState) -> AgentState:
    state["final_answer"] = (
        "I cannot help with that request because it appears unsafe or restricted."
    )
    state["tool_name"] = None
    state["tool_args"] = {}
    state["tool_result"] = None
    state.setdefault("path", []).append("guardrail")
    return state


def final_response_node(state: AgentState) -> AgentState:
    """Synthesize final answer from state and tool result."""

    tool_result = state.get("tool_result")
    if tool_result is None:
        state.setdefault("path", []).append("final_response")
        return state

    if not tool_result["success"]:
        state["final_answer"] = (
            f"I could not complete the request because {tool_result['tool_name']} "
            f"failed: {tool_result['error']}."
        )
    elif state["tool_name"] == "calculator":
        state["final_answer"] = (
            f"The result of {tool_result['data']['expression']} is "
            f"{tool_result['data']['result']}."
        )
    elif state["tool_name"] == "customer_lookup":
        data = tool_result["data"]
        state["final_answer"] = (
            f"Customer {data['customer_id']} is {data['name']} with "
            f"{data['plan']} plan and status {data['status']}."
        )
    elif state["tool_name"] == "ticket_creation":
        data = tool_result["data"]
        state["final_answer"] = (
            f"Created ticket {data['ticket_id']} with {data['priority']} priority."
        )
    elif state["tool_name"] == "knowledge_search":
        state["final_answer"] = tool_result["data"]["context"]
    else:
        state["final_answer"] = "Tool completed successfully."

    state.setdefault("path", []).append("final_response")
    return state


def run_graph_agent(user_input: str, use_openai: bool = False) -> GraphRunResult:
    """Run the mini graph agent.

    This is deliberately written in a LangGraph-like style even before wiring the
    actual StateGraph object, so the control flow remains easy to read and test.
    """

    state: AgentState = {
        "user_input": user_input,
        "tool_args": {},
        "tool_result": None,
        "errors": [],
        "path": [],
    }

    state = classifier_node(state, use_openai=use_openai)
    route = route_from_intent(state)

    if route == "guardrail":
        state = guardrail_node(state)
    elif route == "direct_answer":
        state = direct_answer_node(state)
    else:
        try:
            state = tool_call_node(state)
        except Exception as error:
            state.setdefault("errors", []).append(str(error))
            state["final_answer"] = f"Tool execution failed: {error}"
            state.setdefault("path", []).append("tool_error")

    state = final_response_node(state)

    return GraphRunResult(
        user_input=state["user_input"],
        intent=state["intent"],
        path=state["path"],
        tool_name=state.get("tool_name"),
        tool_args=state.get("tool_args", {}),
        tool_result=state.get("tool_result"),
        final_answer=state.get("final_answer", ""),
        errors=state.get("errors", []),
    )


def print_graph_result(user_input: str, use_openai: bool = False) -> None:
    """Helper for notebook/manual practice."""

    result = run_graph_agent(user_input, use_openai=use_openai)
    print(json.dumps(result.to_dict(), indent=2))
