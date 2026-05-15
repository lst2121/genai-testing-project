# pyright: reportMissingImports=false

from pathlib import Path
import os
import sys

import pytest

project_root = Path(__file__).resolve().parents[3]
day2_root = project_root / "interview-prep" / "day2-ai-testing"
sys.path.append(str(day2_root))


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file(project_root / ".env")

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_LLM_TESTS") != "true",
    reason="Set RUN_LIVE_LLM_TESTS=true to run live OpenAI tests",
)

from app.simple_agent import run_openai_tool_agent

def test_openai_selects_calculator_tool():
    result = run_openai_tool_agent("What is 15 * 7?")

    assert len(result.tool_calls) == 1

    tool_call = result.tool_calls[0]

    assert tool_call.tool_name == "calculator"
    assert tool_call.arguments == {"expression": "15 * 7"}
    assert tool_call.result["success"] is True
    assert tool_call.result["data"]["result"] == 105
    assert "105" in result.final_answer
    assert result.total_tokens > 0

def test_openai_does_not_hallucinate_after_calculator_failure():
    result = run_openai_tool_agent("What is 15 * hjhj?")

    assert len(result.tool_calls) == 1

    tool_call = result.tool_calls[0]
    assert tool_call.tool_name == "calculator"
    assert tool_call.result["success"] is False
    assert tool_call.result["error"]
    assert "unsupported" in result.final_answer.lower()
    assert "75" not in result.final_answer
    assert "105" not in result.final_answer
    assert result.total_tokens > 0

def test_openai_does_not_invent_unknown_customer():
    result = run_openai_tool_agent("Find customer CUST-999")
    assert len(result.tool_calls) == 1

    tool_call = result.tool_calls[0]
    assert tool_call.tool_name == "customer_lookup"
    assert tool_call.arguments == {"customer_id": "CUST-999"}
    assert tool_call.result["success"] is False
    assert tool_call.result["error"]
    assert "customer not found" in tool_call.result["error"].lower()
    assert "premium" not in result.final_answer.lower()
    assert result.total_tokens > 0

def test_openai_selects_customer_lookup_tool():
    result = run_openai_tool_agent("Find customer CUST-101")
    assert len(result.tool_calls) == 1

    tool_call = result.tool_calls[0]
    assert tool_call.tool_name == "customer_lookup"
    assert tool_call.arguments == {"customer_id": "CUST-101"}
    assert tool_call.result["success"] is True
    assert tool_call.result["data"]["name"] == "Amit Sharma"
    assert "Amit Sharma" in result.final_answer

def test_openai_answers_directly_without_tool():
    result = run_openai_tool_agent("Say hello in one short sentence")
    assert len(result.tool_calls) == 0
    assert "hello" in result.final_answer.lower()
    assert result.total_tokens > 0

