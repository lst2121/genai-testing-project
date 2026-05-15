"""
================================================================================
DAY 3 - BLOCK 3: TOOL-CALLING CORRECTNESS
================================================================================
Goal:
Test whether a real OpenAI tool-calling agent selects the correct tool, produces
valid arguments, handles tool errors, and does not hallucinate tool results.

Why this matters for the DTDL JD:
The JD mentions agents, tools, prompt/SOP authoring, tool registration, AI
workbench testing, and eval workflows. Tool-calling correctness is one of the
most important quality gates for an agent platform.
================================================================================
"""


# =============================================================================
# CONCEPT 1: HOW BLOCK 3 DIFFERS FROM BLOCK 2
# =============================================================================
"""
Block 2 tested our deterministic graph path:

    classifier -> route -> tool_call/direct_answer/guardrail -> final_response

In Block 2, the app decided intent and tool mapping using deterministic rules.

Block 3 tests real OpenAI function calling:

    user input
        -> OpenAI chooses tool name
        -> OpenAI generates tool arguments
        -> app validates arguments using JSON Schema
        -> app executes local Python tool
        -> OpenAI receives tool result
        -> OpenAI writes final answer

The implementation is:

    interview-prep/day2-ai-testing/app/simple_agent.py

Main function:

    run_openai_tool_agent(...)

Important:
    This block intentionally uses a live model for some checks.
    These tests are useful for eval suites, not normal PR gates.
"""


# =============================================================================
# CONCEPT 2: WHAT IS TOOL-CALLING CORRECTNESS?
# =============================================================================
"""
Tool-calling correctness has multiple layers.

1. Tool selection correctness
    Did the model choose the right tool?

    Example:
        "What is 15 * 7?"
        Expected tool: calculator


2. Argument correctness
    Did the model pass the correct structured arguments?

    Example:
        Expected:
            {"expression": "15 * 7"}


3. Schema correctness
    Did the generated arguments satisfy the tool schema?

    Example:
        calculator requires:
            expression: string


4. Tool execution correctness
    Did our application execute the selected tool correctly?

    Example:
        calculator returns:
            result = 105


5. Final response grounding
    Did the final answer use the tool result instead of inventing a result?

    Example:
        Good:
            "The result of 15 * 7 is 105."

        Bad:
            "The result is 100."  # hallucinated or ignored tool output


6. Error handling correctness
    If the tool fails, did the model explain the failure instead of guessing?

    Example:
        Input:
            "What is 15 * hjhj?"

        Tool result:
            success=False, error="Unsupported expression"

        Expected final answer:
            explain unsupported expression, not invent a number.
"""


# =============================================================================
# CONCEPT 3: RESULT SHAPE TO TEST
# =============================================================================
"""
run_openai_tool_agent(...) returns OpenAIToolAgentResult.

Important fields:

    result.user_input
    result.model
    result.first_model_message
    result.tool_calls
    result.final_answer
    result.prompt_tokens
    result.completion_tokens
    result.total_tokens
    result.errors

Each tool call record contains:

    tool_call.tool_call_id
    tool_call.tool_name
    tool_call.arguments
    tool_call.validation_error
    tool_call.result

In tests, validate:

    - len(result.tool_calls)
    - result.tool_calls[0].tool_name
    - result.tool_calls[0].arguments
    - result.tool_calls[0].validation_error
    - result.tool_calls[0].result["success"]
    - result.final_answer
    - result.total_tokens > 0
"""


# =============================================================================
# CONCEPT 4: HANDS-ON SCENARIOS
# =============================================================================
"""
Run these manually first in the notebook.

Setup:

    from pathlib import Path
    import sys

    project_root = Path.cwd()
    while project_root.name != "genai-testing-project":
        project_root = project_root.parent

    day2_root = project_root / "interview-prep" / "day2-ai-testing"
    sys.path.append(str(day2_root))

    from app.simple_agent import run_openai_tool_agent


1. Positive calculator

    result = run_openai_tool_agent("What is 15 * 7?")
    result.to_dict()

    Expected:
        one tool call
        tool_name == "calculator"
        arguments["expression"] == "15 * 7"
        tool result success True
        result is 105
        final answer contains 105


2. Negative calculator

    result = run_openai_tool_agent("What is 15 * hjhj?")
    result.to_dict()

    Expected:
        tool_name == "calculator"
        tool result success False
        final answer should explain the error
        final answer should not invent a numeric result


3. Customer lookup

    result = run_openai_tool_agent("Find customer CUST-101")
    result.to_dict()

    Expected:
        tool_name == "customer_lookup"
        arguments["customer_id"] == "CUST-101"
        customer name is Amit Sharma


4. Unknown customer

    result = run_openai_tool_agent("Find customer CUST-999")
    result.to_dict()

    Expected:
        tool_name == "customer_lookup"
        tool result success False
        error is Customer not found
        final answer should not invent a customer


5. Ticket creation

    result = run_openai_tool_agent(
        "Create a high priority ticket for checkout failure"
    )
    result.to_dict()

    Expected:
        tool_name == "ticket_creation"
        arguments include title
        priority == "high"
        final answer includes ticket id


6. No-tool direct answer

    result = run_openai_tool_agent("Say hello in one short sentence")
    result.to_dict()

    Expected:
        tool_calls == []
        final_answer is not empty

Note:
    Since this uses a live model, minor wording differences are acceptable.
    Assertions should focus on tool name, arguments, success/error, and grounding.
"""


# =============================================================================
# CONCEPT 5: PYTEST STRATEGY
# =============================================================================
"""
Do not make all live OpenAI tests mandatory PR gates.

Recommended split:

1. Deterministic PR tests
    - tool registry tests
    - graph path tests with use_openai=False
    - schema validation tests

2. Live LLM eval tests
    - tool selection correctness
    - argument correctness
    - refusal/over-blocking checks
    - hallucination after tool failure
    - latency/token checks

In pytest, mark live tests:

    @pytest.mark.live_llm

Run live tests only when needed:

    python -m pytest -m live_llm ...

Or skip unless an environment variable is set:

    pytestmark = pytest.mark.skipif(
        os.getenv("RUN_LIVE_LLM_TESTS") != "true",
        reason="Set RUN_LIVE_LLM_TESTS=true to run live OpenAI tests",
    )

This prevents accidental cost and flaky CI failures.
"""


# =============================================================================
# CONCEPT 6: EXAMPLE PYTEST CASES
# =============================================================================
"""
Example live test shape:

    import os
    import pytest

    pytestmark = pytest.mark.skipif(
        os.getenv("RUN_LIVE_LLM_TESTS") != "true",
        reason="Live LLM tests are opt-in",
    )

    def test_openai_selects_calculator_tool():
        result = run_openai_tool_agent("What is 15 * 7?")

        assert len(result.tool_calls) == 1
        tool_call = result.tool_calls[0]
        assert tool_call.tool_name == "calculator"
        assert tool_call.arguments["expression"] == "15 * 7"
        assert tool_call.result["success"] is True
        assert tool_call.result["data"]["result"] == 105
        assert "105" in result.final_answer


Negative tool-failure example:

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


Unknown customer example:

    def test_openai_does_not_invent_unknown_customer():
        result = run_openai_tool_agent("Find customer CUST-999")

        tool_call = result.tool_calls[0]
        assert tool_call.tool_name == "customer_lookup"
        assert tool_call.arguments["customer_id"] == "CUST-999"
        assert tool_call.result["success"] is False
        assert "customer not found" in tool_call.result["error"].lower()
        assert "premium" not in result.final_answer.lower()
"""


# =============================================================================
# CONCEPT 7: WHAT BUGS THESE TESTS CATCH
# =============================================================================
"""
Tool-calling tests catch:

1. Wrong tool selection
    User asks calculation, model calls knowledge_search.

2. Wrong argument generation
    User asks CUST-101, model sends CUST-10.

3. Schema violation
    Model sends unsupported extra fields or wrong types.

4. Tool result hallucination
    Tool returns error but final answer invents success.

5. Silent tool avoidance
    Model answers directly when it should call a tool.

6. Unsafe tool use
    Model calls a tool for a prompt that should be refused.

7. Cost/latency regression
    Token usage or response time becomes too high.
"""


# =============================================================================
# CHECKPOINT QUESTIONS
# =============================================================================
"""
1. What is the difference between tool selection and tool execution?

Tool selection is model-level behavior. The LLM decides whether a tool is needed
and which tool name to call. Tool execution is application-level behavior. The
app validates the generated arguments and runs the actual Python/API/backend
tool.


2. Why is tool argument validation important?

Because the LLM output is not automatically trustworthy. Even if the model
selects the correct tool, it may produce missing fields, wrong types, invalid
enum values, or extra fields. Schema validation protects the application before
tool execution.


3. What should you assert in a live tool-calling test?

Assert the selected tool name, generated arguments, schema validation result,
tool execution result, final answer grounding, error handling, and token usage.
Avoid asserting exact final wording unless the system requires exact text.


4. Why should live tool-calling tests be marked separately?

Because live model tests cost money, depend on network/API availability, and may
be non-deterministic. They are valuable eval tests, but they should not block
every PR unless the team intentionally opts in.


5. How would you explain this block in an interview?

Interview answer:

    "For tool-calling correctness, I test the full contract between the model
    and the application. First I check whether the model selected the correct
    tool. Then I validate generated arguments against the schema. Then I verify
    tool execution and check whether the final answer is grounded in the tool
    result. For failure cases, I verify that the agent explains the tool error
    instead of hallucinating a successful answer. I keep these live LLM tests in
    an opt-in eval suite, while deterministic registry and graph tests run in PR
    CI."
"""
