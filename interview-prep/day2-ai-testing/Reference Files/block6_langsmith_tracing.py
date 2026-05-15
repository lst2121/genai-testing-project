"""
================================================================================
BLOCK 6: LANGSMITH TRACING AND DEBUGGING
================================================================================
Goal:
Enable real LangSmith tracing for the Day 2 OpenAI tool-calling agent and learn
how to inspect traces from an SDET/debugging perspective.

This block covers:
1. Runs, traces, spans, datasets, evaluators
2. Debugging prompts, tool calls, latency, tokens, and errors
3. How to use trace evidence in agent testing
4. What Day 3 trace validation should check
================================================================================
"""


# =============================================================================
# CONCEPT 1: WHAT IS LANGSMITH?
# =============================================================================
"""
LangSmith is an observability and evaluation platform for LLM applications.

It helps answer:

    - What prompt was sent?
    - Which model was called?
    - Which tool was selected?
    - What arguments were passed?
    - What did the tool return?
    - Did the final answer use the tool result?
    - Where did latency/cost come from?
    - Which run failed and why?

For agent testing, LangSmith is valuable because final output alone is not
enough. We need to inspect the full trajectory.
"""


# =============================================================================
# CONCEPT 2: CORE TERMS
# =============================================================================
"""
Trace:
    Full execution record for one workflow/run.

Run:
    One recorded operation inside the trace.

Span:
    A child operation inside a trace, such as a tool call or LLM call.

Project:
    A workspace grouping related traces.

Dataset:
    A set of examples used for evaluation/regression testing.

Evaluator:
    A function or LLM judge that scores outputs against expected behavior.
"""


# =============================================================================
# CONCEPT 3: ENVIRONMENT SETUP
# =============================================================================
"""
Add these to your .env file:

    OPENAI_API_KEY=...
    LANGSMITH_API_KEY=...
    LANGSMITH_TRACING=true
    LANGSMITH_PROJECT=day2-agent-workbench

Optional:

    LANGSMITH_ENDPOINT=https://api.smith.langchain.com

Important:
    Never commit real API keys.
"""


# =============================================================================
# CONCEPT 4: APP CODE FOR REAL TRACING
# =============================================================================
"""
The tracing wrapper is implemented in:

    interview-prep/day2-ai-testing/app/tracing.py

It provides:

    traced_execute_tool()
        Creates a real LangSmith tool span.

    traced_openai_tool_agent_run()
        Creates a real LangSmith chain span around the full tool-calling run.

    run_with_langsmith()
        Convenience function for notebooks.

The OpenAI tool-calling agent remains in:

    interview-prep/day2-ai-testing/app/simple_agent.py

When tracing is enabled:

    User input
        -> traced chain span
        -> OpenAI tool-selection call
        -> traced tool span
        -> OpenAI final-answer call
        -> final result
"""


# =============================================================================
# PRACTICE SNIPPET
# =============================================================================
"""
Use this in the Block 6 notebook:

    from pathlib import Path
    import sys
    from dotenv import load_dotenv

    project_root = Path(r"C:\\Users\\Lokender.Singh\\genai-testing-project")
    day2_root = project_root / "interview-prep" / "day2-ai-testing"

    load_dotenv(project_root / ".env")
    sys.path.append(str(day2_root))

    from app.tracing import run_with_langsmith

    result = run_with_langsmith(
        "What is 15 * 7?",
        project_name="day2-agent-workbench"
    )

    result

Try a failure case:

    result = run_with_langsmith(
        "What is 15 * hjhj?",
        project_name="day2-agent-workbench"
    )

    result

Then open LangSmith and inspect the project:

    day2-agent-workbench
"""


# =============================================================================
# WHAT TO INSPECT IN LANGSMITH
# =============================================================================
"""
For each run, inspect:

1. Input
    Was the user query captured correctly?

2. Model call
    Which model was used?
    Was temperature set correctly?

3. Tool call
    Which tool was selected?
    Were arguments correct?

4. Tool output
    Did the tool succeed or fail?
    Was the error captured?

5. Final answer
    Was it grounded in the tool output?
    Did it hallucinate after failure?

6. Latency
    Which step was slow?

7. Token usage
    How many tokens were used?

8. Errors
    Did the trace capture exceptions or validation failures?
"""


# =============================================================================
# DEBUGGING EXAMPLES
# =============================================================================
"""
Case 1: Wrong final answer

    Check:
        - Was the correct tool selected?
        - Were arguments correct?
        - Did tool return correct data?
        - Did final model ignore the tool result?


Case 2: Tool not called

    Check:
        - Was tool schema available?
        - Did the prompt tell the model to use tools?
        - Did the model decide no tool was needed?


Case 3: Tool error but confident answer

    Check:
        - Tool span error
        - Final answer content
        - Groundedness against tool output


Case 4: High latency

    Check:
        - First model call latency
        - Tool execution latency
        - Final model call latency
        - Token count
"""


# =============================================================================
# TESTING ANGLE
# =============================================================================
"""
How LangSmith helps SDETs:

    - Gives evidence for agent trajectory.
    - Helps debug flaky LLM behavior.
    - Makes tool-call failures visible.
    - Supports comparison across prompt/model versions.
    - Provides data for eval datasets and regression checks.

Day 3 test ideas:

    - Verify every critical agent run produces a trace.
    - Verify trace contains expected tool call.
    - Verify failed tool calls are visible in trace.
    - Verify latency/token thresholds.
    - Use trace data to debug hallucination and wrong-tool bugs.
"""


# =============================================================================
# INTERVIEW ANSWER
# =============================================================================
"""
Question:
    How would you use LangSmith for agent testing?

Answer:
    I would use LangSmith to trace the complete agent execution path. For each
    run, I would inspect the input, prompt/model call, tool selection, tool
    arguments, tool output, final answer, errors, latency, and token usage.

    This helps identify whether a failure came from prompt design, model tool
    selection, argument generation, tool execution, state handling, or final
    response synthesis.

    For CI and evaluation workflows, I would use LangSmith projects/datasets to
    compare runs over time, detect regressions, and debug failures using actual
    trace evidence rather than only final output.
"""


# =============================================================================
# CHECKPOINT QUESTIONS
# =============================================================================
"""
Answer before moving to Block 7:

1. What is LangSmith used for?
2. What should you inspect in an agent trace?
3. How does tracing help debug hallucination?
4. How does tracing help debug wrong tool calls?
5. How would you use LangSmith in CI/eval workflows?
"""


# =============================================================================
# CHECKPOINT ANSWERS - INTERVIEW READY
# =============================================================================
"""
1. What is LangSmith used for?

    LangSmith is used for observability, tracing, debugging, and evaluation of
    LLM and agent workflows.

    It captures the full execution path:

        - user input
        - model calls
        - prompts/messages
        - tool calls
        - tool arguments
        - tool outputs
        - handoffs, if any
        - guardrail results
        - final answer
        - latency
        - token usage
        - errors

    In agent testing, LangSmith is useful because final output alone does not
    show why the agent behaved a certain way. The trace shows the trajectory.


2. What should you inspect in an agent trace?

    I would inspect:

    1. Input:
        Was the user query captured correctly?

    2. Model call:
        Which model was used?
        Was the prompt/instruction correct?

    3. Tool selection:
        Did the model choose the correct tool?

    4. Tool arguments:
        Were arguments complete and correct?

    5. Tool output:
        Did the tool succeed or fail?
        What exact data/error did it return?

    6. Final answer:
        Was the response grounded in the tool output?

    7. Usage:
        Token usage, latency, and errors.

    Example from our run:

        Parent trace:
            user_input = "What is 15 * 7?"
            tool_name = "calculator"
            final_answer = "The result of 15 * 7 is 105."
            total_tokens = 348

        Child tool span:
            tool_name = "calculator"
            arguments = {"expression": "15 * 7"}
            output = {"success": true, "result": 105}


3. How does tracing help debug hallucination?

    Tracing helps debug hallucination by showing whether the final answer is
    supported by the actual context or tool output.

    Example:

        Tool result:
            {"success": false, "error": "Customer not found"}

        Bad final answer:
            "Customer CUST-999 is Amit with Premium plan."

        Trace evidence:
            The tool never returned Amit or Premium, so the final answer is
            ungrounded and hallucinated.

    With LangSmith, I can compare the final answer against the tool span output
    and identify whether hallucination happened during final synthesis.


4. How does tracing help debug wrong tool calls?

    Tracing shows the exact tool selected and the arguments generated by the
    model.

    Example:

        User:
            "Find customer CUST-101"

        Expected:
            tool_name = "customer_lookup"
            arguments = {"customer_id": "CUST-101"}

        Trace shows:
            tool_name = "knowledge_search"

        This tells me the issue is tool selection, not tool execution.

    Another example:

        Trace shows:
            tool_name = "customer_lookup"
            arguments = {"customer_id": "101"}

        Then tool selection was correct, but argument generation was wrong.

    This separation is very important for debugging agent failures.


5. How would you use LangSmith in CI/eval workflows?

    I would use LangSmith in CI/eval workflows to track agent behavior over a
    fixed dataset of representative prompts.

    For each run, I would capture:

        - expected tool
        - actual tool
        - expected arguments
        - actual arguments
        - tool success/failure
        - final answer
        - latency
        - token usage
        - evaluator scores

    CI gates could check:

        - tool selection accuracy >= threshold
        - no unauthorized tool calls
        - no hallucinated answer after tool failure
        - latency below threshold
        - token usage below budget
        - prompt-injection cases blocked

    LangSmith traces provide evidence for failures, making it easier to debug
    whether the issue is prompt design, model behavior, tool schema, tool
    execution, or final response synthesis.


Interview summary:

    I used LangSmith to trace the full agent execution. The parent trace
    captured user input, model, final answer, token usage, and tool-call
    records. The child span captured actual tool execution with tool name,
    arguments, output, and success/error. This makes agent debugging much easier
    because I can see whether the issue came from tool selection, argument
    generation, tool execution, or final answer synthesis.
"""
