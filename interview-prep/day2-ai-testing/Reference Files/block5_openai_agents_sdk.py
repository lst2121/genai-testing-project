"""
================================================================================
BLOCK 5: OPENAI AGENTS SDK OVERVIEW
================================================================================
Goal:
Understand OpenAI Agents SDK at interview level: what it provides, how it
differs from LangGraph/LangChain, and how an SDET should test an SDK-based
agent implementation.

This block covers:
1. Agents, instructions, tools, handoffs, guardrails, tracing
2. How OpenAI Agents SDK differs from LangGraph and LangChain
3. What an SDET should test in an Agents SDK based implementation
================================================================================
"""


# =============================================================================
# CONCEPT 1: WHAT IS OPENAI AGENTS SDK?
# =============================================================================
"""
OpenAI Agents SDK is a framework for building and running tool-using agents.

Core pieces:

    Agent:
        Defines name, instructions, tools, handoffs, guardrails, and output type.

    Runner:
        Executes the agent loop.

    Tools:
        Let the agent take actions.

    Handoffs:
        Let one agent delegate to another specialist agent.

    Guardrails:
        Validate inputs, outputs, or tool calls.

    Tracing:
        Captures model calls, tool calls, guardrails, handoffs, latency, and
        execution steps.

Simple mental model:

    User input
        -> Runner
        -> Agent
        -> LLM decides final answer / tool call / handoff
        -> Runner executes tools or handoff
        -> loop continues
        -> final output
"""


# =============================================================================
# CONCEPT 2: AGENT AND RUNNER
# =============================================================================
"""
Basic SDK shape:

    from agents import Agent, Runner

    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant."
    )

    result = Runner.run_sync(agent, "Write a haiku about testing.")
    print(result.final_output)

Runner execution loop:

    1. LLM produces output.
    2. If final output is ready, run ends.
    3. If LLM produces tool calls, Runner executes tools and loops again.
    4. If LLM produces handoff, Runner switches to the target agent and loops.

Testing angle:
    The Runner hides a lot of orchestration. As an SDET, I still need to inspect
    the trace/run result to validate tool calls, handoffs, guardrails, and final
    output.
"""


# =============================================================================
# CONCEPT 3: TOOLS IN OPENAI AGENTS SDK
# =============================================================================
"""
The SDK supports several tool categories:

1. Hosted OpenAI tools

    Examples:
        WebSearchTool
        FileSearchTool
        CodeInterpreterTool
        HostedMCPTool
        ImageGenerationTool

    These run alongside OpenAI-hosted model infrastructure.


2. Function tools

    Python functions exposed as tools.

    Example:

        from agents import function_tool

        @function_tool
        def calculator(expression: str) -> str:
            \"\"\"Calculate a simple arithmetic expression.\"\"\"
            return str(eval(expression))


3. Agents as tools

    One agent can call another agent as a tool without a full handoff.


4. Local/runtime execution tools

    Examples:
        computer-use style tools
        shell tools
        patch/apply tools


Testing angle:
    For any tool type, test:

        - Was the correct tool selected?
        - Were arguments valid?
        - Was execution authorized?
        - Was tool output handled correctly?
        - Did the final response stay grounded in tool output?
"""


# =============================================================================
# CONCEPT 4: HANDOFFS
# =============================================================================
"""
Handoff means one agent delegates the conversation/task to another specialist.

Example:

    Triage Agent:
        Decides whether the request is billing, technical, or security.

    Billing Agent:
        Handles invoice/refund issues.

    Security Agent:
        Handles suspicious account access.

Flow:

    User -> Triage Agent -> handoff to Security Agent -> final response

Why handoffs matter:
    They are useful for multi-agent systems and role-specialized workflows.

Testing angle:
    Validate:

        - correct handoff target
        - no unnecessary handoff
        - handoff context is complete
        - restricted data is not leaked between agents
        - final answer matches specialist agent responsibility
"""


# =============================================================================
# CONCEPT 5: GUARDRAILS
# =============================================================================
"""
Guardrails validate behavior before or after model/tool execution.

Types:

    Input guardrail:
        Runs on the initial user input.
        Example: block prompt injection or unsafe request.

    Output guardrail:
        Runs on final agent output.
        Example: ensure no PII or toxic content.

    Tool guardrail:
        Runs when a function tool is invoked.
        Example: block unauthorized tool arguments.

Testing angle:
    Guardrail tests should include:

        - safe input allowed
        - unsafe input blocked
        - prompt injection blocked
        - sensitive output blocked
        - unauthorized tool call blocked
        - guardrail failure produces clear user-facing response
"""


# =============================================================================
# CONCEPT 6: TRACING
# =============================================================================
"""
The SDK has built-in tracing for agent runs.

Traces can capture:

    - full Runner operation
    - each agent run
    - LLM generation
    - function tool calls
    - handoffs
    - guardrails
    - errors
    - latency / timing

Why tracing matters:
    Agent failures are often not visible from final output alone.
    Tracing helps answer:

        - What prompt was sent?
        - Which tool was selected?
        - What arguments were passed?
        - What did the tool return?
        - Did a guardrail run?
        - Was there a handoff?
        - Where did latency/cost come from?

Testing angle:
    In CI or evaluation runs, traces become evidence for trajectory validation.
"""


# =============================================================================
# CONCEPT 7: OPENAI AGENTS SDK vs LANGGRAPH vs LANGCHAIN
# =============================================================================
"""
LangChain:
    General LLM application framework.
    Good for chains, tools, retrievers, prompt templates, model wrappers.

LangGraph:
    Graph orchestration layer.
    Best when you want explicit state, nodes, edges, conditional routing,
    cycles, retries, and human-in-the-loop.

OpenAI Agents SDK:
    Integrated agent runtime from OpenAI.
    Gives Agent, Runner, tools, handoffs, guardrails, and tracing in one SDK.

Comparison:

    LangGraph:
        More explicit control over graph shape and state transitions.
        Testing focus: nodes, edges, paths, state transitions.

    OpenAI Agents SDK:
        More integrated runtime around agents/tools/handoffs/guardrails/tracing.
        Testing focus: instructions, tool calls, handoffs, guardrails, traces,
        and final output.

Interview-safe answer:

    I would use LangGraph when I need explicit control over workflow state and
    branching. I would use OpenAI Agents SDK when I want an integrated agent
    runtime with tools, handoffs, guardrails, and tracing. From QA perspective,
    both require trajectory validation, but the test focus differs slightly.
"""


# =============================================================================
# CONCEPT 8: WHAT TO TEST IN AN OPENAI AGENTS SDK IMPLEMENTATION
# =============================================================================
"""
1. Agent instructions

    Test:
        Does the agent follow its role and constraints?

    Example:
        A support agent should not answer legal advice if instructed not to.


2. Tool selection

    Test:
        Does the agent select the correct tool for each user request?

    Example:
        "Find customer CUST-101" should call customer_lookup.


3. Tool arguments

    Test:
        Are arguments complete, correctly typed, and semantically correct?

    Example:
        customer_id should be "CUST-101", not "101".


4. Tool errors

    Test:
        If a tool fails, does the agent avoid hallucination?

    Example:
        Tool returns "Customer not found".
        Agent should not invent customer details.


5. Handoffs

    Test:
        Does triage route to the correct specialist agent?

    Example:
        Security issue -> Security Agent, not Billing Agent.


6. Guardrails

    Test:
        Prompt injection, unsafe requests, and unauthorized tool calls are blocked.


7. Traces

    Test:
        Trace contains expected tool calls, handoffs, guardrail results, and errors.


8. Final output

    Test:
        Final answer is accurate, safe, and grounded in tool/context output.
"""


# =============================================================================
# MINI EXAMPLE - CONCEPTUAL
# =============================================================================
"""
Example shape:

    from agents import Agent, Runner, function_tool

    @function_tool
    def customer_lookup(customer_id: str) -> str:
        \"\"\"Look up customer by id.\"\"\"
        return "Customer CUST-101 is Amit with Premium plan."

    support_agent = Agent(
        name="Support Agent",
        instructions=(
            "You are a support agent. Use tools for customer-specific data. "
            "Do not invent customer details."
        ),
        tools=[customer_lookup],
    )

    result = Runner.run_sync(
        support_agent,
        "Find customer CUST-101"
    )

    print(result.final_output)

Testing expectation:

    - Tool called: customer_lookup
    - Args: {"customer_id": "CUST-101"}
    - Final answer contains only returned customer data
"""


# =============================================================================
# INTERVIEW ANSWER
# =============================================================================
"""
Question:
    What do you know about OpenAI Agents SDK and how would you test it?

Answer:
    OpenAI Agents SDK provides a structured way to build agents using Agent,
    Runner, tools, handoffs, guardrails, and tracing. The Runner manages the
    agent loop: model output, tool calls, handoffs, and final output.

    From a testing perspective, I would validate agent instructions, tool
    selection, tool arguments, tool execution results, handoff behavior,
    guardrails, traceability, latency, token usage, and final output quality.
    I would also test negative cases like wrong tool selection, invalid
    arguments, tool failure, prompt injection, unauthorized tool access, and
    hallucination after tool failure.

    I have deeper familiarity with LangGraph-style orchestration, but I
    understand that OpenAI Agents SDK gives a more integrated runtime for
    agents, tools, handoffs, guardrails, and tracing.
"""


# =============================================================================
# CHECKPOINT QUESTIONS
# =============================================================================
"""
Answer before moving to Block 6:

1. What are the main components of OpenAI Agents SDK?
2. What does Runner do?
3. What is a handoff?
4. What are guardrails?
5. How is OpenAI Agents SDK different from LangGraph?
6. What would you test in an Agents SDK based implementation?
"""
