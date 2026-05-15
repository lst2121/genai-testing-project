"""
================================================================================
BLOCK 7: AGENT WORKBENCH DESIGN FOR INTERVIEW DISCUSSION
================================================================================
Goal:
Connect everything from Day 2 to the Deutsche Telekom Digital Labs JD and form
an interview-ready explanation of how you would test an AI agent workbench.

This block connects the mini app to the DTDL job description:
1. Agent management
2. Prompt and SOP authoring
3. Tool registration
4. Testing surfaces and eval workflows
5. Analytics dashboards and data integrity
================================================================================
"""


# =============================================================================
# CONCEPT 1: WHAT IS THE AI WORKBENCH IN THE JD?
# =============================================================================
"""
The JD is describing a workbench/platform where users can create and manage:

    - agents
    - prompts
    - SOPs / workflows
    - tool registrations
    - testing/evaluation surfaces
    - analytics dashboards
    - users/roles/permissions

This is not only a chatbot application.
It is a platform for configuring and operating agentic systems.

Mental model:

    Authoring layer:
        Create agents, prompts, SOPs, tools, permissions.

    Runtime layer:
        Agent executes, calls tools, produces conversation traces.

    Evaluation layer:
        Validate tool selection, outputs, hallucination, guardrails, metrics.

    Analytics layer:
        Capture runtime events and display dashboards.

    Automation layer:
        AI-assisted Playwright generation, selector maintenance, failure triage.
"""


# =============================================================================
# CONCEPT 2: HOW OUR MINI APP MAPS TO THE JD
# =============================================================================
"""
Our mini Day 2 app maps to the JD like this:

    app/graph_agent.py
        Agent workflow / LangGraph-style orchestration.
        Maps to: agent management, runtime execution, branching paths.

    app/simple_agent.py
        Real OpenAI tool-calling flow.
        Maps to: LLM tool selection and argument generation.

    app/tools.py
        Deterministic backend tools.
        Maps to: registered business/runtime tools.

    app/mcp_schemas.py
        MCP-style tool descriptors.
        Maps to: tool registration and schema validation.

    app/tool_registry.py
        Tool registry with enable/disable, role access, schema validation.
        Maps to: workbench tool lifecycle and permissions.

    app/tracing.py
        LangSmith tracing wrapper.
        Maps to: eval workflows, trace debugging, observability.

This gives you an interview story:

    "I built a mini agent workbench model with tool registration, real OpenAI
    tool calling, LangGraph-style routing, role-based tool access, and LangSmith
    tracing. Then I designed tests around the full agent trajectory, not just
    final output."
"""


# =============================================================================
# CONCEPT 3: TEST STRATEGY FOR AN AI WORKBENCH
# =============================================================================
"""
A good SDET test strategy should cover these layers:

1. UI workflow tests

    Test:
        - create agent
        - edit agent
        - create prompt
        - create SOP/workflow
        - register tool
        - run test/eval
        - inspect trace
        - view analytics dashboard

    Tool:
        Playwright


2. API tests

    Test:
        - agent CRUD APIs
        - prompt/SOP APIs
        - tool registry APIs
        - run/eval APIs
        - permissions APIs

    Tool:
        pytest + requests / Playwright APIRequestContext


3. Agent behavior tests

    Test:
        - intent classification
        - correct graph path
        - correct tool selection
        - correct tool arguments
        - fallback behavior
        - final answer grounding

    Tool:
        pytest + deterministic model stubs + smaller live LLM suite


4. Tool registry tests

    Test:
        - register/update/delete/disable/enable
        - invalid schema
        - duplicate names
        - missing required fields
        - role-based discovery
        - unauthorized execution

    Tool:
        pytest + JSON Schema validation


5. Eval / metric tests

    Test:
        - hallucination
        - groundedness
        - faithfulness
        - answer relevancy
        - tool-calling correctness
        - agent trajectory correctness
        - prompt injection resistance

    Tool:
        RAGAS / LangSmith evaluators / custom pytest assertions


6. Analytics validation

    Test:
        - event emitted from runtime
        - event ingested by pipeline
        - event transformed/aggregated correctly
        - dashboard shows correct count/value
        - no silent data loss

    Tool:
        UI + API + DB checks, similar to your EVPNxGen data integrity pattern.
"""


# =============================================================================
# CONCEPT 4: AI-ASSISTED UI AUTOMATION STRATEGY
# =============================================================================
"""
The JD explicitly asks for AI-assisted UI automation:

    "LLM-driven agents that generate, execute, and maintain Playwright tests;
    adapt selectors as UI evolves; triage failures into actionable bug reports."

How to explain your approach:

    Base layer:
        Keep Playwright as deterministic automation framework.

    AI assistance layer:
        Use LLMs/Cursor skills to:
            - generate test scenarios from tickets/AC
            - generate Playwright test skeletons
            - suggest locator fixes when selectors break
            - summarize traces/screenshots/logs
            - classify failures as product bug, test bug, infra issue
            - draft bug reports

    Guardrail:
        Do not let AI blindly decide CI pass/fail for critical flows.
        Keep assertions and quality gates deterministic.

Interview answer:

    "I would use AI to accelerate authoring, maintenance, and triage, but I
    would keep critical assertions, API validations, and CI gates deterministic."
"""


# =============================================================================
# CONCEPT 5: SYNTHETIC TEST DATA GENERATION
# =============================================================================
"""
The JD asks for synthetic test data:

    - agents
    - prompts
    - SOPs
    - tool definitions
    - conversation traces

Testing strategy:

    1. Generate realistic data with schemas.
    2. Validate generated data before inserting it.
    3. Include positive and negative cases.
    4. Seed environments consistently.
    5. Use data to exercise dashboards and analytics.

Example generated tool:

    {
        "name": "customer_lookup",
        "description": "Look up customer by ID",
        "inputSchema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string"}
            },
            "required": ["customer_id"]
        }
    }

Examples of negative synthetic data:

    - invalid prompt with injection text
    - tool missing required schema
    - conversation trace with failed tool call
    - agent with unauthorized tool access
    - dashboard events with missing fields
"""


# =============================================================================
# CONCEPT 6: ANALYTICS / DASHBOARD VALIDATION
# =============================================================================
"""
The JD emphasizes analytics:

    "Validate events emitted from runtime, ingested by analytics pipeline,
    surfaced in dashboards. Catch silent data-loss and aggregation bugs early."

Testing flow:

    Agent runtime event:
        agent_run_completed

    Event fields:
        run_id
        agent_id
        user_id
        tool_calls_count
        latency_ms
        success
        timestamp

    Validation:
        1. Trigger agent run.
        2. Verify event emitted.
        3. Verify event ingested.
        4. Verify aggregation count increments.
        5. Verify dashboard displays correct value.

Your EVPNxGen mapping:

    You already have a strong story around:
        UI value -> API response -> DB query validation

    Use that same story for AI workbench analytics:
        dashboard metric -> analytics API -> event store / warehouse
"""


# =============================================================================
# JD-SPECIFIC INTERVIEW ANSWERS
# =============================================================================
"""
1. How would you test an AI workbench?

    I would test it across multiple layers: UI workflows, backend APIs, agent
    behavior, prompt/SOP authoring, tool registration, eval workflows, role-based
    access, and analytics dashboards.

    For agent behavior, I would validate the full trajectory: user input,
    planning, graph path, tool selection, tool arguments, tool result, final
    answer, guardrails, and trace. For quality, I would combine deterministic
    assertions with LLM evaluation metrics like groundedness, faithfulness, and
    prompt-injection resistance.


2. How would you test tool registration?

    I would validate the full lifecycle: create, update, delete, enable, disable,
    discover, and execute. I would test schema correctness, required fields,
    invalid parameter types, duplicate tool names, role-based discovery,
    unauthorized execution, disabled tools, unavailable tools, and prompt
    injection attempting to call restricted tools.


3. How would you test an agent's tool-calling behavior?

    I would test whether the agent selects the correct tool, generates correct
    arguments, passes schema validation, handles tool success/failure, and
    produces a final answer grounded in the actual tool output. I would inspect
    traces in LangSmith to separate failures in tool selection, argument
    generation, tool execution, and final response synthesis.


4. Where does AI-assisted Playwright help and where does it not?

    AI helps with test generation, scenario brainstorming, locator suggestions,
    selector-healing recommendations, failure triage, log summarization, and bug
    report drafting.

    I would not blindly trust AI for critical assertions, security-sensitive
    validations, or CI gate decisions. Those should remain deterministic and
    reviewable.


5. How would you test analytics dashboards?

    I would validate event correctness end to end: runtime emits the event, the
    pipeline ingests it, transformations/aggregations are correct, and dashboard
    values match backend/API/DB sources. I would test missing events, duplicate
    events, delayed events, filters, time ranges, and aggregation correctness.


6. How would you test hallucination and groundedness?

    I would compare final answers against source context or tool output. If a
    tool fails or returns no data, the agent should not invent a successful
    result. It should retry, fallback, or clearly explain the failure. I would
    use trace evidence from LangSmith and evaluation metrics like groundedness
    and faithfulness to detect unsupported claims.
"""


# =============================================================================
# YOUR FINAL DAY 2 STORY
# =============================================================================
"""
Use this as your concise project/practice story:

    "For interview prep, I built a mini AI workbench model. It had a
    LangGraph-style agent flow, real OpenAI tool calling, deterministic backend
    tools, MCP-style tool schemas, a tool registry with role-based access, and
    LangSmith tracing.

    I tested and inspected the full flow: model-selected tool, generated
    arguments, schema validation, tool execution, final answer grounding, and
    trace evidence. I also tested negative cases like invalid arguments,
    disabled tools, unauthorized roles, failed tool calls, and prompt injection.

    This helped me reason about how to test a real agent workbench beyond just
    checking final LLM text."
"""


# =============================================================================
# DAY 3 HANDOFF
# =============================================================================
"""
Day 3 should convert this mini workbench into tests:

    tests/test_tool_registry.py
        create/update/delete/disable/permission/schema tests

    tests/test_agent_paths.py
        graph path and state transition tests

    tests/test_tool_calling.py
        OpenAI tool selection and argument validation tests

    tests/test_groundedness.py
        final answer vs tool result checks

    tests/test_prompt_injection.py
        restricted request / guardrail checks

    tests/test_langsmith_trace.py
        trace existence and key span validation
"""


# =============================================================================
# CHECKPOINT QUESTIONS
# =============================================================================
"""
Final Day 2 reflection:

1. How does the mini workbench map to the DTDL JD?
2. What would be your end-to-end test strategy for this workbench?
3. What would you automate at UI level vs API level vs agent/eval level?
4. What is your strongest interview story from Day 2?
5. What should Day 3 test first?
"""


# =============================================================================
# CHECKPOINT ANSWERS - INTERVIEW READY
# =============================================================================
"""
1. How does the mini workbench map to the DTDL JD?

    The mini workbench maps closely to the DTDL JD because it models the same
    core surfaces: agents, tools, tool registration, execution traces, eval
    workflows, and testability.

    Mapping:

        app/graph_agent.py
            Maps to agent orchestration and agent management. It models how an
            agent classifies user input, chooses a route, calls tools, handles
            guardrails, and produces a final response.

        app/simple_agent.py
            Maps to real LLM tool calling. OpenAI selects the tool and generates
            arguments, while the application validates and executes the tool.

        app/mcp_schemas.py
            Maps to MCP-style tool definitions and tool registration contracts.

        app/tool_registry.py
            Maps to the workbench tool lifecycle: register, update, delete,
            enable, disable, discover, validate, authorize, and execute.

        app/tracing.py
            Maps to LangSmith-style observability and eval workflows.

    Interview answer:

        "I built a small version of an AI workbench where agents can call tools,
        tools are registered with MCP-style schemas, tool usage is permissioned,
        and runs are traced in LangSmith. This helped me reason about the same
        surfaces mentioned in the JD: agent management, tool registration,
        testing surfaces, eval workflows, and analytics/debugging."


2. What would be your end-to-end test strategy for this workbench?

    I would test it layer by layer:

    UI layer:
        Validate major workbench journeys with Playwright:
            - create/edit agent
            - create/edit prompt
            - create/edit SOP/workflow
            - register tool
            - run agent test
            - inspect trace
            - view dashboard

    API layer:
        Validate backend contracts:
            - agent CRUD
            - prompt/SOP CRUD
            - tool registry APIs
            - permissions
            - run/eval APIs

    Agent behavior layer:
        Validate:
            - intent classification
            - graph path
            - tool selection
            - tool arguments
            - tool failure handling
            - final answer grounding

    Tool registry layer:
        Validate:
            - schema correctness
            - duplicate prevention
            - enable/disable
            - delete/update
            - role-based discovery
            - unauthorized execution

    Eval layer:
        Validate:
            - hallucination
            - groundedness
            - faithfulness
            - answer relevancy
            - tool-calling correctness
            - prompt injection resistance

    Analytics layer:
        Validate:
            - runtime events emitted
            - pipeline ingestion
            - aggregation correctness
            - dashboard values
            - no silent data loss

    CI/CD:
        Run deterministic pytest gates on every PR.
        Run smaller live LLM/LangSmith eval suites on schedule or pre-release.


3. What would you automate at UI level vs API level vs agent/eval level?

    UI level:
        Automate only critical user journeys and visual/workflow behaviors:
            - agent creation
            - tool registration form
            - prompt/SOP authoring
            - run test button
            - trace viewer
            - dashboard filters

        Tool:
            Playwright

    API level:
        Automate backend correctness and contracts:
            - CRUD validations
            - schema validation
            - permissions
            - status codes
            - payload contracts
            - tool registry lifecycle

        Tool:
            pytest + requests / Playwright APIRequestContext

    Agent/eval level:
        Automate agent behavior and quality:
            - correct tool selected
            - correct arguments generated
            - final answer grounded
            - prompt injection blocked
            - hallucination checks
            - trace generated
            - eval metrics above thresholds

        Tool:
            pytest, LangSmith, RAGAS/custom evaluators

    Interview answer:

        "I would not test everything through UI. I would use UI tests for
        critical workbench journeys, API tests for backend contracts, and
        agent/eval tests for LLM behavior. This keeps the suite fast, stable,
        and meaningful."


4. What is your strongest interview story from Day 2?

    Strongest story:

        "I built a mini agent workbench to understand and test modern agentic
        systems. The app had deterministic graph routing, real OpenAI tool
        calling, MCP-style tool schemas, a tool registry with permissions and
        enable/disable behavior, and LangSmith tracing.

        I tested positive and negative scenarios: successful calculator calls,
        invalid calculator expressions, customer lookup success/failure,
        invalid ticket priority behavior, disabled tools, unauthorized roles,
        and prompt-injection style requests.

        The most important learning was that final-answer testing is not enough
        for agents. I need to validate the full trajectory: model-selected tool,
        generated arguments, schema validation, tool execution, final response
        grounding, and trace evidence."

    Why this is strong:

        - It maps directly to the JD.
        - It shows hands-on understanding, not only theory.
        - It uses OpenAI tool calling, MCP-style tools, LangSmith, and testing.
        - It gives you Day 3 material for pytest gates.


5. What should Day 3 test first?

    Day 3 should start with deterministic pytest gates because these are stable
    and CI-friendly.

    Recommended order:

        1. Tool registry tests
            - schema validation
            - duplicate tool
            - disabled tool
            - unauthorized role
            - unknown tool

        2. Deterministic graph path tests
            - calculation route
            - customer lookup route
            - ticket route
            - guardrail route
            - direct answer route

        3. Tool-calling tests
            - OpenAI selects correct tool
            - arguments are correct
            - failure does not hallucinate

        4. Groundedness tests
            - final answer matches tool output
            - failed tool response is not converted into fake success

        5. Prompt injection tests
            - restricted requests blocked
            - unauthorized tools not called

        6. LangSmith trace tests
            - trace is created
            - tool span exists
            - metadata/tokens/errors visible

    Interview answer:

        "I would start Day 3 with deterministic pytest gates around the tool
        registry and graph paths, because those should run reliably in CI. Then
        I would add a smaller live LLM eval suite for tool selection,
        groundedness, and prompt-injection behavior."
"""
