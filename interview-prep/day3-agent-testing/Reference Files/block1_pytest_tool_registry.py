"""
================================================================================
DAY 3 - BLOCK 1: PYTEST GATES FOR TOOL REGISTRY
================================================================================
Goal:
Convert the Day 2 MCP-style tool registry into deterministic pytest gates.

Why this matters for the DTDL JD:
The JD explicitly mentions tool registration, agent management, user/role
management, and CI/CD test gates. Tool registry tests are the cleanest starting
point because they are deterministic, fast, and CI-friendly.
================================================================================
"""


# =============================================================================
# CONCEPT 1: WHAT IS A PYTEST GATE?
# =============================================================================
"""
A pytest gate is an automated pass/fail check that protects the system from
regressions.

For AI/agent platforms, not every test should call a live LLM. Many critical
rules are deterministic and should run on every PR:

    - schema validation
    - permission checks
    - disabled tool behavior
    - duplicate tool prevention
    - role-based discovery
    - unknown tool blocking

These tests are perfect CI gates because:

    - fast
    - stable
    - cheap
    - not dependent on OpenAI/LangSmith availability
"""


# =============================================================================
# CONCEPT 2: SYSTEM UNDER TEST
# =============================================================================
"""
The Day 2 tool registry lives here:

    interview-prep/day2-ai-testing/app/tool_registry.py

It models:

    - tool registration
    - tool updates
    - tool deletion
    - enable/disable
    - role-based discovery
    - role-based execution
    - JSON Schema argument validation

Important functions/classes:

    build_default_registry()
        Creates a registry with calculator, customer_lookup, ticket_creation,
        and knowledge_search.

    ToolRegistry.execute_tool()
        Validates tool existence, enabled state, role permission, schema, then
        executes the tool.

    ToolRegistry.list_tools()
        Returns discoverable tools based on role and enabled state.

    Custom exceptions:
        ToolAlreadyExistsError
        ToolNotFoundError
        ToolDisabledError
        ToolPermissionError
        ToolSchemaError
"""


# =============================================================================
# CONCEPT 3: TEST CASES TO IMPLEMENT
# =============================================================================
"""
Positive tests:

1. test_default_registry_lists_tools_for_tester

    Given:
        default registry

    When:
        list_tools(role="tester")

    Then:
        calculator, customer_lookup, ticket_creation, knowledge_search are listed


2. test_execute_calculator_with_valid_args

    Given:
        calculator tool registered

    When:
        execute_tool("calculator", {"expression": "15 * 7"}, role="tester")

    Then:
        success is True
        result is 105


3. test_admin_can_execute_customer_lookup

    Given:
        customer_lookup tool registered

    When:
        admin executes customer_lookup with CUST-101

    Then:
        returned customer is Amit Sharma / Premium / active


Negative tests:

4. test_duplicate_tool_registration_fails

    Given:
        calculator already registered

    When:
        register calculator again

    Then:
        ToolAlreadyExistsError is raised


5. test_missing_required_argument_fails

    Given:
        calculator schema requires expression

    When:
        execute_tool("calculator", {}, role="tester")

    Then:
        ToolSchemaError is raised


6. test_extra_argument_fails

    Given:
        calculator schema has additionalProperties=False

    When:
        execute_tool("calculator", {"expression": "15 * 7", "extra": "bad"})

    Then:
        ToolSchemaError is raised


7. test_disabled_tool_cannot_execute

    Given:
        calculator is disabled

    When:
        execute calculator

    Then:
        ToolDisabledError is raised


8. test_viewer_cannot_execute_customer_lookup

    Given:
        viewer role is not allowed

    When:
        viewer executes customer_lookup

    Then:
        ToolPermissionError is raised


9. test_unknown_tool_fails

    Given:
        delete_all_users is not registered

    When:
        execute_tool("delete_all_users", {}, role="admin")

    Then:
        ToolNotFoundError is raised


10. test_disabled_tool_hidden_from_discovery

    Given:
        calculator is disabled

    When:
        list_tools(role="tester")

    Then:
        calculator is not returned unless include_disabled=True
"""


# =============================================================================
# CONCEPT 4: PYTEST PATTERNS
# =============================================================================
"""
Fixture pattern:

    import pytest

    @pytest.fixture
    def registry():
        return build_default_registry()

Why fixture?
    Each test gets a fresh registry.
    Tests do not leak state into each other.


Exception assertion:

    with pytest.raises(ToolSchemaError):
        registry.execute_tool("calculator", {}, role="tester")


Parametrize pattern:

    @pytest.mark.parametrize("tool_name,args", [
        ("calculator", {"expression": "15 * 7"}),
        ("customer_lookup", {"customer_id": "CUST-101"}),
    ])
    def test_tools_execute_successfully(registry, tool_name, args):
        result = registry.execute_tool(tool_name, args, role="tester")
        assert result["success"] is True


CI-friendly rule:
    These tests should not call OpenAI or LangSmith.
    They should run on every commit.
"""


# =============================================================================
# CONCEPT 5: WHY THIS IS SENIOR SDET-LEVEL
# =============================================================================
"""
This block is not just unit testing.

It maps directly to real risks in an AI workbench:

    - An agent calls a disabled tool.
    - A viewer role discovers admin-only tools.
    - A malformed tool schema reaches production.
    - A duplicate tool name causes wrong tool execution.
    - A prompt injection asks the agent to call an unregistered tool.

Tool registry tests protect the control plane of the agent platform.

Interview answer:

    "For tool registration, I would start with deterministic pytest gates around
    schema validation, duplicate prevention, enabled/disabled state, role-based
    discovery, and authorized execution. These tests are stable enough to run on
    every PR and prevent unsafe or invalid tools from reaching runtime."
"""


# =============================================================================
# PRACTICE NOTEBOOK TASKS
# =============================================================================
"""
In the Block 1 notebook:

1. Import build_default_registry.
2. List tester tools.
3. Execute calculator with valid args.
4. Try missing required argument and catch ToolSchemaError.
5. Disable calculator and verify ToolDisabledError.
6. Try viewer role and verify ToolPermissionError.
7. Convert two examples into pytest-style assertions manually.
"""


# =============================================================================
# ACTUAL PYTEST FILE
# =============================================================================
"""
We will implement the real pytest version in:

    interview-prep/day3-agent-testing/tests/test_tool_registry.py

For now, use the notebook to understand behavior first.
Then we will convert it into real tests when you say to proceed.
"""


# =============================================================================
# CHECKPOINT QUESTIONS
# =============================================================================
"""
1. Why are tool registry tests good CI gates?

Tool registry tests are good CI gates because not every agent-platform test
needs to call an LLM. Many critical rules are deterministic:

    - schema validation
    - permission checks
    - disabled tool behavior
    - duplicate tool prevention
    - unknown tool blocking
    - role-based tool discovery

These checks should run on every PR because they are fast, stable, cheap, and
do not depend on OpenAI, LangSmith, network availability, or model behavior.

Interview answer:

    "For agent platforms, I would not put every test behind a live LLM call.
    Tool registry behavior like schema validation, permissions, disabled tools,
    duplicate tools, and unknown tool handling is deterministic. These checks
    are ideal CI gates because they are fast, reliable, low-cost, and protect
    the platform before the agent runtime even starts."


2. Why should each pytest test get a fresh registry fixture?

Each test should get a fresh registry because the registry has mutable state.
For example, one test may disable a tool, update permissions, or delete a tool.
If another test reuses the same registry instance, it may fail because of state
left behind by the previous test.

Fresh fixtures give test isolation:

    - one test cannot affect another test
    - failures become easier to debug
    - test order does not matter
    - CI results stay stable

Small example:

    @pytest.fixture
    def registry():
        return build_default_registry()

    def test_disable_tool(registry):
        registry.disable_tool("calculator")

    def test_calculator_still_available(registry):
        # This should pass because this test gets a new registry.
        assert registry.get_tool("calculator").enabled is True

Interview answer:

    "I would create a fresh registry fixture per test because registry tests
    mutate state. Some tests disable tools, delete tools, or change permissions.
    A fresh fixture gives isolation and makes the suite independent of execution
    order."


3. What is the difference between schema validation and tool execution failure?

Schema validation checks whether the tool input is structurally valid before
the tool runs.

Examples of schema validation failures:

    - missing required argument
    - wrong data type
    - extra unsupported property
    - invalid enum value

Tool execution failure happens after schema validation passes, but the tool
itself cannot complete the work.

Examples of tool execution failures:

    - customer id has valid format but customer does not exist
    - calculator receives a syntactically valid string but unsupported expression
    - ticket system is unavailable
    - downstream API returns an error

Simple distinction:

    Schema validation asks:
        "Is the input shape allowed?"

    Tool execution asks:
        "Could the tool complete the requested operation?"

Interview answer:

    "Schema validation is a pre-execution contract check. It verifies required
    fields, data types, allowed values, and extra properties. Tool execution
    failure happens after valid arguments reach the tool, but the business
    operation fails, such as customer not found or downstream service error."


4. Which registry tests map most directly to security?

The most security-relevant registry tests are:

    - role-based discovery:
        A viewer should not even see admin-only tools.

    - role-based execution:
        A restricted role should not execute sensitive tools.

    - disabled tool blocking:
        A disabled tool should not be callable by an agent.

    - unknown tool blocking:
        Prompt injection should not make the platform execute unregistered tools.

    - schema validation:
        Extra parameters should be rejected so attackers cannot smuggle hidden
        instructions or unsupported fields into tool calls.

    - duplicate tool prevention:
        A malicious or badly configured tool should not overwrite a trusted tool
        with the same name.

Interview answer:

    "The security-heavy registry tests are permission checks, restricted tool
    discovery, disabled tool blocking, unknown tool blocking, schema validation,
    and duplicate tool prevention. These protect the control plane of the agent
    platform and prevent prompt injection or misconfiguration from reaching
    sensitive tool execution."


5. How would you explain this block in an interview?

Interview answer:

    "In Day 3 Block 1, I focused on converting the tool registry into
    deterministic pytest gates. The registry is responsible for registering
    tools, exposing them by role, validating schemas, enforcing permissions, and
    blocking disabled or unknown tools. I would test these rules without calling
    an LLM because they are application-level controls and should be stable in
    CI. This gives fast protection for the agent workbench before we move to
    more expensive live LLM or LangSmith evaluation tests."

Short version:

    "This block tests the control plane of an agent platform. Before testing
    whether the LLM selected the right tool, I first verify that the platform
    only exposes valid, enabled, authorized tools and rejects unsafe calls."

"""
