"""
================================================================================
BLOCK 2: TOOL CALLING - @tool, FUNCTION SCHEMAS, AND TOOL ARGUMENTS
================================================================================
Goal:
Understand how agents use tools, how tools are represented in different
frameworks, and what an SDET should validate in tool-calling systems.

This block covers:
1. LangChain `@tool` style definitions
2. OpenAI function/tool schema style definitions
3. How agents choose tools
4. How to design tool inputs/outputs
5. What Day 3 tests should validate
================================================================================
"""


# =============================================================================
# CONCEPT 1: WHAT IS TOOL CALLING?
# =============================================================================
"""
Tool calling means an LLM can request an external capability instead of trying
to answer everything from memory.

Example:

    User:
        "What is 15 * 7?"

    Agent decision:
        calculator(expression="15 * 7")

    Tool result:
        {"result": 105}

    Final answer:
        "15 * 7 equals 105."

Why tool calling matters:
    - Math should be done by a calculator, not guessed by the model.
    - Customer data should come from a database, not model memory.
    - Tickets should be created by an API, not simulated in text.

Testing angle:
    We do not only test the final answer.
    We validate:
        - Was a tool needed?
        - Was the correct tool selected?
        - Were the arguments correct?
        - Did the application validate and execute the tool safely?
        - Was the final response grounded in the tool result?
"""


# =============================================================================
# CONCEPT 2: THREE COMMON TOOL DEFINITION STYLES
# =============================================================================
"""
1. LangChain @tool style

    from langchain_core.tools import tool

    @tool
    def get_weather(city: str) -> str:
        \"\"\"Get weather for a city.\"\"\"
        return f"Weather in {city}: Sunny"

    Pros:
        - Clean Python syntax
        - Docstring becomes tool description
        - Good with LangChain/LangGraph


2. OpenAI function/tool schema style

    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    }

    Pros:
        - Direct OpenAI API format
        - Explicit JSON schema
        - Good for understanding the protocol-level shape


3. MCP-style tool schema

    {
        "name": "get_weather",
        "description": "Get weather for a city",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }
    }

    Pros:
        - Tool can live outside the app
        - Client can discover tools from the MCP server
        - Reusable across Cursor, Claude Desktop, custom clients


Interview framing:
    These are different representations of the same idea:
    a tool name, description, input schema, and executable function.
"""


# =============================================================================
# CONCEPT 3: TOOL SELECTION vs ARGUMENT GENERATION
# =============================================================================
"""
Tool selection:
    The LLM decides which tool should handle the request.

    Example:
        User: "Find customer CUST-101"
        Expected tool: customer_lookup

Argument generation:
    The LLM extracts and structures the inputs for that tool.

    Example:
        Expected arguments:
            {"customer_id": "CUST-101"}

Both can fail independently.

Examples:
    Wrong tool:
        User asks for customer lookup, agent calls calculator.

    Right tool, wrong argument:
        Agent calls customer_lookup, but passes {"customer_id": "101"}
        instead of {"customer_id": "CUST-101"}.

Testing angle:
    Day 3 tests should separately assert:
        - expected_tool == actual_tool
        - expected_arguments == actual_arguments
"""


# =============================================================================
# CONCEPT 4: TOOL OUTPUT DESIGN
# =============================================================================
"""
Good tool output should be structured and predictable.

Bad output:
    "Done"

Better output:
    {
        "success": true,
        "tool_name": "ticket_creation",
        "data": {
            "ticket_id": "TCK-1001",
            "priority": "high"
        },
        "error": null
    }

Why structured output matters:
    - Easier for the agent to interpret
    - Easier for tests to assert
    - Easier to debug in traces
    - Easier to display in dashboards

Testing angle:
    Every tool should return a consistent response shape:
        success
        tool_name
        data
        error
"""


# =============================================================================
# MINI APP CODE FOR THIS BLOCK
# =============================================================================
"""
The actual mini app code for this block is in:

    interview-prep/day2-ai-testing/app/tools.py
    interview-prep/day2-ai-testing/app/mcp_schemas.py

Tools created:
    - calculator_tool(expression)
    - customer_lookup_tool(customer_id)
    - ticket_creation_tool(title, priority)
    - knowledge_search_tool(query)

Schemas created:
    - CALCULATOR_SCHEMA
    - CUSTOMER_LOOKUP_SCHEMA
    - TICKET_CREATION_SCHEMA
    - KNOWLEDGE_SEARCH_SCHEMA

These are deterministic so Day 3 tests can reliably validate behavior.
"""


# =============================================================================
# CONCEPT 5: WHAT TO TEST FOR TOOL CALLING
# =============================================================================
"""
Positive tests:
    1. Agent selects calculator for math query.
    2. Agent selects customer_lookup for customer query.
    3. Agent selects ticket_creation for support ticket request.
    4. Agent passes required arguments.
    5. Agent final response uses tool result.

Negative tests:
    1. Missing required argument.
    2. Invalid argument type.
    3. Unsupported enum value.
    4. Disabled tool.
    5. Tool timeout.
    6. Tool returns error.
    7. User asks for unauthorized/restricted tool.
    8. Prompt injection tries to force unsafe tool usage.

Trace checks:
    - tool_called
    - tool_arguments
    - tool_result.success
    - tool_result.error
    - final_answer grounded in tool_result
"""


# =============================================================================
# INTERVIEW ANSWER
# =============================================================================
"""
Question:
    How would you test tool calling in an AI agent?

Answer:
    I would test tool calling at multiple levels. First, I would validate tool
    schemas: names, descriptions, required parameters, allowed types, and enum
    values. Then I would test tool selection by giving representative user
    queries and checking whether the agent selected the correct tool. After that,
    I would validate generated arguments against the schema and expected values.
    I would also test execution paths such as success, invalid arguments,
    timeouts, unavailable tools, and tool errors. Finally, I would verify that
    the final response is grounded in the actual tool output and does not
    hallucinate results.
"""


# =============================================================================
# CHECKPOINT QUESTIONS
# =============================================================================
"""
Answer before moving to Block 3:

1. What is the difference between tool selection and argument generation?
2. Why is a tool schema important?
3. What are three negative test cases for tool calling?
4. Why should tool output be structured?
5. How would you test that the final answer used the tool result correctly?
"""


# =============================================================================
# CHECKPOINT ANSWERS - INTERVIEW READY
# =============================================================================
"""
1. What is the difference between tool selection and argument generation?

    Tool selection is the process where the LLM understands the user's intent
    and decides whether a tool is needed. If a tool is needed, it selects the
    most appropriate tool for the task.

    Argument generation happens after tool selection. The LLM extracts or
    creates the input values required by that tool.

    Tool selection answers:
        "Which tool should I call?"

    Argument generation answers:
        "What inputs should I pass to that tool?"


2. Why is a tool schema important?

    A tool schema is important because it acts like an API contract between the
    agent and the application.

    It defines the tool name, description, required parameters, parameter types,
    allowed values, and constraints.

    The application can validate generated arguments against the schema before
    execution. This prevents invalid calls, wrong types, missing fields, and
    unsafe tool usage.


3. What are three negative test cases for tool calling?

    Negative test cases include:

    1. The agent calls a tool when no tool is needed.
    2. The agent selects the wrong tool.
    3. Required arguments are missing.
    4. Argument types are incorrect.
    5. Enum values are unsupported.
    6. The tool is unavailable.
    7. The tool times out.
    8. The tool returns an error.
    9. The agent hallucinates a successful result after a failed tool call.


4. Why should tool output be structured?

    Every tool should return a structured and consistent response shape, such as:

        success
        tool_name
        data
        error

    This makes it easier for the LLM to interpret the result, easier for tests
    to assert behavior, easier to debug in traces, and easier to feed into
    dashboards or analytics pipelines.


5. How would you test that the final answer used the tool result correctly?

    I would compare the final answer against the actual tool result.

    If the tool returns data, the final answer should only use that returned
    data. If the tool fails, the agent should not invent a result. It should
    either retry, fallback, or clearly explain the failure based on the tool
    error.

    This validates groundedness and helps prevent hallucination after tool
    execution.
"""
