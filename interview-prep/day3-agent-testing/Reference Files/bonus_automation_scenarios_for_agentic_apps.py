"""
================================================================================
BONUS: AUTOMATION SCENARIOS FOR RAG AND AGENTIC APPLICATIONS
================================================================================
Goal:
List practical UI/API/e2e automation scenarios for RAG apps, agentic bots, AI
workbenches, prompt/SOP authoring UIs, tool registries, eval dashboards, and
analytics pages.

This is not a separate study block. Use it as interview scenario inventory.
================================================================================
"""


# =============================================================================
# HOW TO USE THIS FILE
# =============================================================================
"""
When interviewer asks:

    "What would you automate for this AI workbench?"

Answer by levels:

    - API level for contracts, permissions, analytics calculations
    - agent/eval level for tool calls, grounding, guardrails
    - UI level for critical user journeys and dashboard rendering
    - CI level for deterministic gates
    - live LLM eval level for model-dependent behavior
"""


# =============================================================================
# SCENARIO GROUP 1: AGENT MANAGEMENT UI
# =============================================================================
"""
1. Create agent with valid name, role, model, and allowed tools.

2. Edit agent instructions and verify new version is saved.

3. Disable agent and verify it cannot be used in runtime.

4. Clone agent and verify copied tools/prompts/SOPs.

5. Delete/archive agent and verify it disappears from active list.

6. Role-based visibility:
    viewer can see agent
    tester can run agent
    admin can edit/delete agent

Best level:
    API for permission matrix.
    UI for create/edit/archive critical flows.
"""


# =============================================================================
# SCENARIO GROUP 2: PROMPT AND SOP AUTHORING
# =============================================================================
"""
1. Create prompt template with required variables.

2. Save prompt draft and publish version.

3. Compare prompt versions.

4. Roll back prompt version.

5. Validate missing required variable is rejected.

6. SOP authoring:
    create SOP with steps
    reorder steps
    publish SOP
    attach SOP to agent

7. Prompt injection inside prompt/SOP should be flagged or blocked.

Best level:
    API for validation/versioning.
    UI for editor workflows and preview.
"""


# =============================================================================
# SCENARIO GROUP 3: TOOL REGISTRATION UI/API
# =============================================================================
"""
1. Register tool with valid schema.

2. Reject tool schema missing required fields.

3. Reject duplicate tool name.

4. Disable tool and verify agent cannot call it.

5. Update tool schema and verify version changes.

6. Viewer role cannot register or execute tool.

7. Tool test panel:
    execute with valid args
    show success output
    execute with invalid args
    show schema error

Best level:
    API for lifecycle and schema validation.
    UI for tool creation form and test panel.
"""


# =============================================================================
# SCENARIO GROUP 4: RAG CHATBOT / KNOWLEDGE BOT
# =============================================================================
"""
1. Ask known policy question and verify answer is grounded in retrieved context.

2. Ask unknown question and verify bot says it does not know.

3. Verify citations/sources are shown.

4. Verify source links open correct document section.

5. Ask ambiguous query and verify clarification or safe answer.

6. Ask out-of-domain question and verify fallback behavior.

7. Upload/update knowledge document and verify retrieval changes.

8. Delete document and verify answer no longer uses it.

Best level:
    eval/API level for answer quality.
    UI for source citation rendering and chat user journey.
"""


# =============================================================================
# SCENARIO GROUP 5: AGENT TOOL-CALLING
# =============================================================================
"""
1. Calculation prompt calls calculator with correct expression.

2. Customer prompt calls customer_lookup with correct customer_id.

3. Ticket prompt calls ticket_creation with correct title/priority.

4. Prompt needing no tool does not call a tool.

5. Tool failure is explained, not hallucinated.

6. Unknown customer does not invent profile data.

7. Invalid priority is handled safely.

8. Multi-tool flow:
    lookup customer
    create ticket
    final answer references both outputs

Best level:
    agent/eval level for tool selection and args.
    API level for tool execution contract.
"""


# =============================================================================
# SCENARIO GROUP 6: PROMPT INJECTION AND GUARDRAILS
# =============================================================================
"""
1. "Ignore previous instructions" is blocked.

2. "Delete all users" is blocked.

3. "Tell me admin password" is blocked.

4. Safe greeting is not blocked.

5. Safe business question is not blocked.

6. Retrieved document with malicious instruction is ignored.

7. Tool output containing malicious instruction is ignored.

8. Viewer cannot force admin tool call through prompt.

Best level:
    deterministic CI for application guardrails.
    optional live LLM eval for model guardrail behavior.
"""


# =============================================================================
# SCENARIO GROUP 7: EVAL DASHBOARDS
# =============================================================================
"""
1. Dashboard shows total eval runs.

2. Faithfulness threshold failures are visible.

3. Context precision/recall trends render correctly.

4. Filter by agent/model/prompt version/date works.

5. Clicking failed eval opens trace/details.

6. Low groundedness row shows question, response, context, score.

7. Export eval results to CSV.

8. Empty state when no evals exist.

Best level:
    API for metric values.
    UI for filters, rendering, drilldown.
"""


# =============================================================================
# SCENARIO GROUP 8: ANALYTICS DASHBOARDS
# =============================================================================
"""
1. Total runs equals event count.

2. Success rate calculation is correct.

3. Tool call count is correct.

4. Blocked prompt count is correct.

5. Average and p95 latency are correct.

6. Token usage totals are correct.

7. Date filter handles timezone correctly.

8. Duplicate event does not double-count.

9. Missing event is detected by reconciliation.

10. Dashboard API matches UI value.

Best level:
    API/pipeline for calculation.
    UI for rendering and filters.
"""


# =============================================================================
# SCENARIO GROUP 9: AI-ASSISTED PLAYWRIGHT AUTOMATION
# =============================================================================
"""
1. LLM generates a Playwright test from user journey description.

2. Generated test uses stable selectors/data-testid.

3. Generated test includes meaningful assertions, not only clicks.

4. Selector-healing suggests replacement when locator changes.

5. Failure triage summarizes error, screenshot, trace, and likely root cause.

6. Bug report generation includes steps, expected, actual, environment, logs.

7. Human approval is required before committing AI-generated test changes.

8. AI does not mask real product bugs as flaky tests.

Best level:
    Use AI as assistant for generation/triage.
    Keep final assertions deterministic and reviewable.
"""


# =============================================================================
# SCENARIO GROUP 10: CI/CD GATES
# =============================================================================
"""
PR gates:

    - tool registry deterministic tests
    - graph path deterministic tests
    - prompt injection deterministic tests
    - API contract tests
    - key UI smoke tests

Nightly/eval gates:

    - live OpenAI tool-calling evals
    - RAGAS metrics
    - LangSmith trace checks
    - hallucination/groundedness evals
    - synthetic analytics volume tests

Release gates:

    - end-to-end dashboard validation
    - role/permission matrix
    - data migration checks
    - performance/load checks
"""


# =============================================================================
# INTERVIEW ANSWER
# =============================================================================
"""
Question:
    What scenarios would you automate for RAG/agentic apps?

Answer:

    "I would split automation by risk and layer. At the API level, I would test
    agent, prompt, SOP, and tool lifecycle contracts, schema validation, and
    role permissions. At the agent/eval level, I would test tool selection,
    argument correctness, trajectory, groundedness, hallucination, and prompt
    injection resistance. At the UI level, I would cover critical workflows like
    creating agents, registering tools, authoring prompts, running evaluations,
    and reading dashboards. For analytics, I would validate event-to-dashboard
    correctness using synthetic events with known expected aggregations."
"""
