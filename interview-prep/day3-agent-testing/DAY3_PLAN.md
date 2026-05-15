# Day 3: Agent Workbench Testing

## Main Goal

Convert the Day 2 mini agent workbench into an SDET-style testing strategy with pytest gates, evaluation checks, synthetic data, prompt injection tests, and analytics validation.

## Blocks

| Block | Topic | Output |
|---|---|---|
| 1 | Pytest gates for tool registry | Deterministic CI-friendly tests for tool lifecycle |
| 2 | Pytest gates for graph paths/state | Trajectory and state transition tests |
| 3 | Tool-calling correctness | Mock/live LLM tool selection and argument checks |
| 4 | Groundedness, hallucination, RAGAS/eval metrics | Metric concepts, dashboards, confusion matrices |
| 5 | Prompt injection and guardrails | Security and restricted-tool tests |
| 6 | Synthetic test data generation | Agents, tools, prompts, SOPs, traces, events |
| 7 | Analytics dashboard/event validation | Event-to-dashboard testing strategy |

## Workflow For Each Block

1. Read the reference file.
2. Try examples in the notebook.
3. Convert important examples into pytest.
4. Discuss interview answer.

## Day 2 App Under Test

The tests will target:

```text
interview-prep/day2-ai-testing/app/
```

Key modules:

- `tool_registry.py`
- `graph_agent.py`
- `simple_agent.py`
- `tracing.py`
- `mcp_schemas.py`
- `tools.py`
