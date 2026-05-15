# Day 2: Agent Workbench Build + Understanding

## Main Goal

Build enough of a mini agent workbench to understand how modern agent systems work, then reuse it on Day 3 as the system under test.

Day 2 is not pure theory and not full production development. The focus is balanced:

- Learn the concepts.
- Build small working pieces.
- Keep every piece testable for Day 3.

## Schedule

| Block | Time | Topic | Outcome |
|---|---:|---|---|
| 1 | 1 hr | Agent architecture, ReAct, failure points | Explain agent flow and what can break |
| 2 | 1 hr | Tool calling with `@tool` and schemas | Understand tool definitions and arguments |
| 3 | 1.25 hrs | LangGraph basics | Build classifier -> tool/direct -> final flow |
| 4 | 1 hr | MCP concepts and tool registry | Model tool registration and discovery |
| 5 | 45 min | OpenAI Agents SDK overview | Compare Agents SDK with LangGraph |
| 6 | 1 hr | LangSmith concepts | Understand traces, runs, datasets, evaluators |
| 7 | 1 hr | Agent workbench design | Prepare an interview-ready architecture story |

## Mini App We Will Build

```text
User Query
   |
   v
Classifier Node
   |-- simple question -> Direct Answer
   |-- calculation -> Calculator Tool
   |-- customer query -> Customer Lookup Tool
   |-- ticket request -> Ticket Creation Tool
   |-- unsafe request -> Guardrail Response
   v
Final Response
   |
   v
Trace Record
```

## Day 3 Reuse

Day 3 will test this same mini app:

- Tool selection correctness
- Tool argument correctness
- Tool schema validation
- LangGraph branch/path validation
- Fallback and retry handling
- Hallucination and groundedness checks
- Prompt injection and guardrail checks
- Pytest quality gates
