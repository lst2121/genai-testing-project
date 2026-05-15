"""
================================================================================
DAY 4 - BLOCK 7: FINAL MOCK AND CHEAT SHEET
================================================================================
Goal:
Use this as the last-hour offline revision sheet.
================================================================================
"""


# =============================================================================
# 1. PYTHON QUICK QUESTIONS
# =============================================================================
"""
1. Reverse string.
2. Check palindrome.
3. Count frequency.
4. First non-repeating char.
5. Check anagram.
6. Remove duplicates.
7. Find duplicates.
8. Common elements.
9. Second largest.
10. Move zeroes.
11. Merge sorted lists.
12. Group anagrams.
13. Flatten nested list.
14. Valid parentheses.
15. Retry decorator.

Answer pattern:
    clarify input -> mention data structure -> code -> edge cases -> complexity
"""


# =============================================================================
# 2. OOPS QUICK ANSWERS
# =============================================================================
"""
OOP pillars:
    encapsulation, abstraction, inheritance, polymorphism

staticmethod:
    utility, no self/cls

classmethod:
    receives cls, class-level behavior or alternative constructor

class variable:
    shared by all instances

instance variable:
    per object

POM:
    encapsulates locators/actions in page classes

composition:
    object has another object, flexible for frameworks
"""


# =============================================================================
# 3. PYTEST QUICK ANSWERS
# =============================================================================
"""
fixture:
    reusable setup/teardown

conftest:
    shared fixtures/hooks

parametrize:
    same test with multiple data

pytest.raises:
    exception testing

markers:
    categorize smoke/regression/live_llm

CI:
    fast deterministic tests in PR, heavy/live tests in nightly
"""


# =============================================================================
# 4. PLAYWRIGHT QUICK ANSWERS
# =============================================================================
"""
Why Playwright:
    auto-waiting, locators, context isolation, tracing, API support

Context:
    isolated browser session

Locator:
    auto-waiting element handle

Auth:
    storageState, role-based fixtures, worker isolation

Flaky debugging:
    trace, screenshot, video, network/console logs, data isolation

Why TypeScript:
    type safety, maintainability, refactoring, typed POM/fixtures/API payloads
"""


# =============================================================================
# 5. AI METRICS QUICK ANSWERS
# =============================================================================
"""
context_precision:
    how much retrieved context is relevant

context_recall:
    how much required context was retrieved

faithfulness:
    answer claims supported by context

groundedness:
    answer backed by evidence/tool output

hallucination:
    unsupported invented information

tool_selection_accuracy:
    right tool selected

argument_accuracy:
    right tool args generated

guardrail false positive:
    safe prompt blocked

guardrail false negative:
    unsafe prompt allowed

LangSmith:
    trace/debug individual run

RAGAS:
    dataset-level RAG quality metrics
"""


# =============================================================================
# 6. MOCK QUESTIONS
# =============================================================================
"""
Coding:
    1. First non-repeating char
    2. Group anagrams
    3. Valid parentheses

OOP:
    Design BasePage and LoginPage.

pytest:
    Explain fixtures, parametrize, pytest.raises.

Playwright:
    How do you handle auth and parallel execution?

Project:
    Tell me about your Playwright framework.

AI:
    How do you test an agent workbench?

Dashboard:
    Context recall low, faithfulness low. What is wrong?
"""


# =============================================================================
# 7. FINAL SELF INTRO
# =============================================================================
"""
Use this structure:

    "I am a Senior SDET with strong experience in Playwright, TypeScript,
    API testing, CI/CD, and test framework design. I have worked on complex web
    automation with page objects, custom fixtures, parallel execution, and
    reporting. Recently I have also been working on GenAI testing concepts:
    agents, tool calling, MCP-style schemas, LangGraph-style flows, LangSmith
    tracing, prompt injection, groundedness, and eval metrics. I like choosing
    the right test layer instead of pushing everything through UI."
"""


# =============================================================================
# 8. LAST 30-MINUTE REVISION ORDER
# =============================================================================
"""
1. Python problems: 1, 4, 5, 9, 13, 14, 15.
2. OOP: class vs instance, staticmethod/classmethod, POM, retry decorator.
3. pytest: fixtures, parametrize, raises, markers.
4. Playwright: context, locators, auth, trace, TS benefits.
5. Project stories: framework, parallel auth, data validation, CI, AI-assisted.
6. AI metrics: context precision/recall, faithfulness, hallucination, LangSmith vs RAGAS.
"""
