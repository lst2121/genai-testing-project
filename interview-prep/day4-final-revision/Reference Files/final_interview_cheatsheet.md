# Final Interview Cheat Sheet

Use this file for last revision before the first interview. The focus is Round 1: Python coding, OOP, pytest, Playwright, TypeScript, and project discussion. AI/GenAI is included as a strong bonus area because the JD is AI-workbench focused.

## 1. Final 60-Second Intro

Sure. I am Lokender Singh. I have around 7 years of full-time experience in QA and SDET roles across healthcare, telecom, e-commerce, privacy, and analytics domains.

My core expertise is automation testing, API testing, database validation, dashboard validation, CI/CD quality gates, and framework development. I have mainly worked with Playwright, Cypress, TypeScript, JavaScript, Python, SQL, Postman, and CI tools like Azure DevOps.

Currently, I am working at Norstella on a healthcare analytics application. My work includes Playwright automation with TypeScript, Page Object Model, reusable fixtures, API validation, data integrity checks between UI, API, and database, smoke and regression suites, and CI/CD execution with reporting.

I have also worked on AI-assisted QA workflows using Cursor and MCP-style integrations, where we use AI to support test scenario generation, automation script creation, CI failure analysis, bug verification, data validation, and UAT readiness.

Along with that, I have tested AI assistant and GenAI workflows involving RAG, tool-calling, and MCP-style interactions. My testing focus there includes response correctness, groundedness, hallucination checks, context handling, tool selection, permissions, fallback behavior, and application-side validation.

## 2. Python Coding Patterns For SDET Interviews

Do not present yourself like a competitive DSA candidate. Present yourself as someone who can solve practical problems using Python strings, lists, dictionaries, sets, OOP, and test utilities.

Answer pattern:

1. Clarify input and edge cases.
2. Choose data structure.
3. Write clean code.
4. Mention time complexity.
5. Mention how this applies in automation or validation.

### String Pattern 1: Reverse String

```python
def reverse_string(text: str) -> str:
    return text[::-1]
```

Concepts: slicing, string immutability.

Interview line: Strings are immutable in Python, so slicing creates a new reversed string.

### String Pattern 2: Palindrome With Normalization

```python
def is_palindrome(text: str) -> bool:
    cleaned = ""
    for char in text:
        if char.isalnum():
            cleaned += char.lower()
    return cleaned == cleaned[::-1]
```

Two-pointer version:

```python
def is_palindrome_two_pointers(text: str) -> bool:
    left, right = 0, len(text) - 1

    while left < right:
        while left < right and not text[left].isalnum():
            left += 1
        while left < right and not text[right].isalnum():
            right -= 1

        if text[left].lower() != text[right].lower():
            return False

        left += 1
        right -= 1

    return True
```

Concepts: `isalnum()`, `lower()`, two pointers, normalization.

### String Pattern 3: Character Frequency

```python
def char_frequency(text: str) -> dict[str, int]:
    freq = {}
    for char in text:
        char = char.lower()
        if char.isalnum():
            freq[char] = freq.get(char, 0) + 1
    return freq
```

Concepts: dictionary, `.get()`, frequency counting.

Automation use: Count error types from logs or failed test names.

### String Pattern 4: First Non-Repeating Character

```python
def first_non_repeating_char(text: str) -> str | None:
    freq = {}

    for char in text:
        freq[char] = freq.get(char, 0) + 1

    for char in text:
        if freq[char] == 1:
            return char

    return None
```

Concepts: two-pass dictionary solution, `O(n)` time.

### String Pattern 5: First Repeating Character

```python
def first_repeating_char(text: str) -> str | None:
    seen = set()

    for char in text:
        if char in seen:
            return char
        seen.add(char)

    return None
```

Concepts: set membership, uniqueness, `O(1)` average lookup.

### String Pattern 6: Anagram

Sorting version:

```python
def is_anagram(a: str, b: str) -> bool:
    return sorted(a.lower()) == sorted(b.lower())
```

Frequency version:

```python
def is_anagram_frequency(a: str, b: str) -> bool:
    if len(a) != len(b):
        return False

    freq = {}
    for char in a:
        freq[char] = freq.get(char, 0) + 1

    for char in b:
        if char not in freq:
            return False
        freq[char] -= 1
        if freq[char] < 0:
            return False

    return True
```

Concepts: sorting vs hashmap, time complexity.

### List Pattern 1: Remove Duplicates While Preserving Order

```python
def remove_duplicates(items: list[int]) -> list[int]:
    seen = set()
    result = []

    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)

    return result
```

Concepts: list for order, set for membership.

Interview line: If order does not matter, `set(items)` is enough. If order matters, use a set plus result list.

### List Pattern 2: Second Largest Unique Number

```python
def second_largest(numbers: list[int]) -> int | None:
    unique_numbers = list(set(numbers))

    if len(unique_numbers) < 2:
        return None

    unique_numbers.sort(reverse=True)
    return unique_numbers[1]
```

Single-pass version:

```python
def second_largest_single_pass(numbers: list[int]) -> int | None:
    first = None
    second = None

    for number in numbers:
        if first is None or number > first:
            second = first
            first = number
        elif number != first and (second is None or number > second):
            second = number

    return second
```

Concepts: sorting, uniqueness, edge cases, single-pass tracking.

### List Pattern 3: Move Zeroes

```python
def move_zeroes(numbers: list[int]) -> list[int]:
    non_zero = []
    zero_count = 0

    for number in numbers:
        if number == 0:
            zero_count += 1
        else:
            non_zero.append(number)

    return non_zero + [0] * zero_count
```

Concepts: list building, preserving order.

### List Pattern 4: Flatten Nested List

```python
def flatten(items: list) -> list:
    result = []

    for item in items:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)

    return result
```

Concepts: recursion, `isinstance`, `append` vs `extend`.

### Dict Pattern 1: Group Anagrams

```python
def group_anagrams(words: list[str]) -> dict[str, list[str]]:
    groups = {}

    for word in words:
        key = "".join(sorted(word))
        groups.setdefault(key, []).append(word)

    return groups
```

Concepts: grouping, dictionary key, `setdefault`.

### Dict Pattern 2: Two Sum

```python
def two_sum(numbers: list[int], target: int) -> tuple[int, int] | None:
    seen = {}

    for index, number in enumerate(numbers):
        required = target - number
        if required in seen:
            return seen[required], index
        seen[number] = index

    return None
```

Concepts: hashmap lookup, `enumerate`, complement logic.

Keep this as light DSA only.

### Automation Pattern 1: Validate API Response Required Keys

```python
def has_required_keys(response: dict, required_keys: list[str]) -> bool:
    for key in required_keys:
        if key not in response:
            return False
    return True
```

Example:

```python
user_response = {"id": 101, "name": "Lokender", "role": "qa"}
assert has_required_keys(user_response, ["id", "name", "role"]) is True
```

Concepts: dictionary membership, schema-style validation.

### Automation Pattern 2: Compare UI Value And API Value

```python
def normalize_amount(value: str) -> float:
    cleaned = value.replace("$", "").replace(",", "").strip()
    return float(cleaned)


def values_match(ui_value: str, api_value: float) -> bool:
    return normalize_amount(ui_value) == api_value
```

Automation use: dashboard value validation.

### Automation Pattern 3: Count Test Statuses

```python
def count_test_statuses(results: list[dict]) -> dict[str, int]:
    summary = {}

    for result in results:
        status = result["status"]
        summary[status] = summary.get(status, 0) + 1

    return summary
```

Example:

```python
results = [
    {"name": "login", "status": "passed"},
    {"name": "checkout", "status": "failed"},
    {"name": "search", "status": "passed"},
]

assert count_test_statuses(results) == {"passed": 2, "failed": 1}
```

### Automation Pattern 4: Find Flaky Tests

```python
def find_flaky_tests(run_results: list[dict]) -> list[str]:
    history = {}

    for row in run_results:
        test_name = row["test_name"]
        status = row["status"]
        history.setdefault(test_name, set()).add(status)

    flaky = []
    for test_name, statuses in history.items():
        if "passed" in statuses and "failed" in statuses:
            flaky.append(test_name)

    return flaky
```

Concepts: dictionary of sets, test triage.

### Python Concepts To Revise

- `list`: ordered, mutable, allows duplicates.
- `tuple`: ordered, immutable, allows duplicates.
- `set`: unordered, unique values, fast membership.
- `dict`: key-value pairs, fast lookup by key.
- `==`: compares values.
- `is`: compares object identity.
- `sort()`: modifies list in place and returns `None`.
- `sorted()`: returns a new sorted list.
- Shallow copy: copies outer object, nested objects are shared.
- Deep copy: recursively copies nested objects.
- `*args`: variable positional arguments.
- `**kwargs`: variable keyword arguments.
- List comprehension: concise list building.
- `enumerate`: index plus value.
- `zip`: combine parallel iterables.

## 3. OOP Quick Revision

OOP pillars:

### Encapsulation

Encapsulation means keeping related data and behavior together inside one class.

In automation, Page Object Model is the best example. Instead of spreading locators and actions across multiple test files, we keep the login page locators and login behavior inside `LoginPage`.

```python
class LoginPage:
    def __init__(self, page):
        self.page = page
        self.username_input = page.get_by_label("Username")
        self.password_input = page.get_by_label("Password")
        self.login_button = page.get_by_role("button", name="Login")

    def login(self, username, password):
        self.username_input.fill(username)
        self.password_input.fill(password)
        self.login_button.click()
```

Interview line:

```text
Encapsulation means wrapping data and methods inside a class. In automation, POM is a good example because locators and page actions are kept inside one page class instead of being scattered across test files.
```

### Abstraction

Abstraction means hiding internal complexity and exposing only simple, useful behavior.

The test should not need to know every internal step of login. It should simply call `login_page.login()`.

```python
class LoginPage:
    def login(self, username, password):
        self.enter_username(username)
        self.enter_password(password)
        self.click_login()

    def enter_username(self, username):
        print(f"Entering username: {username}")

    def enter_password(self, password):
        print("Entering password")

    def click_login(self):
        print("Clicking login button")
```

Interview line:

```text
Abstraction means exposing only necessary behavior and hiding implementation details. In automation, a test calls login_page.login() without caring about locators, waits, fill steps, or click details.
```

### Inheritance

Inheritance means a child class can reuse behavior from a parent class.

In an API testing framework, common behavior like base URL, token, headers, and request helpers can stay in `BaseApiClient`. Specific API clients can inherit from it.

```python
class BaseApiClient:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.token = token

    def build_headers(self):
        return {"Authorization": f"Bearer {self.token}"}


class CustomerApiClient(BaseApiClient):
    def get_customer_url(self, customer_id):
        return f"{self.base_url}/customers/{customer_id}"
```

Interview line:

```text
Inheritance allows a child class to reuse common behavior from a parent class. In an API framework, BaseApiClient can hold common auth/header logic and CustomerApiClient can add customer-specific endpoints.
```

### Polymorphism

Polymorphism means different classes can expose the same method name but implement it differently.

In testing, different validators can all have a `validate()` method, but each validator checks a different thing.

```python
class StatusCodeValidator:
    def validate(self, response):
        return response["status_code"] == 200


class SchemaValidator:
    def validate(self, response):
        required_keys = ["id", "name", "email"]
        return all(key in response for key in required_keys)


class DatabaseRecordValidator:
    def validate(self, record):
        return record is not None
```

Usage:

```python
validators = [
    StatusCodeValidator(),
    SchemaValidator(),
    DatabaseRecordValidator(),
]

for validator in validators:
    validator.validate(response_or_record)
```

Interview line:

```text
Polymorphism means the same method name can behave differently based on the object. In testing, different validators can all expose validate(), but one validates status code, another validates schema, and another validates database records.
```

Quick memory:

- Encapsulation: keep related data and methods together.
- Abstraction: hide internal steps, expose simple methods.
- Inheritance: reuse parent class behavior.
- Polymorphism: same method name, different behavior.

Strong automation answer:

```text
Page Object Model uses encapsulation and abstraction. Base page/API client classes use inheritance. Different validators or page components exposing the same method name show polymorphism.
```

### `__init__` vs `__new__`

- `__new__` creates the object.
- `__init__` initializes the object after creation.
- In normal automation framework code, you mostly use `__init__`.

### Instance Variable vs Class Variable

```python
class TestRun:
    environment = "uat"  # class variable

    def __init__(self, name: str):
        self.name = name  # instance variable
```

### Static Method vs Class Method

```python
class UrlBuilder:
    base_url = "https://example.com"

    @staticmethod
    def clean_path(path: str) -> str:
        return path.strip("/")

    @classmethod
    def build_url(cls, path: str) -> str:
        return f"{cls.base_url}/{cls.clean_path(path)}"
```

### Page Object Model Example

```python
class LoginPage:
    def __init__(self, page):
        self.page = page
        self.username = page.get_by_label("Username")
        self.password = page.get_by_label("Password")
        self.submit = page.get_by_role("button", name="Login")

    async def login(self, user: str, password: str):
        await self.username.fill(user)
        await self.password.fill(password)
        await self.submit.click()
```

Interview line: POM keeps locators and page actions in one class so tests remain readable and maintainable.

### API Client Example

```python
class BaseApiClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.token = token

    def build_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"}


class CustomerApiClient(BaseApiClient):
    def get_customer_url(self, customer_id: str) -> str:
        return f"{self.base_url}/customers/{customer_id}"
```

## 4. Pytest Cheat Sheet

### Basic Test

```python
def add(a, b):
    return a + b


def test_add_two_numbers():
    assert add(2, 3) == 5
```

### Fixture

```python
import pytest


@pytest.fixture
def user_payload():
    return {"id": 1, "name": "Lokender", "role": "qa"}


def test_user_has_role(user_payload):
    assert user_payload["role"] == "qa"
```

### Parametrize

```python
import pytest


@pytest.mark.parametrize(
    "text, expected",
    [
        ("madam", True),
        ("hello", False),
    ],
)
def test_palindrome(text, expected):
    assert is_palindrome(text) == expected
```

### `pytest.raises`

```python
def divide(a, b):
    if b == 0:
        raise ValueError("b cannot be zero")
    return a / b


def test_divide_by_zero():
    with pytest.raises(ValueError, match="zero"):
        divide(10, 0)
```

### `conftest.py`

Use `conftest.py` for shared fixtures, hooks, and test configuration. Tests can use fixtures from `conftest.py` without importing them.

### CI Gate vs Live Eval Gate

- PR CI: deterministic tests, schema checks, permission checks, unit tests, stable Playwright smoke tests.
- Nightly or manual: live LLM tests, RAGAS evals, LangSmith trace checks, long regression, performance tests.

## 5. TypeScript And JavaScript For Playwright

### Why TypeScript?

TypeScript gives type safety, better autocomplete, safer refactoring, typed page objects, typed fixtures, and typed API payloads. In large automation frameworks, this reduces maintenance risk.

### `let`, `const`, `var`

- `const`: use when reassignment is not needed.
- `let`: use when value changes.
- `var`: avoid in modern TypeScript because of function scope and hoisting issues.

### Type vs Interface

```typescript
type UserRole = "admin" | "viewer";

interface User {
  id: string;
  name: string;
  role: UserRole;
}
```

Simple answer: both can describe object shapes. I often use `interface` for object contracts and `type` for unions or aliases.

### Async/Await

```typescript
test("user can login", async ({ page }) => {
  await page.goto("/login");
  await page.getByLabel("Username").fill("user1");
  await page.getByLabel("Password").fill("secret");
  await page.getByRole("button", { name: "Login" }).click();
  await expect(page.getByText("Dashboard")).toBeVisible();
});
```

Important: Playwright actions are async, so use `await`.

### Useful Array Methods

```typescript
const failed = results.filter((r) => r.status === "failed");
const names = results.map((r) => r.name);
const checkout = results.find((r) => r.name === "checkout");
```

### Type-Safe Page Object

```typescript
import { Page, Locator } from "@playwright/test";

export class LoginPage {
  readonly page: Page;
  readonly username: Locator;
  readonly password: Locator;

  constructor(page: Page) {
    this.page = page;
    this.username = page.getByLabel("Username");
    this.password = page.getByLabel("Password");
  }
}
```

## 6. Playwright Final Revision

### Why Playwright?

Playwright supports auto-waiting, reliable locators, browser contexts, cross-browser execution, tracing, screenshots, videos, API testing, network interception, parallel execution, and good CI support.

### Locator Strategy

Preferred order:

1. `getByRole`
2. `getByLabel`
3. `getByText`
4. `getByTestId`
5. CSS/XPath only when needed

Good locator:

```typescript
await page.getByRole("button", { name: "Submit" }).click();
```

Avoid:

```typescript
await page.locator("//div[3]/button[2]").click();
```

### Auto-Waiting

Playwright waits for elements to be actionable before clicking, filling, or asserting. Still, you should assert meaningful business outcomes.

### Avoid Hard Waits

Avoid:

```typescript
await page.waitForTimeout(5000);
```

Prefer:

```typescript
await expect(page.getByText("Saved successfully")).toBeVisible();
```

### Fixtures

Fixtures provide reusable setup like logged-in page, API client, test data, or role-based sessions.

### Auth Handling

Use `storageState` or role-based fixtures to avoid logging in through UI for every test. This makes CI faster and more stable.

### API Testing With Playwright

```typescript
test("customer api returns valid response", async ({ request }) => {
  const response = await request.get("/api/customers/123");
  expect(response.status()).toBe(200);

  const body = await response.json();
  expect(body).toHaveProperty("id");
  expect(body).toHaveProperty("name");
});
```

### Flaky Test Debugging

Check:

- Trace viewer
- Screenshot and video
- Network logs
- Console errors
- Test data isolation
- Wrong waits
- Parallel execution conflicts
- Environment instability

## 7. AI / GenAI / Agent Testing Cheat Sheet

### RAG vs Agentic Workflow

RAG:

```text
user question -> retrieve context -> LLM generates answer
```

Agentic workflow:

```text
user question -> intent -> tool selection -> tool execution -> tool result -> final answer
```

Interview line: Ella is best described as an AI assistant with RAG and agentic/tool-calling capabilities.

### MCP

MCP is like a standard protocol that allows AI agents to connect with external tools and data sources. Instead of hardcoding every tool inside one app, MCP exposes tools through a common interface.

Testing MCP/tool workflows:

- Tool discovery
- Schema validation
- Required arguments
- Permissions
- Tool availability
- Timeout and failure handling
- Final response based on tool output

### Tool Selection vs Tool Execution

- Tool selection: LLM decides which tool to call and with what arguments.
- Tool execution: application validates the tool name, validates arguments, checks permission, runs the tool, and returns result.

### LangGraph

LangGraph helps model agent workflows as nodes, edges, state, conditional routing, and retries. From QA side, test state transitions, branch paths, failure paths, and final output.

### LangSmith

LangSmith helps inspect traces: prompt, context, tool calls, intermediate steps, token usage, latency, errors, and final response. It helps identify whether the issue is retrieval, tool execution, prompt, model, or application logic.

### RAGAS

RAGAS is used to evaluate RAG quality. Important metrics:

- Context precision
- Context recall
- Faithfulness
- Answer relevancy

### Groundedness / Faithfulness / Hallucination

- Groundedness: answer is supported by retrieved context or tool output.
- Faithfulness: claims in the answer are supported by the provided context.
- Hallucination: answer contains unsupported or incorrect claims.

## 8. Metrics And Dashboard Diagnosis

### Context Precision

Formula:

```text
context_precision = relevant_retrieved_chunks / total_retrieved_chunks
```

Meaning: Out of all chunks retrieved, how many were actually useful?

Low CP means many irrelevant chunks were retrieved.

### Context Recall

Formula:

```text
context_recall = retrieved_required_facts / total_required_facts
```

Meaning: Out of all facts needed to answer correctly, how many were retrieved?

Low CR means important facts were missing.

### Faithfulness

Formula:

```text
faithfulness = supported_claims / total_claims
```

Meaning: How much of the final answer is supported by context/tool output?

Low faithfulness means hallucination or unsupported claims.

### Diagnosis Examples

Low CP + High CR:

```text
The retriever found most required facts, but it also retrieved many irrelevant chunks.
Likely issue: ranking, top-k too high, chunking, noisy index.
```

High CP + Low CR:

```text
Retrieved chunks are relevant, but many required facts are missing.
Likely issue: missing documents, metadata filter issue, top-k too low, embedding/indexing problem.
```

High CP + High CR + Low Faithfulness:

```text
Retrieval is good, but generation is hallucinating.
Likely issue: prompt, grounding instruction, model behavior, answer synthesis.
```

Low CP + Low CR + Low Faithfulness:

```text
Retrieval is weak and generation is unreliable.
First debug retrieval, then tune prompt/model behavior.
```

High Faithfulness + Low Answer Relevance:

```text
Answer may be grounded, but it does not answer the user's actual question.
Likely issue: intent understanding or prompt alignment.
```

### Confusion Matrix For Prompt Injection Guardrail

Assume positive means "dangerous prompt".

```text
TP: dangerous prompt blocked
FP: safe prompt blocked
TN: safe prompt allowed
FN: dangerous prompt allowed
```

Precision:

```text
precision = TP / (TP + FP)
```

Recall:

```text
recall = TP / (TP + FN)
```

High false positives:

```text
System is over-blocking safe prompts.
User experience problem.
```

High false negatives:

```text
Dangerous prompts are bypassing guardrail.
Security/safety problem.
```

Interview line: For safety systems, recall is often more critical because false negatives mean unsafe prompts are allowed. But too many false positives hurt user experience, so thresholds need balancing.

## 9. Agent Tool-Calling Metrics

### Tool Selection Accuracy

```text
correct_tool_selected / total_tool_selection_cases
```

Example: User asks "create a ticket", agent should call `ticket_creation`, not `knowledge_search`.

### Argument Accuracy

```text
correct_arguments / total_tool_calls
```

Example: User says priority critical, tool argument should be `priority="critical"`.

### Tool Execution Success Rate

```text
successful_tool_executions / total_tool_executions
```

Low success rate may mean schema mismatch, bad arguments, unavailable tools, permission issues, or backend failures.

### Final Answer Groundedness After Tool Output

The final answer should be based on the actual tool result. If tool fails, agent should not invent success.

Bad:

```text
Tool failed, but assistant says ticket created.
```

Good:

```text
I could not create the ticket because priority is invalid. Please choose low, medium, high, or critical.
```

### Agent Trajectory Correctness

Trajectory means the path the agent followed:

```text
intent -> selected tool -> arguments -> execution -> final answer
```

Final answer alone is not enough. For agents, you must validate the path also.

## 10. Project Story Answers

### Playwright Framework Ownership

I worked on a Playwright and TypeScript automation framework using Page Object Model, reusable fixtures, role/session handling, API helpers, reporting, and CI execution. My focus was to make the suite maintainable, fast, and stable in CI.

### UI/API/DB Validation

For data-heavy features, I validate the complete chain: source data or database, API response, and UI display. This catches issues where the UI looks correct but aggregation or backend data is wrong.

### Dashboard Validation

I validate filters, date ranges, counts, aggregations, sorting, empty states, and role-based visibility. For analytics dashboards, I also validate event payloads, ingestion, aggregation, and final dashboard values.

### CI/CD And Reporting

I use CI gates for smoke and regression confidence. Stable deterministic tests run in PR or deployment pipelines, while heavier suites, performance checks, and live AI evals can run nightly or manually. Reporting helps triage failures faster.

### Cursor Skills And MCP Workflow

I used Cursor skills and MCP-style workflows to support QA activities like Jira story analysis, test scenario generation, automation creation, CI failure analysis, bug verification, data validation, and UAT readiness. AI speeds up repetitive work, but final quality comes from deterministic assertions and review.

### Ella AI Assistant Testing

I would describe Ella as an AI assistant with RAG and tool-calling/MCP-style capabilities. I test response correctness, groundedness, hallucination, context handling, tool selection, arguments, permission checks, tool failure handling, fallback behavior, and whether the final response is based on retrieved context or tool output.

### Why You Fit DTDL

This role needs strong SDET fundamentals plus AI/GenAI testing understanding. I bring Playwright, TypeScript, Python, API testing, data validation, dashboard validation, CI/CD gates, and framework design experience. I also understand agentic workflows, MCP/tool calling, LangGraph, LangSmith, RAGAS, prompt injection, hallucination checks, and AI-assisted automation.

## 11. Last-Hour Rapid Recall

- Python: string/list/dict/set patterns, API validation utilities, OOP basics.
- Pytest: fixture, parametrize, `pytest.raises`, markers, `conftest.py`.
- Playwright: locators, auto-waiting, POM, fixtures, auth, API testing, trace debugging.
- TypeScript: type safety, async/await, interface/type, typed POM.
- AI testing: RAG vs agent, MCP, tool selection vs execution, groundedness, hallucination, CP/CR, guardrail confusion matrix.
- Project: EVPNxGen/Norstella story, data validation story, Cursor/MCP workflow, Ella assistant testing.
