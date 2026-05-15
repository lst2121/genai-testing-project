"""
================================================================================
DAY 4 - BLOCK 3: PYTEST AND TEST FRAMEWORK REVISION
================================================================================
Goal:
Revise pytest concepts and be ready to explain how to design/maintain a Python
automation framework.
================================================================================
"""


# =============================================================================
# 1. BASIC TEST STRUCTURE
# =============================================================================
"""
Pytest discovers:
    files named test_*.py or *_test.py
    functions named test_*

Basic test:

    def test_addition():
        assert 2 + 2 == 4

Good tests:
    - clear name
    - arrange/act/assert structure
    - independent
    - deterministic
"""


# =============================================================================
# 2. FIXTURES
# =============================================================================
"""
Fixture:
    Reusable setup/teardown provider.

Example:

    import pytest

    @pytest.fixture
    def user():
        return {"name": "Lokender", "role": "tester"}

    def test_user_role(user):
        assert user["role"] == "tester"

Fixture scopes:
    function - default, new fixture per test
    class
    module
    package
    session

Interview line:
    Use function scope for isolation. Use session scope for expensive setup like
    browser, DB connection, or auth token when safe.
"""


# =============================================================================
# 3. CONFTST.PY
# =============================================================================
"""
conftest.py:
    Shared fixtures and pytest hooks for a test directory tree.

Use for:
    browser/page fixtures
    API client fixtures
    test data fixtures
    custom markers
    hooks for reporting

Do not import conftest manually. Pytest discovers it automatically.
"""


# =============================================================================
# 4. PARAMETRIZE
# =============================================================================
"""
Use parametrize to run same test with many inputs.

Example:

    @pytest.mark.parametrize("text,expected", [
        ("madam", True),
        ("hello", False),
    ])
    def test_palindrome(text, expected):
        assert is_palindrome(text) is expected

Interview line:
    Parametrize reduces duplicate tests and makes data-driven testing simple.
"""


# =============================================================================
# 5. PYTEST.RAISES
# =============================================================================
"""
Use pytest.raises for exception testing.

Example:

    def test_invalid_role_raises():
        with pytest.raises(ValueError):
            create_user(role="superadmin")

SDET relevance:
    Useful for schema validation, permission errors, bad API payloads, and
    negative tests.
"""


# =============================================================================
# 6. MARKERS
# =============================================================================
"""
Markers categorize tests:

    @pytest.mark.smoke
    @pytest.mark.regression
    @pytest.mark.live_llm
    @pytest.mark.flaky

Run:
    pytest -m smoke
    pytest -m "not live_llm"

Interview line:
    I use markers to separate PR gates, nightly tests, live LLM evals, and slow
    regression tests.
"""


# =============================================================================
# 7. MOCKING
# =============================================================================
"""
Mocking replaces external dependencies.

Use when:
    - API is unstable
    - third-party service costs money
    - you need deterministic failure
    - live LLM should not run in PR CI

Example idea:
    Mock payment API response.
    Mock OpenAI response for deterministic unit test.
"""


# =============================================================================
# 8. API TESTING WITH PYTEST
# =============================================================================
"""
API test checks:
    status code
    response schema
    required fields
    business rules
    auth/permissions
    error messages

Example shape:

    def test_get_customer(api_client):
        response = api_client.get("/customers/CUST-101")
        assert response.status_code == 200
        body = response.json()
        assert body["customer_id"] == "CUST-101"
"""


# =============================================================================
# 9. FRAMEWORK STRUCTURE
# =============================================================================
"""
Example Python automation framework:

    tests/
        conftest.py
        test_customers.py
        test_tickets.py
    clients/
        base_client.py
        customer_client.py
    pages/
        base_page.py
        login_page.py
    data/
        test_users.json
    utils/
        retry.py
        assertions.py
    pytest.ini

Key principles:
    - keep tests readable
    - put API logic in clients
    - put UI locators/actions in pages
    - put reusable checks in helpers
    - avoid sleep
    - isolate test data
"""


# =============================================================================
# 10. CI STRATEGY
# =============================================================================
"""
PR gate:
    smoke tests
    deterministic API/unit tests
    schema/permission tests

Nightly:
    broader regression
    live integrations
    live LLM evals

Release:
    full regression
    performance/accessibility/security checks if needed

Keep CI green:
    - avoid flaky waits
    - isolate data
    - use retries only for infra instability
    - quarantine flaky tests with ownership
    - report clearly
"""


# =============================================================================
# LIKELY INTERVIEW QUESTIONS
# =============================================================================
"""
1. What is fixture?
    Reusable setup/teardown dependency injected into tests.

2. Why conftest.py?
    To share fixtures/hooks across tests without manual imports.

3. Parametrize use case?
    Run same test logic with multiple data sets.

4. How to test exception?
    pytest.raises.

5. How to manage flaky tests?
    Find root cause, add proper waits, isolate data, improve selectors, use
    retries only for infra, quarantine with tracking.

6. What should run in PR?
    Fast deterministic smoke/API/unit tests. Not slow/flaky/live-cost tests.

7. How to design test framework?
    Separate tests, pages, API clients, fixtures, test data, utilities, and CI
    config.
"""


# =============================================================================
# OFFLINE PRACTICE TASKS
# =============================================================================
"""
In tests/test_pytest_patterns.py:

1. Write fixture returning test user.
2. Parametrize palindrome test.
3. Test ValueError using pytest.raises.
4. Add marker examples.
5. Write small API response validation using a fake dict.
"""
