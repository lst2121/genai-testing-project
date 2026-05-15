"""
================================================================================
BLOCK 6: PYTEST FUNDAMENTALS (1 hr)
================================================================================
Essential pytest concepts for SDET interviews
Note: pytest must be run from command line, not Jupyter notebooks
================================================================================
"""

# =============================================================================
# CONCEPT 1: BASIC TEST STRUCTURE
# =============================================================================
"""
KEY POINTS:
- Test files: test_*.py or *_test.py
- Test functions: def test_*()
- Test classes: class Test* (no __init__)
- Use assert for assertions (not assertEqual)
- pytest discovers tests automatically
- Run: pytest, pytest -v, pytest test_file.py

CODE EXAMPLES:
    # Simple test function
    def test_addition():
        assert 1 + 1 == 2
    
    # Test with descriptive name
    def test_user_can_login_with_valid_credentials():
        result = login("user", "pass")
        assert result.success is True
    
    # Test class
    class TestCalculator:
        def test_add(self):
            assert add(2, 3) == 5
        
        def test_subtract(self):
            assert subtract(5, 3) == 2

RUNNING TESTS:
    pytest                      # Run all tests
    pytest -v                   # Verbose output
    pytest test_file.py         # Specific file
    pytest test_file.py::test_func  # Specific test
    pytest -k "login"           # Tests matching pattern
    pytest -x                   # Stop on first failure
    pytest --tb=short           # Shorter tracebacks
"""


# =============================================================================
# CONCEPT 2: FIXTURES
# =============================================================================
"""
KEY POINTS:
- @pytest.fixture decorator
- Provide test data or setup/teardown
- Scopes: function (default), class, module, session
- yield for teardown (setup before, teardown after)
- conftest.py for shared fixtures
- Fixtures can use other fixtures

CODE EXAMPLES:
    import pytest
    
    # Basic fixture
    @pytest.fixture
    def sample_user():
        return {"name": "John", "email": "john@test.com"}
    
    def test_user_name(sample_user):
        assert sample_user["name"] == "John"
    
    # Fixture with teardown
    @pytest.fixture
    def db_connection():
        conn = create_connection()
        yield conn  # Test runs here
        conn.close()  # Teardown
    
    # Session-scoped fixture (runs once)
    @pytest.fixture(scope="session")
    def browser():
        driver = webdriver.Chrome()
        yield driver
        driver.quit()
    
    # Fixture using another fixture
    @pytest.fixture
    def logged_in_user(browser, test_user):
        browser.get("/login")
        login(browser, test_user)
        return test_user

FIXTURE SCOPES:
- function: New for each test (default)
- class: Once per test class
- module: Once per test file
- session: Once per entire test run
"""


# =============================================================================
# CONCEPT 3: PARAMETRIZE
# =============================================================================
"""
KEY POINTS:
- @pytest.mark.parametrize for data-driven tests
- Run same test with multiple inputs
- Each parameter set = separate test
- Can combine multiple parametrize decorators
- ids parameter for readable test names

CODE EXAMPLES:
    import pytest
    
    # Basic parametrize
    @pytest.mark.parametrize("input,expected", [
        (1, 2),
        (2, 4),
        (3, 6),
    ])
    def test_double(input, expected):
        assert input * 2 == expected
    
    # Multiple parameters with ids
    @pytest.mark.parametrize("username,password,expected", [
        ("admin", "admin123", True),
        ("user", "wrong", False),
        ("", "", False),
    ], ids=["valid_admin", "wrong_password", "empty_credentials"])
    def test_login(username, password, expected):
        result = login(username, password)
        assert result == expected
    
    # Combining parametrize (cartesian product)
    @pytest.mark.parametrize("x", [1, 2])
    @pytest.mark.parametrize("y", [10, 20])
    def test_multiply(x, y):
        assert x * y > 0
    # Runs: (1,10), (1,20), (2,10), (2,20)
"""


# =============================================================================
# CONCEPT 4: MARKERS
# =============================================================================
"""
KEY POINTS:
- @pytest.mark.* for categorizing tests
- Built-in: skip, skipif, xfail, parametrize
- Custom markers for filtering (smoke, regression, etc.)
- Register custom markers in pytest.ini
- Run specific markers: pytest -m "smoke"

CODE EXAMPLES:
    import pytest
    import sys
    
    # Skip unconditionally
    @pytest.mark.skip(reason="Not implemented yet")
    def test_future_feature():
        pass
    
    # Skip based on condition
    @pytest.mark.skipif(sys.platform == "win32", reason="Unix only")
    def test_unix_feature():
        pass
    
    # Expected failure
    @pytest.mark.xfail(reason="Known bug #123")
    def test_known_bug():
        assert buggy_function() == expected
    
    # Custom markers
    @pytest.mark.smoke
    def test_homepage_loads():
        pass
    
    @pytest.mark.regression
    @pytest.mark.slow
    def test_full_checkout_flow():
        pass

pytest.ini:
    [pytest]
    markers =
        smoke: Quick smoke tests
        regression: Full regression suite
        slow: Tests that take > 1 minute

RUNNING WITH MARKERS:
    pytest -m smoke              # Only smoke tests
    pytest -m "not slow"         # Exclude slow tests
    pytest -m "smoke or regression"  # Either marker
"""


# =============================================================================
# CONCEPT 5: CONFTEST.PY
# =============================================================================
"""
KEY POINTS:
- Special file for shared fixtures
- Auto-discovered by pytest (no import needed)
- Can have multiple conftest.py at different levels
- Fixtures available to all tests in same directory and below
- Good for: browser setup, API clients, test data

CODE EXAMPLES:
    # conftest.py
    import pytest
    
    @pytest.fixture(scope="session")
    def api_client():
        client = APIClient(base_url="https://api.test.com")
        client.authenticate()
        yield client
        client.close()
    
    @pytest.fixture
    def test_user(api_client):
        user = api_client.create_user(name="Test User")
        yield user
        api_client.delete_user(user.id)
    
    # Hook for adding custom CLI options
    def pytest_addoption(parser):
        parser.addoption(
            "--env",
            action="store",
            default="test",
            help="Environment to run tests against"
        )
    
    @pytest.fixture
    def env(request):
        return request.config.getoption("--env")

DIRECTORY STRUCTURE:
    tests/
    ├── conftest.py          # Root fixtures (browser, API)
    ├── api/
    │   ├── conftest.py      # API-specific fixtures
    │   └── test_users.py
    └── ui/
        ├── conftest.py      # UI-specific fixtures
        └── test_login.py
"""


# =============================================================================
# CONCEPT 6: ASSERTIONS & MATCHERS
# =============================================================================
"""
KEY POINTS:
- Plain assert statements (pytest rewrites for better output)
- pytest.raises for exception testing
- pytest.approx for floating point comparison
- Detailed assertion introspection (shows values on failure)

CODE EXAMPLES:
    import pytest
    
    # Basic assertions
    def test_assertions():
        assert 1 == 1
        assert "hello" in "hello world"
        assert [1, 2, 3] == [1, 2, 3]
        assert {"a": 1} == {"a": 1}
    
    # Exception testing
    def test_raises_value_error():
        with pytest.raises(ValueError):
            int("not a number")
    
    def test_raises_with_message():
        with pytest.raises(ValueError, match="invalid literal"):
            int("abc")
    
    # Floating point comparison
    def test_float():
        assert 0.1 + 0.2 == pytest.approx(0.3)
        assert [0.1, 0.2] == pytest.approx([0.1, 0.2])
    
    # Custom message on failure
    def test_with_message():
        result = calculate()
        assert result > 0, f"Expected positive, got {result}"
"""


# =============================================================================
# CONCEPT 7: PYTEST PLUGINS & FEATURES
# =============================================================================
"""
KEY POINTS:
- pytest-html: HTML reports
- pytest-xdist: Parallel execution
- pytest-rerunfailures: Retry flaky tests
- pytest-cov: Code coverage
- pytest-mock: Mocking support

USEFUL COMMANDS:
    pytest --html=report.html       # HTML report
    pytest -n auto                  # Parallel (xdist)
    pytest --reruns 3               # Retry failures
    pytest --cov=src --cov-report=html  # Coverage
    pytest --timeout=60             # Test timeout
    pytest -v --tb=short            # Verbose, short traceback
    pytest --collect-only           # List tests without running
    pytest --durations=10           # Show 10 slowest tests

CODE EXAMPLES:
    # Using pytest-mock
    def test_with_mock(mocker):
        mock_api = mocker.patch('module.api_call')
        mock_api.return_value = {"status": "ok"}
        
        result = function_that_calls_api()
        
        assert result == "ok"
        mock_api.assert_called_once()
    
    # Retry flaky tests
    @pytest.mark.flaky(reruns=3, reruns_delay=1)
    def test_flaky_api():
        response = call_unstable_api()
        assert response.status == 200
"""


# =============================================================================
# SAMPLE TEST FILE STRUCTURE
# =============================================================================
"""
PROJECT STRUCTURE:
    my_project/
    ├── src/
    │   ├── __init__.py
    │   ├── calculator.py
    │   └── user_service.py
    ├── tests/
    │   ├── __init__.py
    │   ├── conftest.py
    │   ├── test_calculator.py
    │   └── test_user_service.py
    ├── pytest.ini
    └── requirements.txt
"""

# =============================================================================
# EXAMPLE: conftest.py
# =============================================================================

import pytest
from typing import Generator, Any


# Session-scoped: runs once for entire test session
@pytest.fixture(scope="session")
def base_url() -> str:
    """Base URL for API tests"""
    return "https://api.example.com"


# Function-scoped: fresh for each test
@pytest.fixture
def sample_user() -> dict:
    """Sample user data for tests"""
    return {
        "id": 1,
        "name": "Test User",
        "email": "test@example.com",
        "is_active": True
    }


# Fixture with setup and teardown
@pytest.fixture
def temp_file(tmp_path) -> Generator[str, None, None]:
    """Create a temporary file for testing"""
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("test content")
    
    yield str(file_path)
    
    # Teardown: file automatically cleaned by tmp_path


# Fixture that depends on another fixture
@pytest.fixture
def authenticated_client(base_url: str) -> dict:
    """Return authenticated API client config"""
    return {
        "base_url": base_url,
        "token": "test-auth-token",
        "headers": {"Authorization": "Bearer test-auth-token"}
    }


# =============================================================================
# EXAMPLE: test_calculator.py
# =============================================================================

class Calculator:
    """Simple calculator for demonstration"""
    
    def add(self, a: int, b: int) -> int:
        return a + b
    
    def subtract(self, a: int, b: int) -> int:
        return a - b
    
    def multiply(self, a: int, b: int) -> int:
        return a * b
    
    def divide(self, a: int, b: int) -> float:
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b


class TestCalculator:
    """Test class for Calculator"""
    
    @pytest.fixture
    def calc(self) -> Calculator:
        """Calculator instance for each test"""
        return Calculator()
    
    def test_add(self, calc: Calculator):
        assert calc.add(2, 3) == 5
    
    def test_subtract(self, calc: Calculator):
        assert calc.subtract(5, 3) == 2
    
    @pytest.mark.parametrize("a,b,expected", [
        (2, 3, 6),
        (0, 5, 0),
        (-1, 5, -5),
        (10, 10, 100),
    ])
    def test_multiply(self, calc: Calculator, a: int, b: int, expected: int):
        assert calc.multiply(a, b) == expected
    
    def test_divide(self, calc: Calculator):
        assert calc.divide(10, 2) == 5.0
    
    def test_divide_by_zero_raises(self, calc: Calculator):
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            calc.divide(10, 0)
    
    @pytest.mark.skip(reason="Feature not implemented")
    def test_power(self, calc: Calculator):
        assert calc.power(2, 3) == 8


# =============================================================================
# EXAMPLE: test_user_service.py (API Testing Pattern)
# =============================================================================

class TestUserAPI:
    """API tests demonstrating pytest patterns"""
    
    @pytest.mark.smoke
    def test_get_user_returns_correct_data(self, sample_user: dict):
        """Verify user data structure"""
        assert "id" in sample_user
        assert "name" in sample_user
        assert "email" in sample_user
    
    @pytest.mark.parametrize("email,is_valid", [
        ("valid@email.com", True),
        ("invalid-email", False),
        ("", False),
        ("test@domain.co.uk", True),
    ], ids=["valid", "no_at_sign", "empty", "uk_domain"])
    def test_email_validation(self, email: str, is_valid: bool):
        """Test email validation with multiple cases"""
        result = self._validate_email(email)
        assert result == is_valid
    
    @staticmethod
    def _validate_email(email: str) -> bool:
        """Simple email validation"""
        return "@" in email and "." in email.split("@")[-1]
    
    @pytest.mark.regression
    def test_user_can_be_deactivated(self, sample_user: dict):
        """Test user deactivation flow"""
        sample_user["is_active"] = False
        assert sample_user["is_active"] is False


# =============================================================================
# pytest.ini EXAMPLE
# =============================================================================
"""
[pytest]
# Test discovery
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Output options
addopts = -v --tb=short --strict-markers

# Custom markers
markers =
    smoke: Quick smoke tests for CI
    regression: Full regression suite
    slow: Tests that take > 30 seconds
    api: API integration tests
    ui: UI/browser tests
    wip: Work in progress (skip in CI)

# Logging
log_cli = true
log_cli_level = INFO

# Timeout (requires pytest-timeout)
timeout = 300
"""


# =============================================================================
# QUICK REFERENCE CHEAT SHEET
# =============================================================================
"""
PYTEST CHEAT SHEET:

FIXTURES:
    @pytest.fixture                    # Function scope
    @pytest.fixture(scope="session")   # Session scope
    yield value                        # Setup + teardown
    
PARAMETRIZE:
    @pytest.mark.parametrize("a,b", [(1,2), (3,4)])
    
MARKERS:
    @pytest.mark.skip(reason="...")
    @pytest.mark.skipif(condition, reason="...")
    @pytest.mark.xfail(reason="...")
    @pytest.mark.smoke  # Custom
    
ASSERTIONS:
    assert x == y
    assert x in collection
    with pytest.raises(Exception):
    pytest.approx(0.3)
    
COMMANDS:
    pytest -v                  # Verbose
    pytest -k "pattern"        # Filter by name
    pytest -m smoke            # Filter by marker
    pytest -x                  # Stop on first failure
    pytest --collect-only      # List tests
    pytest -n auto             # Parallel (xdist)
"""


# =============================================================================
# TEST CASES (To verify examples work)
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("BLOCK 6: PYTEST FUNDAMENTALS")
    print("=" * 60)
    print("\nThis file contains pytest examples.")
    print("Run with: pytest block6_pytest_fundamentals.py -v")
    print("\nKey concepts covered:")
    print("1. Basic test structure")
    print("2. Fixtures (setup/teardown)")
    print("3. Parametrize (data-driven)")
    print("4. Markers (skip, xfail, custom)")
    print("5. conftest.py (shared fixtures)")
    print("6. Assertions & matchers")
    print("7. Plugins & CLI options")
    print("\n" + "=" * 60)
