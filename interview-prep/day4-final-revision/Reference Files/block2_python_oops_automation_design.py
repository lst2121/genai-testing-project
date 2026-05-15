"""
================================================================================
DAY 4 - BLOCK 2: PYTHON OOPS AND AUTOMATION DESIGN
================================================================================
Goal:
Revise Python OOP concepts through SDET-friendly examples: Page Object Model,
API clients, test result models, retry decorators, factories, and framework
design.

Interview expectation:
You should be able to explain OOP theory and connect it to automation framework
design.
================================================================================
"""


# =============================================================================
# 1. CLASS AND OBJECT
# =============================================================================
"""
Class:
    Blueprint.

Object:
    Instance created from the class.

Example:
"""


class TestResult:
    def __init__(self, test_name: str, status: str, duration_seconds: float):
        self.test_name = test_name
        self.status = status
        self.duration_seconds = duration_seconds

    def is_passed(self) -> bool:
        return self.status.lower() == "passed"


"""
Interview line:
    A class defines behavior and data; an object is a runtime instance with its
    own state. In automation, Page Objects, API clients, and test result models
    are common classes.
"""


# =============================================================================
# 2. __init__ VS __new__
# =============================================================================
"""
__new__:
    Creates the object.

__init__:
    Initializes the object after it is created.

Most automation code uses __init__, not __new__.

Example:
"""


class BasePage:
    def __init__(self, page):
        self.page = page


"""
Interview line:
    __new__ controls object creation, while __init__ sets initial state. I use
    __init__ commonly in page objects to inject the Playwright page instance.
"""


# =============================================================================
# 3. INSTANCE VARIABLE VS CLASS VARIABLE
# =============================================================================
"""
Instance variable:
    Unique per object.

Class variable:
    Shared by all instances.
"""


class ApiClient:
    default_timeout = 30  # class variable

    def __init__(self, base_url: str):
        self.base_url = base_url  # instance variable


"""
Interview warning:
    Do not use mutable class variables like [] or {} for per-test state, because
    they are shared across all instances and can leak data between tests.
"""


# =============================================================================
# 4. STATICMETHOD VS CLASSMETHOD
# =============================================================================
"""
@staticmethod:
    Utility function inside a class. Does not receive self or cls.

@classmethod:
    Receives cls. Often used for alternative constructors or class-level config.
"""


class UserFactory:
    domain = "example.com"

    @staticmethod
    def is_valid_role(role: str) -> bool:
        return role in {"admin", "tester", "viewer"}

    @classmethod
    def create_email(cls, username: str) -> str:
        return f"{username}@{cls.domain}"


"""
Interview line:
    I use staticmethod for stateless validation helpers and classmethod when the
    method needs class-level configuration or returns an instance of the class.
"""


# =============================================================================
# 5. INHERITANCE AND POLYMORPHISM
# =============================================================================
"""
Inheritance:
    Child class reuses behavior from parent class.

Polymorphism:
    Different classes expose the same method name but implement it differently.
"""


class BaseApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def build_url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"


class CustomerApiClient(BaseApiClient):
    def get_customer_url(self, customer_id: str) -> str:
        return self.build_url(f"/customers/{customer_id}")


class TicketApiClient(BaseApiClient):
    def get_ticket_url(self, ticket_id: str) -> str:
        return self.build_url(f"/tickets/{ticket_id}")


"""
Interview line:
    In automation frameworks, inheritance is useful for common behavior like
    navigation, logging, request building, or assertions. But too much inheritance
    makes frameworks rigid, so composition is often better.
"""


# =============================================================================
# 6. ABSTRACTION WITH ABC
# =============================================================================
"""
Abstraction hides details and exposes a contract.

In Python, use ABC for abstract classes.
"""

from abc import ABC, abstractmethod


class ReportPublisher(ABC):
    @abstractmethod
    def publish(self, results: list[TestResult]) -> None:
        pass


class ConsoleReportPublisher(ReportPublisher):
    def publish(self, results: list[TestResult]) -> None:
        for result in results:
            print(result.test_name, result.status)


# =============================================================================
# 7. ENCAPSULATION
# =============================================================================
"""
Encapsulation means keeping data and behavior together and controlling access.

Python uses convention:
    _internal
    __private_name_mangled
"""


class TokenStore:
    def __init__(self):
        self._token = None

    def set_token(self, token: str) -> None:
        if not token:
            raise ValueError("token cannot be empty")
        self._token = token

    def get_auth_header(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}


# =============================================================================
# 8. DUNDER METHODS
# =============================================================================
"""
__str__:
    User-friendly string.

__repr__:
    Developer/debug representation.

__eq__:
    Equality comparison.

__hash__:
    Allows object to be used in set/dict key if immutable.
"""


class User:
    def __init__(self, user_id: str, role: str):
        self.user_id = user_id
        self.role = role

    def __repr__(self) -> str:
        return f"User(user_id={self.user_id!r}, role={self.role!r})"

    def __eq__(self, other) -> bool:
        return isinstance(other, User) and self.user_id == other.user_id

    def __hash__(self) -> int:
        return hash(self.user_id)


# =============================================================================
# 9. COMPOSITION VS INHERITANCE
# =============================================================================
"""
Inheritance:
    "is-a" relationship.

Composition:
    "has-a" relationship.

Automation example:
    LoginPage has a Logger.
    ApiClient has a TokenStore.

Composition is often better for frameworks because it is flexible and avoids
deep inheritance chains.
"""


class Logger:
    def info(self, message: str) -> None:
        print(f"[INFO] {message}")


class LoginPage:
    def __init__(self, page, logger: Logger):
        self.page = page
        self.logger = logger

    def login(self, username: str, password: str) -> None:
        self.logger.info(f"Logging in as {username}")
        # page interactions would go here


# =============================================================================
# 10. PAGE OBJECT MODEL DESIGN
# =============================================================================
"""
POM goal:
    Keep locators and page behavior in page classes, not scattered across tests.

Good Page Object:
    - exposes business actions
    - hides locator details
    - does not contain too many assertions
    - is reusable
"""


class ExampleLoginPage:
    def __init__(self, page):
        self.page = page
        self.username_input = page.get_by_label("Username")
        self.password_input = page.get_by_label("Password")
        self.submit_button = page.get_by_role("button", name="Login")

    async def login(self, username: str, password: str) -> None:
        await self.username_input.fill(username)
        await self.password_input.fill(password)
        await self.submit_button.click()


"""
Interview line:
    POM applies encapsulation. Tests call business methods like login(), while
    locator details stay inside the page class.
"""


# =============================================================================
# 11. RETRY DECORATOR
# =============================================================================
"""
Retry decorators are common in automation, but should be used carefully.
Do not hide product bugs with retries.
"""


def retry(times: int = 3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_error = None
            for _ in range(times):
                try:
                    return func(*args, **kwargs)
                except Exception as error:
                    last_error = error
            raise last_error

        return wrapper

    return decorator


# =============================================================================
# LIKELY INTERVIEW QUESTIONS
# =============================================================================
"""
1. Explain OOPs pillars.

    Encapsulation, abstraction, inheritance, polymorphism.


2. How does POM use OOP?

    Page classes encapsulate locators and page actions. Tests use page methods
    instead of raw selectors.


3. staticmethod vs classmethod?

    staticmethod does not receive class or instance. classmethod receives cls and
    can access class-level config or create objects.


4. Class variable vs instance variable?

    Class variable is shared by all instances. Instance variable belongs to one
    object.


5. Why composition over inheritance?

    Composition avoids deep inheritance and lets classes use small reusable
    services like logger, token store, API client, or config provider.


6. What are dunder methods?

    Special methods like __repr__, __eq__, and __hash__ that customize object
    behavior.


7. Design BasePage.

    BasePage should hold page instance and common actions like navigate, click,
    fill, wait, screenshot. Specific pages inherit or compose it.


8. Design API client.

    Base client handles base_url, headers, auth, and request methods. Domain
    clients like CustomerApiClient add business-specific methods.
"""


# =============================================================================
# OFFLINE PRACTICE TASKS
# =============================================================================
"""
Write these in the notebook or tests/test_oops_design.py:

1. Create TestResult class with is_passed().
2. Create User class with __repr__ and __eq__.
3. Create BaseApiClient and CustomerApiClient.
4. Create retry decorator.
5. Create BasePage/LoginPage skeleton.
6. Create UserFactory with staticmethod and classmethod.
7. Explain each design in 2-3 lines as if in interview.
"""
