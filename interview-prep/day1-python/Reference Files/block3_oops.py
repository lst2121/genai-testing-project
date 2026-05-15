"""
================================================================================
BLOCK 3: OOPs CONCEPTS + CODING (2 hrs)
================================================================================
Object-Oriented Programming concepts critical for SDET interviews
Theory + 6 practical coding tasks
================================================================================
"""

# =============================================================================
# CONCEPT 1: __init__ vs __new__
# =============================================================================
"""
KEY POINTS:
- __new__: Creates and returns a new instance (allocates memory)
- __init__: Initializes the instance (sets attributes)
- __new__ is called BEFORE __init__
- __new__ receives class (cls), __init__ receives instance (self)
- Override __new__ for: Singletons, immutable types, metaclasses
- 99% of the time you only need __init__

CODE EXAMPLES:
    class MyClass:
        def __new__(cls, *args, **kwargs):
            print("__new__ called")
            instance = super().__new__(cls)
            return instance
        
        def __init__(self, value):
            print("__init__ called")
            self.value = value
    
    obj = MyClass(10)
    # Output:
    # __new__ called
    # __init__ called

INTERVIEW QUESTION: When would you override __new__?
ANSWER: 
1. Singleton pattern (return existing instance)
2. Subclassing immutables (str, int, tuple) - must set value in __new__
3. Metaclasses and factory patterns
"""


# =============================================================================
# CONCEPT 2: CLASS vs INSTANCE VARIABLES
# =============================================================================
"""
KEY POINTS:
- Class variable: Defined in class body, shared by ALL instances
- Instance variable: Defined in __init__ with self, unique per instance
- Modifying class var through instance creates instance var (shadows it)
- Use self.__class__.var to modify class variable from instance
- Class variables are stored in ClassName.__dict__
- Instance variables are stored in instance.__dict__

CODE EXAMPLES:
    class Dog:
        species = "Canis familiaris"  # Class variable - shared
        
        def __init__(self, name):
            self.name = name  # Instance variable - unique
    
    dog1 = Dog("Buddy")
    dog2 = Dog("Max")
    
    print(dog1.species)  # "Canis familiaris"
    print(dog2.species)  # "Canis familiaris"
    
    Dog.species = "Wolf"  # Changes for ALL instances
    print(dog1.species)  # "Wolf"
    print(dog2.species)  # "Wolf"
    
    dog1.species = "Dog"  # Creates INSTANCE variable, shadows class var
    print(dog1.species)  # "Dog" (instance)
    print(dog2.species)  # "Wolf" (still class)

INTERVIEW TIP: Common gotcha with mutable class variables!
    class BadExample:
        items = []  # DANGER: Shared mutable!
        
        def add(self, item):
            self.items.append(item)  # Modifies shared list!
    
    # FIX: Initialize mutable in __init__
    class GoodExample:
        def __init__(self):
            self.items = []  # Each instance gets its own list
"""


# =============================================================================
# CONCEPT 3: @staticmethod vs @classmethod
# =============================================================================
"""
KEY POINTS:
- @staticmethod: No implicit first argument, like regular function in class
- @classmethod: First argument is class (cls), not instance
- Instance method: First argument is instance (self)
- @classmethod can access/modify class state, @staticmethod cannot
- Use @classmethod for factory methods (alternative constructors)
- Use @staticmethod for utility functions that don't need class/instance

CODE EXAMPLES:
    class Date:
        def __init__(self, year, month, day):
            self.year = year
            self.month = month
            self.day = day
        
        # Instance method - needs self
        def display(self):
            return f"{self.year}-{self.month}-{self.day}"
        
        # Class method - factory/alternative constructor
        @classmethod
        def from_string(cls, date_string):
            year, month, day = map(int, date_string.split('-'))
            return cls(year, month, day)  # Creates new instance
        
        # Static method - utility, no self/cls
        @staticmethod
        def is_valid_date(year, month, day):
            return 1 <= month <= 12 and 1 <= day <= 31
    
    # Usage
    d1 = Date(2024, 1, 15)
    d2 = Date.from_string("2024-01-15")  # classmethod
    valid = Date.is_valid_date(2024, 13, 1)  # staticmethod, returns False

INTERVIEW QUESTION: When to use each?
- Instance method: Needs to access/modify instance state
- @classmethod: Factory methods, needs access to class (inheritance-friendly)
- @staticmethod: Utility functions, logically belongs to class but needs no state
"""


# =============================================================================
# CONCEPT 4: DUNDER (MAGIC) METHODS
# =============================================================================
"""
KEY POINTS:
- Dunder = "double underscore" (__method__)
- Enable operator overloading and Python protocols
- __str__: Human-readable string (print, str())
- __repr__: Developer-friendly string (debugging, repr())
- __eq__: Equality comparison (==)
- __hash__: Hash value for dict/set membership
- __len__: Length (len())
- __getitem__: Indexing (obj[key])
- __iter__: Make object iterable
- __call__: Make object callable like function

CODE EXAMPLES:
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y
        
        def __repr__(self):
            return f"Point({self.x}, {self.y})"
        
        def __str__(self):
            return f"({self.x}, {self.y})"
        
        def __eq__(self, other):
            if not isinstance(other, Point):
                return NotImplemented
            return self.x == other.x and self.y == other.y
        
        def __hash__(self):
            return hash((self.x, self.y))
        
        def __add__(self, other):
            return Point(self.x + other.x, self.y + other.y)
    
    p1 = Point(1, 2)
    p2 = Point(1, 2)
    print(p1)        # (1, 2) - uses __str__
    print(repr(p1))  # Point(1, 2) - uses __repr__
    print(p1 == p2)  # True - uses __eq__
    print(p1 + p2)   # (2, 4) - uses __add__

CRITICAL FOR SDET:
- __eq__ is essential for test assertions (assertEqual, assert obj1 == obj2)
- Without __eq__, Python compares by identity (memory address)
"""


# =============================================================================
# CONCEPT 5: INHERITANCE & POLYMORPHISM
# =============================================================================
"""
KEY POINTS:
- Inheritance: class Child(Parent) - inherits attributes and methods
- super(): Access parent class methods
- Method Resolution Order (MRO): Order in which classes are searched
- Multiple inheritance: class Child(Parent1, Parent2)
- Polymorphism: Same interface, different implementations
- isinstance(obj, Class): Check if obj is instance of Class
- issubclass(Child, Parent): Check inheritance relationship

CODE EXAMPLES:
    class Animal:
        def __init__(self, name):
            self.name = name
        
        def speak(self):
            raise NotImplementedError("Subclass must implement")
    
    class Dog(Animal):
        def speak(self):
            return f"{self.name} says Woof!"
    
    class Cat(Animal):
        def speak(self):
            return f"{self.name} says Meow!"
    
    # Polymorphism - same interface, different behavior
    animals = [Dog("Buddy"), Cat("Whiskers")]
    for animal in animals:
        print(animal.speak())
    
    # MRO for multiple inheritance
    class A: pass
    class B(A): pass
    class C(A): pass
    class D(B, C): pass
    
    print(D.__mro__)  # (D, B, C, A, object)

INTERVIEW TIP: Know the difference between:
- is-a relationship (inheritance): Dog IS-A Animal
- has-a relationship (composition): Car HAS-A Engine
"""


# =============================================================================
# CONCEPT 6: ABSTRACT CLASSES (ABC)
# =============================================================================
"""
KEY POINTS:
- Abstract class: Cannot be instantiated, blueprint for subclasses
- from abc import ABC, abstractmethod
- @abstractmethod: Method that MUST be implemented by subclass
- Can have both abstract and concrete methods
- Use for defining interfaces/contracts in test frameworks

CODE EXAMPLES:
    from abc import ABC, abstractmethod
    
    class Shape(ABC):
        @abstractmethod
        def area(self):
            pass
        
        @abstractmethod
        def perimeter(self):
            pass
        
        def description(self):  # Concrete method
            return f"I am a shape with area {self.area()}"
    
    class Rectangle(Shape):
        def __init__(self, width, height):
            self.width = width
            self.height = height
        
        def area(self):
            return self.width * self.height
        
        def perimeter(self):
            return 2 * (self.width + self.height)
    
    # shape = Shape()  # TypeError: Can't instantiate abstract class
    rect = Rectangle(5, 3)
    print(rect.area())  # 15

SDET USE CASE:
- Abstract BasePage class for Page Object Model
- Abstract BaseTest class with common setup/teardown
"""


# =============================================================================
# CODING TASK 1: IMPLEMENT SINGLETON PATTERN
# =============================================================================
"""
PROBLEM: Implement a Logger class using Singleton pattern.
- Only ONE instance should exist throughout the application
- All calls to Logger() should return the same instance
- Should have methods: info(), warning(), error()

Use case: In test frameworks, you want a single logger instance
that all test classes share for consistent logging.
"""

class Logger:
    """Singleton Logger using __new__"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._logs = []
        return cls._instance
    
    def info(self, message: str) -> None:
        self._log("INFO", message)
    
    def warning(self, message: str) -> None:
        self._log("WARNING", message)
    
    def error(self, message: str) -> None:
        self._log("ERROR", message)
    
    def _log(self, level: str, message: str) -> None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        self._logs.append(log_entry)
        print(log_entry)
    
    def get_logs(self) -> list[str]:
        return self._logs.copy()


class LoggerMeta(type):
    """Alternative: Singleton using metaclass"""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


# =============================================================================
# CODING TASK 2: PAGE OBJECT MODEL BASE CLASS
# =============================================================================
"""
PROBLEM: Design a BasePage class for Page Object Model pattern.
- Should store page URL and driver reference
- Common methods: navigate(), get_title(), is_loaded()
- Subclasses will add page-specific locators and methods

This is the most common OOP question for SDET interviews!
"""

class BasePage:
    """Base class for Page Object Model"""
    
    def __init__(self, driver, base_url: str = ""):
        self.driver = driver
        self.base_url = base_url
        self.timeout = 10
    
    def navigate(self, path: str = "") -> None:
        """Navigate to page URL"""
        url = f"{self.base_url}{path}"
        self.driver.get(url)
    
    def get_title(self) -> str:
        """Get page title"""
        return self.driver.title
    
    def get_current_url(self) -> str:
        """Get current URL"""
        return self.driver.current_url
    
    def is_loaded(self) -> bool:
        """Check if page is loaded - override in subclass"""
        raise NotImplementedError("Subclass must implement is_loaded()")
    
    def find_element(self, locator: tuple):
        """Find element with wait"""
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        
        wait = WebDriverWait(self.driver, self.timeout)
        return wait.until(EC.presence_of_element_located(locator))
    
    def click(self, locator: tuple) -> None:
        """Click element"""
        element = self.find_element(locator)
        element.click()
    
    def type_text(self, locator: tuple, text: str) -> None:
        """Type text into element"""
        element = self.find_element(locator)
        element.clear()
        element.send_keys(text)


class LoginPage(BasePage):
    """Example page class extending BasePage"""
    
    # Locators (would use By.ID, By.CSS_SELECTOR, etc.)
    USERNAME_INPUT = ("id", "username")
    PASSWORD_INPUT = ("id", "password")
    LOGIN_BUTTON = ("id", "login-btn")
    ERROR_MESSAGE = ("class", "error-msg")
    
    def __init__(self, driver, base_url: str):
        super().__init__(driver, base_url)
        self.path = "/login"
    
    def is_loaded(self) -> bool:
        """Check if login page is loaded"""
        return "/login" in self.get_current_url()
    
    def login(self, username: str, password: str) -> None:
        """Perform login action"""
        self.navigate(self.path)
        self.type_text(self.USERNAME_INPUT, username)
        self.type_text(self.PASSWORD_INPUT, password)
        self.click(self.LOGIN_BUTTON)
    
    def get_error_message(self) -> str:
        """Get error message if login failed"""
        return self.find_element(self.ERROR_MESSAGE).text


# =============================================================================
# CODING TASK 3: TEST RESULT CLASS WITH EQUALITY
# =============================================================================
"""
PROBLEM: Create a TestResult class that can be compared for equality.
- Store: test_name, status (pass/fail), duration, error_message
- Implement __eq__ so two results with same values are equal
- Implement __repr__ for debugging
- Implement __hash__ so TestResult can be used in sets/dicts
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TestResultDataclass:
    """Using dataclass - automatically generates __eq__, __repr__, __hash__"""
    test_name: str
    status: str
    duration: float
    error_message: Optional[str] = None
    
    def is_passed(self) -> bool:
        return self.status == "pass"
    
    def is_failed(self) -> bool:
        return self.status == "fail"


class TestResult:
    """Manual implementation showing all dunder methods"""
    
    def __init__(self, test_name: str, status: str, duration: float, 
                 error_message: str = None):
        self.test_name = test_name
        self.status = status
        self.duration = duration
        self.error_message = error_message
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, TestResult):
            return NotImplemented
        return (self.test_name == other.test_name and 
                self.status == other.status and
                self.duration == other.duration and
                self.error_message == other.error_message)
    
    def __hash__(self) -> int:
        return hash((self.test_name, self.status, self.duration, self.error_message))
    
    def __repr__(self) -> str:
        return (f"TestResult(test_name={self.test_name!r}, status={self.status!r}, "
                f"duration={self.duration}, error_message={self.error_message!r})")
    
    def __str__(self) -> str:
        status_emoji = "✓" if self.status == "pass" else "✗"
        return f"{status_emoji} {self.test_name} ({self.duration:.2f}s)"
    
    def is_passed(self) -> bool:
        return self.status == "pass"
    
    def is_failed(self) -> bool:
        return self.status == "fail"


# =============================================================================
# CODING TASK 4: RETRY DECORATOR
# =============================================================================
"""
PROBLEM: Implement a retry decorator for flaky tests.
- Takes max_attempts and delay parameters
- Retries function if it raises an exception
- Should work with any function (tests, API calls, etc.)
- Log each retry attempt

This is a VERY common SDET interview question!
"""

import time
import functools


def retry(max_attempts: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    """
    Retry decorator for flaky operations.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Seconds to wait between retries
        exceptions: Tuple of exceptions to catch and retry
    
    Usage:
        @retry(max_attempts=3, delay=2)
        def flaky_test():
            # test code that might fail intermittently
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        print(f"Attempt {attempt}/{max_attempts} failed: {e}")
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        print(f"All {max_attempts} attempts failed")
            
            raise last_exception
        
        return wrapper
    return decorator


def retry_simple(max_attempts: int = 3):
    """Simpler version without delay parameter"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"Retry {attempt + 1}/{max_attempts}: {e}")
        return wrapper
    return decorator


# Example usage
@retry(max_attempts=3, delay=0.5)
def flaky_api_call():
    """Simulates a flaky API that sometimes fails"""
    import random
    if random.random() < 0.7:  # 70% chance of failure
        raise ConnectionError("API temporarily unavailable")
    return {"status": "success"}


# =============================================================================
# CODING TASK 5: ABSTRACT BASE PAGE WITH ABC
# =============================================================================
"""
PROBLEM: Create an abstract BasePage class using ABC module.
- Define abstract methods that ALL page classes must implement
- Provide concrete helper methods
- Demonstrate proper use of abstractmethod

This shows understanding of interfaces/contracts in test frameworks.
"""

from abc import ABC, abstractmethod
from typing import Any


class AbstractBasePage(ABC):
    """Abstract base class for all page objects"""
    
    def __init__(self, driver: Any):
        self.driver = driver
        self._validate_page()
    
    @property
    @abstractmethod
    def url_path(self) -> str:
        """Each page must define its URL path"""
        pass
    
    @property
    @abstractmethod
    def page_title(self) -> str:
        """Each page must define its expected title"""
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Each page must implement its own load check"""
        pass
    
    @abstractmethod
    def get_page_elements(self) -> dict:
        """Each page must return its locators"""
        pass
    
    def _validate_page(self) -> None:
        """Concrete method - validates page is correct"""
        if not self.is_loaded():
            raise PageNotLoadedError(f"Page {self.url_path} did not load correctly")
    
    def navigate_to(self) -> None:
        """Concrete method - navigates to this page"""
        base_url = getattr(self.driver, 'base_url', '')
        self.driver.get(f"{base_url}{self.url_path}")
    
    def take_screenshot(self, filename: str) -> str:
        """Concrete method - takes screenshot"""
        path = f"screenshots/{filename}.png"
        self.driver.save_screenshot(path)
        return path


class PageNotLoadedError(Exception):
    """Custom exception for page load failures"""
    pass


class ConcreteHomePage(AbstractBasePage):
    """Concrete implementation of a home page"""
    
    @property
    def url_path(self) -> str:
        return "/home"
    
    @property
    def page_title(self) -> str:
        return "Welcome - Home"
    
    def is_loaded(self) -> bool:
        return self.page_title in self.driver.title
    
    def get_page_elements(self) -> dict:
        return {
            "search_box": ("id", "search"),
            "nav_menu": ("class", "navigation"),
            "user_profile": ("id", "user-profile")
        }


# =============================================================================
# CODING TASK 6: INHERITANCE - API CLIENT HIERARCHY
# =============================================================================
"""
PROBLEM: Build an API client hierarchy with inheritance.
- BaseAPIClient: Common functionality (headers, auth, logging)
- RestClient: REST-specific methods (GET, POST, PUT, DELETE)
- GraphQLClient: GraphQL-specific methods (query, mutation)

Demonstrates proper use of inheritance and super().
"""

import json
from typing import Optional
from dataclasses import dataclass


@dataclass
class APIResponse:
    """Response object from API calls"""
    status_code: int
    body: dict
    headers: dict
    elapsed_time: float
    
    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300
    
    def json(self) -> dict:
        return self.body


class BaseAPIClient:
    """Base class for all API clients"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = None  # Would be requests.Session()
        self._default_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    def set_auth_token(self, token: str) -> None:
        """Set authorization header"""
        self._default_headers["Authorization"] = f"Bearer {token}"
    
    def set_header(self, key: str, value: str) -> None:
        """Set custom header"""
        self._default_headers[key] = value
    
    def _build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint"""
        endpoint = endpoint.lstrip('/')
        return f"{self.base_url}/{endpoint}"
    
    def _log_request(self, method: str, url: str, **kwargs) -> None:
        """Log outgoing request"""
        print(f"[REQUEST] {method} {url}")
        if 'json' in kwargs:
            print(f"[BODY] {json.dumps(kwargs['json'], indent=2)}")
    
    def _log_response(self, response: APIResponse) -> None:
        """Log incoming response"""
        print(f"[RESPONSE] Status: {response.status_code}")


class RestClient(BaseAPIClient):
    """REST API client with standard HTTP methods"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        super().__init__(base_url, timeout)
    
    def get(self, endpoint: str, params: dict = None) -> APIResponse:
        """HTTP GET request"""
        url = self._build_url(endpoint)
        self._log_request("GET", url)
        
        # Simulated response (would use requests library)
        return APIResponse(
            status_code=200,
            body={"message": "success"},
            headers={},
            elapsed_time=0.1
        )
    
    def post(self, endpoint: str, data: dict = None) -> APIResponse:
        """HTTP POST request"""
        url = self._build_url(endpoint)
        self._log_request("POST", url, json=data)
        
        return APIResponse(
            status_code=201,
            body={"id": 1, **data} if data else {},
            headers={},
            elapsed_time=0.15
        )
    
    def put(self, endpoint: str, data: dict = None) -> APIResponse:
        """HTTP PUT request"""
        url = self._build_url(endpoint)
        self._log_request("PUT", url, json=data)
        
        return APIResponse(
            status_code=200,
            body=data or {},
            headers={},
            elapsed_time=0.12
        )
    
    def delete(self, endpoint: str) -> APIResponse:
        """HTTP DELETE request"""
        url = self._build_url(endpoint)
        self._log_request("DELETE", url)
        
        return APIResponse(
            status_code=204,
            body={},
            headers={},
            elapsed_time=0.08
        )


class GraphQLClient(BaseAPIClient):
    """GraphQL API client"""
    
    def __init__(self, base_url: str, endpoint: str = "/graphql"):
        super().__init__(base_url)
        self.graphql_endpoint = endpoint
    
    def query(self, query: str, variables: dict = None) -> APIResponse:
        """Execute GraphQL query"""
        url = self._build_url(self.graphql_endpoint)
        payload = {
            "query": query,
            "variables": variables or {}
        }
        self._log_request("POST (GraphQL Query)", url, json=payload)
        
        return APIResponse(
            status_code=200,
            body={"data": {}},
            headers={},
            elapsed_time=0.2
        )
    
    def mutation(self, mutation: str, variables: dict = None) -> APIResponse:
        """Execute GraphQL mutation"""
        url = self._build_url(self.graphql_endpoint)
        payload = {
            "query": mutation,
            "variables": variables or {}
        }
        self._log_request("POST (GraphQL Mutation)", url, json=payload)
        
        return APIResponse(
            status_code=200,
            body={"data": {}},
            headers={},
            elapsed_time=0.25
        )


# =============================================================================
# TEST CASES
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("BLOCK 3: OOPs CONCEPTS - TEST CASES")
    print("=" * 60)
    
    # Task 1: Singleton Logger
    print("\n1. Singleton Logger:")
    logger1 = Logger()
    logger2 = Logger()
    assert logger1 is logger2, "Should be same instance"
    logger1.info("Test message")
    print(f"   Same instance: {logger1 is logger2}")
    print("   Singleton pattern works!")
    
    # Task 2: Page Object Model
    print("\n2. Page Object Model:")
    print("   BasePage and LoginPage classes created")
    print("   Methods: navigate(), find_element(), click(), type_text()")
    
    # Task 3: TestResult with equality
    print("\n3. TestResult with equality:")
    result1 = TestResult("test_login", "pass", 1.5)
    result2 = TestResult("test_login", "pass", 1.5)
    result3 = TestResult("test_login", "fail", 1.5)
    assert result1 == result2, "Same values should be equal"
    assert result1 != result3, "Different status should not be equal"
    assert hash(result1) == hash(result2), "Equal objects should have same hash"
    print(f"   result1 == result2: {result1 == result2}")
    print(f"   result1 == result3: {result1 == result3}")
    print(f"   str(result1): {str(result1)}")
    
    # Task 4: Retry Decorator
    print("\n4. Retry Decorator:")
    print("   @retry(max_attempts=3, delay=1.0) decorator created")
    print("   Catches exceptions and retries automatically")
    
    # Task 5: Abstract Base Page
    print("\n5. Abstract Base Page:")
    print("   AbstractBasePage with @abstractmethod decorators")
    print("   ConcreteHomePage implements all abstract methods")
    
    # Task 6: API Client Hierarchy
    print("\n6. API Client Hierarchy:")
    rest_client = RestClient("https://api.example.com")
    rest_client.set_auth_token("test-token")
    response = rest_client.post("/users", {"name": "John"})
    print(f"   POST response: {response.body}")
    assert response.is_success
    
    graphql_client = GraphQLClient("https://api.example.com")
    query = "query { users { id name } }"
    response = graphql_client.query(query)
    print(f"   GraphQL query executed: {response.is_success}")
    
    print("\n" + "=" * 60)
    print("BLOCK 3 COMPLETE - ALL 6 CODING TASKS IMPLEMENTED!")
    print("=" * 60)
