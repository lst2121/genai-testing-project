import pytest


@pytest.fixture
def test_user():
    return {"name": "Lokender", "role": "tester"}


def is_palindrome(text: str) -> bool:
    return text == text[::-1]


def create_user(role: str) -> dict:
    if role not in {"admin", "tester", "viewer"}:
        raise ValueError(f"Unsupported role: {role}")
    return {"role": role}


def test_fixture_returns_user(test_user):
    assert test_user["role"] == "tester"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("madam", True),
        ("hello", False),
    ],
)
def test_parametrize_palindrome(text, expected):
    assert is_palindrome(text) is expected


def test_pytest_raises_for_invalid_role():
    with pytest.raises(ValueError):
        create_user("superadmin")


def test_fake_api_response_shape():
    response = {"status_code": 200, "body": {"customer_id": "CUST-101"}}
    assert response["status_code"] == 200
    assert response["body"]["customer_id"] == "CUST-101"


def add_numbers(a, b):
    return a + b

def test_add_numbers():
    assert add_numbers(1, 2) == 3
    assert add_numbers(1, -2) == -1
    assert add_numbers(0, 0) == 0
    assert add_numbers(-1, -1) == -2


@pytest.fixture
def test_user():
    return {"name": "Lokender", "role": "tester", "active": True}

def test_user_is_active(test_user):
    assert test_user["active"] is True

def test_user_is_not_active(test_user):
    assert test_user["active"] is False

def test_user_is_admin(test_user):
    assert test_user["role"] == "admin"

def test_user_is_tester(test_user):
    assert test_user["role"] == "tester"
    assert test_user["active"] is True

def test_user_is_viewer(test_user):
    assert test_user["role"] == "viewer"

def test_user_is_invalid(test_user):
    assert test_user["role"] == "invalid"

def is_even(number):
    return number % 2 == 0

@pytest.mark.parametrize(
    "number,expected",
    [
        (2, True),
        (3, False),
        (0, True),
        (-1, False),
    ],
)

def test_is_even(number, expected):
    assert is_even(number) is expected

def divide_numbers(a, b):
    return a / b

@pytest.mark.parametrize(
    "a,b,expected",
    [
        (1, 2, 0.5),
        (1, -2, -0.5),
        (-1, -1, 1),
    ],
)
def test_divide_numbers(a, b, expected):
    assert divide_numbers(a, b) == expected

def divide_numbers(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def test_divide_by_zero_raises_error():
    with pytest.raises(ValueError):
        divide_numbers(10, 0)

def test_customer_api_response_shape():
    response = {"status_code": 200, "body": {"customer_id": "CUST-101"}}
    assert response["status_code"] == 200
    assert response["body"]["customer_id"] == "CUST-101"

