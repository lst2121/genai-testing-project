class ResultRecord:
    def __init__(self, test_name: str, status: str):
        self.test_name = test_name
        self.status = status

    def is_passed(self) -> bool:
        return self.status.lower() == "passed"


class BaseApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def build_url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"


class CustomerApiClient(BaseApiClient):
    def get_customer_url(self, customer_id: str) -> str:
        return self.build_url(f"/customers/{customer_id}")


def test_result_is_passed():
    result = ResultRecord("login test", "passed")
    assert result.is_passed() is True


def test_base_api_client_builds_url():
    client = BaseApiClient("https://api.example.com/")
    assert client.build_url("/health") == "https://api.example.com/health"


def test_customer_api_client_inherits_base_url_builder():
    client = CustomerApiClient("https://api.example.com")
    assert client.get_customer_url("CUST-101") == "https://api.example.com/customers/CUST-101"


# Practice next:
# 1. Add User class with __repr__ and __eq__.
# 2. Add UserFactory with staticmethod/classmethod.
# 3. Add retry decorator test.
