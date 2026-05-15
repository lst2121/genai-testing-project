class BaseApiClient:
    def __init__(self, base_url, headers):
        self.base_url = base_url
        self.headers = headers

    def build_url(self, endpoint):
        return f"{self.base_url}/{endpoint}"
    
class CustomerApiClient(BaseApiClient):
    def get_customer_url(self, customer_id):
        return self.build_url(f"/customers/{customer_id}")

class TicketApiClient(BaseApiClient):
    def get_ticket_url(self, ticket_id):
        return self.build_url(f"/tickets/{ticket_id}")

customer_api_client = CustomerApiClient("https://api.example.com", {"Authorization": "Bearer 1234567890"})
ticket_api_client = TicketApiClient("https://api.example.com", {"Authorization": "Bearer 1234567890"})

print(customer_api_client.get_customer_url("123"))
print(ticket_api_client.get_ticket_url("456"))
