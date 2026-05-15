class FakePage:
    def click(self, locator):
        print(f"Clicking on {locator}")
    
    def fill(self, locator, text):
        print(f"Filling {text} into {locator}")


class BasePage:
    def __init__(self, page):
        self.page = page

    def click(self, locator):
        self.page.click(locator)
    
    def fill(self, locator, text):
        self.page.fill(locator, text)
    
class LoginPage(BasePage):
    def login(self, username, password):
        self.fill("username", username)
        self.fill("password", password)
        self.click("login")

page = FakePage()
page = LoginPage(page)
page.login("testuser", "testpassword")