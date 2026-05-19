# Block 1: Playwright Basics From Scratch

## 1. Installation And Setup

### Step 1: Create Project Folder
```bash
mkdir playwright-practice
cd playwright-practice
```

### Step 2: Initialize Node Project
```bash
npm init -y
```

### Step 3: Install Playwright
```bash
npm init playwright@latest
```

When prompted:
- Choose JavaScript
- Add GitHub Actions workflow: Yes
- Install browsers: Yes

### Step 4: Verify Installation
```bash
npx playwright --version
npx playwright test
npx playwright show-report
```

---

## 2. Project Structure

After setup, your folder looks like:

```
playwright-practice/
├── node_modules/
├── tests/
│   └── example.spec.js
├── tests-examples/
│   └── demo-todo-app.spec.js
├── playwright.config.js
├── package.json
└── package-lock.json
```

Key files:
- `playwright.config.js` — configuration (browsers, timeout, baseURL, reporters)
- `tests/` — your test files go here
- `package.json` — project dependencies

---

## 3. Basic Test Structure

```javascript
// tests/my-first-test.spec.js

const { test, expect } = require("@playwright/test");

test("has title", async ({ page }) => {
  await page.goto("https://playwright.dev/");
  await expect(page).toHaveTitle(/Playwright/);
});

test("get started link", async ({ page }) => {
  await page.goto("https://playwright.dev/");
  await page.getByRole("link", { name: "Get started" }).click();
  await expect(page).toHaveURL(/.*intro/);
});
```

Run:
```bash
npx playwright test
npx playwright test --headed
npx playwright test --ui
```

---

## 4. Locator Strategies

### Priority Order (Best to Worst)

| Priority | Locator Type | Example |
|----------|--------------|---------|
| 1 | getByRole | `page.getByRole("button", { name: "Login" })` |
| 2 | getByLabel | `page.getByLabel("Username")` |
| 3 | getByPlaceholder | `page.getByPlaceholder("Enter email")` |
| 4 | getByText | `page.getByText("Welcome")` |
| 5 | getByTestId | `page.getByTestId("submit-btn")` |
| 6 | CSS | `page.locator("#username")` |
| 7 | XPath | `page.locator("//button[@type='submit']")` |

### getByRole Examples

```javascript
// Buttons
page.getByRole("button", { name: "Submit" })
page.getByRole("button", { name: /submit/i })  // case insensitive

// Links
page.getByRole("link", { name: "Sign up" })

// Text inputs
page.getByRole("textbox", { name: "Email" })

// Checkboxes
page.getByRole("checkbox", { name: "Remember me" })

// Radio buttons
page.getByRole("radio", { name: "Male" })

// Dropdowns
page.getByRole("combobox", { name: "Country" })

// Headings
page.getByRole("heading", { name: "Welcome" })
```

### getByLabel Examples

```javascript
// Input with associated label
page.getByLabel("Username")
page.getByLabel("Password")
page.getByLabel("Email Address")
```

### getByText Examples

```javascript
// Exact match
page.getByText("Login", { exact: true })

// Partial match
page.getByText("Welcome")

// Regex
page.getByText(/welcome/i)
```

### CSS Locator Examples

```javascript
// ID
page.locator("#username")

// Class
page.locator(".login-btn")

// Attribute
page.locator("[data-qa='login']")
page.locator("[type='submit']")

// Combined
page.locator("input#username")
page.locator("button.primary")
```

---

## 5. Common Actions

```javascript
// Navigation
await page.goto("https://example.com");
await page.goBack();
await page.goForward();
await page.reload();

// Click
await page.getByRole("button", { name: "Submit" }).click();
await page.getByRole("button", { name: "Submit" }).dblclick();
await page.getByRole("button", { name: "Submit" }).click({ button: "right" });

// Fill input
await page.getByLabel("Username").fill("testuser");
await page.getByLabel("Username").clear();

// Type with delay (simulates real typing)
await page.getByLabel("Username").pressSequentially("testuser", { delay: 100 });

// Select dropdown
await page.getByRole("combobox").selectOption("option1");
await page.getByRole("combobox").selectOption({ label: "Option 1" });
await page.getByRole("combobox").selectOption({ value: "opt1" });

// Checkbox
await page.getByRole("checkbox").check();
await page.getByRole("checkbox").uncheck();

// Hover
await page.getByText("Menu").hover();

// Focus
await page.getByLabel("Email").focus();

// Press key
await page.getByLabel("Search").press("Enter");

// Screenshot
await page.screenshot({ path: "screenshot.png" });
await page.getByRole("button").screenshot({ path: "button.png" });
```

---

## 6. Assertions

```javascript
// Element visible
await expect(page.getByText("Welcome")).toBeVisible();

// Element hidden
await expect(page.getByText("Loading")).toBeHidden();

// Element enabled/disabled
await expect(page.getByRole("button")).toBeEnabled();
await expect(page.getByRole("button")).toBeDisabled();

// Checkbox checked
await expect(page.getByRole("checkbox")).toBeChecked();
await expect(page.getByRole("checkbox")).not.toBeChecked();

// Text content
await expect(page.getByRole("heading")).toHaveText("Welcome");
await expect(page.getByRole("heading")).toContainText("Welcome");

// Input value
await expect(page.getByLabel("Username")).toHaveValue("testuser");

// URL
await expect(page).toHaveURL("https://example.com/dashboard");
await expect(page).toHaveURL(/dashboard/);

// Title
await expect(page).toHaveTitle("Dashboard");
await expect(page).toHaveTitle(/Dashboard/);

// Count
await expect(page.getByRole("listitem")).toHaveCount(5);

// Attribute
await expect(page.getByRole("link")).toHaveAttribute("href", "/about");
```

---

## 7. Workflows On the-internet.herokuapp.com

### 7.1 Login Page

URL: https://the-internet.herokuapp.com/login

```javascript
const { test, expect } = require("@playwright/test");

test.describe("Login Page Tests", () => {
  
  test.beforeEach(async ({ page }) => {
    await page.goto("https://the-internet.herokuapp.com/login");
  });

  test("successful login", async ({ page }) => {
    // Fill credentials
    await page.getByLabel("Username").fill("tomsmith");
    await page.getByLabel("Password").fill("SuperSecretPassword!");
    
    // Click login
    await page.getByRole("button", { name: "Login" }).click();
    
    // Verify success
    await expect(page.getByText("You logged into a secure area!")).toBeVisible();
    await expect(page).toHaveURL(/secure/);
  });

  test("failed login with invalid username", async ({ page }) => {
    await page.getByLabel("Username").fill("invaliduser");
    await page.getByLabel("Password").fill("SuperSecretPassword!");
    await page.getByRole("button", { name: "Login" }).click();
    
    await expect(page.getByText("Your username is invalid!")).toBeVisible();
  });

  test("failed login with invalid password", async ({ page }) => {
    await page.getByLabel("Username").fill("tomsmith");
    await page.getByLabel("Password").fill("wrongpassword");
    await page.getByRole("button", { name: "Login" }).click();
    
    await expect(page.getByText("Your password is invalid!")).toBeVisible();
  });

  test("logout after login", async ({ page }) => {
    await page.getByLabel("Username").fill("tomsmith");
    await page.getByLabel("Password").fill("SuperSecretPassword!");
    await page.getByRole("button", { name: "Login" }).click();
    
    await page.getByRole("link", { name: "Logout" }).click();
    
    await expect(page).toHaveURL(/login/);
    await expect(page.getByText("You logged out of the secure area!")).toBeVisible();
  });
});
```

### 7.2 Dropdown

URL: https://the-internet.herokuapp.com/dropdown

```javascript
test.describe("Dropdown Tests", () => {

  test("select option by value", async ({ page }) => {
    await page.goto("https://the-internet.herokuapp.com/dropdown");
    
    // Select by value
    await page.locator("#dropdown").selectOption("1");
    
    // Verify
    await expect(page.locator("#dropdown")).toHaveValue("1");
  });

  test("select option by label", async ({ page }) => {
    await page.goto("https://the-internet.herokuapp.com/dropdown");
    
    // Select by visible text
    await page.locator("#dropdown").selectOption({ label: "Option 2" });
    
    // Verify
    await expect(page.locator("#dropdown")).toHaveValue("2");
  });

  test("verify all options exist", async ({ page }) => {
    await page.goto("https://the-internet.herokuapp.com/dropdown");
    
    const options = page.locator("#dropdown option");
    await expect(options).toHaveCount(3); // includes "Please select an option"
  });
});
```

### 7.3 Checkboxes

URL: https://the-internet.herokuapp.com/checkboxes

```javascript
test.describe("Checkbox Tests", () => {

  test("check and uncheck boxes", async ({ page }) => {
    await page.goto("https://the-internet.herokuapp.com/checkboxes");
    
    const checkbox1 = page.locator("input[type='checkbox']").first();
    const checkbox2 = page.locator("input[type='checkbox']").last();
    
    // Initial state
    await expect(checkbox1).not.toBeChecked();
    await expect(checkbox2).toBeChecked();
    
    // Check first checkbox
    await checkbox1.check();
    await expect(checkbox1).toBeChecked();
    
    // Uncheck second checkbox
    await checkbox2.uncheck();
    await expect(checkbox2).not.toBeChecked();
  });
});
```

### 7.4 File Upload

URL: https://the-internet.herokuapp.com/upload

```javascript
const path = require("path");

test.describe("File Upload Tests", () => {

  test("upload a file", async ({ page }) => {
    await page.goto("https://the-internet.herokuapp.com/upload");
    
    // Create a test file path
    const filePath = path.join(__dirname, "test-file.txt");
    
    // Upload file
    await page.locator("#file-upload").setInputFiles(filePath);
    
    // Submit
    await page.locator("#file-submit").click();
    
    // Verify
    await expect(page.locator("#uploaded-files")).toHaveText("test-file.txt");
  });

  test("upload multiple files", async ({ page }) => {
    await page.goto("https://the-internet.herokuapp.com/upload");
    
    // Upload multiple files
    await page.locator("#file-upload").setInputFiles([
      path.join(__dirname, "file1.txt"),
      path.join(__dirname, "file2.txt")
    ]);
  });

  test("remove uploaded file", async ({ page }) => {
    await page.goto("https://the-internet.herokuapp.com/upload");
    
    await page.locator("#file-upload").setInputFiles(path.join(__dirname, "test-file.txt"));
    
    // Remove file
    await page.locator("#file-upload").setInputFiles([]);
  });
});
```

### 7.5 JavaScript Alerts

URL: https://the-internet.herokuapp.com/javascript_alerts

```javascript
test.describe("JavaScript Alert Tests", () => {

  test("handle simple alert", async ({ page }) => {
    await page.goto("https://the-internet.herokuapp.com/javascript_alerts");
    
    // Listen for dialog before triggering
    page.on("dialog", async dialog => {
      expect(dialog.type()).toBe("alert");
      expect(dialog.message()).toBe("I am a JS Alert");
      await dialog.accept();
    });
    
    await page.getByRole("button", { name: "Click for JS Alert" }).click();
    
    await expect(page.locator("#result")).toHaveText("You successfully clicked an alert");
  });

  test("handle confirm - accept", async ({ page }) => {
    await page.goto("https://the-internet.herokuapp.com/javascript_alerts");
    
    page.on("dialog", async dialog => {
      expect(dialog.type()).toBe("confirm");
      await dialog.accept();
    });
    
    await page.getByRole("button", { name: "Click for JS Confirm" }).click();
    
    await expect(page.locator("#result")).toHaveText("You clicked: Ok");
  });

  test("handle confirm - dismiss", async ({ page }) => {
    await page.goto("https://the-internet.herokuapp.com/javascript_alerts");
    
    page.on("dialog", async dialog => {
      await dialog.dismiss();
    });
    
    await page.getByRole("button", { name: "Click for JS Confirm" }).click();
    
    await expect(page.locator("#result")).toHaveText("You clicked: Cancel");
  });

  test("handle prompt", async ({ page }) => {
    await page.goto("https://the-internet.herokuapp.com/javascript_alerts");
    
    page.on("dialog", async dialog => {
      expect(dialog.type()).toBe("prompt");
      await dialog.accept("Lokender");
    });
    
    await page.getByRole("button", { name: "Click for JS Prompt" }).click();
    
    await expect(page.locator("#result")).toHaveText("You entered: Lokender");
  });
});
```

### 7.6 Dynamic Loading

URL: https://the-internet.herokuapp.com/dynamic_loading/1

```javascript
test.describe("Dynamic Loading Tests", () => {

  test("wait for hidden element to appear", async ({ page }) => {
    await page.goto("https://the-internet.herokuapp.com/dynamic_loading/1");
    
    // Click start
    await page.getByRole("button", { name: "Start" }).click();
    
    // Wait for loading to finish and text to appear
    await expect(page.getByText("Hello World!")).toBeVisible({ timeout: 10000 });
  });

  test("wait for element rendered after load", async ({ page }) => {
    await page.goto("https://the-internet.herokuapp.com/dynamic_loading/2");
    
    await page.getByRole("button", { name: "Start" }).click();
    
    // Element is rendered after loading
    await expect(page.locator("#finish h4")).toHaveText("Hello World!");
  });
});
```

### 7.7 Frames / iFrames

URL: https://the-internet.herokuapp.com/iframe

```javascript
test.describe("iFrame Tests", () => {

  test("interact with iframe content", async ({ page }) => {
    await page.goto("https://the-internet.herokuapp.com/iframe");
    
    // Get iframe
    const frame = page.frameLocator("#mce_0_ifr");
    
    // Clear and type in iframe
    await frame.locator("#tinymce").clear();
    await frame.locator("#tinymce").fill("Hello from Playwright!");
    
    // Verify
    await expect(frame.locator("#tinymce")).toHaveText("Hello from Playwright!");
  });
});
```

### 7.8 New Window / Tab

URL: https://the-internet.herokuapp.com/windows

```javascript
test.describe("New Window Tests", () => {

  test("handle new tab", async ({ page, context }) => {
    await page.goto("https://the-internet.herokuapp.com/windows");
    
    // Wait for new page to open
    const [newPage] = await Promise.all([
      context.waitForEvent("page"),
      page.getByRole("link", { name: "Click Here" }).click()
    ]);
    
    // Wait for new page to load
    await newPage.waitForLoadState();
    
    // Verify new page
    await expect(newPage).toHaveURL(/windows\/new/);
    await expect(newPage.getByText("New Window")).toBeVisible();
    
    // Close new page
    await newPage.close();
    
    // Continue on original page
    await expect(page.getByText("Opening a new window")).toBeVisible();
  });
});
```

---

## 8. API Testing With Playwright (Using reqres.in)

### Basic GET Request

```javascript
const { test, expect } = require("@playwright/test");

test.describe("API Tests - reqres.in", () => {

  test("GET single user", async ({ request }) => {
    const response = await request.get("https://reqres.in/api/users/2");
    
    // Status check
    expect(response.status()).toBe(200);
    
    // Body check
    const body = await response.json();
    expect(body.data.id).toBe(2);
    expect(body.data.email).toBe("janet.weaver@reqres.in");
    expect(body.data.first_name).toBe("Janet");
  });

  test("GET list of users", async ({ request }) => {
    const response = await request.get("https://reqres.in/api/users?page=2");
    
    expect(response.status()).toBe(200);
    
    const body = await response.json();
    expect(body.page).toBe(2);
    expect(body.data.length).toBeGreaterThan(0);
  });

  test("GET user not found", async ({ request }) => {
    const response = await request.get("https://reqres.in/api/users/999");
    
    expect(response.status()).toBe(404);
  });
});
```

### POST Request

```javascript
test("POST create user", async ({ request }) => {
  const response = await request.post("https://reqres.in/api/users", {
    data: {
      name: "Lokender",
      job: "QA Lead"
    }
  });
  
  expect(response.status()).toBe(201);
  
  const body = await response.json();
  expect(body.name).toBe("Lokender");
  expect(body.job).toBe("QA Lead");
  expect(body.id).toBeDefined();
  expect(body.createdAt).toBeDefined();
});
```

### PUT Request

```javascript
test("PUT update user", async ({ request }) => {
  const response = await request.put("https://reqres.in/api/users/2", {
    data: {
      name: "Lokender Updated",
      job: "Senior QA Lead"
    }
  });
  
  expect(response.status()).toBe(200);
  
  const body = await response.json();
  expect(body.name).toBe("Lokender Updated");
  expect(body.job).toBe("Senior QA Lead");
  expect(body.updatedAt).toBeDefined();
});
```

### PATCH Request

```javascript
test("PATCH partial update user", async ({ request }) => {
  const response = await request.patch("https://reqres.in/api/users/2", {
    data: {
      job: "Lead QA Engineer"
    }
  });
  
  expect(response.status()).toBe(200);
  
  const body = await response.json();
  expect(body.job).toBe("Lead QA Engineer");
});
```

### DELETE Request

```javascript
test("DELETE user", async ({ request }) => {
  const response = await request.delete("https://reqres.in/api/users/2");
  
  expect(response.status()).toBe(204);
});
```

### POST With Headers

```javascript
test("POST with custom headers", async ({ request }) => {
  const response = await request.post("https://reqres.in/api/users", {
    headers: {
      "Content-Type": "application/json",
      "Authorization": "Bearer fake-token-12345",
      "X-Custom-Header": "CustomValue"
    },
    data: {
      name: "Test User",
      job: "Tester"
    }
  });
  
  expect(response.status()).toBe(201);
});
```

### Validate Response Schema

```javascript
test("validate response schema", async ({ request }) => {
  const response = await request.get("https://reqres.in/api/users/2");
  const body = await response.json();
  
  // Check required fields exist
  expect(body).toHaveProperty("data");
  expect(body).toHaveProperty("support");
  
  // Check data structure
  expect(body.data).toHaveProperty("id");
  expect(body.data).toHaveProperty("email");
  expect(body.data).toHaveProperty("first_name");
  expect(body.data).toHaveProperty("last_name");
  expect(body.data).toHaveProperty("avatar");
  
  // Check data types
  expect(typeof body.data.id).toBe("number");
  expect(typeof body.data.email).toBe("string");
});
```

---

## 9. Network Interception

### Mock API Response

```javascript
test("mock API response", async ({ page }) => {
  // Intercept and mock
  await page.route("**/api/users/2", route => {
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        data: {
          id: 2,
          email: "mocked@test.com",
          first_name: "Mocked",
          last_name: "User"
        }
      })
    });
  });
  
  await page.goto("https://reqres.in");
  // Now any call to /api/users/2 returns mocked data
});
```

### Mock Empty Response

```javascript
test("mock empty list", async ({ page }) => {
  await page.route("**/api/users*", route => {
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        page: 1,
        data: [],
        total: 0
      })
    });
  });
  
  await page.goto("https://some-app.com/users");
  // App should show "No users found"
});
```

### Mock Error Response

```javascript
test("mock server error", async ({ page }) => {
  await page.route("**/api/users", route => {
    route.fulfill({
      status: 500,
      contentType: "application/json",
      body: JSON.stringify({
        error: "Internal Server Error"
      })
    });
  });
  
  await page.goto("https://some-app.com/users");
  // App should show error message
});
```

### Modify Request

```javascript
test("modify request headers", async ({ page }) => {
  await page.route("**/api/**", route => {
    const headers = {
      ...route.request().headers(),
      "X-Custom-Header": "injected-value"
    };
    route.continue({ headers });
  });
  
  await page.goto("https://example.com");
});
```

### Wait For API Response

```javascript
test("wait for API and validate", async ({ page }) => {
  await page.goto("https://some-app.com");
  
  // Wait for specific API call
  const responsePromise = page.waitForResponse("**/api/users");
  
  await page.getByRole("button", { name: "Load Users" }).click();
  
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  
  const data = await response.json();
  expect(data.length).toBeGreaterThan(0);
});
```

---

## 10. Page Object Model

### Structure

```
playwright-practice/
├── pages/
│   ├── LoginPage.js
│   ├── DashboardPage.js
│   └── BasePage.js
├── tests/
│   └── login.spec.js
└── playwright.config.js
```

### BasePage.js

```javascript
// pages/BasePage.js

class BasePage {
  constructor(page) {
    this.page = page;
  }

  async navigate(path) {
    await this.page.goto(path);
  }

  async getTitle() {
    return await this.page.title();
  }

  async waitForPageLoad() {
    await this.page.waitForLoadState("networkidle");
  }
}

module.exports = BasePage;
```

### LoginPage.js

```javascript
// pages/LoginPage.js

const BasePage = require("./BasePage");

class LoginPage extends BasePage {
  constructor(page) {
    super(page);
    
    // Locators
    this.usernameInput = page.getByLabel("Username");
    this.passwordInput = page.getByLabel("Password");
    this.loginButton = page.getByRole("button", { name: "Login" });
    this.errorMessage = page.locator("#flash");
    this.successMessage = page.locator(".flash.success");
  }

  async goto() {
    await this.navigate("https://the-internet.herokuapp.com/login");
  }

  async login(username, password) {
    await this.usernameInput.fill(username);
    await this.passwordInput.fill(password);
    await this.loginButton.click();
  }

  async getErrorMessage() {
    return await this.errorMessage.textContent();
  }

  async isErrorVisible() {
    return await this.errorMessage.isVisible();
  }
}

module.exports = LoginPage;
```

### DashboardPage.js

```javascript
// pages/DashboardPage.js

const BasePage = require("./BasePage");

class DashboardPage extends BasePage {
  constructor(page) {
    super(page);
    
    this.logoutButton = page.getByRole("link", { name: "Logout" });
    this.secureAreaMessage = page.locator(".flash.success");
  }

  async logout() {
    await this.logoutButton.click();
  }

  async isLoggedIn() {
    return await this.secureAreaMessage.isVisible();
  }
}

module.exports = DashboardPage;
```

### Test Using POM

```javascript
// tests/login.spec.js

const { test, expect } = require("@playwright/test");
const LoginPage = require("../pages/LoginPage");
const DashboardPage = require("../pages/DashboardPage");

test.describe("Login Tests with POM", () => {
  let loginPage;
  let dashboardPage;

  test.beforeEach(async ({ page }) => {
    loginPage = new LoginPage(page);
    dashboardPage = new DashboardPage(page);
    await loginPage.goto();
  });

  test("successful login", async ({ page }) => {
    await loginPage.login("tomsmith", "SuperSecretPassword!");
    
    expect(await dashboardPage.isLoggedIn()).toBeTruthy();
    await expect(page).toHaveURL(/secure/);
  });

  test("failed login shows error", async () => {
    await loginPage.login("invalid", "invalid");
    
    expect(await loginPage.isErrorVisible()).toBeTruthy();
  });

  test("logout after login", async ({ page }) => {
    await loginPage.login("tomsmith", "SuperSecretPassword!");
    await dashboardPage.logout();
    
    await expect(page).toHaveURL(/login/);
  });
});
```

---

## 11. Running Tests

```bash
# Run all tests
npx playwright test

# Run specific file
npx playwright test tests/login.spec.js

# Run with headed browser
npx playwright test --headed

# Run specific browser
npx playwright test --project=chromium

# Run with UI mode
npx playwright test --ui

# Run with debug
npx playwright test --debug

# Generate report
npx playwright show-report

# Run with trace
npx playwright test --trace on
```

---

## Quick Reference Commands

| Action | Code |
|--------|------|
| Navigate | `await page.goto(url)` |
| Click | `await locator.click()` |
| Fill | `await locator.fill("text")` |
| Select | `await locator.selectOption("value")` |
| Check | `await locator.check()` |
| Uncheck | `await locator.uncheck()` |
| Assert visible | `await expect(locator).toBeVisible()` |
| Assert text | `await expect(locator).toHaveText("text")` |
| Assert URL | `await expect(page).toHaveURL(/pattern/)` |
| Screenshot | `await page.screenshot({ path: "file.png" })` |
| Wait for element | `await locator.waitFor()` |
| Get text | `await locator.textContent()` |
| Get value | `await locator.inputValue()` |
