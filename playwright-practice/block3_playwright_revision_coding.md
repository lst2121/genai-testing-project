# Block 3: Playwright Revision And Coding Hands-On

## 1. Playwright Config Deep Dive

### Full Config Example

```javascript
// playwright.config.js

const { defineConfig, devices } = require("@playwright/test");

module.exports = defineConfig({
  // Test directory
  testDir: "./tests",

  // Run tests in parallel
  fullyParallel: true,

  // Fail the build on CI if test.only is found
  forbidOnly: !!process.env.CI,

  // Retry failed tests
  retries: process.env.CI ? 2 : 0,

  // Parallel workers
  workers: process.env.CI ? 1 : undefined,

  // Reporter
  reporter: [
    ["html"],
    ["list"],
    ["junit", { outputFile: "results/junit.xml" }]
  ],

  // Global timeout
  timeout: 30000,

  // Expect timeout
  expect: {
    timeout: 5000
  },

  // Shared settings for all projects
  use: {
    // Base URL
    baseURL: "https://the-internet.herokuapp.com",

    // Browser options
    headless: true,
    
    // Viewport
    viewport: { width: 1280, height: 720 },

    // Artifacts
    screenshot: "only-on-failure",
    video: "retain-on-failure",
    trace: "retain-on-failure",

    // Navigation timeout
    navigationTimeout: 30000,
    actionTimeout: 10000,
  },

  // Projects for different browsers
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
    {
      name: "firefox",
      use: { ...devices["Desktop Firefox"] },
    },
    {
      name: "webkit",
      use: { ...devices["Desktop Safari"] },
    },
    {
      name: "mobile-chrome",
      use: { ...devices["Pixel 5"] },
    },
  ],
});
```

### Key Config Options

| Option | Purpose |
|--------|---------|
| `testDir` | Where test files are located |
| `timeout` | Max time per test |
| `retries` | Number of retries on failure |
| `workers` | Parallel test workers |
| `fullyParallel` | Run tests in same file in parallel |
| `baseURL` | Base URL for `page.goto("/path")` |
| `headless` | Run without browser UI |
| `screenshot` | When to capture screenshots |
| `video` | When to record video |
| `trace` | When to capture trace |

---

## 2. Fixtures

### Built-in Fixtures

```javascript
const { test, expect } = require("@playwright/test");

test("uses built-in fixtures", async ({ page, browser, context, request }) => {
  // page - isolated page for each test
  // browser - shared browser instance
  // context - isolated browser context
  // request - API request context
});
```

### Custom Fixtures

```javascript
// fixtures.js
const { test: base } = require("@playwright/test");
const LoginPage = require("./pages/LoginPage");

const test = base.extend({
  // Custom fixture: loginPage
  loginPage: async ({ page }, use) => {
    const loginPage = new LoginPage(page);
    await use(loginPage);
  },

  // Custom fixture: logged in user
  loggedInPage: async ({ page }, use) => {
    const loginPage = new LoginPage(page);
    await loginPage.goto();
    await loginPage.login("tomsmith", "SuperSecretPassword!");
    await use(page);
  },

  // Custom fixture with setup/teardown
  testUser: async ({ request }, use) => {
    // Setup: create user
    const response = await request.post("/api/users", {
      data: { name: "Test User", email: "test@test.com" }
    });
    const user = await response.json();
    
    // Provide user to test
    await use(user);
    
    // Teardown: delete user
    await request.delete(`/api/users/${user.id}`);
  }
});

module.exports = { test };
```

### Using Custom Fixtures

```javascript
const { test } = require("./fixtures");
const { expect } = require("@playwright/test");

test("login test using fixture", async ({ loginPage }) => {
  await loginPage.goto();
  await loginPage.login("tomsmith", "SuperSecretPassword!");
  // ...
});

test("test with logged in page", async ({ loggedInPage }) => {
  // Already logged in
  await expect(loggedInPage).toHaveURL(/secure/);
});
```

---

## 3. Hooks

```javascript
const { test, expect } = require("@playwright/test");

test.describe("Test Suite", () => {

  // Runs once before all tests in this describe block
  test.beforeAll(async () => {
    console.log("Before all tests");
  });

  // Runs before each test
  test.beforeEach(async ({ page }) => {
    await page.goto("https://the-internet.herokuapp.com");
  });

  // Runs after each test
  test.afterEach(async ({ page }) => {
    console.log("After each test");
  });

  // Runs once after all tests
  test.afterAll(async () => {
    console.log("After all tests");
  });

  test("test 1", async ({ page }) => {
    // ...
  });

  test("test 2", async ({ page }) => {
    // ...
  });
});
```

### Skip And Only

```javascript
// Skip a test
test.skip("skipped test", async ({ page }) => {
  // This won't run
});

// Run only this test
test.only("only this runs", async ({ page }) => {
  // Only this runs
});

// Conditional skip
test("conditional skip", async ({ page, browserName }) => {
  test.skip(browserName === "firefox", "Not supported on Firefox");
  // ...
});

// Slow test
test("slow test", async ({ page }) => {
  test.slow(); // Triples timeout
  // ...
});
```

---

## 4. Assertions Cheat Sheet

### Element Assertions

```javascript
// Visible / Hidden
await expect(locator).toBeVisible();
await expect(locator).toBeHidden();
await expect(locator).not.toBeVisible();

// Enabled / Disabled
await expect(locator).toBeEnabled();
await expect(locator).toBeDisabled();

// Checked (checkbox/radio)
await expect(locator).toBeChecked();
await expect(locator).not.toBeChecked();

// Focused
await expect(locator).toBeFocused();

// Editable
await expect(locator).toBeEditable();

// Empty
await expect(locator).toBeEmpty();

// Attached to DOM
await expect(locator).toBeAttached();
```

### Text Assertions

```javascript
// Exact text
await expect(locator).toHaveText("Exact text");

// Partial text
await expect(locator).toContainText("partial");

// Regex
await expect(locator).toHaveText(/pattern/i);

// Multiple elements
await expect(locator).toHaveText(["Item 1", "Item 2", "Item 3"]);
```

### Value Assertions

```javascript
// Input value
await expect(locator).toHaveValue("expected value");

// Partial value
await expect(locator).toHaveValue(/partial/);
```

### Attribute Assertions

```javascript
// Has attribute
await expect(locator).toHaveAttribute("href", "/about");
await expect(locator).toHaveAttribute("class", /active/);

// Has CSS
await expect(locator).toHaveCSS("color", "rgb(255, 0, 0)");

// Has class
await expect(locator).toHaveClass(/active/);
await expect(locator).toHaveClass("btn primary");
```

### Page Assertions

```javascript
// URL
await expect(page).toHaveURL("https://example.com/dashboard");
await expect(page).toHaveURL(/dashboard/);

// Title
await expect(page).toHaveTitle("Dashboard");
await expect(page).toHaveTitle(/Dashboard/);
```

### Count Assertions

```javascript
await expect(locator).toHaveCount(5);
await expect(page.getByRole("listitem")).toHaveCount(10);
```

### API Response Assertions

```javascript
expect(response.status()).toBe(200);
expect(response.ok()).toBeTruthy();

const body = await response.json();
expect(body).toHaveProperty("id");
expect(body.name).toBe("Lokender");
expect(Array.isArray(body.items)).toBeTruthy();
expect(body.items.length).toBeGreaterThan(0);
```

### Soft Assertions

```javascript
// Continue test even if assertion fails
await expect.soft(locator).toHaveText("text");
await expect.soft(page).toHaveURL(/pattern/);

// All soft assertions are reported at end
```

---

## 5. Python Playwright Syntax

### Installation

```bash
pip install playwright
playwright install
```

### Basic Test Structure

```python
# test_example.py
from playwright.sync_api import sync_playwright, expect

def test_basic():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        
        page.goto("https://the-internet.herokuapp.com/login")
        page.get_by_label("Username").fill("tomsmith")
        page.get_by_label("Password").fill("SuperSecretPassword!")
        page.get_by_role("button", name="Login").click()
        
        expect(page).to_have_url("/secure")
        
        browser.close()
```

### With Pytest

```python
# conftest.py
import pytest
from playwright.sync_api import sync_playwright

@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        yield browser
        browser.close()

@pytest.fixture
def page(browser):
    context = browser.new_context()
    page = context.new_page()
    yield page
    context.close()
```

```python
# test_login.py
from playwright.sync_api import expect

def test_successful_login(page):
    page.goto("https://the-internet.herokuapp.com/login")
    page.get_by_label("Username").fill("tomsmith")
    page.get_by_label("Password").fill("SuperSecretPassword!")
    page.get_by_role("button", name="Login").click()
    
    expect(page).to_have_url("/secure")
    expect(page.get_by_text("You logged into a secure area!")).to_be_visible()
```

### JavaScript vs Python Comparison

| JavaScript | Python |
|------------|--------|
| `await page.goto(url)` | `page.goto(url)` |
| `await locator.click()` | `locator.click()` |
| `await locator.fill("text")` | `locator.fill("text")` |
| `await expect(locator).toBeVisible()` | `expect(locator).to_be_visible()` |
| `await expect(page).toHaveURL(/pattern/)` | `expect(page).to_have_url(re.compile(r"pattern"))` |
| `await expect(locator).toHaveText("text")` | `expect(locator).to_have_text("text")` |
| `page.getByRole("button", { name: "Login" })` | `page.get_by_role("button", name="Login")` |
| `page.getByLabel("Username")` | `page.get_by_label("Username")` |
| `page.getByText("Welcome")` | `page.get_by_text("Welcome")` |
| `page.locator("#id")` | `page.locator("#id")` |

### Python API Testing

```python
from playwright.sync_api import sync_playwright, expect

def test_api():
    with sync_playwright() as p:
        request_context = p.request.new_context(
            base_url="https://reqres.in"
        )
        
        # GET
        response = request_context.get("/api/users/2")
        assert response.status == 200
        body = response.json()
        assert body["data"]["id"] == 2
        
        # POST
        response = request_context.post("/api/users", data={
            "name": "Lokender",
            "job": "QA Lead"
        })
        assert response.status == 201
        body = response.json()
        assert body["name"] == "Lokender"
        
        request_context.dispose()
```

---

## 6. Python Coding Quick Reference

### Reverse String

```python
def reverse(text):
    return text[::-1]

# Loop version
def reverse_loop(text):
    reversed_text = ""
    for char in text:
        reversed_text = char + reversed_text
    return reversed_text
```

### Palindrome

```python
def is_palindrome(text):
    cleaned = "".join(c for c in text.lower() if c.isalnum())
    return cleaned == cleaned[::-1]
```

### Character Frequency

```python
def char_frequency(text):
    freq = {}
    for char in text.lower():
        if char.isalnum():
            freq[char] = freq.get(char, 0) + 1
    return freq
```

### First Non-Repeating Character

```python
def first_non_repeating(text):
    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1
    
    for char in text:
        if freq[char] == 1:
            return char
    
    return None
```

### First Repeating Character

```python
def first_repeating(text):
    seen = set()
    for char in text:
        if char in seen:
            return char
        seen.add(char)
    return None
```

### Anagram

```python
def is_anagram(a, b):
    clean_a = "".join(c for c in a.lower() if c.isalnum())
    clean_b = "".join(c for c in b.lower() if c.isalnum())
    return sorted(clean_a) == sorted(clean_b)
```

### Remove Duplicates

```python
def remove_duplicates(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result
```

### Second Largest

```python
def second_largest(numbers):
    unique = list(set(numbers))
    if len(unique) < 2:
        return None
    unique.sort(reverse=True)
    return unique[1]
```

### FizzBuzz

```python
def fizzbuzz(n):
    result = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(i)
    return result
```

### Factorial

```python
def factorial(n):
    if n < 0:
        return None
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
```

### Fibonacci

```python
def fibonacci(n):
    if n <= 0:
        return []
    if n == 1:
        return [0]
    
    result = [0, 1]
    while len(result) < n:
        result.append(result[-1] + result[-2])
    return result
```

### Validate Required Keys

```python
def has_required_keys(obj, keys):
    return all(key in obj for key in keys)
```

---

## 7. Combined UI + API Test Scenarios

### Scenario 1: Create User Via API, Verify In UI

```javascript
const { test, expect } = require("@playwright/test");

test("create user via API and verify in UI", async ({ page, request }) => {
  // Step 1: Create user via API
  const apiResponse = await request.post("https://reqres.in/api/users", {
    data: {
      name: "Lokender",
      job: "QA Lead"
    }
  });
  expect(apiResponse.status()).toBe(201);
  const createdUser = await apiResponse.json();
  
  // Step 2: Navigate to UI
  await page.goto("https://some-app.com/users");
  
  // Step 3: Verify user appears in UI
  await expect(page.getByText(createdUser.name)).toBeVisible();
});
```

### Scenario 2: Mock API And Test UI Behavior

```javascript
test("test UI with mocked API response", async ({ page }) => {
  // Mock API to return empty list
  await page.route("**/api/users", route => {
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ data: [] })
    });
  });
  
  await page.goto("https://some-app.com/users");
  
  // Verify empty state message
  await expect(page.getByText("No users found")).toBeVisible();
});
```

### Scenario 3: Compare UI Value With API Response

```javascript
test("compare UI value with API", async ({ page, request }) => {
  // Get data from API
  const apiResponse = await request.get("https://reqres.in/api/users/2");
  const apiData = await apiResponse.json();
  
  // Navigate to UI
  await page.goto("https://some-app.com/users/2");
  
  // Compare
  await expect(page.getByTestId("user-email")).toHaveText(apiData.data.email);
  await expect(page.getByTestId("user-name")).toHaveText(
    `${apiData.data.first_name} ${apiData.data.last_name}`
  );
});
```

### Scenario 4: Test Error Handling

```javascript
test("test error handling when API fails", async ({ page }) => {
  // Mock API failure
  await page.route("**/api/users", route => {
    route.fulfill({
      status: 500,
      contentType: "application/json",
      body: JSON.stringify({ error: "Server error" })
    });
  });
  
  await page.goto("https://some-app.com/users");
  
  // Verify error message shown to user
  await expect(page.getByText("Something went wrong")).toBeVisible();
  await expect(page.getByRole("button", { name: "Retry" })).toBeVisible();
});
```

---

## 8. Practice Scenarios

### Practice 1: Login Flow

Write a test that:
1. Goes to https://the-internet.herokuapp.com/login
2. Enters valid credentials
3. Verifies successful login
4. Logs out
5. Verifies back on login page

### Practice 2: Dynamic Table

Write a test that:
1. Goes to https://the-internet.herokuapp.com/tables
2. Finds a specific row by text
3. Reads a value from that row
4. Asserts the value

### Practice 3: API CRUD

Write tests for https://reqres.in:
1. GET single user, verify fields
2. POST new user, verify created
3. PUT update user, verify updated
4. DELETE user, verify deleted

### Practice 4: Network Mock

Write a test that:
1. Mocks an API to return custom data
2. Opens a page that uses that API
3. Verifies UI shows the mocked data

### Practice 5: File Download

Write a test that:
1. Goes to https://the-internet.herokuapp.com/download
2. Downloads a file
3. Verifies file was downloaded

```javascript
const { test, expect } = require("@playwright/test");
const path = require("path");

test("download file", async ({ page }) => {
  await page.goto("https://the-internet.herokuapp.com/download");
  
  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("link", { name: "some-file.txt" }).click()
  ]);
  
  // Save to specific path
  const filePath = path.join(__dirname, "downloads", download.suggestedFilename());
  await download.saveAs(filePath);
  
  // Verify
  expect(download.suggestedFilename()).toBe("some-file.txt");
});
```

---

## 9. Commands Quick Reference

### Run Tests

```bash
# All tests
npx playwright test

# Specific file
npx playwright test tests/login.spec.js

# Specific test by name
npx playwright test -g "login"

# Headed mode
npx playwright test --headed

# Specific browser
npx playwright test --project=chromium

# Debug mode
npx playwright test --debug

# UI mode
npx playwright test --ui

# With trace
npx playwright test --trace on
```

### Reports

```bash
# Show HTML report
npx playwright show-report

# Generate specific reporters
npx playwright test --reporter=list
npx playwright test --reporter=html
```

### Codegen

```bash
# Generate code by recording
npx playwright codegen https://the-internet.herokuapp.com

# Save to file
npx playwright codegen -o tests/recorded.spec.js https://example.com
```

---

## 10. Interview Questions

### Q1: How do you handle flaky tests?

> I analyze the failure using trace viewer, screenshots, and video. Common causes include timing issues, dynamic content, shared state, or test data conflicts. I fix by using proper waits, stable locators, isolating test data, and avoiding hard waits.

### Q2: How do you decide between UI and API tests?

> I use UI tests for critical user journeys and visual validation. I use API tests for business logic, data validation, and faster feedback. API tests are faster and more stable, so I push as much validation to API level as possible.

### Q3: How do you structure a large Playwright framework?

> I use Page Object Model for maintainability, custom fixtures for reusable setup, config for environment handling, and tags/projects for test organization. I separate page objects, fixtures, utilities, and tests into clear folders.

### Q4: How do you handle authentication?

> I use storageState to save authenticated session and reuse across tests. For different roles, I create separate fixtures. This avoids logging in via UI for every test, making tests faster and more stable.

### Q5: How do you run tests in CI?

> I configure Playwright to run headless with retries, capture artifacts on failure, and generate reports. I use parallel workers for speed and ensure tests are independent so they can run in any order.
