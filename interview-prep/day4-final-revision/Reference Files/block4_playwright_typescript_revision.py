"""
================================================================================
DAY 4 - BLOCK 4: PLAYWRIGHT AND TYPESCRIPT/JAVASCRIPT REVISION
================================================================================
Goal:
Revise Playwright and TypeScript theory for Senior SDET interview discussions.
This block is theory-heavy because your actual work experience is already strong.
================================================================================
"""


# =============================================================================
# 1. WHY PLAYWRIGHT?
# =============================================================================
"""
Playwright strengths:

    - reliable auto-waiting
    - modern locator strategy
    - browser context isolation
    - parallel execution
    - network interception
    - API testing support
    - trace viewer, screenshots, videos
    - cross-browser support

Interview line:
    I prefer Playwright for modern web automation because it reduces flakiness
    with auto-waiting and locator assertions, supports isolated browser contexts,
    and gives strong debugging through traces.
"""


# =============================================================================
# 2. BROWSER, CONTEXT, PAGE
# =============================================================================
"""
Browser:
    Actual browser process.

Context:
    Isolated browser session. Like a fresh incognito profile.

Page:
    A tab inside a context.

Why context matters:
    Each test can get isolated cookies/localStorage/session state.
    This helps parallel execution.
"""


# =============================================================================
# 3. LOCATORS VS SELECTORS
# =============================================================================
"""
Selector:
    Raw query string like "#login".

Locator:
    Playwright object that auto-waits and retries.

Prefer:
    page.getByRole()
    page.getByLabel()
    page.getByTestId()
    page.locator()

Interview line:
    I prefer user-facing locators and test IDs over brittle CSS/XPath. Locators
    improve reliability because Playwright auto-waits for actionability.
"""


# =============================================================================
# 4. AUTO-WAITING
# =============================================================================
"""
Playwright waits for:
    element attached
    visible
    stable
    enabled
    receives events

Avoid:
    hard waits like waitForTimeout()

Use:
    expect(locator).toBeVisible()
    expect(locator).toHaveText()
"""


# =============================================================================
# 5. PAGE OBJECT MODEL
# =============================================================================
"""
POM keeps locators/actions inside page classes.

Example TypeScript shape:

    export class LoginPage {
      constructor(private page: Page) {}

      username = this.page.getByLabel('Username')
      password = this.page.getByLabel('Password')
      submit = this.page.getByRole('button', { name: 'Login' })

      async login(user: string, pass: string) {
        await this.username.fill(user)
        await this.password.fill(pass)
        await this.submit.click()
      }
    }

Good POM:
    business methods
    stable locators
    minimal assertions
    reusable flows
"""


# =============================================================================
# 6. FIXTURES IN PLAYWRIGHT
# =============================================================================
"""
Fixtures provide reusable setup.

Examples:
    page
    context
    request
    authenticatedPage
    apiClient
    testData

Use fixtures for:
    auth setup
    page objects
    API clients
    test data
"""


# =============================================================================
# 7. AUTH AND SESSION REUSE
# =============================================================================
"""
Common pattern:
    login once
    save storageState
    reuse storageState in tests

Benefits:
    faster tests
    less login flakiness
    supports parallel workers

Risk:
    stale session
    shared state across tests
    role leakage

Senior answer:
    I isolate auth per role/worker and avoid sharing mutable account state across
    parallel tests.
"""


# =============================================================================
# 8. API TESTING IN PLAYWRIGHT
# =============================================================================
"""
Playwright has request context.

Use for:
    setup data
    cleanup data
    validate backend APIs
    avoid slow UI flows

Judgement:
    Do not test everything through UI. Use API where UI adds no value.
"""


# =============================================================================
# 9. NETWORK INTERCEPTION
# =============================================================================
"""
Use route/intercept to:
    mock API response
    simulate error
    validate request payload
    test loading/empty/error states

Example scenarios:
    500 response shows error toast
    slow response shows loader
    empty list shows empty state
"""


# =============================================================================
# 10. DEBUGGING FLAKY TESTS
# =============================================================================
"""
Tools:
    trace viewer
    screenshots
    video
    console logs
    network logs
    retries for evidence, not masking

Common causes:
    bad locator
    hard wait
    shared test data
    animation/timing
    backend delay
    parallel data collision
"""


# =============================================================================
# 11. WHY TYPESCRIPT WITH PLAYWRIGHT?
# =============================================================================
"""
TypeScript benefits:

    - type safety
    - better autocomplete
    - safer refactoring
    - typed page objects
    - typed API payloads
    - typed fixtures
    - catches mistakes before runtime

Interview line:
    TypeScript makes large Playwright frameworks easier to maintain because page
    object methods, fixtures, test data, and API payloads are typed and safer to
    refactor.
"""


# =============================================================================
# 12. JS/TS TOPICS TO REVISE
# =============================================================================
"""
let vs const vs var:
    const = cannot reassign
    let = block-scoped variable
    var = function-scoped, avoid in modern code

Promise:
    Represents async result.

async/await:
    Cleaner syntax for promises.

Array methods:
    map, filter, reduce, find, some, every

Optional chaining:
    user?.profile?.name

Interface/type:
    Define shape of objects.

Generics:
    Reusable typed functions/classes.
"""


# =============================================================================
# LIKELY INTERVIEW QUESTIONS
# =============================================================================
"""
1. Why Playwright over Selenium?
    Auto-waiting, modern locators, context isolation, tracing, API support.

2. What is browser context?
    Isolated session with its own cookies/local storage.

3. How do you reduce flakiness?
    Stable locators, no hard waits, isolated data, trace debugging, proper
    assertions, controlled network.

4. Why TypeScript?
    Type safety and maintainability for large framework code.

5. What should be UI vs API?
    UI for critical user journeys; API for setup, contracts, permissions, and
    backend validation.

6. How do you handle authentication?
    storageState, per-role fixtures, worker isolation.

7. How do you debug a failed Playwright test?
    Trace viewer, screenshot, video, console/network logs, reproduce locally.
"""


# =============================================================================
# OFFLINE PRACTICE TASKS
# =============================================================================
"""
Prepare 2-minute answers for:

1. Playwright framework architecture.
2. How you manage auth and parallel execution.
3. How you choose UI vs API automation.
4. Why TypeScript is useful.
5. How you debug flaky tests.
6. How AI-assisted automation helps and where it should not be trusted blindly.
"""


# =============================================================================
# 13. DEEPER PLAYWRIGHT ARCHITECTURE ANSWER
# =============================================================================
"""
Question:
    How would you design a Playwright automation framework?

Strong answer:

    "I would design the framework around test isolation, reusable page objects,
    stable fixtures, API helpers, and CI-friendly execution. Specs should stay
    thin and readable. Page classes should own locators and page actions. API
    clients should handle backend setup and validation. Fixtures should provide
    authenticated pages, role-specific contexts, and test data. Utilities should
    centralize constants, routes, and common assertions.

    I would separate smoke, regression, integration, and live/e2e tests using
    tags or projects. For debugging, I would keep traces, screenshots, videos,
    console logs, and network logs. For CI, I would keep PR gates fast and move
    heavier regression or live-system checks to scheduled pipelines."


Framework layers:

    tests/
        Thin scenario files.

    pages/
        Page objects, locators, reusable actions.

    fixtures/
        Authenticated sessions, users, test data, page objects.

    utils/
        constants, route config, helper assertions, data builders.

    api/
        API clients for setup, cleanup, backend validation.

    reporting/
        traces, screenshots, ReportPortal/HTML reports.


What makes it senior-level:
    - choosing UI vs API level correctly
    - avoiding duplication
    - designing for parallel execution
    - making failures diagnosable
    - keeping CI fast and trustworthy
"""


# =============================================================================
# 14. YOUR EVPNXGEN-WEB PROJECT PATTERN
# =============================================================================
"""
Your real project story should mention:

    - React/TypeScript application
    - Playwright e2e tests
    - Page Object Model
    - custom fixtures
    - API validations
    - route/config constants
    - data integrity checks
    - CI analysis
    - ReportPortal / pipeline reporting
    - AI-assisted automation skills


Your Cursor skills framework is a strong differentiator.

You built/used skills for the QA pipeline:

    e2e-automation-generator:
        Generate Playwright tests from Jira stories using project POM patterns.
        Enforces no inline locators in specs, code reuse, hard assertions, API
        validation, and Playwright MCP verification.

    e2e-ci-test-analyzer:
        Analyze Azure DevOps CI failures, classify failures using a deterministic
        Node.js script, and produce structured triage output.

    e2e-data-integrity-validator:
        Validate UI values against PostgreSQL/database queries with matching
        filters, sort order, transformation logic, and tolerance rules.

    e2e-uat-readiness-validator:
        Triple-check UI = API = DB before UAT, including console errors, refresh
        behavior, API validation, DB validation, and defect reporting.

    e2e-pr-reviewer:
        Review e2e PRs for scope, POM adherence, no test.only/page.pause, no
        hardcoded routes, locator strategy, API usage, and AC coverage.

    e2e-bug-verifier:
        Reproduce bug steps, classify bug type, verify fix with evidence, and
        use Playwright/Figma/Postgres depending on bug category.


Interview framing:

    "I have not only written Playwright tests manually; I also created an
    AI-assisted QA workflow around the framework. The skills enforce our project
    rules: reuse page objects, keep locators out of specs, validate APIs where
    relevant, analyze CI failures, validate data integrity, review e2e PRs, and
    verify bugs with evidence. That helped turn AI from a code generator into a
    controlled QA assistant."
"""


# =============================================================================
# 15. PLAYWRIGHT + AI-ASSISTED AUTOMATION STORY
# =============================================================================
"""
Question:
    Where does AI-assisted automation help in Playwright?

Answer:

    "AI helps most when it accelerates repetitive engineering work but still
    operates inside strict framework rules. In my project, I used Cursor/Claude-
    style workflows to generate Playwright test skeletons from Jira stories,
    analyze existing page objects before adding new code, suggest locators,
    triage CI failures, prepare PR review feedback, validate data integrity, and
    generate bug verification steps.

    But I do not trust AI blindly. Generated tests must follow our POM pattern,
    use shared constants, avoid inline locators, include hard assertions, add API
    validation where relevant, and pass Playwright/MCP verification. Critical
    assertions, data validation, and CI gate behavior must remain deterministic
    and reviewable."


Where AI pays off:

    - scenario brainstorming from Jira ACs
    - first draft of test skeleton
    - finding similar existing specs
    - locator suggestions
    - failure triage
    - bug report drafting
    - PR review checklist


Where AI should not be blindly trusted:

    - final assertions
    - production data handling
    - security-sensitive tests
    - permission checks
    - CI gate pass/fail decisions
    - changing app code while generating e2e tests
"""


# =============================================================================
# 16. TYPESCRIPT QUESTIONS WITH BETTER ANSWERS
# =============================================================================
"""
1. Why TypeScript over JavaScript in automation?

    TypeScript catches mistakes earlier. In a large Playwright framework, it
    helps type page object methods, fixtures, API payloads, test data, and helper
    return values. It improves autocomplete and refactoring, which is important
    when hundreds of tests use the same shared methods.


2. What are interfaces used for?

    Interfaces define object contracts.

    Example:

        interface Customer {
          customerId: string;
          name: string;
          status: 'active' | 'inactive';
        }

    This helps API tests validate payload shape and helps page/API helpers accept
    predictable objects.


3. What is async/await?

    Playwright actions return promises because browser operations are async.
    async/await makes promise code readable and ensures each action completes
    before the next dependent step.


4. What is the risk of missing await?

    The test may continue before the browser action finishes. This can create
    flaky tests or false failures.


5. Why avoid hard waits?

    Hard waits make tests slow and still flaky. A fixed delay may be too short on
    slow CI and too long locally. Locator assertions and auto-waiting are better.


6. What is optional chaining useful for?

    It safely accesses nested values:

        response?.body?.customer?.name

    Useful when API responses may omit optional objects.
"""


# =============================================================================
# 17. QUICK MOCK ANSWERS
# =============================================================================
"""
Question:
    How do you decide UI vs API testing?

Answer:
    "I use UI tests for critical user journeys and visual/user interaction
    confidence. I use API tests for business rules, data setup, permissions,
    schema validation, and negative cases. If the UI is only displaying backend
    data, I validate most logic at API or DB layer and keep UI checks focused on
    rendering and workflow."


Question:
    How do you debug a flaky Playwright test?

Answer:
    "I start with the trace viewer and check whether the failure is locator,
    timing, data, network, or environment related. Then I inspect screenshots,
    video, console logs, and network requests. I avoid adding blind waits; I fix
    the root cause with better locators, proper assertions, API setup, or data
    isolation."


Question:
    How do you make Playwright tests parallel-safe?

Answer:
    "I avoid shared mutable data, isolate browser contexts, use worker-aware test
    data, separate role/session state, and avoid ordering dependencies. Tests
    should be able to run independently in any order."
"""
