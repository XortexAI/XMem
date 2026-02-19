## 2024-05-23 - Testing in restricted environment
**Learning:** This environment lacks core dependencies (pydantic, langchain, etc.) needed for unit tests to run. To verify changes, extensive `sys.modules` patching is required.
**Action:** When testing, be prepared to mock `sys.modules` aggressively, but ensure these mocks are removed before submission if the target environment is expected to have dependencies.
