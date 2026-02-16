## 2026-02-16 - [Testing Caching with MagicMock]
**Learning:** `MagicMock` by default returns the *same* child mock instance for repeated calls. This can cause false positives when testing if a function caches its results (i.e., you might think caching is working because the return value is identical, but it's just the default mock behavior).
**Action:** When testing caching logic (like `@lru_cache`) with mocks, explicitly configure the mock to return *different* instances on each call using `side_effect = lambda **kwargs: MagicMock()`. Then, verify that the cached function returns the *same* instance for identical inputs but *different* instances for different inputs.

## 2026-02-16 - [PII Logging in Profiler Agent]
**Learning:** Found that `ProfilerAgent` was logging extracted user facts (including PII like phone numbers) at `INFO` level. This violates security guidelines.
**Action:** Changed the log level to `DEBUG` for sensitive fact details. Always audit logging statements in agents that handle user data to ensure PII is not leaked to production logs.
