## 2025-02-14 - Python Dictionary Access Optimization
**Learning:** Using `list(d.items())[0]` to access the first item of a dictionary creates a full list of all items, which is O(N) memory and time. Using `next(iter(d.items()))` is O(1) and cleaner.
**Action:** Always prefer `next(iter(d.items()))` when accessing a single item from a dictionary, especially in loops or frequently called helper methods.
