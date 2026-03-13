## 2024-05-24 - Optimizing AST Parsing
**Learning:** Nested `ast.walk` loops in AST parsing (like in `PythonParser._extract_calls`) can cause O(N^2) traversal overhead, heavily impacting indexing performance on large files.
**Action:** Use `ast.NodeVisitor` for a single-pass O(N) traversal over the AST tree when extracting calls or computing complexity.
