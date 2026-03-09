
## 2025-03-09 - AST Parsing Performance Bottleneck with `ast.walk`
**Learning:** Using nested `ast.walk` loops to traverse AST subtrees for symbol operations (like call extraction) results in massive O(N^2) performance bottlenecks, as `ast.walk` lacks targeted node visitation and must queue/yield the entire subtree repeatedly. It also incurs high generator overhead for simpler operations like complexity computation.
**Action:** Always prefer `ast.NodeVisitor` over `ast.walk` for AST traversals, especially when processing nested structures or specific node types, as it provides a single O(N) pass with minimal overhead.
