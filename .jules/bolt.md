## 2024-05-24 - [AST Traversal Optimization]
**Learning:** O(n²) behavior in abstract syntax tree parsing when using `ast.walk` nested inside another `ast.walk` loop inside Python scripts causes very poor performance on larger codebases. In `src/scanner/ast_parser.py`, `_extract_calls`, `_compute_complexity`, and `_extract_imports` all used `ast.walk` inefficiently.
**Action:** Used `ast.NodeVisitor` classes to replace `ast.walk`, resulting in a ~2-3x speedup. `NodeVisitor` is the idiomatic way to traverse AST nodes in Python and is significantly faster, as well as structurally cleaner.
