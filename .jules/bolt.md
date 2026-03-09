
### 2026-03-09
* Refactored `_extract_calls` and `_compute_complexity` in `src/scanner/ast_parser.py` using `ast.NodeVisitor` to eliminate nested `ast.walk` traversals, avoiding an O(N^2) evaluation time scaling for heavily nested functions, improving performance significantly (execution dropped from 13.0s to 6.7s for 100k calls benchmark).
