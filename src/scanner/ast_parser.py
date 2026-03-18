"""
AST Parser — extracts symbols, imports, and call edges from source code.

Uses Python's built-in ``ast`` module for Python files (zero dependencies,
maximum reliability). Uses tree-sitter for TypeScript/JavaScript parsing.
Returns language-agnostic data structures that the indexer writes to
Pinecone, Neo4j, and MongoDB.

No LLM calls. Everything is deterministic software logic.

Supported languages:
  - Python (via stdlib ast)
  - TypeScript / TSX (via tree-sitter-typescript)
  - JavaScript / JSX (via tree-sitter-javascript)
"""

from __future__ import annotations

import ast
import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Tree-sitter imports (optional — graceful degradation if not installed)
try:
    from tree_sitter import Language, Parser, Node
    import tree_sitter_typescript as ts_typescript
    import tree_sitter_javascript as ts_javascript
    import tree_sitter_go as ts_go
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

logger = logging.getLogger("xmem.scanner.parser")


# ---------------------------------------------------------------------------
# Output data structures (language-agnostic)
# ---------------------------------------------------------------------------

@dataclass
class ParsedSymbol:
    """A function, method, class, or other symbol extracted from source code."""
    name: str
    qualified_name: str          # e.g. "ClassName.method_name"
    symbol_type: str             # function | method | class
    signature: str               # reconstructed signature string
    docstring: str = ""
    raw_code: str = ""           # the actual source code of this symbol
    start_line: int = 0
    end_line: int = 0
    line_count: int = 0
    parent_class: Optional[str] = None
    is_public: bool = True
    is_entrypoint: bool = False
    complexity: int = 0          # cyclomatic complexity
    decorators: List[str] = field(default_factory=list)
    parameters: List[str] = field(default_factory=list)
    return_type: str = ""
    content_hash: str = ""       # sha256 of raw_code for change detection

    @property
    def complexity_bucket(self) -> str:
        if self.complexity <= 5:
            return "low"
        elif self.complexity <= 15:
            return "medium"
        return "high"

    @property
    def summary(self) -> str:
        """Build a summary from docstring or signature. No LLM needed."""
        if self.docstring:
            first_line = self.docstring.strip().split("\n")[0].strip()
            if len(first_line) > 200:
                return first_line[:200] + "..."
            return first_line
        # Fallback: construct from signature
        return f"{self.symbol_type} {self.qualified_name}({', '.join(self.parameters)})"

    @property
    def searchable_text(self) -> str:
        """Text to embed in Pinecone for semantic search."""
        parts = [self.qualified_name, self.signature]
        if self.docstring:
            parts.append(self.docstring[:500])
        return " ".join(parts)


@dataclass
class ParsedImport:
    """An import statement found in a source file."""
    module: str                  # e.g. "os.path" or "src.utils.retry"
    names: List[str] = field(default_factory=list)  # e.g. ["join", "exists"]
    is_relative: bool = False
    alias: Optional[str] = None


@dataclass
class ParsedCall:
    """A function/method call found inside a symbol's body."""
    caller_name: str             # the symbol that makes the call
    callee_name: str             # the symbol being called
    is_direct: bool = True


@dataclass
class ParsedFile:
    """All extracted data from a single source file."""
    file_path: str
    language: str
    raw_content: str = ""
    total_lines: int = 0
    symbols: List[ParsedSymbol] = field(default_factory=list)
    imports: List[ParsedImport] = field(default_factory=list)
    calls: List[ParsedCall] = field(default_factory=list)
    parse_errors: List[str] = field(default_factory=list)

    @property
    def symbol_names(self) -> List[str]:
        return [s.qualified_name for s in self.symbols]

    @property
    def summary(self) -> str:
        """File-level summary built from symbols. No LLM."""
        if not self.symbols:
            return f"File {self.file_path} ({self.language})"
        names = ", ".join(s.qualified_name for s in self.symbols[:10])
        suffix = f" and {len(self.symbols) - 10} more" if len(self.symbols) > 10 else ""
        return f"Defines: {names}{suffix}"

    @property
    def searchable_text(self) -> str:
        parts = [self.file_path, self.summary]
        for s in self.symbols[:20]:
            if s.docstring:
                parts.append(s.docstring[:100])
        return " ".join(parts)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.raw_content.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Python AST Parser (uses stdlib only)
# ---------------------------------------------------------------------------

ENTRYPOINT_DECORATORS = {
    "app.route", "app.get", "app.post", "app.put", "app.delete", "app.patch",
    "router.get", "router.post", "router.put", "router.delete",
    "api_view", "action", "task", "celery_task",
    "click.command", "click.group",
    "pytest.fixture",
}


class PythonParser:
    """Parse Python files using the built-in ast module."""

    def parse_file(self, file_path: str, content: str) -> ParsedFile:
        result = ParsedFile(
            file_path=file_path,
            language="python",
            raw_content=content,
            total_lines=content.count("\n") + 1,
        )

        try:
            tree = ast.parse(content, filename=file_path)
        except SyntaxError as e:
            result.parse_errors.append(f"SyntaxError: {e}")
            logger.warning("Failed to parse %s: %s", file_path, e)
            return result

        source_lines = content.splitlines()

        result.imports = self._extract_imports(tree)
        result.symbols = self._extract_symbols(tree, source_lines)
        result.calls = self._extract_calls(tree, result.symbols)

        return result

    # -- Imports -----------------------------------------------------------

    def _extract_imports(self, tree: ast.Module) -> List[ParsedImport]:
        imports: List[ParsedImport] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(ParsedImport(
                        module=alias.name,
                        alias=alias.asname,
                    ))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [a.name for a in node.names]
                imports.append(ParsedImport(
                    module=module,
                    names=names,
                    is_relative=node.level > 0,
                ))

        return imports

    # -- Symbols -----------------------------------------------------------

    def _extract_symbols(
        self, tree: ast.Module, source_lines: List[str],
    ) -> List[ParsedSymbol]:
        symbols: List[ParsedSymbol] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                symbols.append(self._parse_function(node, source_lines, parent_class=None))
            elif isinstance(node, ast.ClassDef):
                symbols.append(self._parse_class(node, source_lines))
                for item in ast.iter_child_nodes(node):
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        symbols.append(self._parse_function(
                            item, source_lines, parent_class=node.name,
                        ))

        return symbols

    def _parse_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        source_lines: List[str],
        parent_class: Optional[str],
    ) -> ParsedSymbol:
        is_async = isinstance(node, ast.AsyncFunctionDef)
        name = node.name
        qualified = f"{parent_class}.{name}" if parent_class else name
        sym_type = "method" if parent_class else "function"

        # Signature
        params = self._extract_params(node.args)
        return_type = self._annotation_to_str(node.returns) if node.returns else ""
        async_prefix = "async " if is_async else ""
        ret_suffix = f" -> {return_type}" if return_type else ""
        signature = f"{async_prefix}def {name}({', '.join(params)}){ret_suffix}"

        # Docstring
        docstring = ast.get_docstring(node) or ""

        # Raw code
        start = node.lineno - 1
        end = node.end_lineno or node.lineno
        raw_code = "\n".join(source_lines[start:end])
        content_hash = hashlib.sha256(raw_code.encode()).hexdigest()[:16]

        # Decorators
        decorators = [self._decorator_to_str(d) for d in node.decorator_list]

        # Is public?
        is_public = not name.startswith("_")

        # Is entrypoint?
        is_entrypoint = (
            name == "main"
            or any(d.lower() in ENTRYPOINT_DECORATORS for d in decorators)
        )

        # Cyclomatic complexity
        complexity = self._compute_complexity(node)

        return ParsedSymbol(
            name=name,
            qualified_name=qualified,
            symbol_type=sym_type,
            signature=signature,
            docstring=docstring,
            raw_code=raw_code,
            start_line=node.lineno,
            end_line=end,
            line_count=end - node.lineno + 1,
            parent_class=parent_class,
            is_public=is_public,
            is_entrypoint=is_entrypoint,
            complexity=complexity,
            decorators=decorators,
            parameters=[p.split(":")[0].strip() for p in params],
            return_type=return_type,
            content_hash=content_hash,
        )

    def _parse_class(
        self, node: ast.ClassDef, source_lines: List[str],
    ) -> ParsedSymbol:
        docstring = ast.get_docstring(node) or ""
        start = node.lineno - 1
        end = node.end_lineno or node.lineno
        raw_code = "\n".join(source_lines[start:end])
        content_hash = hashlib.sha256(raw_code.encode()).hexdigest()[:16]

        bases = [self._annotation_to_str(b) for b in node.bases]
        signature = f"class {node.name}"
        if bases:
            signature += f"({', '.join(bases)})"

        methods = []
        for item in ast.iter_child_nodes(node):
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(item.name)

        decorators = [self._decorator_to_str(d) for d in node.decorator_list]

        return ParsedSymbol(
            name=node.name,
            qualified_name=node.name,
            symbol_type="class",
            signature=signature,
            docstring=docstring,
            raw_code=raw_code,
            start_line=node.lineno,
            end_line=end,
            line_count=end - node.lineno + 1,
            is_public=not node.name.startswith("_"),
            decorators=decorators,
            parameters=methods,
            content_hash=content_hash,
        )

    # -- Calls -------------------------------------------------------------

    def _extract_calls(
        self, tree: ast.Module, symbols: List[ParsedSymbol],
    ) -> List[ParsedCall]:
        """Extract function calls from within each symbol's AST subtree."""
        calls: List[ParsedCall] = []
        known_names = {s.name for s in symbols}

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            caller = node.name
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue

                callee = self._call_to_name(child)
                if callee and callee != caller:
                    calls.append(ParsedCall(
                        caller_name=caller,
                        callee_name=callee,
                        is_direct=True,
                    ))

        return calls

    def _call_to_name(self, node: ast.Call) -> Optional[str]:
        """Extract the function name from a Call node."""
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        elif isinstance(func, ast.Attribute):
            # e.g. self.method() or module.func()
            parts = []
            current = func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                if current.id == "self":
                    return parts[-1] if parts else None
                parts.append(current.id)
            parts.reverse()
            return ".".join(parts) if parts else None
        return None

    # -- Helpers -----------------------------------------------------------

    def _extract_params(self, args: ast.arguments) -> List[str]:
        """Build parameter strings from function arguments."""
        params: List[str] = []
        defaults_offset = len(args.args) - len(args.defaults)

        for i, arg in enumerate(args.args):
            if arg.arg == "self" or arg.arg == "cls":
                continue
            p = arg.arg
            if arg.annotation:
                p += f": {self._annotation_to_str(arg.annotation)}"
            default_idx = i - defaults_offset
            if default_idx >= 0 and default_idx < len(args.defaults):
                p += " = ..."
            params.append(p)

        if args.vararg:
            params.append(f"*{args.vararg.arg}")
        if args.kwarg:
            params.append(f"**{args.kwarg.arg}")

        return params

    def _annotation_to_str(self, node: Optional[ast.expr]) -> str:
        """Convert an AST annotation node to a readable string."""
        if node is None:
            return ""
        try:
            return ast.unparse(node)
        except Exception:
            return ""

    def _decorator_to_str(self, node: ast.expr) -> str:
        try:
            return ast.unparse(node)
        except Exception:
            return ""

    def _compute_complexity(self, node: ast.AST) -> int:
        """Compute cyclomatic complexity from AST nodes. No LLM needed."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.IfExp)):
                complexity += 1
            elif isinstance(child, (ast.For, ast.AsyncFor, ast.While)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.Assert):
                complexity += 1
        return complexity


# ---------------------------------------------------------------------------
# Tree-sitter Parser for TypeScript / JavaScript
# ---------------------------------------------------------------------------

# Entrypoint decorators/patterns for JS/TS (Express, Next.js, NestJS, etc.)
JS_ENTRYPOINT_PATTERNS = {
    "app.get", "app.post", "app.put", "app.delete", "app.patch", "app.use",
    "router.get", "router.post", "router.put", "router.delete",
    "express.Router", "createHandler", "handler",
    "getServerSideProps", "getStaticProps", "getStaticPaths",
    "@Get", "@Post", "@Put", "@Delete", "@Patch", "@Controller",
    "test", "it", "describe", "beforeEach", "afterEach",
}


class TreeSitterParser:
    """
    Parse TypeScript/JavaScript files using tree-sitter.
    
    Supports: .ts, .tsx, .js, .jsx
    
    Extracts:
      - Function declarations (function foo() {})
      - Arrow functions assigned to const/let/var
      - Class declarations and their methods
      - Import statements
      - Function call edges
    """

    def __init__(self, language_name: str):
        """
        Initialize parser for a specific language.
        
        Args:
            language_name: One of "typescript", "javascript"
        """
        if not TREE_SITTER_AVAILABLE:
            raise ImportError(
                "tree-sitter is not installed. Install with: "
                "pip install tree-sitter tree-sitter-typescript tree-sitter-javascript"
            )
        
        self.language_name = language_name
        self._parser = Parser()
        
        # Select the appropriate language grammar
        if language_name == "typescript":
            # TSX grammar handles both .ts and .tsx files
            lang = Language(ts_typescript.language_tsx())
        else:
            # JavaScript grammar handles both .js and .jsx
            lang = Language(ts_javascript.language())
        
        self._parser.language = lang

    def parse_file(self, file_path: str, content: str) -> ParsedFile:
        """Parse a TypeScript/JavaScript file and extract all symbols."""
        result = ParsedFile(
            file_path=file_path,
            language=self.language_name,
            raw_content=content,
            total_lines=content.count("\n") + 1,
        )

        try:
            tree = self._parser.parse(bytes(content, "utf-8"))
        except Exception as e:
            result.parse_errors.append(f"Parse error: {e}")
            logger.warning("Failed to parse %s: %s", file_path, e)
            return result

        root = tree.root_node
        source_bytes = bytes(content, "utf-8")
        source_lines = content.splitlines()

        result.imports = self._extract_imports(root, source_bytes)
        result.symbols = self._extract_symbols(root, source_bytes, source_lines)
        result.calls = self._extract_calls(root, source_bytes, result.symbols)

        return result

    # -- Imports -----------------------------------------------------------

    def _extract_imports(self, root: "Node", source_bytes: bytes) -> List[ParsedImport]:
        """Extract import statements from the AST."""
        imports: List[ParsedImport] = []

        def visit(node: "Node") -> None:
            # import x from 'module'
            # import { x, y } from 'module'
            # import * as x from 'module'
            if node.type == "import_statement":
                module = ""
                names: List[str] = []
                alias: Optional[str] = None
                
                source_node = node.child_by_field_name("source")
                if source_node:
                    module = self._get_node_text(source_node, source_bytes).strip("'\"")
                
                # Extract imported names
                for child in node.children:
                    if child.type == "import_clause":
                        for clause_child in child.children:
                            if clause_child.type == "identifier":
                                names.append(self._get_node_text(clause_child, source_bytes))
                            elif clause_child.type == "named_imports":
                                for spec in clause_child.children:
                                    if spec.type == "import_specifier":
                                        name_node = spec.child_by_field_name("name")
                                        if name_node:
                                            names.append(self._get_node_text(name_node, source_bytes))
                            elif clause_child.type == "namespace_import":
                                # import * as X
                                for ns_child in clause_child.children:
                                    if ns_child.type == "identifier":
                                        alias = self._get_node_text(ns_child, source_bytes)
                
                imports.append(ParsedImport(
                    module=module,
                    names=names,
                    is_relative=module.startswith("."),
                    alias=alias,
                ))
            
            # const x = require('module')
            elif node.type == "call_expression":
                func = node.child_by_field_name("function")
                if func and self._get_node_text(func, source_bytes) == "require":
                    args = node.child_by_field_name("arguments")
                    if args and args.child_count > 0:
                        first_arg = args.children[1] if args.child_count > 1 else None
                        if first_arg and first_arg.type == "string":
                            module = self._get_node_text(first_arg, source_bytes).strip("'\"")
                            imports.append(ParsedImport(
                                module=module,
                                is_relative=module.startswith("."),
                            ))
            
            for child in node.children:
                visit(child)
        
        visit(root)
        return imports

    # -- Symbols -----------------------------------------------------------

    def _extract_symbols(
        self, root: "Node", source_bytes: bytes, source_lines: List[str],
    ) -> List[ParsedSymbol]:
        """Extract function and class declarations from the AST."""
        symbols: List[ParsedSymbol] = []
        
        def visit(node: "Node", parent_class: Optional[str] = None) -> None:
            # function declaration: function foo() {}
            if node.type == "function_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    symbols.append(self._parse_function_node(
                        node, name_node, source_bytes, source_lines, parent_class, is_async=False
                    ))
            
            # generator function: function* foo() {}
            elif node.type == "generator_function_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    symbols.append(self._parse_function_node(
                        node, name_node, source_bytes, source_lines, parent_class, is_async=False
                    ))
            
            # Arrow function or function expression assigned to variable
            # const foo = () => {} or const foo = function() {}
            elif node.type in ("lexical_declaration", "variable_declaration"):
                for decl in node.children:
                    if decl.type == "variable_declarator":
                        name_node = decl.child_by_field_name("name")
                        value_node = decl.child_by_field_name("value")
                        
                        if name_node and value_node:
                            if value_node.type in ("arrow_function", "function_expression", "function"):
                                is_async = any(
                                    c.type == "async" for c in value_node.children
                                ) or self._get_node_text(value_node, source_bytes).strip().startswith("async")
                                symbols.append(self._parse_arrow_function(
                                    decl, name_node, value_node, source_bytes, source_lines, parent_class, is_async
                                ))
            
            # Class declaration: class Foo {}
            elif node.type == "class_declaration":
                name_node = node.child_by_field_name("name")
                class_name = self._get_node_text(name_node, source_bytes) if name_node else "AnonymousClass"
                symbols.append(self._parse_class_node(node, class_name, source_bytes, source_lines))
                
                # Visit methods inside the class body
                body = node.child_by_field_name("body")
                if body:
                    for child in body.children:
                        if child.type == "method_definition":
                            method_name_node = child.child_by_field_name("name")
                            if method_name_node:
                                is_async = any(c.type == "async" for c in child.children)
                                symbols.append(self._parse_method_node(
                                    child, method_name_node, source_bytes, source_lines, class_name, is_async
                                ))
                        elif child.type == "public_field_definition":
                            # Handle class field arrow functions: foo = () => {}
                            name_n = child.child_by_field_name("name")
                            value_n = child.child_by_field_name("value")
                            if name_n and value_n and value_n.type == "arrow_function":
                                is_async = any(c.type == "async" for c in value_n.children)
                                symbols.append(self._parse_arrow_function(
                                    child, name_n, value_n, source_bytes, source_lines, class_name, is_async
                                ))
                    return  # Don't recurse into class body again
            
            # Export statements may contain declarations
            elif node.type in ("export_statement", "export_default_declaration"):
                for child in node.children:
                    visit(child, parent_class)
                return
            
            # Recurse into children (skip class bodies, handled above)
            for child in node.children:
                visit(child, parent_class)
        
        visit(root)
        return symbols

    def _parse_function_node(
        self,
        node: "Node",
        name_node: "Node",
        source_bytes: bytes,
        source_lines: List[str],
        parent_class: Optional[str],
        is_async: bool,
    ) -> ParsedSymbol:
        """Parse a function declaration node."""
        name = self._get_node_text(name_node, source_bytes)
        qualified = f"{parent_class}.{name}" if parent_class else name
        sym_type = "method" if parent_class else "function"
        
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        raw_code = "\n".join(source_lines[node.start_point[0]:node.end_point[0] + 1])
        content_hash = hashlib.sha256(raw_code.encode()).hexdigest()[:16]
        
        # Extract parameters
        params = self._extract_parameters(node, source_bytes)
        
        # Extract return type (TypeScript)
        return_type = self._extract_return_type(node, source_bytes)
        
        # Build signature
        async_prefix = "async " if is_async else ""
        ret_suffix = f": {return_type}" if return_type else ""
        signature = f"{async_prefix}function {name}({', '.join(params)}){ret_suffix}"
        
        # Extract JSDoc comment
        docstring = self._extract_jsdoc(node, source_lines)
        
        # Check if entrypoint
        is_entrypoint = self._is_entrypoint(name, node, source_bytes)
        
        # Compute complexity
        complexity = self._compute_complexity(node)
        
        return ParsedSymbol(
            name=name,
            qualified_name=qualified,
            symbol_type=sym_type,
            signature=signature,
            docstring=docstring,
            raw_code=raw_code,
            start_line=start_line,
            end_line=end_line,
            line_count=end_line - start_line + 1,
            parent_class=parent_class,
            is_public=not name.startswith("_"),
            is_entrypoint=is_entrypoint,
            complexity=complexity,
            parameters=[p.split(":")[0].strip() for p in params],
            return_type=return_type,
            content_hash=content_hash,
        )

    def _parse_arrow_function(
        self,
        decl_node: "Node",
        name_node: "Node",
        value_node: "Node",
        source_bytes: bytes,
        source_lines: List[str],
        parent_class: Optional[str],
        is_async: bool,
    ) -> ParsedSymbol:
        """Parse an arrow function or function expression assigned to a variable."""
        name = self._get_node_text(name_node, source_bytes)
        qualified = f"{parent_class}.{name}" if parent_class else name
        sym_type = "method" if parent_class else "function"
        
        # Use the full declaration for line range
        node = decl_node
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        raw_code = "\n".join(source_lines[node.start_point[0]:node.end_point[0] + 1])
        content_hash = hashlib.sha256(raw_code.encode()).hexdigest()[:16]
        
        # Extract parameters from the arrow function
        params = self._extract_parameters(value_node, source_bytes)
        
        # Extract return type
        return_type = self._extract_return_type(value_node, source_bytes)
        
        # Build signature
        async_prefix = "async " if is_async else ""
        ret_suffix = f": {return_type}" if return_type else ""
        signature = f"const {name} = {async_prefix}({', '.join(params)}){ret_suffix} => ..."
        
        # Extract JSDoc
        docstring = self._extract_jsdoc(node, source_lines)
        
        # Check entrypoint
        is_entrypoint = self._is_entrypoint(name, node, source_bytes)
        
        # Complexity
        complexity = self._compute_complexity(value_node)
        
        return ParsedSymbol(
            name=name,
            qualified_name=qualified,
            symbol_type=sym_type,
            signature=signature,
            docstring=docstring,
            raw_code=raw_code,
            start_line=start_line,
            end_line=end_line,
            line_count=end_line - start_line + 1,
            parent_class=parent_class,
            is_public=not name.startswith("_"),
            is_entrypoint=is_entrypoint,
            complexity=complexity,
            parameters=[p.split(":")[0].strip() for p in params],
            return_type=return_type,
            content_hash=content_hash,
        )

    def _parse_method_node(
        self,
        node: "Node",
        name_node: "Node",
        source_bytes: bytes,
        source_lines: List[str],
        parent_class: str,
        is_async: bool,
    ) -> ParsedSymbol:
        """Parse a class method definition."""
        name = self._get_node_text(name_node, source_bytes)
        qualified = f"{parent_class}.{name}"
        
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        raw_code = "\n".join(source_lines[node.start_point[0]:node.end_point[0] + 1])
        content_hash = hashlib.sha256(raw_code.encode()).hexdigest()[:16]
        
        # Extract parameters
        params = self._extract_parameters(node, source_bytes)
        
        # Extract return type
        return_type = self._extract_return_type(node, source_bytes)
        
        # Build signature
        async_prefix = "async " if is_async else ""
        ret_suffix = f": {return_type}" if return_type else ""
        signature = f"{async_prefix}{name}({', '.join(params)}){ret_suffix}"
        
        # Check for static/getter/setter
        is_static = any(c.type == "static" for c in node.children)
        is_getter = any(c.type == "get" for c in node.children)
        is_setter = any(c.type == "set" for c in node.children)
        
        if is_static:
            signature = f"static {signature}"
        if is_getter:
            signature = f"get {signature}"
        if is_setter:
            signature = f"set {signature}"
        
        # Extract JSDoc
        docstring = self._extract_jsdoc(node, source_lines)
        
        # Complexity
        complexity = self._compute_complexity(node)
        
        return ParsedSymbol(
            name=name,
            qualified_name=qualified,
            symbol_type="method",
            signature=signature,
            docstring=docstring,
            raw_code=raw_code,
            start_line=start_line,
            end_line=end_line,
            line_count=end_line - start_line + 1,
            parent_class=parent_class,
            is_public=not name.startswith("_") and not name.startswith("#"),
            is_entrypoint=False,
            complexity=complexity,
            parameters=[p.split(":")[0].strip() for p in params],
            return_type=return_type,
            content_hash=content_hash,
        )

    def _parse_class_node(
        self,
        node: "Node",
        class_name: str,
        source_bytes: bytes,
        source_lines: List[str],
    ) -> ParsedSymbol:
        """Parse a class declaration."""
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        raw_code = "\n".join(source_lines[node.start_point[0]:node.end_point[0] + 1])
        content_hash = hashlib.sha256(raw_code.encode()).hexdigest()[:16]
        
        # Extract base classes (extends, implements)
        bases: List[str] = []
        heritage = node.child_by_field_name("heritage")
        if heritage:
            bases.append(self._get_node_text(heritage, source_bytes))
        
        # Build signature
        signature = f"class {class_name}"
        if bases:
            signature += f" extends {', '.join(bases)}"
        
        # Extract method names
        methods: List[str] = []
        body = node.child_by_field_name("body")
        if body:
            for child in body.children:
                if child.type == "method_definition":
                    name_node = child.child_by_field_name("name")
                    if name_node:
                        methods.append(self._get_node_text(name_node, source_bytes))
        
        # Extract JSDoc
        docstring = self._extract_jsdoc(node, source_lines)
        
        # Check for decorators (TypeScript)
        decorators: List[str] = []
        for child in node.children:
            if child.type == "decorator":
                decorators.append(self._get_node_text(child, source_bytes))
        
        return ParsedSymbol(
            name=class_name,
            qualified_name=class_name,
            symbol_type="class",
            signature=signature,
            docstring=docstring,
            raw_code=raw_code,
            start_line=start_line,
            end_line=end_line,
            line_count=end_line - start_line + 1,
            is_public=not class_name.startswith("_"),
            decorators=decorators,
            parameters=methods,  # Store method names in parameters for classes
            content_hash=content_hash,
        )

    # -- Calls -------------------------------------------------------------

    def _extract_calls(
        self, root: "Node", source_bytes: bytes, symbols: List[ParsedSymbol],
    ) -> List[ParsedCall]:
        """Extract function calls from the AST."""
        calls: List[ParsedCall] = []
        known_names = {s.name for s in symbols}
        
        # Map line ranges to symbol names
        symbol_ranges: List[Tuple[int, int, str]] = []
        for s in symbols:
            if s.symbol_type in ("function", "method"):
                symbol_ranges.append((s.start_line, s.end_line, s.name))
        
        def find_caller(line: int) -> Optional[str]:
            for start, end, name in symbol_ranges:
                if start <= line <= end:
                    return name
            return None
        
        def visit(node: "Node") -> None:
            if node.type == "call_expression":
                func = node.child_by_field_name("function")
                if func:
                    callee = self._call_to_name(func, source_bytes)
                    if callee:
                        line = node.start_point[0] + 1
                        caller = find_caller(line)
                        if caller and caller != callee:
                            calls.append(ParsedCall(
                                caller_name=caller,
                                callee_name=callee,
                                is_direct=True,
                            ))
            
            for child in node.children:
                visit(child)
        
        visit(root)
        return calls

    def _call_to_name(self, node: "Node", source_bytes: bytes) -> Optional[str]:
        """Extract the function name from a call expression's function node."""
        if node.type == "identifier":
            return self._get_node_text(node, source_bytes)
        elif node.type == "member_expression":
            # obj.method() -> extract "method" or "obj.method"
            prop = node.child_by_field_name("property")
            obj = node.child_by_field_name("object")
            if prop:
                prop_name = self._get_node_text(prop, source_bytes)
                if obj and obj.type == "this":
                    return prop_name
                elif obj:
                    obj_name = self._get_node_text(obj, source_bytes)
                    return f"{obj_name}.{prop_name}"
                return prop_name
        return None

    # -- Helpers -----------------------------------------------------------

    def _get_node_text(self, node: "Node", source_bytes: bytes) -> str:
        """Extract the text content of a node."""
        return source_bytes[node.start_byte:node.end_byte].decode("utf-8")

    def _extract_parameters(self, node: "Node", source_bytes: bytes) -> List[str]:
        """Extract function parameters from a node."""
        params: List[str] = []
        
        # Find the parameters/formal_parameters node
        params_node = node.child_by_field_name("parameters")
        if not params_node:
            # Try looking for formal_parameters
            for child in node.children:
                if child.type == "formal_parameters":
                    params_node = child
                    break
        
        if not params_node:
            return params
        
        for child in params_node.children:
            if child.type in ("identifier", "required_parameter", "optional_parameter", "rest_parameter"):
                param_text = self._get_node_text(child, source_bytes)
                # Clean up the parameter
                param_text = param_text.strip()
                if param_text and param_text not in ("(", ")", ","):
                    params.append(param_text)
            elif child.type == "assignment_pattern":
                # Default parameter: x = value
                left = child.child_by_field_name("left")
                if left:
                    params.append(f"{self._get_node_text(left, source_bytes)} = ...")
        
        return params

    def _extract_return_type(self, node: "Node", source_bytes: bytes) -> str:
        """Extract TypeScript return type annotation if present."""
        # Look for type_annotation or return_type child
        for child in node.children:
            if child.type == "type_annotation":
                # Skip the colon
                for tc in child.children:
                    if tc.type != ":":
                        return self._get_node_text(tc, source_bytes)
        
        return_type = node.child_by_field_name("return_type")
        if return_type:
            return self._get_node_text(return_type, source_bytes)
        
        return ""

    def _extract_jsdoc(self, node: "Node", source_lines: List[str]) -> str:
        """Extract JSDoc comment preceding a node."""
        start_line = node.start_point[0]
        if start_line == 0:
            return ""
        
        # Look backwards for a comment block
        jsdoc_lines: List[str] = []
        for i in range(start_line - 1, max(0, start_line - 20), -1):
            line = source_lines[i].strip()
            if line.endswith("*/"):
                jsdoc_lines.insert(0, line)
                # Continue collecting until we find the start
                for j in range(i - 1, max(0, i - 50), -1):
                    prev_line = source_lines[j].strip()
                    jsdoc_lines.insert(0, prev_line)
                    if prev_line.startswith("/**") or prev_line.startswith("/*"):
                        break
                break
            elif line and not line.startswith("*") and not line.startswith("//"):
                # Not a comment, stop looking
                break
        
        if jsdoc_lines:
            # Clean up JSDoc
            doc = "\n".join(jsdoc_lines)
            # Remove comment markers
            doc = re.sub(r'^/\*\*?\s*', '', doc)
            doc = re.sub(r'\s*\*/$', '', doc)
            doc = re.sub(r'^\s*\*\s?', '', doc, flags=re.MULTILINE)
            return doc.strip()
        
        return ""

    def _is_entrypoint(self, name: str, node: "Node", source_bytes: bytes) -> bool:
        """Check if a function is likely an entrypoint."""
        # Check name patterns
        if name in ("main", "handler", "default"):
            return True
        
        # Check for export default
        parent = node.parent
        while parent:
            if parent.type in ("export_statement", "export_default_declaration"):
                if "default" in self._get_node_text(parent, source_bytes):
                    return True
            parent = parent.parent
        
        # Check for common framework patterns
        for pattern in JS_ENTRYPOINT_PATTERNS:
            if name == pattern or name.lower() == pattern.lower():
                return True
        
        return False

    def _compute_complexity(self, node: "Node") -> int:
        """Compute cyclomatic complexity from tree-sitter AST."""
        complexity = 1
        
        def visit(n: "Node") -> None:
            nonlocal complexity
            
            # Decision points
            if n.type in ("if_statement", "ternary_expression", "conditional_expression"):
                complexity += 1
            elif n.type in ("for_statement", "for_in_statement", "while_statement", "do_statement"):
                complexity += 1
            elif n.type == "switch_case":
                complexity += 1
            elif n.type == "catch_clause":
                complexity += 1
            elif n.type == "binary_expression":
                # Count && and || operators
                op = n.child_by_field_name("operator")
                if op:
                    op_text = self._get_node_text(op, node.parent.parent if node.parent and node.parent.parent else node)
                    # This is a bit hacky, but tree-sitter stores the operator as text
                    if "&&" in str(n) or "||" in str(n):
                        complexity += 1
            
            for child in n.children:
                visit(child)
        
        visit(node)
        return complexity

class GoParser:
    """Parse Go files using tree-sitter."""

    def __init__(self):
        if not TREE_SITTER_AVAILABLE:
            raise ImportError("tree-sitter-go is not installed")
        
        self._parser = Parser()
        self._parser.language = Language(ts_go.language())

    def parse_file(self, file_path: str, content: str) -> ParsedFile:
        result = ParsedFile(
            file_path=file_path,
            language="go",
            raw_content=content,
            total_lines=content.count("\n") + 1,
        )

        try:
            tree = self._parser.parse(bytes(content, "utf-8"))
        except Exception as e:
            result.parse_errors.append(f"Parse error: {e}")
            return result

        root = tree.root_node
        source_bytes = bytes(content, "utf-8")
        source_lines = content.splitlines()

        result.imports = self._extract_imports(root, source_bytes)
        result.symbols = self._extract_symbols(root, source_bytes, source_lines)
        result.calls = self._extract_calls(root, source_bytes, result.symbols)

        return result

    def _extract_imports(self, root, source_bytes: bytes) -> List[ParsedImport]:
        imports: List[ParsedImport] = []
        
        def visit(node):
            # import "fmt" or import ( "fmt" "os" )
            if node.type == "import_declaration":
                for child in node.children:
                    if child.type == "import_spec":
                        path = child.child_by_field_name("path")
                        if path:
                            module = self._get_node_text(path, source_bytes).strip('"')
                            alias_node = child.child_by_field_name("name")
                            alias = self._get_node_text(alias_node, source_bytes) if alias_node else None
                            imports.append(ParsedImport(module=module, alias=alias))
                    elif child.type == "import_spec_list":
                        for spec in child.children:
                            if spec.type == "import_spec":
                                path = spec.child_by_field_name("path")
                                if path:
                                    module = self._get_node_text(path, source_bytes).strip('"')
                                    alias_node = spec.child_by_field_name("name")
                                    alias = self._get_node_text(alias_node, source_bytes) if alias_node else None
                                    imports.append(ParsedImport(module=module, alias=alias))
            
            for child in node.children:
                visit(child)
        
        visit(root)
        return imports

    def _extract_symbols(self, root, source_bytes: bytes, source_lines: List[str]) -> List[ParsedSymbol]:
        symbols: List[ParsedSymbol] = []
        
        def visit(node, receiver_type: Optional[str] = None):
            # func foo() {} or func (r *Receiver) foo() {}
            if node.type == "function_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    symbols.append(self._parse_function(node, name_node, source_bytes, source_lines, None))
            
            # Method: func (r *Type) Method() {}
            elif node.type == "method_declaration":
                name_node = node.child_by_field_name("name")
                receiver = node.child_by_field_name("receiver")
                parent_class = None
                if receiver:
                    # Extract receiver type
                    for child in receiver.children:
                        if child.type == "parameter_declaration":
                            type_node = child.child_by_field_name("type")
                            if type_node:
                                parent_class = self._get_node_text(type_node, source_bytes).lstrip("*")
                
                if name_node:
                    symbols.append(self._parse_function(node, name_node, source_bytes, source_lines, parent_class))
            
            # type Foo struct {} or type Foo interface {}
            elif node.type == "type_declaration":
                for child in node.children:
                    if child.type == "type_spec":
                        name_node = child.child_by_field_name("name")
                        type_node = child.child_by_field_name("type")
                        if name_node and type_node:
                            if type_node.type in ("struct_type", "interface_type"):
                                symbols.append(self._parse_type(child, name_node, type_node, source_bytes, source_lines))
            
            for child in node.children:
                visit(child)
        
        visit(root)
        return symbols

    def _parse_function(self, node, name_node, source_bytes: bytes, source_lines: List[str], parent_class: Optional[str]) -> ParsedSymbol:
        name = self._get_node_text(name_node, source_bytes)
        qualified = f"{parent_class}.{name}" if parent_class else name
        sym_type = "method" if parent_class else "function"
        
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        raw_code = "\n".join(source_lines[node.start_point[0]:node.end_point[0] + 1])
        content_hash = hashlib.sha256(raw_code.encode()).hexdigest()[:16]
        
        # Extract parameters
        params: List[str] = []
        params_node = node.child_by_field_name("parameters")
        if params_node:
            for child in params_node.children:
                if child.type == "parameter_declaration":
                    params.append(self._get_node_text(child, source_bytes))
        
        # Extract return type
        result_node = node.child_by_field_name("result")
        return_type = self._get_node_text(result_node, source_bytes) if result_node else ""
        
        # Build signature
        ret_suffix = f" {return_type}" if return_type else ""
        signature = f"func {name}({', '.join(params)}){ret_suffix}"
        
        # Extract doc comment (Go uses // comments above function)
        docstring = self._extract_doc_comment(node, source_lines)
        
        # Check if exported (starts with uppercase)
        is_public = name[0].isupper() if name else False
        
        # Check if entrypoint
        is_entrypoint = name == "main" or name.startswith("Test") or name.startswith("Benchmark")
        
        return ParsedSymbol(
            name=name,
            qualified_name=qualified,
            symbol_type=sym_type,
            signature=signature,
            docstring=docstring,
            raw_code=raw_code,
            start_line=start_line,
            end_line=end_line,
            line_count=end_line - start_line + 1,
            parent_class=parent_class,
            is_public=is_public,
            is_entrypoint=is_entrypoint,
            parameters=[p.split(" ")[0] for p in params],
            return_type=return_type,
            content_hash=content_hash,
        )

    def _parse_type(self, spec_node, name_node, type_node, source_bytes: bytes, source_lines: List[str]) -> ParsedSymbol:
        name = self._get_node_text(name_node, source_bytes)
        type_kind = "struct" if type_node.type == "struct_type" else "interface"
        
        start_line = spec_node.start_point[0] + 1
        end_line = spec_node.end_point[0] + 1
        raw_code = "\n".join(source_lines[spec_node.start_point[0]:spec_node.end_point[0] + 1])
        content_hash = hashlib.sha256(raw_code.encode()).hexdigest()[:16]
        
        signature = f"type {name} {type_kind}"
        docstring = self._extract_doc_comment(spec_node, source_lines)
        is_public = name[0].isupper() if name else False
        
        # Extract field/method names
        fields: List[str] = []
        for child in type_node.children:
            if child.type == "field_declaration_list":
                for field in child.children:
                    if field.type == "field_declaration":
                        name_n = field.child_by_field_name("name")
                        if name_n:
                            fields.append(self._get_node_text(name_n, source_bytes))
        
        return ParsedSymbol(
            name=name,
            qualified_name=name,
            symbol_type="class",  # Map struct/interface to class for consistency
            signature=signature,
            docstring=docstring,
            raw_code=raw_code,
            start_line=start_line,
            end_line=end_line,
            line_count=end_line - start_line + 1,
            is_public=is_public,
            parameters=fields,
            content_hash=content_hash,
        )

    def _extract_calls(self, root, source_bytes: bytes, symbols: List[ParsedSymbol]) -> List[ParsedCall]:
        calls: List[ParsedCall] = []
        symbol_ranges = [(s.start_line, s.end_line, s.name) for s in symbols if s.symbol_type in ("function", "method")]
        
        def find_caller(line: int) -> Optional[str]:
            for start, end, name in symbol_ranges:
                if start <= line <= end:
                    return name
            return None
        
        def visit(node):
            if node.type == "call_expression":
                func = node.child_by_field_name("function")
                if func:
                    callee = self._get_node_text(func, source_bytes)
                    # Handle method calls like obj.Method()
                    if "." in callee:
                        callee = callee.split(".")[-1]
                    
                    line = node.start_point[0] + 1
                    caller = find_caller(line)
                    if caller and caller != callee:
                        calls.append(ParsedCall(caller_name=caller, callee_name=callee, is_direct=True))
            
            for child in node.children:
                visit(child)
        
        visit(root)
        return calls

    def _extract_doc_comment(self, node, source_lines: List[str]) -> str:
        start_line = node.start_point[0]
        if start_line == 0:
            return ""
        
        doc_lines: List[str] = []
        for i in range(start_line - 1, max(0, start_line - 20), -1):
            line = source_lines[i].strip()
            if line.startswith("//"):
                doc_lines.insert(0, line[2:].strip())
            elif line == "":
                continue
            else:
                break
        
        return "\n".join(doc_lines)

    def _get_node_text(self, node, source_bytes: bytes) -> str:
        return source_bytes[node.start_byte:node.end_byte].decode("utf-8")
# ---------------------------------------------------------------------------
# Multi-language dispatcher
# ---------------------------------------------------------------------------

_python_parser = PythonParser()

# Initialize tree-sitter parsers if available
_ts_parser: Optional[TreeSitterParser] = None
_js_parser: Optional[TreeSitterParser] = None
_go_parser: Optional[GoParser] = None

if TREE_SITTER_AVAILABLE:
    try:
        _ts_parser = TreeSitterParser("typescript")
        _js_parser = TreeSitterParser("javascript")
        _go_parser = GoParser()
        logger.info("Tree-sitter parsers initialized for TypeScript and JavaScript")
    except Exception as e:
        logger.warning("Failed to initialize tree-sitter parsers: %s", e)

PARSERS: Dict[str, Any] = {
    "python": _python_parser,
}

# Add tree-sitter parsers if available
if _ts_parser:
    PARSERS["typescript"] = _ts_parser
if _js_parser:
    PARSERS["javascript"] = _js_parser
if _go_parser:
    PARSERS["go"] = _go_parser 


def parse_file(file_path: str, content: str, language: str) -> ParsedFile:
    """Parse a source file and extract all symbols, imports, and calls.

    Falls back to a minimal ParsedFile (just metadata, no symbols)
    if no parser is available for the language.
    """
    parser = PARSERS.get(language)
    if parser:
        return parser.parse_file(file_path, content)

    # No parser available — return file-level metadata only
    return ParsedFile(
        file_path=file_path,
        language=language,
        raw_content=content,
        total_lines=content.count("\n") + 1,
    )


def compute_content_hash(content: str) -> str:
    """Compute a short hash for change detection."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]
