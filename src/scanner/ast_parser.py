"""
AST Parser — extracts symbols, imports, and call edges from source code.

Uses Python's built-in ``ast`` module for Python files (zero dependencies,
maximum reliability). Returns language-agnostic data structures that the
indexer writes to Pinecone, Neo4j, and MongoDB.

No LLM calls. Everything is deterministic software logic.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

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

        class CallVisitor(ast.NodeVisitor):
            def __init__(self, parser):
                self.parser = parser
                self.caller_stack = []

            def visit_FunctionDef(self, node):
                self.caller_stack.append(node.name)
                self.generic_visit(node)
                self.caller_stack.pop()

            def visit_AsyncFunctionDef(self, node):
                self.caller_stack.append(node.name)
                self.generic_visit(node)
                self.caller_stack.pop()

            def visit_Call(self, node):
                callee = self.parser._call_to_name(node)
                if callee:
                    for caller in self.caller_stack:
                        if callee != caller:
                            calls.append(ParsedCall(
                                caller_name=caller,
                                callee_name=callee,
                                is_direct=True,
                            ))
                self.generic_visit(node)

        visitor = CallVisitor(self)
        visitor.visit(tree)

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
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 1

            def visit_If(self, node):
                self.complexity += 1
                self.generic_visit(node)

            def visit_IfExp(self, node):
                self.complexity += 1
                self.generic_visit(node)

            def visit_For(self, node):
                self.complexity += 1
                self.generic_visit(node)

            def visit_AsyncFor(self, node):
                self.complexity += 1
                self.generic_visit(node)

            def visit_While(self, node):
                self.complexity += 1
                self.generic_visit(node)

            def visit_ExceptHandler(self, node):
                self.complexity += 1
                self.generic_visit(node)

            def visit_BoolOp(self, node):
                self.complexity += len(node.values) - 1
                self.generic_visit(node)

            def visit_Assert(self, node):
                self.complexity += 1
                self.generic_visit(node)

        visitor = ComplexityVisitor()
        visitor.visit(node)
        return visitor.complexity


# ---------------------------------------------------------------------------
# Multi-language dispatcher
# ---------------------------------------------------------------------------

_python_parser = PythonParser()

PARSERS = {
    "python": _python_parser,
}


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
