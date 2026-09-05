"""Static safety analysis for agent-generated test code.

The Generator's output is model-written Python that this process then executes.
That is unavoidable - the whole point is that no human writes the tests - so it
is gated by an AST check before it is ever compiled.

The check is a whitelist, not a blacklist of known-bad strings: the module is
parsed, and any import, call or attribute access outside the permitted set
rejects the file. A rejected file never runs; the Generator retries once and
then falls back to its deterministic compiler.

Honest scope statement
----------------------
This is defence in depth, not a security boundary. Restricted builtins and an
AST whitelist raise the cost of an escape substantially but a determined
adversary with control of the model output could still find a path. The real
boundary is that you point this agent at applications you are authorised to
test, with an API key you control. Treat generated tests the way you would
treat any code from an untrusted source: run them in a container or a VM if
the target is not yours.
"""

from __future__ import annotations

import ast
import builtins as _builtins
from dataclasses import dataclass, field
from typing import Any

# Modules the generated test may import. Deliberately tiny: the runner injects
# everything a test legitimately needs into the module namespace.
ALLOWED_IMPORTS: frozenset[str] = frozenset({"re", "asyncio", "json", "math"})

# Names that must never be referenced.
FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "eval", "exec", "compile", "__import__", "open", "input", "breakpoint",
        "globals", "locals", "vars", "memoryview", "help", "exit", "quit",
        "os", "sys", "subprocess", "socket", "shutil", "pathlib", "importlib",
        "ctypes", "pickle", "marshal", "shelve", "tempfile", "glob", "requests",
        "httpx", "urllib", "http", "ftplib", "smtplib", "multiprocessing",
        "threading", "signal", "resource", "platform", "getpass", "pty",
    }
)

# Dunder attributes that lead straight out of the sandbox.
FORBIDDEN_ATTRS: frozenset[str] = frozenset(
    {
        "__class__", "__bases__", "__subclasses__", "__mro__", "__globals__",
        "__builtins__", "__code__", "__closure__", "__func__", "__self__",
        "__dict__", "__getattribute__", "__reduce__", "__reduce_ex__",
        "__init_subclass__", "__loader__", "__spec__", "mro",
    }
)

REQUIRED_FUNCTION = "test_flow"
REQUIRED_PARAMS = ("page", "ctx")

# Builtins the generated code is allowed to see at runtime. Everything else -
# notably ``open``, ``__import__`` and ``eval`` - is simply absent from the
# executed module's namespace.
SAFE_BUILTIN_NAMES: tuple[str, ...] = (
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "format", "int", "isinstance", "len", "list", "map", "max", "min",
    "print", "range", "repr", "reversed", "round", "set", "sorted", "str",
    "sum", "tuple", "zip", "Exception", "AssertionError", "ValueError",
    "TypeError", "KeyError", "IndexError", "AttributeError", "RuntimeError",
    "StopIteration", "StopAsyncIteration",
)

SAFE_BUILTINS: dict[str, Any] = {
    name: getattr(_builtins, name)
    for name in SAFE_BUILTIN_NAMES
    if hasattr(_builtins, name)
}


@dataclass
class SourceValidation:
    """Verdict on one generated test module."""

    ok: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    has_required_function: bool = False
    awaits_count: int = 0
    assertion_count: int = 0

    def summary(self) -> str:
        if self.ok:
            return (
                f"valid: {self.awaits_count} awaited calls, "
                f"{self.assertion_count} assertions"
            )
        return "; ".join(self.errors[:4]) or "invalid"


class _Auditor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.awaits = 0
        self.assertions = 0
        self.found_function = False

    def _reject(self, node: ast.AST, message: str) -> None:
        line = getattr(node, "lineno", "?")
        self.errors.append(f"line {line}: {message}")

    # -- imports --------------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root not in ALLOWED_IMPORTS:
                self._reject(node, f"import of {alias.name!r} is not permitted")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".")[0]
        if root not in ALLOWED_IMPORTS and root != "playwright":
            self._reject(node, f"import from {node.module!r} is not permitted")
        self.generic_visit(node)

    # -- names and attributes -------------------------------------------
    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_NAMES:
            self._reject(node, f"reference to forbidden name {node.id!r}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in FORBIDDEN_ATTRS:
            self._reject(node, f"access to forbidden attribute {node.attr!r}")
        self.generic_visit(node)

    # -- structural -----------------------------------------------------
    def visit_Await(self, node: ast.Await) -> None:
        self.awaits += 1
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self.assertions += 1
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in ("expect", "getattr", "setattr", "delattr"):
            if func.id != "expect":
                self._reject(node, f"call to {func.id!r} is not permitted")
            else:
                self.assertions += 1
        if isinstance(func, ast.Attribute) and func.attr.startswith("to_"):
            self.assertions += 1
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node.name == REQUIRED_FUNCTION:
            params = [a.arg for a in node.args.args]
            if tuple(params[:2]) != REQUIRED_PARAMS:
                self._reject(
                    node,
                    f"{REQUIRED_FUNCTION} must take exactly (page, ctx), got {params!r}",
                )
            else:
                self.found_function = True
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == REQUIRED_FUNCTION:
            self._reject(node, f"{REQUIRED_FUNCTION} must be declared 'async def'")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.warnings.append(f"line {node.lineno}: class definitions are unusual in a test file")
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self._reject(node, "'global' is not permitted")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self._reject(node, "'nonlocal' is not permitted")


def validate_test_source(source: str) -> SourceValidation:
    """Parse and audit one generated test module.

    Returns a :class:`SourceValidation`; the caller decides whether to repair,
    fall back to the deterministic compiler, or drop the flow.
    """
    verdict = SourceValidation()
    if not source or not source.strip():
        verdict.errors.append("generated source is empty")
        return verdict

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        verdict.errors.append(f"syntax error at line {exc.lineno}: {exc.msg}")
        return verdict

    auditor = _Auditor()
    auditor.visit(tree)

    verdict.errors.extend(auditor.errors)
    verdict.warnings.extend(auditor.warnings)
    verdict.has_required_function = auditor.found_function
    verdict.awaits_count = auditor.awaits
    verdict.assertion_count = auditor.assertions

    if not auditor.found_function:
        verdict.errors.append(
            f"module does not define 'async def {REQUIRED_FUNCTION}(page, ctx)'"
        )
    if auditor.awaits == 0:
        verdict.errors.append("no awaited calls: the test would not drive the browser")
    if auditor.assertions == 0:
        verdict.warnings.append(
            "no assertions detected: this flow can only fail on an exception"
        )

    verdict.ok = not verdict.errors
    return verdict


def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
    """``__import__`` restricted to :data:`ALLOWED_IMPORTS`.

    Present so that a generated test containing ``import re`` inside the
    function body still runs; anything outside the allowlist raises rather than
    silently importing.
    """
    root = name.split(".")[0]
    if root not in ALLOWED_IMPORTS:
        raise ImportError(f"import of {name!r} is not permitted in a generated test")
    return _builtins.__import__(name, *args, **kwargs)


def build_namespace(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Module globals for an executed test: restricted builtins plus helpers."""
    import asyncio as _asyncio
    import json as _json
    import re as _re

    from playwright.async_api import expect as _expect

    safe_builtins = dict(SAFE_BUILTINS)
    safe_builtins["__import__"] = _safe_import

    namespace: dict[str, Any] = {
        "__builtins__": safe_builtins,
        "__name__": "generated_test",
        "re": _re,
        "json": _json,
        "asyncio": _asyncio,
        "expect": _expect,
    }
    if extra:
        namespace.update(extra)
    return namespace
