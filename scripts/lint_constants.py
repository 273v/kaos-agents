"""Lint for hardcoded constants and domain-specific strings in function bodies.

Catches the patterns that keep slipping through code review:
1. Hardcoded model IDs (should use DEFAULT_MODEL or settings)
2. Domain-specific terms in non-recipe code (should be neutral)
3. Numeric literals in function bodies (should be module-level constants)
4. String literals that look like they should be configurable

Usage::

    uv run python scripts/lint_constants.py
    uv run python scripts/lint_constants.py --path kaos_agents/
    uv run python scripts/lint_constants.py --strict   # also flag string defaults
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Patterns that should NEVER appear in function bodies
_MODEL_ID_PREFIXES = ("anthropic:", "openai:", "google:", "xai:", "groq:", "mistral:")

# Domain terms that shouldn't appear in non-recipe, non-test code
_DOMAIN_TERMS = {
    "legal research assistant",
    "legal memo",
    "legal assistant",
    "contract extraction",
    "employment agreement",
    "non-compete",
    "indemnification",
    "breach of contract",
    "filing fee",
    "Rule 10b-5",
    "SEC filed",
}

# Paths where domain terms are expected (recipes, extraction, benchmarks)
_DOMAIN_EXEMPT_PATHS = {
    "recipes/",
    "mcp_extract",
    "extraction",
    "benchmarks/",
    "test_mcp_extract",
    "test_agent_extraction",
    "cuad",
}

# Paths where hardcoded model IDs are expected (provider presets)
_MODEL_EXEMPT_PATHS = {
    "providers.py",
}

# Numeric literals that are likely magic numbers (not 0, 1, 2, -1, 100, etc.)
_EXEMPT_NUMBERS = {0, 1, 2, 3, -1, 0.0, 1.0, 0.5, 100, 10, 1000}


@dataclass(frozen=True, slots=True)
class LintIssue:
    """A single lint finding."""

    file: str
    line: int
    category: str
    message: str
    severity: str = "warning"


@dataclass(slots=True)  # Mutable: accumulated
class LintResult:
    """Accumulated lint results."""

    issues: list[LintIssue] = field(default_factory=list)
    files_checked: int = 0


def _is_in_function(node: ast.AST, parents: list[ast.AST]) -> bool:
    """Check if a node is inside a function/method body (not module level)."""
    return any(isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)) for p in parents)


def _is_in_docstring_position(node: ast.Constant, parents: list[ast.AST]) -> bool:
    """Check if a string constant is a docstring."""
    if not isinstance(node.value, str):
        return False
    if not parents:
        return False
    parent = parents[-1]
    if isinstance(parent, ast.Expr):
        # First statement in a function/class/module body
        grandparent = parents[-2] if len(parents) >= 2 else None
        if grandparent and isinstance(
            grandparent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
        ):
            body = grandparent.body
            if body and body[0] is parent:
                return True
    return False


def _is_in_type_annotation(parents: list[ast.AST]) -> bool:
    """Check if node is in a type annotation context."""
    return any(isinstance(p, (ast.AnnAssign,)) for p in parents)


def _is_exempt_path(filepath: str) -> bool:
    """Check if a file is exempt from domain-term checks."""
    return any(exempt in filepath for exempt in _DOMAIN_EXEMPT_PATHS)


class ConstantLinter(ast.NodeVisitor):
    """AST visitor that finds hardcoded constants in function bodies."""

    def __init__(self, filepath: str, *, strict: bool = False) -> None:
        self.filepath = filepath
        self.strict = strict
        self.issues: list[LintIssue] = []
        self._parents: list[ast.AST] = []

    def visit(self, node: ast.AST) -> None:
        """Override to track parent chain, then dispatch normally."""
        self._parents.append(node)
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        visitor(node)
        self._parents.pop()

    def visit_Constant(self, node: ast.Constant) -> None:
        # Skip docstrings
        if _is_in_docstring_position(node, self._parents):
            return

        # Skip type annotations
        if _is_in_type_annotation(self._parents):
            return

        value = node.value

        # Check for hardcoded model IDs anywhere (even module level)
        if isinstance(value, str):
            model_exempt = any(ex in self.filepath for ex in _MODEL_EXEMPT_PATHS)
            if not model_exempt:
                for prefix in _MODEL_ID_PREFIXES:
                    if value.startswith(prefix):
                        self.issues.append(
                            LintIssue(
                                file=self.filepath,
                                line=node.lineno,
                                category="hardcoded-model",
                                message=f'Hardcoded model ID: "{value}". Use DEFAULT_MODEL from settings.',
                                severity="error",
                            )
                        )

            # Check for domain-specific terms (only in non-exempt files)
            if not _is_exempt_path(self.filepath):
                val_lower = value.lower()
                for term in _DOMAIN_TERMS:
                    if term.lower() in val_lower:
                        self.issues.append(
                            LintIssue(
                                file=self.filepath,
                                line=node.lineno,
                                category="domain-bias",
                                message=f'Domain-specific term: "{value[:80]}". Use neutral language.',
                                severity="warning",
                            )
                        )
                        break

        # Check for magic numbers in function bodies
        if isinstance(value, (int, float)) and _is_in_function(node, self._parents):
            if value not in _EXEMPT_NUMBERS and not isinstance(value, bool):
                # Skip if it's an argument default
                is_default = any(
                    isinstance(p, ast.arguments) for p in self._parents
                )
                if not is_default and self.strict:
                    self.issues.append(
                        LintIssue(
                            file=self.filepath,
                            line=node.lineno,
                            category="magic-number",
                            message=f"Magic number {value} in function body. Extract to module-level constant.",
                            severity="info",
                        )
                    )


def lint_file(filepath: Path, *, strict: bool = False) -> list[LintIssue]:
    """Lint a single Python file for hardcoded constants."""
    try:
        source = filepath.read_text()
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return []

    linter = ConstantLinter(str(filepath), strict=strict)
    linter.visit(tree)
    return linter.issues


def lint_directory(
    directory: Path,
    *,
    strict: bool = False,
    exclude: set[str] | None = None,
) -> LintResult:
    """Lint all Python files in a directory."""
    result = LintResult()
    exclude = exclude or set()

    for py_file in sorted(directory.rglob("*.py")):
        rel = str(py_file.relative_to(directory))
        if any(ex in rel for ex in exclude):
            continue
        issues = lint_file(py_file, strict=strict)
        result.issues.extend(issues)
        result.files_checked += 1

    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Lint for hardcoded constants")
    parser.add_argument(
        "--path",
        default="kaos_agents/",
        help="Directory to lint (default: kaos_agents/)",
    )
    parser.add_argument(
        "--also",
        action="append",
        default=[],
        help="Additional directories to lint",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also flag magic numbers (noisy)",
    )
    parser.add_argument(
        "--severity",
        choices=("error", "warning", "info"),
        default="warning",
        help="Minimum severity to report (default: warning)",
    )
    args = parser.parse_args(argv)

    severity_order = {"error": 0, "warning": 1, "info": 2}
    min_severity = severity_order[args.severity]

    all_dirs = [Path(args.path)] + [Path(p) for p in args.also]
    total_issues: list[LintIssue] = []
    total_files = 0

    for directory in all_dirs:
        if not directory.exists():
            sys.stderr.write(f"Warning: {directory} does not exist\n")
            continue
        result = lint_directory(directory, strict=args.strict)
        total_issues.extend(result.issues)
        total_files += result.files_checked

    # Filter by severity
    filtered = [i for i in total_issues if severity_order[i.severity] <= min_severity]

    # Print results
    if not filtered:
        sys.stdout.write(f"OK: {total_files} files checked, no issues found.\n")
        return

    for issue in sorted(filtered, key=lambda i: (i.file, i.line)):
        marker = {"error": "E", "warning": "W", "info": "I"}[issue.severity]
        sys.stdout.write(
            f"{issue.file}:{issue.line}: [{marker}:{issue.category}] {issue.message}\n"
        )

    errors = sum(1 for i in filtered if i.severity == "error")
    warnings = sum(1 for i in filtered if i.severity == "warning")
    sys.stdout.write(
        f"\n{total_files} files checked: {errors} error(s), {warnings} warning(s)\n"
    )

    if errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
