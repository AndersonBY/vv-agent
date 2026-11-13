"""Static W9 documentation and capability checks; see the capability matrix for syntax."""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path

MATRIX = "docs/session-kernel-capability-matrix.md"
CAPABILITY_ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
INLINE_CODE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
REPOSITORY_PATH = re.compile(r"(?:src|tests|scripts|docs|examples)/[\w./-]*\Z")
SYMBOL = re.compile(r"((?:src|tests|scripts)/[\w/-]+\.py)::([A-Za-z_]\w*(?:(?:::|\.)[A-Za-z_]\w*)*)\Z")
EXCLUDED_DOC_ROOTS = {"releases"}


def markdown_lines(text: str):
    """Yield non-fenced lines and preserve their source locations."""
    fence = None
    for number, line in enumerate(text.splitlines(), 1):
        delimiter = re.match(r"^\s{0,3}(`{3,}|~{3,})(.*)$", line)
        if delimiter:
            run, rest = delimiter.groups()
            if fence is None:
                if run[0] == "`" and "`" in rest:
                    yield number, line
                    continue
                fence = run
            elif run[0] == fence[0] and len(run) >= len(fence) and not rest.strip():
                fence = None
            continue
        if fence is None:
            yield number, line


def code_spans(text: str):
    for number, line in markdown_lines(text):
        for match in INLINE_CODE.finditer(line):
            yield number, match.group(1)


def definitions(body: list[ast.stmt], prefix: str = ""):
    """Lexical definitions only; never infer imports, aliases, attributes or runtime exports."""
    for node in body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            name = prefix + node.name
            yield name, node
            if isinstance(node, ast.ClassDef):
                yield from definitions(node.body, name + ".")


def check_kernel_boundaries(root: Path) -> list[str]:
    """Check every kernel source, including unimported files and conditional imports.

    Syntax boundary: imports, Name/Attribute calls, import aliases and simple
    name-to-name/attribute assignments. Effectful stdlib API families are
    conservatively rejected, including pure helpers within those families.
    No scope/control-flow inference, reflection, re-export tracing, unlisted
    API effect inference or arbitrary execution proof. This is a maintenance
    gate, not a sandbox. Only the two concrete storage adapters may open
    connections/create synchronous RLocks; PostgreSQL driver imports must be
    function-local or type-only. Raw connection, cursor and SQL execution
    names are limited to the exact storage implementations. Other core
    modules retain the pure boundary.
    """
    root = root.resolve()
    core = "vv_agent.kernel"
    effect_families = (
        "_thread",
        "threading",
        "concurrent.futures",
        "multiprocessing",
        "subprocess",
        "time",
        "os",
        "io",
        "pathlib",
        "shutil",
        "tempfile",
        "socket",
        "socketserver",
        "ssl",
        "urllib.request",
        "http.client",
        "http.server",
        "asyncio",
        "sqlite3",
        "psycopg",
    )
    effect_calls = {
        "open",
        "builtins.open",
        "print",
        "builtins.print",
        "sys.stdout.write",
        "sys.stdout.writelines",
        "sys.stderr.write",
        "sys.stderr.writelines",
        "datetime.datetime.now",
        "datetime.datetime.utcnow",
        "datetime.datetime.today",
        "datetime.date.today",
        "__import__",
        "builtins.__import__",
        "importlib.import_module",
        "eval",
        "exec",
        "builtins.eval",
        "builtins.exec",
    }
    storage_calls = {
        "src/vv_agent/kernel/_sqlite.py": {"sqlite3.connect", "threading.RLock"},
        "src/vv_agent/kernel/_postgres.py": {
            "threading.RLock",
            "psycopg.connect",
            "psycopg.conninfo.make_conninfo",
            "psycopg.conninfo.conninfo_to_dict",
        },
    }
    storage_owners = {
        "src/vv_agent/kernel/_sql.py",
        "src/vv_agent/kernel/_sqlite.py",
        "src/vv_agent/kernel/_postgres.py",
    }
    storage_api_names = {
        "connection",
        "_conn",
        "_one",
        "_rows",
        "_transaction",
        "execute",
        "executemany",
        "executescript",
        "transaction",
    }
    postgres_imports = {"psycopg", "psycopg.conninfo", "psycopg.rows", "psycopg.pq"}
    errors = []
    for file in sorted((root / "src/vv_agent/kernel").rglob("*.py")):
        path = file.relative_to(root).as_posix()
        if not file.resolve().is_relative_to(root):
            errors.append(f"{path}: kernel source escapes repository root")
            continue
        try:
            module = ast.parse(file.read_text(encoding="utf-8"), filename=path)
        except (SyntaxError, UnicodeError, OSError) as exc:
            errors.append(f"{path}: cannot parse kernel Python: {exc}")
            continue
        package = ".".join(file.parent.relative_to(root / "src").parts)
        parents = {child: node for node in ast.walk(module) for child in ast.iter_child_nodes(node)}

        def in_function(node: ast.AST, parents=parents) -> bool:
            while node in parents:
                child, node = node, parents[node]
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and child in node.body:
                    return True
            return False

        def type_only(node: ast.AST, parents=parents) -> bool:
            while node in parents:
                child, node = node, parents[node]
                if isinstance(node, ast.If) and name(node.test) == "typing.TYPE_CHECKING" and child in node.body:
                    return True
            return False

        bindings: dict[str, str] = {}

        def name(node: ast.AST, bindings=bindings) -> str:
            if isinstance(node, ast.Name):
                return bindings.get(node.id, node.id)
            if isinstance(node, ast.Attribute) and (parent := name(node.value)):
                return parent + "." + node.attr
            return ""

        def check_import(target: str, node: ast.Import | ast.ImportFrom, path=path) -> None:
            if target == core or target.startswith(core + "."):
                return
            if (
                path == "src/vv_agent/kernel/_postgres.py"
                and target in postgres_imports
                and (
                    in_function(node)
                    or (
                        isinstance(node, ast.ImportFrom)
                        and target == "psycopg"
                        and all(alias.name == "Connection" for alias in node.names)
                        and type_only(node)
                    )
                )
            ):
                return
            if target.split(".")[0] not in sys.stdlib_module_names:
                errors.append(f"{path}:{node.lineno}: forbidden kernel import: {target}")

        for node in sorted(ast.walk(module), key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0))):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    check_import(alias.name, node)
                    bindings[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
            elif isinstance(node, ast.ImportFrom):
                try:
                    target = importlib.util.resolve_name("." * node.level + (node.module or ""), package)
                except ImportError:
                    errors.append(f"{path}:{node.lineno}: kernel relative import escapes package")
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        errors.append(f"{path}:{node.lineno}: kernel wildcard import cannot be checked")
                    check_import(target + "." + alias.name if target == "vv_agent" else target, node)
                    bindings[alias.asname or alias.name] = target + "." + alias.name
            elif isinstance(node, ast.Assign) and (target := name(node.value)):
                for assigned in node.targets:
                    if isinstance(assigned, ast.Name):
                        bindings[assigned.id] = target
            elif isinstance(node, ast.Attribute):
                target = name(node)
                if path not in storage_owners and node.attr in storage_api_names:
                    errors.append(f"{path}:{node.lineno}: forbidden kernel storage API: {target}")
            elif isinstance(node, ast.Call):
                target = name(node.func)
                if path not in storage_owners and target.endswith(".cursor"):
                    errors.append(f"{path}:{node.lineno}: forbidden kernel storage API: {target}")
                if (
                    target in effect_calls
                    or any(target == family or target.startswith(family + ".") for family in effect_families)
                ) and not (target in storage_calls.get(path, set()) and in_function(node)):
                    errors.append(f"{path}:{node.lineno}: forbidden kernel effect call: {target}")
    return sorted(set(errors))


def check_repository(root: Path) -> list[str]:
    root = root.resolve()
    errors: list[str] = check_kernel_boundaries(root)
    trees: dict[str, ast.Module | None] = {}

    def error(where: str, message: str):
        errors.append(f"{where}: {message}")

    def path_exists(path: str, where: str) -> bool:
        target = root / path
        if not target.resolve().is_relative_to(root):
            error(where, f"repository path escapes root: {path}")
            return False
        if not target.exists():
            error(where, f"missing repository path: {path}")
            return False
        return True

    def tree(path: str, where: str):
        if path not in trees:
            trees[path] = None
            if path_exists(path, where):
                try:
                    trees[path] = ast.parse((root / path).read_text(encoding="utf-8"), filename=path)
                except (SyntaxError, UnicodeError, OSError) as exc:
                    error(where, f"cannot parse Python file {path}: {exc}")
        return trees[path]

    tests: dict[str, set[str]] = {}
    marked: set[tuple[str, str]] = set()
    for file in sorted((root / "tests").rglob("test_*.py")):
        path = file.relative_to(root).as_posix()
        module = tree(path, path)
        if module is None:
            continue
        accepted_marks = set()
        for name, node in definitions(module.body):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            parts = name.split(".")
            if not parts[-1].startswith("test_") or any(not part.startswith("Test") for part in parts[:-1]):
                continue
            reference = path + "::" + name.replace(".", "::")
            if reference in tests:
                error(f"{path}:{node.lineno}", f"duplicate test definition: {reference}")
            tests[reference] = set()
            for decorator in node.decorator_list:
                call = decorator.func if isinstance(decorator, ast.Call) else decorator
                if ast.unparse(call) != "pytest.mark.capability":
                    continue
                accepted_marks.add(id(call))
                where = f"{path}:{decorator.lineno}"
                if (
                    not isinstance(decorator, ast.Call)
                    or len(decorator.args) != 1
                    or decorator.keywords
                    or not isinstance(decorator.args[0], ast.Constant)
                    or not isinstance(decorator.args[0].value, str)
                    or not CAPABILITY_ID.fullmatch(decorator.args[0].value)
                ):
                    error(where, 'capability requires one literal kebab-case ID: @pytest.mark.capability("id")')
                    continue
                capability = decorator.args[0].value
                if capability in tests[reference]:
                    error(where, f"duplicate capability marker: {capability}")
                tests[reference].add(capability)
                marked.add((capability, reference))
        for node in ast.walk(module):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "capability"
                and ast.unparse(node) == "pytest.mark.capability"
                and id(node) not in accepted_marks
            ):
                error(f"{path}:{node.lineno}", "capability marks must be direct decorators on test functions")

    expected: set[tuple[str, str]] = set()
    matrix_ids: set[str] = set()
    matrix = root / MATRIX
    if path_exists(MATRIX, MATRIX):
        columns = None
        for number, line in markdown_lines(matrix.read_text(encoding="utf-8")):
            if not line.startswith("|"):
                columns = None
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if cells[0] == "Capability ID" and "Tests" in cells:
                columns = cells.index("Tests"), len(cells)
                continue
            if columns is None or all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                continue
            where = f"{MATRIX}:{number}"
            if len(cells) != columns[1]:
                error(where, "malformed capability row")
                continue
            capability = cells[0].strip("`")
            if not CAPABILITY_ID.fullmatch(capability):
                error(where, f"invalid capability ID: {capability}")
                continue
            if capability in matrix_ids:
                error(where, f"duplicate capability ID: {capability}")
            matrix_ids.add(capability)
            references = [value for _, value in code_spans(cells[columns[0]]) if value.startswith("tests/") and "::" in value]
            if not references:
                error(where, f"capability has no explicit test nodes: {capability}")
            for reference in references:
                pair = capability, reference
                if pair in expected:
                    error(where, f"duplicate capability test reference: {capability} -> {reference}")
                expected.add(pair)
                if reference not in tests:
                    error(where, f"missing test definition: {reference}")
                elif capability not in tests[reference]:
                    error(where, f"missing capability marker: {capability} -> {reference}")
        if not matrix_ids:
            error(MATRIX, "no capability rows found (expected a Capability ID / Tests table)")
    for capability, reference in sorted(marked - expected):
        error(reference, f"capability marker absent from matrix: {capability}")

    for file in sorted((root / "docs").rglob("*.md")):
        if file.relative_to(root / "docs").parts[0] in EXCLUDED_DOC_ROOTS:
            continue
        path = file.relative_to(root).as_posix()
        for number, value in code_spans(file.read_text(encoding="utf-8")):
            where = f"{path}:{number}"
            if value.startswith("py:") or (value.startswith("tests/") and "::" in value):
                target = value.removeprefix("py:")
                match = SYMBOL.fullmatch(target)
                if match is None:
                    error(where, f"invalid Python reference: {value}")
                    continue
                source, symbol = match.groups()
                module = tree(source, where)
                if module is not None and symbol.replace("::", ".") not in dict(definitions(module.body)):
                    error(where, f"missing Python definition: {value}")
            elif value.startswith("repo:"):
                target = value.removeprefix("repo:")
                if not target or Path(target).is_absolute() or any(c in target for c in " *?[]<>\\"):
                    error(where, f"invalid repository reference: {value}")
                else:
                    path_exists(target, where)
            elif REPOSITORY_PATH.fullmatch(value):
                path_exists(value, where)
            elif path == MATRIX and re.fullmatch(r"test_\w+(?:\[.*\])?", value):
                error(where, f"matrix test name needs an explicit tests/path.py::test_name: {value}")
    return sorted(set(errors))


def main() -> int:
    errors = check_repository(Path(__file__).resolve().parents[1])
    for message in errors:
        print(message)
    print(f"Repository contracts: {len(errors)} violation(s).")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
