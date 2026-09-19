"""A structural tripwire over the package's own source.

nnsight ships a traced block as source plus every name the block loads, each
pickled whole. An attribute is not a name — `self.document.model` ships `self` —
so the rule that keeps a remote run small and a local run honest is about the
`ast.Name` nodes of the block, and it is checkable without running anything.

Two properties are pinned here:

1. a trace body loads only its own function's data: parameters, names the block
   binds, and module-level names of the file it lives in;
2. a trace body never reaches the client side of the project — no document, no
   dataset, no tokenizer, no `self`. The block gets the plan and the model.

`cls` is allowed where `self` is not: an engine is a stateless class, and a
class pickles by reference out of a registered package, where an instance
would drag its attributes along with it.
"""

import ast
import pathlib

import pytest

import causalab_mini

PACKAGE = pathlib.Path(causalab_mini.__file__).parent
RUN_METHODS = {"trace", "session", "generate"}

# The client side: the modules that turn documents into plans and plans into
# files. A block that loads one of them has a client-side object in it.
# `plan` is deliberately absent — the plan is pure data and is the one thing a
# block is meant to carry; `address`, `intervene` and `metrics` are block-side code.
CLIENT_SIDE = {"document", "build", "rows", "encoding", "output", "cli"}


def _is_block(node):
    return isinstance(node, ast.With) and any(
        isinstance(item.context_expr, ast.Call)
        and isinstance(item.context_expr.func, ast.Attribute)
        and item.context_expr.func.attr in RUN_METHODS
        for item in node.items
    )


def _blocks():
    """Every `with ….trace(/.session(` block in the package, with its file,
    its enclosing function, and the module it sits in."""
    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(function):
                if _is_block(node):
                    found.append((f"{path.parent.name}/{path.name}", tree, function, node))
    return found


def _bound(node):
    """Every name `node` binds: assignments, for targets, with-as, imports."""
    names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            names.add(child.id)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(child.name)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            for alias in child.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(child, ast.arg):
            names.add(child.arg)
    return names


def _loaded(block):
    return {
        node.id
        for statement in block.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def test_there_are_blocks_to_check():
    methods = {
        item.context_expr.func.attr
        for _, _, _, block in _blocks()
        for item in block.items
    }
    assert {"trace", "session"} <= methods, "the tripwire is vacuous"


@pytest.mark.parametrize("case", _blocks(), ids=lambda case: f"{case[0]}:{case[2].name}")
def test_a_trace_body_loads_only_its_functions_own_data(case):
    filename, tree, function, block = case
    allowed = _bound(function) | {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    allowed |= {
        alias.asname or alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    allowed |= set(dir(__builtins__)) | set(__builtins__.keys() if isinstance(__builtins__, dict) else dir(__builtins__))
    leaked = _loaded(block) - allowed
    assert not leaked, f"{filename}:{function.name} loads {sorted(leaked)} inside a trace body"


@pytest.mark.parametrize("case", _blocks(), ids=lambda case: f"{case[0]}:{case[2].name}")
def test_a_trace_body_never_reaches_the_client_side(case):
    filename, _tree, function, block = case
    forbidden = _loaded(block) & (CLIENT_SIDE | {"self", "tokenizer", "document"})
    assert not forbidden, (
        f"{filename}:{function.name} loads {sorted(forbidden)} inside a trace "
        "body — that object would ship whole"
    )
