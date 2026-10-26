# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""One implementation: the student stubs mirror the pipeline, and nothing else is copied.

The pipeline (``vla_pipeline/``) is the only implementation and the answer
key. ``workshop/project/`` holds one stub per TODO in
``vvla_workshop.feedback.TODO_INDEX``, with the name and parameters of the
pipeline's finished function (a method TODO is a method of a subclass of
the pipeline's class). That stub is the only duplicate allowed:

- every TODO stub has its reference's signature;
- no top-level function, class or constant is defined twice: the same name,
  at least 75 % alike once docstrings are set aside (the rip-out baseline's
  measure);
- no file is a byte-for-byte copy of another.
"""

from __future__ import annotations

import ast
import collections
import difflib
import hashlib
import subprocess
from pathlib import Path

import pytest

from vla_pipeline.utils.config import REPO_ROOT as ROOT

pytestmark = pytest.mark.sim

SCANNED = ("vla_pipeline", "workshop", "so101_sim/src/so101_sim", "scripts", "tests", "arm_calibrator", "launch")
SIMILAR = 0.75


def _todo_index() -> dict:
    from vvla_workshop.feedback import TODO_INDEX

    return TODO_INDEX


def _definition(path: Path, what: str) -> ast.FunctionDef:
    """``what`` (``name`` or ``Class.method``) in ``path``, found in its syntax tree."""
    body = ast.parse(path.read_text(encoding="utf-8")).body
    for part in what.split("."):
        node = next((n for n in body if getattr(n, "name", None) == part), None)
        assert node is not None, f"{path.relative_to(ROOT)} does not define {what}"
        body = getattr(node, "body", [])
    return node


def _parameters(fn: ast.FunctionDef) -> list:
    """Each parameter's kind, name and default; annotations set aside."""
    a = fn.args
    positional = [*a.posonlyargs, *a.args]
    defaults = [None] * (len(positional) - len(a.defaults)) + list(a.defaults)
    out = [("positional", p.arg, ast.dump(d) if d else None) for p, d in zip(positional, defaults)]
    out += [("*", a.vararg.arg, None)] if a.vararg else []
    out += [("keyword", p.arg, ast.dump(d) if d else None) for p, d in zip(a.kwonlyargs, a.kw_defaults)]
    out += [("**", a.kwarg.arg, None)] if a.kwarg else []
    return out


@pytest.mark.parametrize("todo_id", sorted(_todo_index(), key=lambda t: [int(x) for x in t.split(".")]))
def test_every_todo_stub_has_its_references_signature(todo_id):
    where, what, reference, _notebook = _todo_index()[todo_id]
    stub = _definition(ROOT / "workshop" / "project" / where, what)
    finished = _definition(ROOT / reference, what)
    assert _parameters(stub) == _parameters(finished), (
        f"TODO {todo_id}: {what} in project/{where} takes {_parameters(stub)}, "
        f"the pipeline's in {reference} takes {_parameters(finished)}"
    )


def _python_files() -> list[Path]:
    listed = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "--", *SCANNED],
                            cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    return [ROOT / f for f in listed if f.endswith(".py") and (ROOT / f).is_file()]


def _without_docstrings(node) -> str:
    node = ast.parse(ast.unparse(node)).body[0]
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body \
                and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant) \
                and isinstance(n.body[0].value.value, str):
            n.body = n.body[1:] or [ast.Pass()]
    return ast.unparse(node)


def _stub_names() -> set:
    """The project files' TODO definitions (a method TODO's class included): the allowed duplicates."""
    return {(f"workshop/project/{where}", part) for where, what, _ref, _nb in _todo_index().values()
            for part in what.split(".")}


def test_no_definition_is_written_twice():
    defs = collections.defaultdict(list)
    for path in _python_files():
        rel = str(path.relative_to(ROOT))
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = node.name
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                    and node.targets[0].id.isupper():
                name = node.targets[0].id
            else:
                continue
            defs[name].append((rel, _without_docstrings(node), node.end_lineno - node.lineno + 1))
    allowed = _stub_names()
    copies = []
    for name, found in defs.items():
        for i, (a, text_a, lines_a) in enumerate(found):
            for b, text_b, lines_b in found[i + 1:]:
                if a == b or (a, name) in allowed or (b, name) in allowed:
                    continue
                ratio = difflib.SequenceMatcher(None, text_a, text_b).ratio()
                if ratio >= SIMILAR:
                    copies.append((min(lines_a, lines_b), f"{name} ({ratio:.0%} alike): {a} and {b}"))
    total = sum(lines for lines, _ in copies)
    assert not copies, f"{total} duplicated lines; keep one and import it:\n" + "\n".join(
        line for _lines, line in sorted(copies, reverse=True)
    )


def test_no_file_is_a_copy():
    listed = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.split()
    by_content = collections.defaultdict(list)
    for rel in listed:
        path = ROOT / rel
        if not path.is_file():
            continue
        data = path.read_bytes()
        if data.count(b"\n") >= 5:
            by_content[hashlib.sha256(data).hexdigest()].append(rel)
    copies = [names for names in by_content.values() if len(names) > 1]
    assert not copies, "files with the same bytes; keep one:\n" + "\n".join(", ".join(c) for c in copies)


def test_the_twin_defaults_are_the_pipelines_arm():
    """The twin's own rest pose and jaw closed point, for a world built without the pipeline's."""
    from so101_sim.config import DEFAULTS
    from vla_pipeline.robot.arm_interface import GRIPPER_CLOSED_FLOOR, REST_POSE

    assert DEFAULTS["rest_pose"] == REST_POSE
    assert DEFAULTS["gripper_map"]["closed_cal"] == GRIPPER_CLOSED_FLOOR
