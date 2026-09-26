# SPDX-License-Identifier: Apache-2.0
"""The console's JavaScript tests, as one suite.

They are plain `node --test` files. Node runs a `test(...)` opened inside another
one as a *nested* subtest: node 24 runs it to the end, node 22 cancels it as soon
as its parent returns — so a malformed file looks green on the machine that wrote
it and fails in CI. `tests/admin_settings_nav.test.cjs` did exactly that: a
missing `});` nested six tests and their assertions inside the test above them,
and the pull request's CI went red while the local run stayed green.

Everything the files define has to be top level, and the suite has to pass as one
`node --test tests/admin_*.test.cjs` run.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
JS_FILES = sorted(TESTS.glob("admin_*.test.cjs"))


def _test_calls(source: str) -> list:
    """`(line, brace depth)` for every `test(` call in `source`.

    Comments, strings and template literals are skipped, so a brace inside one of
    them does not count towards the depth.
    """
    found = []
    depth = 0
    line = 1
    index = 0
    length = len(source)
    while index < length:
        char = source[index]
        following = source[index + 1] if index + 1 < length else ""
        if char == "\n":
            line += 1
            index += 1
            continue
        if char == "/" and following == "/":
            while index < length and source[index] != "\n":
                index += 1
            continue
        if char == "/" and following == "*":
            index += 2
            while index < length and not (
                source[index] == "*" and index + 1 < length and source[index + 1] == "/"
            ):
                if source[index] == "\n":
                    line += 1
                index += 1
            index += 2
            continue
        if char in "'\"`":
            quote = char
            index += 1
            while index < length:
                if source[index] == "\\":
                    index += 2
                    continue
                if source[index] == quote:
                    index += 1
                    break
                if source[index] == "\n":
                    line += 1
                index += 1
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif source.startswith("test(", index) and (
            index == 0 or not re.match(r"[\w$.]", source[index - 1])
        ):
            found.append((line, depth))
            index += len("test(")
            continue
        index += 1
    return found


def test_the_js_test_files_exist():
    assert JS_FILES, "the console's node tests moved; this guard is watching nothing"


@pytest.mark.parametrize("path", JS_FILES, ids=lambda path: path.name)
def test_every_node_test_is_top_level(path):
    calls = _test_calls(path.read_text(encoding="utf-8"))
    # A file that drives node's test runner from somewhere else has nothing to
    # check here; every other one declares at least one test.
    nested = [(line, depth) for line, depth in calls if depth != 0]
    assert not nested, (
        f"{path.name}: a test is nested inside another one at lines "
        f"{[line for line, _ in nested]} — node closes the outer one first and "
        f"cancels the inner ones"
    )


def test_the_console_js_suite_passes_as_one_run():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the console's JavaScript tests")
    result = subprocess.run(
        [node, "--test", *[str(path) for path in JS_FILES]],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
