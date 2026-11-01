import ast
from pathlib import Path


def load_segments():
    tree = ast.parse(Path("app.py").read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "segments"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {}
    exec(compile(module, "app.py", "exec"), namespace)
    return namespace["segments"]


segments = load_segments()


def test_segments_merges_consecutive_windows():
    assert segments([0, 3, 6], [True, True, False], 10) == [(0, 6.0)]


def test_segments_caps_end_at_duration():
    assert segments([9], [True], 10) == [(9, 10)]


def test_segments_returns_empty_when_none_present():
    assert segments([0, 3], [False, False], 10) == []
