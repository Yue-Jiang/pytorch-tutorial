"""Shared utilities for the day-by-day checkers. You don't need to edit this."""
import inspect


def _source_without_doc(obj) -> str:
    """Source of obj with its docstring removed — docstrings may legally
    NAME banned APIs (e.g. 'Banned: torch.gather'); only code is checked."""
    src = inspect.getsource(obj)
    doc = getattr(obj, "__doc__", None)
    if doc:
        src = src.replace(doc, "")
    return src


def assert_not_used(obj, banned, hint=""):
    """Fail if the source of `obj` (docstring excluded) mentions any banned
    API string."""
    src = _source_without_doc(obj)
    for name in banned:
        assert name not in src, (
            f"Your implementation of `{obj.__name__}` uses the banned API "
            f"`{name}`. The whole point is to build it yourself! {hint}"
        )


def assert_used(obj, required, hint=""):
    """Fail unless the source of `obj` mentions a required API string."""
    src = inspect.getsource(obj)
    for name in required:
        assert name in src, (
            f"`{obj.__name__}` must be implemented with `{name}`. {hint}"
        )
