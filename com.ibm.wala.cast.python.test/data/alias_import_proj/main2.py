# Test wala/ML#1019: the shadowing case, with no other binding of the name `mod` in this module: Python's
# `from shadow import mod` takes the package attribute `shadow/__init__.py` binds, (7, 7), and never
# loads the submodule `shadow/mod.py`, (9, 9).
from shadow import mod


def consume_shadow_direct(t):
    assert t.shape == (7, 7)
    return t


consume_shadow_direct(mod.make())
