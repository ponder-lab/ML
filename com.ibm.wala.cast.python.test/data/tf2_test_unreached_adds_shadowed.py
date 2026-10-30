# A function the call graph does not reach may still run. Here `dict` is the module's own function,
# which returns a shared dict, so its `update` may add an element to that dict.
SHARED = {"a": 1, "b": 2}


def dict():
    return SHARED


def consume(c):
    assert len(c) == 2 and c["a"] == 1


def grow(source):
    dict().update(source)


consume(SHARED)
