# A function the call graph does not reach, but the program names, may still run, and its parameter
# may be any container, so its `update` may add an element to any dict (wala/ML#1255).


def consume(c):
    assert len(c) == 2 and c["a"] == 1


def merge(target, source):
    target.update(source)
    return target


handlers = [merge]

consume({"a": 1, "b": 2})
