# A function the call graph does not reach may still run. Its `update` on what a module function
# other than a builtin returns may add an element to any dict.
SHARED = {"a": 1, "b": 2}


def defaults():
    return SHARED


def consume(c):
    pass


def grow(source):
    defaults().update(source)


assert len(SHARED) == 2 and SHARED["a"] == 1
consume(SHARED)
