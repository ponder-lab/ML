# A function the call graph does not reach may still run. Its `update` on a global that holds a
# list cannot change the list, which has no such method (https://github.com/wala/ML/issues/1328).
ITEMS = [1, 2]


def consume(c):
    assert len(c) == 2 and c[0] == 1


def grow(source):
    ITEMS.update(source)


consume(ITEMS)
