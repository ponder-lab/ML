# A function the call graph does not reach may still run. Its `extend` on what a module function
# returns may add an element to any list (https://github.com/wala/ML/issues/1328).
def consume(c):
    assert len(c) == 2 and c[0] == 1


def load():
    return []


def merge(source):
    items = load()
    items.extend(source)
    return items


consume([1, 2])
