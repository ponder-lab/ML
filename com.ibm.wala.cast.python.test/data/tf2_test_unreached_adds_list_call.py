# A function the call graph does not reach may still run. Its `update` on what a module function
# returns may add an element to any dict or set, but no list has the method, so it changes no list
# (https://github.com/wala/ML/issues/1328).
def consume(c):
    assert len(c) == 2 and c[0] == 1


def load():
    return {}


def merge(source):
    config = load()
    config.update(source)
    return config


consume([1, 2])
