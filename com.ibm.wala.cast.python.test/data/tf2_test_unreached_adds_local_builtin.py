# A function the call graph does not reach may still run. It adds to a dict it builds itself by
# calling `dict`, which only a variable of another function's frame binds to a function of the
# program's own.
REGISTRY = {"a": 1}


def registry():
    return REGISTRY


def make_lookup():
    dict = registry

    def lookup(key):
        return dict()[key]

    return lookup


def consume(c):
    pass


def tally(tokens):
    counts = dict()
    counts.update(tokens)
    return counts


assert make_lookup()("a") == 1
c = [1, 2]
assert len(c) == 2 and c[0] == 1
consume(c)
