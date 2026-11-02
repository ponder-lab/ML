# A function the call graph does not reach may still run. It adds to a dict it builds itself by
# calling the `dict` builtin through a `global` declaration, which the module does not rebind.


def consume(c):
    pass


def tally(tokens):
    global dict
    counts = dict()
    counts.update(tokens)
    return counts


c = [1, 2]
assert len(c) == 2 and c[0] == 1
consume(c)
