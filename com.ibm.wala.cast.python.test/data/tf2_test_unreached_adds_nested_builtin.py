# A function the call graph does not reach may still run. Its `update` on a dict the `dict` builtin
# builds adds to that dict alone, though another function defines a `dict` of its own inside it,
# which binds the name in that function's scope only.


def consume(c):
    pass


def make_lookup():
    def dict(*args):
        return {}

    return dict


def count(tokens):
    counts = dict()
    counts.update({t: 1 for t in tokens})
    return counts


make_lookup()
values = [1, 2]
assert len(values) == 2 and values[0] == 1
consume(values)
