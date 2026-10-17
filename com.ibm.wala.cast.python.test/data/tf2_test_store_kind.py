# wala/ML#1100: an attribute store and a subscript store with the same string key lower to the
# same kind of instruction; each records which syntax it came from.


def stores(y, i):
    y.tag = 1.0
    y["tag"] = 2.0
    y[i] = 3.0
    y["n"] = 0.0
    y["n"] += 1.0
    a = y.tag
    b = y["tag"]
    c = y[i]
    return a, b, c


class Holder(dict):
    pass


h = Holder()
assert stores(h, "k") == (1.0, 2.0, 3.0)
assert h.tag == 1.0 and h["tag"] == 2.0 and h["k"] == 3.0 and h["n"] == 1.0
