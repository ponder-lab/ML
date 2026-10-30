# wala/ML#1117: a destructuring read lowers to the same kind of instruction as an attribute read
# and a subscript read; each records which syntax it came from.


def reads(x, pair, seq, nested, k):
    a, b = pair
    [c, d] = pair
    e = x[0]
    f = x[k]
    g = x.shape
    for h, i in seq:
        pass
    for j in seq:
        pass
    m, (n, o) = nested
    return a, b, c, d, e, f, g, h, i, j, m, n, o


class X(list):
    shape = 7


assert reads(X([1, 2]), (3, 4), [(5, 6)], (8, (9, 10)), 1) == (
    3,
    4,
    3,
    4,
    1,
    2,
    7,
    5,
    6,
    (5, 6),
    8,
    9,
    10,
)
